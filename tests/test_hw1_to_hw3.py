# ==============================================================================
# 🌍🎯📝 TESTS FOR HW 1, HW 2 & HW 3
# ==============================================================================
"""
Перевірка критеріїв прийняття:
- ДЗ 1: Вірш на 6 рядків, чистота української мови, інверсія полярності Так<->Ні,
        математика семплінгу (T, Top-P, Top-K) та експорт звіту Google Docs.
- ДЗ 2: Усі 3 завдання (50 + 25 + 25 = 100 балів), нормалізація Atlas Pro -> Atlas,
        ліміт заголовка <= 8 слів, LCEL RunnableBranch/Parallel.
- ДЗ 3: TODO-агент на create_agent + InMemoryStore, двокрокове видалення (list_tasks -> delete_task),
        limit=1000, store.get() для неіснуючих ID та HITL-симуляція.
"""

from pathlib import Path
import numpy as np
import pytest

from hw_01_model_alignment import (
    HW1BenchmarkRunner,
    analyze_bpe_tokenization_fertility,
    compute_sampling_distribution,
    evaluate_ukrainian_poem,
    export_hw1_jupyter_notebook,
    generate_google_docs_markdown,
)
from hw_02_context_triage import (
    ContextTriageExtractor,
    build_parallel_summary_chain,
    export_hw2_jupyter_notebook,
    simulate_typeddict_stream,
)
from hw_03_todo_agent import (
    TASKS_NAMESPACE,
    TodoAgentOrchestrator,
    build_todo_agent,
    export_hw3_jupyter_notebook,
    simulate_hitl_delete_approval,
)


# ------------------------------------------------------------------------------
# ДЗ 1: Pre-training, Alignment, Sampling & Base vs Instruct
# ------------------------------------------------------------------------------
def test_hw1_poem_evaluator_and_alignment_matrix(isolated_submission_dir: Path) -> None:
    runner = HW1BenchmarkRunner(preferred_tier="reference")
    poem_evals = runner.run_task1_poem_evaluations()
    assert len(poem_evals) == 4

    # Перевіряємо детектор порушень на первинній відповіді меншої моделі та виправлення після репромпту
    qwen_initial = poem_evals[0]
    qwen_fixed = poem_evals[1]
    assert qwen_initial.exact_six_lines is False
    assert qwen_initial.pure_ukrainian is False
    assert qwen_fixed.exact_six_lines is True
    assert qwen_fixed.pure_ukrainian is True

    # Перевіряємо матрицю ціннісних питань (пряме, перефразоване, інверсія Так <-> Ні)
    alignments = runner.run_task2_alignment_matrix()
    assert len(alignments) == 6
    assert all(rec.polarity_consistent for rec in alignments)

    # Перевірка генерації звіту для Google Docs та ноутбука
    md_path = isolated_submission_dir / "Hatsenko_Oleh_DZ1.md"
    md_content = generate_google_docs_markdown(str(md_path))
    assert md_path.exists()
    assert "Задача 1" in md_content and "Задача 2" in md_content and "Задача 3" in md_content

    nb_path = export_hw1_jupyter_notebook(str(isolated_submission_dir / "oleh_hatsenko_hw1.ipynb"))
    assert nb_path.exists()


def test_hw1_sampling_math_and_tokenization() -> None:
    df_samp = compute_sampling_distribution(temperature=0.7, top_p=0.9, top_k=4)
    final_probs = df_samp["Фінальна P(x) після ренормалізації"].to_numpy()
    assert pytest.approx(float(np.sum(final_probs)), abs=1e-3) == 1.0
    assert int(np.sum(final_probs > 0)) <= 4

    df_tok = analyze_bpe_tokenization_fertility()
    assert len(df_tok) == 5


# ------------------------------------------------------------------------------
# ДЗ 2: Context Engineering & Strict JSON Triage (100/100 балів)
# ------------------------------------------------------------------------------
def test_hw2_all_three_tasks_and_lcel(isolated_submission_dir: Path) -> None:
    extractor = ContextTriageExtractor(preferred_tier="reference")
    runs = extractor.run_all_tasks()

    assert set(runs.keys()) == {"task1", "task2", "task3"}
    for task_key in ("task1", "task2", "task3"):
        records = runs[task_key]
        assert len(records) == 5
        assert all(r.is_valid_contract for r in records)
        assert all(r.is_exact_match for r in records)

    # Перевірка нормалізації "Atlas Pro" -> "Atlas" у Завданні 2
    assert runs["task2"][0].parsed_data["product"] == "Atlas"
    # Перевірка ліміту <= 8 слів у заголовках Завдання 3
    assert all(len(r.parsed_data["title"].split()) <= 8 for r in runs["task3"])

    # Перевірка LCEL RunnableParallel + RunnableBranch та TypedDict стримінгу
    parallel_chain = build_parallel_summary_chain()
    routed = parallel_chain.invoke({"text": "Atlas показує старі курси валют"})
    assert routed["routed"]["branch"] == "Task 2: Product + Priority"
    assert routed["stats"]["words"] == 5

    stream_df = simulate_typeddict_stream({"title": "Реліз версії 2.0", "category": "реліз"})
    assert len(stream_df) == 5

    nb_path = export_hw2_jupyter_notebook(runs, str(isolated_submission_dir / "oleh_hatsenko_hw2.ipynb"))
    assert nb_path.exists()


# ------------------------------------------------------------------------------
# ДЗ 3: TODO-Агент на LangChain v1 (create_agent) + InMemoryStore
# ------------------------------------------------------------------------------
def test_hw3_todo_agent_canonical_and_edge_cases(isolated_submission_dir: Path) -> None:
    orchestrator = TodoAgentOrchestrator(preferred_tier="reference")
    logs = orchestrator.run_canonical_suite()

    assert len(logs) == 5
    # Тест 1 і 2: додано 2 завдання
    assert logs[0].store_snapshot_count == 1
    assert logs[1].store_snapshot_count == 2
    # Тест 3: перегляд списку показує 2 завдання з 8-символьними ID
    assert "купити хліб" in logs[2].final_response
    assert "подзвонити лікарю" in logs[2].final_response
    # Тест 4 («Видали завдання про хліб»): обов'язковий двокроковий виклик list_tasks -> delete_task
    assert len(logs[3].tool_calls_chain) == 2
    assert logs[3].tool_calls_chain[0].startswith("list_tasks")
    assert logs[3].tool_calls_chain[1].startswith("delete_task")
    assert logs[3].store_snapshot_count == 1
    # Тест 5 («Що залишилось?»): залишилась 1 задача («подзвонити лікарю»)
    assert logs[4].store_snapshot_count == 1
    assert "лікарю" in logs[4].final_response.lower()

    # Перевірка edge-cases (порожній список, невідомий ID, limit=1000)
    edge_df = TodoAgentOrchestrator.run_edge_case_verification()
    assert all("✅ PASS" in status for status in edge_df["Статус"].to_list())

    # Перевірка масштабованості > 10 елементів у сховищі (limit=1000)
    agent, store, _, _ = build_todo_agent(preferred_tier="reference")
    for idx in range(15):
        store.put(TASKS_NAMESPACE, f"abcd12{idx:02d}", {"id": f"abcd12{idx:02d}", "title": f"Пункт {idx}"})
    res = agent.invoke({"messages": [{"role": "user", "content": "Покажи всі завдання"}]})
    assert "Пункт 14" in str(res["messages"][-1].content)

    hitl_df = simulate_hitl_delete_approval()
    assert len(hitl_df) == 3

    nb_path = export_hw3_jupyter_notebook(logs, str(isolated_submission_dir / "oleh_hatsenko_hw3.ipynb"))
    assert nb_path.exists()
