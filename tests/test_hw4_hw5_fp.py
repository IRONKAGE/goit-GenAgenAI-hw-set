# ==============================================================================
# 📈🛠️🌐 TESTS FOR HW 4 (RAG), HW 5 (LangGraph) & FINAL PROJECT (Excel Enricher)
# ==============================================================================
"""
Перевірка критеріїв прийняття:
- ДЗ 4: Рівно 4 чанки, ChromaDB (finansovi_zvity_kompaniy), k=2, точні цифри для Q1–Q5,
        захист від галюцинацій на Q6 («ЕкоПакування Солюшнс»), RAG-Fusion (RRF) та RAGAS.
- ДЗ 5: Послідовний StateGraph (4 вузли), 1–2 пункти FAQ, довжина <= 50 слів,
        TCR = 100%, Escalation Rate = 40% (запити #2 та #4), Dual Judge та Command/interrupt.
- ФП: Універсальний інтерфейс process_excel() на capitals.xlsx (10 рядків)
      та mountains.xlsx (10 та 4 рядки) з точністю ±10%.
"""

from pathlib import Path
import pandas as pd
import pytest

from fp_excel_enricher import (
    ExcelEnrichmentOrchestrator,
    export_fp_jupyter_notebook,
    process_excel,
)
from hw_04_financial_rag import (
    COLLECTION_NAME_HW4,
    FinancialRAGOrchestrator,
    TEST_QUESTIONS_HW4,
    compute_cosine_similarity_matrix,
    export_hw4_jupyter_notebook,
    run_advanced_rag_comparison,
    split_financial_reports,
)
from hw_05_support_langgraph import (
    SupportAgentBenchmark,
    export_hw5_jupyter_notebook,
    simulate_topic9_hitl_interrupt_resume,
    test_agent,
)


# ------------------------------------------------------------------------------
# ДЗ 4: RAG для фінансових звітів (ChromaDB + 4 чанки + k=2)
# ------------------------------------------------------------------------------
def test_hw4_financial_rag_pipeline(isolated_submission_dir: Path) -> None:
    chunks = split_financial_reports()
    assert len(chunks) == 4
    assert COLLECTION_NAME_HW4 == "finansovi_zvity_kompaniy"

    rag_orch = FinancialRAGOrchestrator(preferred_tier="reference")
    records = rag_orch.evaluate_all_questions()
    assert len(records) == 6
    assert all(r.retrieved_chunks_count == 2 for r in records)

    # Точні цифри для питань 1–5
    assert "450" in records[0].answer
    assert "5100" in records[1].answer
    assert "9.3" in records[2].answer
    assert "Львові" in records[3].answer
    assert "мінус 8" in records[4].answer

    # Питання 6: неіснуюча компанія «ЕкоПакування Солюшнс» не повинна отримати прибуток «ГрінЕнерджі Солюшнс» (156)
    assert "відсутня" in records[5].answer.lower()
    assert "156" not in records[5].answer
    assert records[5].hallucination_guard_triggered is True

    # Перевірка матриці косинусної схожості та просунутих трансформацій (Multi-Query, RRF, HyDE)
    sim_df = compute_cosine_similarity_matrix(TEST_QUESTIONS_HW4, chunks)
    assert len(sim_df) == 6

    adv_df = run_advanced_rag_comparison(rag_orch.retriever)
    assert len(adv_df) == 5

    nb_path = export_hw4_jupyter_notebook(
        chunks, records, str(isolated_submission_dir / "oleh_hatsenko_hw4.ipynb")
    )
    assert nb_path.exists()
    assert (isolated_submission_dir / "Oleh_Hatsenko_DZ7.ipynb").exists()


# ------------------------------------------------------------------------------
# ДЗ 5: Агент технічної підтримки на LangGraph v1 (100/100 балів)
# ------------------------------------------------------------------------------
def test_hw5_support_agent_graph_and_metrics(
    isolated_submission_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bench = SupportAgentBenchmark(preferred_tier="reference")
    rows = bench.run_benchmark()

    assert len(rows) == 5
    # 1. Task Completion Rate = 100% (5/5)
    assert all(bool(r.final_response) for r in rows)
    # 2. Оптимальна кількість FAQ: 1 або 2 пункти для кожного запиту (+5 балів у рубриці)
    assert all(1 <= r.faq_count <= 2 for r in rows)
    # 3. Ліміт довжини <= 50 слів для всіх відповідей (включно з ескальованими)
    assert all(r.within_50_words and r.word_count <= 50 for r in rows)
    # 4. Escalation Rate = 40% (рівно запити #2 та #4)
    escalated_nums = [r.test_num for r in rows if r.needs_escalation]
    assert escalated_nums == [2, 4]
    # 5. Резюме класифікації українською мовою
    assert all(any("а" <= ch.lower() <= "я" or ch.lower() in "іїєґ" for ch in r.summary_ua) for r in rows)
    # 6. Average Answer Relevancy >= 0.80
    avg_rel = sum(r.self_judge_score for r in rows) / len(rows)
    assert avg_rel >= 0.80

    # Перевірка канонічної консольної функції test_agent()
    raw_results = test_agent(preferred_tier="reference")
    captured = capsys.readouterr().out
    assert "Task Completion Rate: 100%" in captured
    assert "Escalation Rate: 2/5 (40%)" in captured
    assert len(raw_results) == 5

    # Перевірка двофазного сценарію Command + interrupt() з ноутбука Теми 9
    hitl_df = simulate_topic9_hitl_interrupt_resume()
    assert len(hitl_df) == 2

    nb_path = export_hw5_jupyter_notebook(str(isolated_submission_dir / "oleh_hatsenko_hw5.ipynb"))
    assert nb_path.exists()
    assert (isolated_submission_dir / "Oleh_Hatsenko_DZ9.ipynb").exists()


# ------------------------------------------------------------------------------
# Фінальний проект: Інтелектуальне збагачення даних з Excel (process_excel)
# ------------------------------------------------------------------------------
def test_final_project_excel_enrichment_both_datasets(
    sample_excel_workspace: dict[str, Path],
    isolated_submission_dir: Path,
    isolated_cache_paths: dict[str, Path],
) -> None:
    # 1. Тестування через фасад process_excel() на capitals.xlsx (10 рядків)
    out_capitals_path = isolated_submission_dir / "capitals_enriched.xlsx"
    saved_cap = process_excel(
        file_path=str(sample_excel_workspace["capitals"]),
        task_description="знайди пряму відстань між столицями в км для колонки distance",
        output_path=str(out_capitals_path),
        preferred_tier="reference",
    )
    df_cap = pd.read_excel(saved_cap)
    assert list(df_cap.columns) == ["Capital_From", "Country_From", "Capital_To", "Country_To", "distance"]
    assert len(df_cap) == 10
    assert df_cap["distance"].notnull().all()
    assert (df_cap["distance"] >= 200).all()

    # 2. Тестування через фасад process_excel() на mountains.xlsx (10 рядків) без зміни коду
    out_mountains_path = isolated_submission_dir / "mountains_enriched.xlsx"
    saved_mnt = process_excel(
        file_path=str(sample_excel_workspace["mountains"]),
        task_description="додай висоту гір у метрах до колонки height",
        output_path=str(out_mountains_path),
        preferred_tier="reference",
    )
    df_mnt = pd.read_excel(saved_mnt)
    assert list(df_mnt.columns) == ["Mountain", "Country", "height"]
    assert len(df_mnt) == 10
    assert df_mnt["height"].notnull().all()
    assert int(df_mnt.loc[0, "height"]) == 8849
    assert int(df_mnt.loc[1, "height"]) == 8611

    # 3. Тестування на 4-рядковому варіанті mountains_pdf.xlsx (Everest, Mont Blanc, Denali, Kilimanjaro)
    orch = ExcelEnrichmentOrchestrator(
        preferred_tier="reference",
        max_workers=4,
        cache_path=str(isolated_cache_paths["json_search"]),
    )
    cap_summary = orch.run_enrichment(
        sample_excel_workspace["capitals"],
        "знайди пряму відстань між столицями в км для колонки distance",
        output_path=out_capitals_path,
    )
    mnt_summary = orch.run_enrichment(
        sample_excel_workspace["mountains"],
        "додай висоту гір у метрах до колонки height",
        output_path=out_mountains_path,
    )
    pdf_mnt_summary = orch.run_enrichment(
        sample_excel_workspace["mountains_pdf"],
        "додай висоту гір у метрах до колонки height",
        output_path=isolated_submission_dir / "mountains_pdf_enriched.xlsx",
    )
    assert pdf_mnt_summary.successful_rows == 4
    heights_4 = pdf_mnt_summary.enriched_df["height"].to_list()
    assert heights_4 == [8849, 4808, 6190, 5895]

    fp_nb = export_fp_jupyter_notebook(
        cap_summary,
        mnt_summary,
        output_path=str(isolated_submission_dir / "oleh_hatsenko_fp.ipynb"),
    )
    assert fp_nb.exists()
