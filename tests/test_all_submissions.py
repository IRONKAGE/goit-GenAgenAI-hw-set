# ==============================================================================
# ✅ END-TO-END SUBMISSION SECURITY & FORMAT AUDIT (tests/test_all_submissions.py)
# ==============================================================================
"""
Фінальний аудит усіх згенерованих файлів для здачі в LMS:
1. Перевіряє валідність JSON-структури Jupyter Notebook (nbformat 4)
2. Перевіряє відсутність витоку реальних API-ключів у коді та виводах комірок
3. Перевіряє наявність усіх 6 ноутбуків Marimo у директорії `notebooks_eda/`
"""

import json
import re
from pathlib import Path

from hw_01_model_alignment import export_hw1_jupyter_notebook
from hw_02_context_triage import ContextTriageExtractor, export_hw2_jupyter_notebook
from hw_03_todo_agent import TodoAgentOrchestrator, export_hw3_jupyter_notebook
from hw_04_financial_rag import FinancialRAGOrchestrator, export_hw4_jupyter_notebook
from hw_05_support_langgraph import export_hw5_jupyter_notebook
from fp_excel_enricher import ExcelEnrichmentOrchestrator, export_fp_jupyter_notebook

# Регулярні вирази для пошуку реальних секретів (OpenAI/OpenRouter, Google Gemini, Tavily)
SECRET_LEAK_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"AIzaSy[A-Za-z0-9\-_]{25,}"),
    re.compile(r"tvly-[A-Za-z0-9\-_]{20,}"),
)


def test_all_marimo_dashboards_exist() -> None:
    """Перевіряє наявність усіх 6 реактивних ноутбуків у папці notebooks_eda/."""
    expected_eda_files = [
        "notebooks_eda/hw_01_eda.py",
        "notebooks_eda/hw_02_eda.py",
        "notebooks_eda/hw_03_eda.py",
        "notebooks_eda/hw_04_eda.py",
        "notebooks_eda/hw_05_eda.py",
        "notebooks_eda/fp_eda.py",
    ]
    for rel_path in expected_eda_files:
        p = Path(rel_path)
        assert p.exists(), f"Відсутній Marimo-дашборд: {rel_path}"
        content = p.read_text(encoding="utf-8")
        assert "__generated_with = \"0.25.2\"" in content


def test_all_exported_notebooks_are_valid_and_secret_free(
    isolated_submission_dir: Path,
    sample_excel_workspace: dict[str, Path],
    isolated_cache_paths: dict[str, Path],
) -> None:
    """Генерує всі 6 ноутбуків для LMS і перевіряє їхню структуру та відсутність API-ключів."""
    nb1 = export_hw1_jupyter_notebook(str(isolated_submission_dir / "oleh_hatsenko_hw1.ipynb"))

    hw2_runs = ContextTriageExtractor(preferred_tier="reference").run_all_tasks()
    nb2 = export_hw2_jupyter_notebook(hw2_runs, str(isolated_submission_dir / "oleh_hatsenko_hw2.ipynb"))

    hw3_logs = TodoAgentOrchestrator(preferred_tier="reference").run_canonical_suite()
    nb3 = export_hw3_jupyter_notebook(hw3_logs, str(isolated_submission_dir / "oleh_hatsenko_hw3.ipynb"))

    hw4_orch = FinancialRAGOrchestrator(preferred_tier="reference")
    hw4_recs = hw4_orch.evaluate_all_questions()
    nb4 = export_hw4_jupyter_notebook(
        hw4_orch.chunks, hw4_recs, str(isolated_submission_dir / "oleh_hatsenko_hw4.ipynb")
    )

    nb5 = export_hw5_jupyter_notebook(str(isolated_submission_dir / "oleh_hatsenko_hw5.ipynb"))

    fp_orch = ExcelEnrichmentOrchestrator(
        preferred_tier="reference",
        cache_path=str(isolated_cache_paths["json_search"]),
    )
    cap_sum = fp_orch.run_enrichment(
        sample_excel_workspace["capitals"],
        "знайди пряму відстань між столицями в км для колонки distance",
        output_path=isolated_submission_dir / "capitals_enriched.xlsx",
    )
    mnt_sum = fp_orch.run_enrichment(
        sample_excel_workspace["mountains"],
        "додай висоту гір у метрах до колонки height",
        output_path=isolated_submission_dir / "mountains_enriched.xlsx",
    )
    nb_fp = export_fp_jupyter_notebook(
        cap_sum, mnt_sum, output_path=str(isolated_submission_dir / "oleh_hatsenko_fp.ipynb")
    )

    for nb_path in (nb1, nb2, nb3, nb4, nb5, nb_fp):
        assert nb_path.exists()
        raw_json = nb_path.read_text(encoding="utf-8")
        parsed = json.loads(raw_json)
        assert parsed["nbformat"] == 4
        assert len(parsed["cells"]) >= 3

        # Жорстка перевірка на відсутність витоку ключів API перед здачею
        for pattern in SECRET_LEAK_PATTERNS:
            assert not pattern.search(raw_json), f"Виявлено потенційний API-ключ у {nb_path.name}!"
