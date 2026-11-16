# ==============================================================================
# ⚙️ PYTEST CONFIGURATION & FIXTURES: Ізольовані кеші, Seed та Еталонні Дані
# ==============================================================================
"""
Спільні фікстури Pytest:
- Фіксація GLOBAL_SEED=42 перед кожною тестовою сесією
- Ізольовані тимчасові директорії для SQLite/JSON кешів, Excel-таблиць та .ipynb
"""

from pathlib import Path
from typing import Dict
import pytest

import os
from core import configure_langsmith_tracing, set_global_seed
from fp_excel_enricher.excel_io import ensure_sample_excel_files

@pytest.fixture(scope="session", autouse=True)
def lock_reproducibility_seed() -> int:
    """Фіксує детермінований seed та вимикає фоновий мережевий трейсинг LangSmith під час pytest."""
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    configure_langsmith_tracing(enable=False)
    return set_global_seed(42)

@pytest.fixture()
def isolated_submission_dir(tmp_path: Path) -> Path:
    """Надає чисту тимчасову директорію для перевірки генерації .ipynb, .md та .xlsx."""
    sub_dir = tmp_path / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    return sub_dir


@pytest.fixture()
def isolated_cache_paths(tmp_path: Path) -> Dict[str, Path]:
    """Надає ізольовані шляхи для SQLite LLM-кешу та JSON пошукового кешу."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "sqlite_llm": cache_dir / "test_llm_cache.sqlite",
        "json_search": cache_dir / "test_search_cache.json",
    }


@pytest.fixture()
def sample_excel_workspace(tmp_path: Path) -> Dict[str, Path]:
    """Створює ізольовані вхідні таблиці capitals.xlsx, mountains.xlsx та mountains_pdf.xlsx."""
    excel_dir = tmp_path / "hw_fp_excel"
    return ensure_sample_excel_files(base_dir=excel_dir)
