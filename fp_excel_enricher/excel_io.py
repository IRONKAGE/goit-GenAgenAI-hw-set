# ==============================================================================
# 📊 FINAL PROJECT EXCEL I/O: Safe Reader, Writer & Template Bootstrapper
# ==============================================================================
"""
Модуль безпечного читання та запису Excel-файлів (.xlsx), синхронізований із `core.datasets`:
- Автоматично створює еталонні `capitals.xlsx`, `mountains.xlsx` та `mountains_pdf.xlsx`
- Зберігає 100% оригінальних колонок та структури даних
- Форматує ширину колонок у вихідних файлах `*_enriched.xlsx` через `core.save_excel_formatted`
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from core import ensure_directory, read_excel_safe, save_excel_formatted
from fp_excel_enricher.prompts import (
    CAPITALS_FULL_10_ROWS,
    MOUNTAINS_FULL_10_ROWS,
    MOUNTAINS_PDF_4_ROWS,
)

logger = logging.getLogger("hw_fp.excel_io")


def ensure_sample_excel_files(base_dir: Union[str, Path] = "data/hw_fp_excel") -> Dict[str, Path]:
    """
    Гарантує наявність вхідних тестових файлів `capitals.xlsx` та `mountains.xlsx`
    (плюс `mountains_pdf.xlsx` для 4-рядкового варіанта з тексту PDF).
    """
    target_dir = ensure_directory(base_dir)

    capitals_path = target_dir / "capitals.xlsx"
    mountains_path = target_dir / "mountains.xlsx"
    mountains_pdf_path = target_dir / "mountains_pdf.xlsx"

    if not capitals_path.exists():
        save_excel_formatted(pd.DataFrame(CAPITALS_FULL_10_ROWS), capitals_path)
    if not mountains_path.exists():
        save_excel_formatted(pd.DataFrame(MOUNTAINS_FULL_10_ROWS), mountains_path)
    if not mountains_pdf_path.exists():
        save_excel_formatted(pd.DataFrame(MOUNTAINS_PDF_4_ROWS), mountains_pdf_path)

    return {
        "capitals": capitals_path,
        "mountains": mountains_path,
        "mountains_pdf": mountains_pdf_path,
    }


def read_excel_with_metadata(
    file_path: Union[str, Path],
) -> Tuple[pd.DataFrame, List[str], List[Dict[str, object]]]:
    """
    Безпечно зчитує Excel-файл і повертає `(DataFrame, список_колонок, перші_3_рядки)`.
    """
    path = Path(file_path)
    if not path.exists():
        alt_path = Path("data/hw_fp_excel") / path.name
        if alt_path.exists():
            path = alt_path
        else:
            ensure_sample_excel_files()
            if not path.exists() and alt_path.exists():
                path = alt_path

    df = read_excel_safe(path)
    columns = list(df.columns)
    sample_rows = df.head(3).where(pd.notnull(df.head(3)), None).to_dict(orient="records")
    return df, columns, sample_rows


def save_enriched_excel(
    df: pd.DataFrame,
    original_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
) -> Path:
    """
    Зберігає збагачений DataFrame у файл `<stem>_enriched.xlsx` (або у вказаний `output_path`)
    через `core.save_excel_formatted`.
    """
    orig = Path(original_path)
    if output_path is None:
        out = orig.with_name(f"{orig.stem}_enriched{orig.suffix}")
    else:
        out = Path(output_path)

    saved = save_excel_formatted(df, out)
    logger.info(f"💾 Збагачений Excel-файл успішно збережено: {saved}")
    return saved
