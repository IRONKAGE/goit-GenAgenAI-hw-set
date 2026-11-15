# ==============================================================================
# 🗃️ CORE DATASETS: Smart Tabular & Excel Reader with Graceful Degradation
# ==============================================================================
"""
Ядро читання та збереження табличних даних (Polars, Pandas, CSV, Parquet, Excel .xlsx).
Забезпечує стійкість до битих рядків, проблем кодування та автоматичне форматування
ширини стовпців у вихідних Excel-файлах через OpenPyXL.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import polars as pl

logger = logging.getLogger("core.datasets")


def ensure_directory(path: Union[str, Path]) -> Path:
    """Створює директорію (разом із батьківськими), якщо вона ще не існує."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_tabular_data_safe(file_path: Union[str, Path]) -> Optional[pl.DataFrame]:
    """
    Блискавичне завантаження CSV / Parquet / Excel через Polars / Pandas
    із патерном Graceful Degradation.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"❌ Файл не знайдено: {path}")
        return None

    try:
        if path.suffix.lower() == ".parquet":
            return pl.read_parquet(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            pdf = pd.read_excel(path)
            return pl.from_pandas(pdf)
        return pl.read_csv(path, null_values=["NA", "N/A", "null", "NaN", ""])
    except Exception as err:
        logger.error(f"❌ Помилка читання таблиці {path.name}: {err}")
        return None


def smart_read_csv(file_path: Union[str, Path], desc_name: str = "dataset", **kwargs) -> pd.DataFrame:
    """
    3-рівневий DRY-рідер CSV (pyarrow -> c -> python) із захистом від помилок UTF-8 та битих рядків.
    """
    engines = ["pyarrow", "c", "python"]
    last_error = None

    for engine in engines:
        attempt_kwargs = kwargs.copy()
        attempt_kwargs["engine"] = engine
        if engine in {"python", "pyarrow"}:
            attempt_kwargs.pop("low_memory", None)
        if engine == "python":
            attempt_kwargs.pop("float_precision", None)

        try:
            return pd.read_csv(file_path, **attempt_kwargs)
        except UnicodeDecodeError:
            attempt_kwargs["encoding"] = "latin1"
            try:
                return pd.read_csv(file_path, **attempt_kwargs)
            except Exception as err:
                last_error = err
        except Exception as err:
            attempt_kwargs["on_bad_lines"] = "skip"
            try:
                return pd.read_csv(file_path, **attempt_kwargs)
            except Exception as fallback_err:
                last_error = fallback_err

    raise RuntimeError(f"❌ Не вдалося прочитати CSV '{desc_name}': {last_error}")


def read_excel_safe(file_path: Union[str, Path]) -> pd.DataFrame:
    """Безпечно зчитує Excel-файл (.xlsx) через engine='openpyxl'."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel-файл не знайдено: {path}")
    return pd.read_excel(path, engine="openpyxl")


def save_excel_formatted(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    sheet_name: str = "Sheet1",
) -> Path:
    """
    Зберігає DataFrame у файл `.xlsx` зі збереженням усіх оригінальних даних
    та автоматичним підбором ширини колонок через OpenPyXL.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        for col_idx, col_name in enumerate(df.columns, 1):
            col_letter = ws.cell(row=1, column=col_idx).column_letter
            values_len = [len(str(v)) for v in df[col_name].dropna().tolist()]
            max_len = max([len(str(col_name))] + values_len)
            ws.column_dimensions[col_letter].width = max(12, min(max_len + 4, 42))

    return out
