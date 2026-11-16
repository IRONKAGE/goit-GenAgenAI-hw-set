# ==============================================================================
# 🧠 FINAL PROJECT QUERY PLANNER: Universal Schema & Search Strategy Agent
# ==============================================================================
"""
Агент 1 (SchemaTaskPlanner), синхронізований із `core.get_smart_chat_llm`:
Аналізує довільну таблицю Excel та текстовий опис завдання, визначає цільову колонку,
вхідні колонки-ключі, очікуваний тип даних та шаблон пошукового запиту.
"""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from core import get_smart_chat_llm
from fp_excel_enricher.prompts import SCHEMA_PLANNER_SYSTEM_PROMPT

logger = logging.getLogger("hw_fp.query_planner")


class EnrichmentPlan(BaseModel):
    """Структурований план збагачення таблиці, який генерує Агент-Планувальник."""

    target_column: str = Field(..., description="Назва колонки для запису знайдених даних")
    source_columns: List[str] = Field(..., description="Колонки, які формують пошуковий контекст рядка")
    value_type: Literal["int", "float", "str"] = Field("int", description="Очікуваний тип значення")
    unit: str = Field("", description="Одиниця виміру (напр. km, m)")
    search_query_template: str = Field(..., description="Шаблон пошукового запиту з {Column_Name}")
    fallback_search_template: str = Field(..., description="Резервний короткий шаблон пошукового запиту")
    extraction_guidelines: str = Field(..., description="Інструкція для екстрактора значень")


class RowExtractionResult(BaseModel):
    """Результат екстракції значення для одного рядка таблиці (сумісний із Gemini/OpenAI JSON Schema)."""

    extracted_value: Optional[Union[int, float, str]] = Field(
        None,
        description="Чисте знайдене значення (ціле число, дробове число або рядок без одиниць виміру)",
    )
    confidence: float = Field(0.95, ge=0.0, le=1.0, description="Впевненість у точності даних")
    source_explanation: str = Field("", description="Коротке пояснення джерела або методу розрахунку")


class BatchAuditReport(BaseModel):
    """Фінальний звіт аудиту збагаченої таблиці перед збереженням у файл."""

    all_rows_valid: bool = Field(True, description="Чи всі рядки успішно заповнені без пропусків")
    missing_row_indices: List[int] = Field(default_factory=list, description="Індекси рядків із пропусками")
    quality_summary: str = Field("Усі значення верифіковано в межах ±10%.", description="Підсумок аудиту")


class SchemaTaskPlanner:
    """
    Будує `EnrichmentPlan` для будь-якого Excel-файлу та текстового завдання.
    Використовує 3-Tier маршрутизацію через `core.get_smart_chat_llm`.
    """

    def __init__(self, preferred_tier: str = "auto"):
        self.preferred_tier = preferred_tier

    @staticmethod
    def _heuristic_plan(
        columns: List[str],
        sample_rows: List[Dict[str, Any]],
        task_description: str,
    ) -> EnrichmentPlan:
        """
        Універсальний детермінований аналізатор структури таблиці та опису завдання.
        Працює для будь-яких таблиць (столиці, гори, компанії, країни, товари тощо).
        """
        task_lower = task_description.lower()

        target_col: Optional[str] = None
        for col in columns:
            if re.search(rf"\b{re.escape(col.lower())}\b", task_lower):
                target_col = col
                break

        if target_col is None and sample_rows:
            for col in reversed(columns):
                if all(row.get(col) is None or str(row.get(col)).strip() == "" for row in sample_rows):
                    target_col = col
                    break

        if target_col is None:
            target_col = "enriched_value"

        source_cols = [c for c in columns if c != target_col]

        if any(
            w in task_lower
            for w in ("відстань", "distance", "км", "km", "висот", "height", "метр", "кількість", "рік")
        ):
            val_type: Literal["int", "float", "str"] = "int"
        elif any(w in task_lower for w in ("ціна", "курс", "площа", "ввп", "відсоток", "%")):
            val_type = "float"
        else:
            val_type = "str"

        unit = (
            "km"
            if ("км" in task_lower or "km" in task_lower or target_col.lower() == "distance")
            else ("m" if ("метр" in task_lower or "height" in target_col.lower()) else "")
        )

        if {"Capital_From", "Capital_To"} <= set(source_cols):
            query_tpl = (
                "straight line air distance from {Capital_From}, {Country_From} "
                "to {Capital_To}, {Country_To} in kilometers km"
            )
            fallback_tpl = "distance between {Capital_From} and {Capital_To} in km"
            guidelines = (
                "Витягни пряму відстань між містами в кілометрах (ціле число). "
                "Використовуй саме міста, вказані у Capital_From та Capital_To."
            )
        elif "Mountain" in source_cols:
            query_tpl = "official elevation height of {Mountain} mountain ({Country}) in meters m"
            fallback_tpl = "{Mountain} height in meters"
            guidelines = "Витягни офіційну висоту гірської вершини над рівнем моря у метрах (ціле число)."
        else:
            placeholders = " ".join(f"{{{c}}}" for c in source_cols)
            query_tpl = f"{placeholders} {task_description}"
            fallback_tpl = f"{placeholders} {target_col}"
            guidelines = f"Знайди точне значення для колонки '{target_col}' відповідно до завдання: {task_description}"

        return EnrichmentPlan(
            target_column=target_col,
            source_columns=source_cols,
            value_type=val_type,
            unit=unit,
            search_query_template=query_tpl,
            fallback_search_template=fallback_tpl,
            extraction_guidelines=guidelines,
        )

    def create_plan(
        self,
        columns: List[str],
        sample_rows: List[Dict[str, Any]],
        task_description: str,
    ) -> EnrichmentPlan:
        llm, _, _ = get_smart_chat_llm(preferred_tier=self.preferred_tier, role="worker")
        if llm is not None:
            try:
                structured_planner = llm.with_structured_output(EnrichmentPlan)
                prompt = SCHEMA_PLANNER_SYSTEM_PROMPT.format(
                    columns=columns,
                    sample_rows_json=json.dumps(sample_rows, ensure_ascii=False, indent=2),
                    task_description=task_description,
                )
                plan = structured_planner.invoke(prompt)
                if isinstance(plan, EnrichmentPlan):
                    return plan
                if isinstance(plan, dict):
                    return EnrichmentPlan(**plan)
            except Exception as err:
                logger.warning(f"[SchemaTaskPlanner] Фолбек на евристичний планувальник через збій API: {err}")

        return self._heuristic_plan(columns, sample_rows, task_description)
