# ==============================================================================
# 🔗 HW 02 R&D EXTENSION: LCEL Branches, Union Schemas & TypedDict Streaming
# ==============================================================================
"""
Додатковий модуль для ДЗ 2 за матеріалами живого ноутбука Теми 5 (`AI_topic_5_lesson (2).ipynb`):
- Поліморфна класифікація через Union[PersonaToneModel, ProductPriorityModel, TitleCategoryModel]
- Стримінг структурованого виводу по частках через TypedDict + Annotated
- Паралельне виконання гілок через RunnableParallel та маршрутизація через RunnableBranch
"""

from typing import Any, Dict, List, Union
from typing_extensions import Annotated, TypedDict

import polars as pl
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

from hw_02_context_triage.schemas import (
    PersonaToneModel,
    ProductPriorityModel,
    TitleCategoryModel,
)


class StreamingTriageDict(TypedDict):
    """Структура тріажу через TypedDict (дозволяє почастковий стримінг `.stream()`, п. 4.2 і 4.5 Теми 5)."""

    product: Annotated[str, ..., "Базова назва продукту: Atlas, Core або Desk"]
    priority: Annotated[str, ..., "Пріоритет впливу на роботу: високий, середній або низький"]


class UnifiedTriageOutput(BaseModel):
    """Union-схема (п. 4.4 Теми 5): модель сама обирає відповідну схему під вхідний текст."""

    final_output: Union[PersonaToneModel, ProductPriorityModel, TitleCategoryModel] = Field(
        ..., description="Структурований результат відповідно до жанру вхідного тексту"
    )


def build_triage_router_chain() -> RunnableBranch:
    """
    Демонструє `RunnableBranch` (п. 3.3 Теми 5): маршрутизує вхідне повідомлення
    між трьома спеціалізованими гілками аналізу.
    """
    persona_branch = RunnableLambda(
        lambda x: {"branch": "Task 1: Persona + Tone", "text": x["text"]}
    )
    product_branch = RunnableLambda(
        lambda x: {"branch": "Task 2: Product + Priority", "text": x["text"]}
    )
    release_branch = RunnableLambda(
        lambda x: {"branch": "Task 3: Title + Category", "text": x["text"]}
    )

    return RunnableBranch(
        (lambda x: any(k in x["text"].lower() for k in ("atlas", "core", "desk")), product_branch),
        (
            lambda x: any(
                k in x["text"].lower()
                for k in ("версія", "профіль", "відгуки", "обслуговування", "автентифікац")
            ),
            release_branch,
        ),
        persona_branch,
    )


def build_parallel_summary_chain() -> RunnableParallel:
    """
    Демонструє `RunnableParallel` (п. 3.2 Теми 5): одночасно обчислює
    кількість слів, символів та визначає гілку маршрутизації.
    """
    router = build_triage_router_chain()
    return RunnableParallel(
        routed=router,
        stats=RunnableLambda(
            lambda x: {
                "words": len(x["text"].split()),
                "chars": len(x["text"]),
            }
        ),
    )


def simulate_typeddict_stream(target_dict: Dict[str, Any]) -> pl.DataFrame:
    """
    Відтворює почастковий стримінг `structured_llm.stream(JokeDict)` із п. 4.5 Теми 5,
    демонструючи, як TypedDict поступово нарощує ключі та токени в реальному часі.
    """
    chunks: List[Dict[str, Any]] = [{"Крок стримінгу": 0, "Частковий об'єкт (TypedDict Chunk)": "{}"}]
    step = 1
    accumulated: Dict[str, Any] = {}

    for key, full_val in target_dict.items():
        words = str(full_val).split()
        partial = ""
        for w in words:
            partial = f"{partial} {w}".strip()
            snap = {**accumulated, key: partial}
            chunks.append(
                {
                    "Крок стримінгу": step,
                    "Частковий об'єкт (TypedDict Chunk)": str(snap),
                }
            )
            step += 1
        accumulated[key] = full_val

    return pl.DataFrame(chunks)
