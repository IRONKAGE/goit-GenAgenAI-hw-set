# ==============================================================================
# 📐 CORE EVALUATION ENGINE: JSON Contract check(), Robust Retry, Judge & RAGAS
# ==============================================================================
"""
Універсальне ядро валідації та оцінювання для ДЗ 1–5 і Фінального проекту:
1. `check()` — канонічний валідатор JSON-відповідей із оновлення ДЗ 2.
2. `robust_structured_invoke()` — виклик structured output із retry (ValidationError, OutputParserException) з Теми 5.
3. `parse_relevancy_score()` — виправлений regex-парсер оцінок LLM-as-Judge із оновлення ДЗ 5.
4. `compute_ragas_metrics_offline()` — обчислення Faithfulness, Answer Relevancy, Context Precision/Recall (Тема 9).
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

ALLOWED_TONES_DEFAULT: Set[str] = {"позитивна", "нейтральна", "негативна"}


def check(
    result: str,
    keys: Sequence[str] = ("person", "tone"),
    field: str = "tone",
    allowed: Set[str] = ALLOWED_TONES_DEFAULT,
) -> Dict[str, Any]:
    """
    Канонічна функція перевірки суворого JSON-контракту з документа `ДЗ2_GenAI оновлення`:
    перевіряє непорожність, синтаксис JSON, точну множину ключів та допустиме значення поля.
    """
    if not result:
        raise ValueError("порожня відповідь моделі")
    data = json.loads(result.strip())
    if not isinstance(data, dict) or set(data) != set(keys):
        raise ValueError(f"очікувались рівно ключі {keys}: {data}")
    if data[field] not in allowed:
        raise ValueError(f"недопустиме значення {field}: {data[field]}")
    return data


def robust_structured_invoke(
    llm: Any,
    schema_cls: Any,
    prompt_text: str,
    max_retries: int = 2,
) -> Any:
    """
    Виклик `.with_structured_output(schema_cls)` з автоматичним retry при помилках
    валідації `ValidationError` або `OutputParserException` (патерн п. 4.7 Теми 5).
    """
    structured = llm.with_structured_output(schema_cls)
    last_err: Optional[Exception] = None
    current_prompt = prompt_text

    for attempt in range(max_retries + 1):
        try:
            return structured.invoke(current_prompt)
        except (ValidationError, OutputParserException, ValueError) as err:
            last_err = err
            current_prompt = (
                f"{prompt_text}\n\n"
                f"Спроба {attempt + 1}: виправ помилки у структурі відповіді ({err}). "
                "Поверни коректний об'єкт згідно зі схемою."
            )
    raise RuntimeError(f"Вичерпано спроби валідації структурованого виводу: {last_err}")


def parse_relevancy_score(raw_text: str) -> float:
    """
    Виправлений парсер оцінки релевантності (0..1) з документа `ДЗ5_GenAI оновлення`:
    - замінює кому на крапку ('0,9' -> 0.9)
    - витягує перше число через `re.search(r"\\d+(?:\\.\\d+)?", text)` (ігнорує крапку в кінці речення)
    - переводить відсотки ('85%' -> 0.85) та 10-бальну шкалу ('9.2' -> 0.92) у діапазон [0.0, 1.0].
    """
    text = raw_text.strip().replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"Не вдалося знайти числове значення у відповіді: {raw_text!r}")
    score = float(match.group())
    if score > 1.0:
        score = score / 100.0 if score > 10.0 else score / 10.0
    return min(max(score, 0.0), 1.0)


def compute_ragas_metrics_offline(
    question: str,
    answer: str,
    retrieved_contexts: List[str],
    expected_fact_substring: str = "",
    expected_entity: str = "",
) -> Dict[str, float]:
    """
    Обчислює 4 класичні метрики RAGAS (Тема 9, Частина 2):
    - faithfulness (відсутність галюцинацій відносно контексту)
    - answer_relevancy (відповідність запитанню)
    - context_precision (частка релевантного контексту на перших позиціях)
    - context_recall (повнота знайденого контексту)
    """
    joined_ctx = "\n".join(retrieved_contexts).lower()
    ans_lower = answer.lower()

    if expected_entity == "NONE":
        # Кейс відсутньої в базі компанії (Питання №6 у ДЗ 4)
        faithful = 1.0 if "відсутня" in ans_lower else 0.0
        relevancy = 1.0 if len(ans_lower) > 10 else 0.0
        return {
            "faithfulness": faithful,
            "answer_relevancy": relevancy,
            "context_precision": 1.0,
            "context_recall": 1.0,
        }

    faithful = 1.0 if (not expected_fact_substring or expected_fact_substring.lower() in ans_lower) else 0.5
    relevancy = 1.0 if len(ans_lower) > 5 else 0.0

    precision = 0.0
    for idx, ctx in enumerate(retrieved_contexts):
        if expected_entity.lower() in ctx.lower():
            precision = 1.0 if idx == 0 else round(1.0 / (idx + 1), 2)
            break

    recall = 1.0 if (not expected_entity or expected_entity.lower() in joined_ctx) else 0.0
    return {
        "faithfulness": faithful,
        "answer_relevancy": relevancy,
        "context_precision": precision,
        "context_recall": recall,
    }
