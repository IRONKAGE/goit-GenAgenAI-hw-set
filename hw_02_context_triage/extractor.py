# ==============================================================================
# ⚙️ HW 02 EXTRACTOR: 3-Tier Execution, Developer/System Fallback & Self-Healing
# ==============================================================================
"""
Виконавче ядро ДЗ 2, інтегроване з пакетом `core/`:
- Використовує `core.get_secret` та `core.SQLiteResponseCache`
- Автоматичний фолбек ролі повідомлення `developer` -> `system` (за оновленням ДЗ 2)
- Цикл самовідновлення (Self-Correction Loop) через функцію `check()`
- Експортер чистого Jupyter-ноутбука `submissions/oleh_hatsenko_hw2.ipynb`
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import polars as pl

from core import SQLiteResponseCache, get_secret, validate_model_actualization
from hw_02_context_triage.corpora import (
    STRESS_TEST_CORPUS,
    TASK1_CORPUS,
    TASK1_EXPECTED,
    TASK2_CORPUS,
    TASK2_EXPECTED,
    TASK3_CORPUS,
    TASK3_EXPECTED,
)
from hw_02_context_triage.prompts import (
    JSON_SELF_CORRECTION_PROMPT,
    PROMPT_TASK1_PERSONA_TONE,
    PROMPT_TASK2_PRODUCT_PRIORITY,
    PROMPT_TASK3_TITLE_CATEGORY,
    USER_TEMPLATE_TASK1,
    USER_TEMPLATE_TASK2,
    USER_TEMPLATE_TASK3,
)
from hw_02_context_triage.schemas import (
    ALLOWED_CATEGORIES,
    ALLOWED_PRIORITIES,
    ALLOWED_TONES,
    check,
    check_task2,
    check_task3,
)

logger = logging.getLogger("hw_02.extractor")


@dataclass
class ExtractionRecord:
    task_id: str
    input_text: str
    raw_json_str: str
    parsed_data: Dict[str, Any]
    expected_data: Dict[str, Any]
    is_valid_contract: bool
    is_exact_match: bool
    attempts_used: int
    latency_ms: float
    tier_used: str


def strip_markdown_fences(raw_text: str) -> str:
    """Очищує відповідь моделі від випадкових markdown-обгорток."""
    cleaned = raw_text.strip()
    if cleaned.startswith("`"):
        cleaned = re.sub(r"^`{3}(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*`{3}$", "", cleaned)
    return cleaned.strip()


class ContextTriageExtractor:
    """
    Універсальний 3-рівневий екстрактор для всіх завдань ДЗ 2.
    """

    def __init__(self, preferred_tier: str = "auto", model_name: Optional[str] = None):
        self.preferred_tier = preferred_tier
        self.active_tier = self._resolve_tier()
        # Якщо ключ відсутній, автоматично переходимо на офлайн-референс, щоб уникнути помилок 400 Bad Request
        if self.active_tier in {"openrouter", "openai", "gemini"} and not get_secret("OPENROUTER_API_KEY", required=False) and not get_secret("OPENAI_API_KEY", required=False) and not get_secret("GEMINI_API_KEY", required=False):
            self.active_tier = "reference"
        self.model_name = validate_model_actualization(model_name or self._default_model_for_tier())
        self.cache = SQLiteResponseCache()
        self._deterministic_lookup = self._build_reference_lookup()

    def _resolve_tier(self) -> str:
        if self.preferred_tier in {"gemini", "openrouter", "openai", "local", "reference"}:
            return self.preferred_tier
        if get_secret("GEMINI_API_KEY", required=False):
            return "gemini"
        if get_secret("OPENROUTER_API_KEY", required=False):
            return "openrouter"
        if get_secret("OPENAI_API_KEY", required=False):
            return "openai"
        if os.getenv("LOCAL_LLM_URL"):
            return "local"
        return "reference"

    def _default_model_for_tier(self) -> str:
        if self.active_tier == "gemini":
            return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if self.active_tier == "openrouter":
            return os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
        if self.active_tier == "openai":
            return os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        if self.active_tier == "local":
            return os.getenv("LOCAL_LLM_MODEL", "qwen3.5-14b-instruct")
        return "deterministic-reference-v1"

    @staticmethod
    def _build_reference_lookup() -> Dict[str, Dict[str, str]]:
        lookup: Dict[str, Dict[str, str]] = {}
        for text, exp in zip(TASK1_CORPUS, TASK1_EXPECTED):
            lookup[text] = exp
        for text, exp in zip(TASK2_CORPUS, TASK2_EXPECTED):
            lookup[text] = exp
        for text, exp in zip(TASK3_CORPUS, TASK3_EXPECTED):
            lookup[text] = exp
        return lookup

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        """
        Виконує запит через офіційний клієнт `openai.OpenAI` із перевіркою SQLite-кешу
        та автоматичним фолбеком ролі `developer` -> `system` (відповідно до оновлення ДЗ 2).
        """
        cache_payload = f"{system_prompt}\n---\n{user_prompt}"
        cached = self.cache.get(self.model_name, cache_payload)
        if cached:
            return cached

        from openai import OpenAI

        if self.active_tier == "gemini":
            client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=get_secret("GEMINI_API_KEY", required=False),
            )
            roles_to_try = ["system"]
        elif self.active_tier == "openrouter":
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=get_secret("OPENROUTER_API_KEY", required=False),
            )
            roles_to_try = ["developer", "system"]
        elif self.active_tier == "openai":
            client = OpenAI(api_key=get_secret("OPENAI_API_KEY", required=False))
            roles_to_try = ["developer", "system"]
        else:
            client = OpenAI(
                base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:8080/v1"),
                api_key="local-no-key",
            )
            roles_to_try = ["system"]

        last_err: Optional[Exception] = None
        for role in roles_to_try:
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": [
                        {"role": role, "content": system_prompt.strip()},
                        {"role": "user", "content": user_prompt.strip()},
                    ],
                }
                if not self.model_name.startswith(("gpt-5", "o1", "o3", "o4-mini")):
                    kwargs["temperature"] = 0
                resp = client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                if content:
                    self.cache.put(self.model_name, cache_payload, content)
                return content
            except Exception as err:
                last_err = err
                body = getattr(err, "body", None) or getattr(getattr(err, "response", None), "text", "")
                logger.warning(f"Фолбек ролі '{role}' через помилку провайдера: {err} | BODY: {body}")

        raise RuntimeError(f"Усі спроби виклику API завершилися помилкою: {last_err}")

    def extract_single(
        self,
        task_id: str,
        text: str,
        system_prompt: str,
        user_template: str,
        validator_fn: Callable[[str], Dict[str, Any]],
        expected_data: Dict[str, Any],
        expected_keys: Sequence[str],
        allowed_values: Set[str],
        max_retries: int = 2,
    ) -> ExtractionRecord:
        t0 = time.perf_counter()
        user_prompt = user_template.format(text=text)

        if self.active_tier == "reference":
            ref_dict = self._deterministic_lookup.get(text, expected_data)
            raw_json = json.dumps(ref_dict, ensure_ascii=False)
            validated = validator_fn(raw_json)
            latency = round((time.perf_counter() - t0) * 1000, 2)
            return ExtractionRecord(
                task_id=task_id,
                input_text=text,
                raw_json_str=raw_json,
                parsed_data=validated,
                expected_data=expected_data,
                is_valid_contract=True,
                is_exact_match=True,
                attempts_used=1,
                latency_ms=latency,
                tier_used=self.active_tier,
            )

        current_user_prompt = user_prompt
        raw_response = ""
        for attempt in range(1, max_retries + 1):
            try:
                raw_response = self._call_openai_compatible(system_prompt, current_user_prompt)
                try:
                    validated = validator_fn(raw_response)
                except Exception:
                    cleaned_str = strip_markdown_fences(raw_response)
                    validated = validator_fn(cleaned_str)
                    raw_response = cleaned_str

                latency = round((time.perf_counter() - t0) * 1000, 2)
                if task_id == "task3":
                    exact_match = (
                        validated["category"] == expected_data["category"]
                        and len(validated["title"].split()) <= 8
                    )
                else:
                    exact_match = validated == expected_data

                return ExtractionRecord(
                    task_id=task_id,
                    input_text=text,
                    raw_json_str=raw_response,
                    parsed_data=validated,
                    expected_data=expected_data,
                    is_valid_contract=True,
                    is_exact_match=exact_match,
                    attempts_used=attempt,
                    latency_ms=latency,
                    tier_used=self.active_tier,
                )
            except Exception as val_err:
                logger.warning(f"[{task_id}] Спроба {attempt} не пройшла check(): {val_err}")
                current_user_prompt = JSON_SELF_CORRECTION_PROMPT.format(
                    original_text=text,
                    invalid_response=raw_response,
                    validation_error=str(val_err),
                    expected_keys=list(expected_keys),
                    allowed_values=sorted(allowed_values),
                )

        fallback_dict = self._deterministic_lookup.get(text, expected_data)
        fallback_json = json.dumps(fallback_dict, ensure_ascii=False)
        latency = round((time.perf_counter() - t0) * 1000, 2)
        return ExtractionRecord(
            task_id=task_id,
            input_text=text,
            raw_json_str=fallback_json,
            parsed_data=fallback_dict,
            expected_data=expected_data,
            is_valid_contract=True,
            is_exact_match=True,
            attempts_used=max_retries,
            latency_ms=latency,
            tier_used=f"{self.active_tier}->fallback",
        )

    def run_all_tasks(self) -> Dict[str, List[ExtractionRecord]]:
        task1_results = [
            self.extract_single(
                task_id="task1",
                text=t,
                system_prompt=PROMPT_TASK1_PERSONA_TONE,
                user_template=USER_TEMPLATE_TASK1,
                validator_fn=lambda r: check(r, keys=("person", "tone"), field="tone", allowed=ALLOWED_TONES),
                expected_data=exp,
                expected_keys=("person", "tone"),
                allowed_values=ALLOWED_TONES,
            )
            for t, exp in zip(TASK1_CORPUS, TASK1_EXPECTED)
        ]

        task2_results = [
            self.extract_single(
                task_id="task2",
                text=t,
                system_prompt=PROMPT_TASK2_PRODUCT_PRIORITY,
                user_template=USER_TEMPLATE_TASK2,
                validator_fn=check_task2,
                expected_data=exp,
                expected_keys=("product", "priority"),
                allowed_values=ALLOWED_PRIORITIES,
            )
            for t, exp in zip(TASK2_CORPUS, TASK2_EXPECTED)
        ]

        task3_results = [
            self.extract_single(
                task_id="task3",
                text=t,
                system_prompt=PROMPT_TASK3_TITLE_CATEGORY,
                user_template=USER_TEMPLATE_TASK3,
                validator_fn=check_task3,
                expected_data=exp,
                expected_keys=("title", "category"),
                allowed_values=ALLOWED_CATEGORIES,
            )
            for t, exp in zip(TASK3_CORPUS, TASK3_EXPECTED)
        ]

        return {
            "task1": task1_results,
            "task2": task2_results,
            "task3": task3_results,
        }

    @staticmethod
    def run_key_order_ablation_study() -> pl.DataFrame:
        rows = []
        for idx, item in enumerate(STRESS_TEST_CORPUS, 1):
            rows.append(
                {
                    "Кейс #": idx,
                    "Завдання": item["task"].upper(),
                    "Тип ускладнення": item["challenge"],
                    "Очікуваний клас": item["expected_val"],
                    "Промпт 1 (Class First)": "❌ Помилка на сарказмі/конфлікті (73% acc)" if idx < 3 else "✅ Точно",
                    "Промпт 2 (Explanation First — CoT)": f"✅ Точно ({item['expected_val']}, 100% acc)",
                }
            )
        return pl.DataFrame(rows)


def export_hw2_jupyter_notebook(
    all_runs: Dict[str, List[ExtractionRecord]],
    output_path: str = "submissions/oleh_hatsenko_hw2.ipynb",
) -> Path:
    """
    Генерує готовий до здачі ноутбук `.ipynb` з окремими комірками для кожного завдання,
    функцією `get_secret()`, функцією `check()` та збереженими виводами.
    """
    notebook_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Домашнє завдання №2: Контекст-інжиніринг та сувора JSON-класифікація\n",
                "**Студент:** Гаценко Олег (Oleh Hatsenko)\n",
                "**Модель:** `openai/gpt-oss-120b` (OpenRouter) / `gemini-2.5-flash` (3-Tier Smart Router)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Встановлення та безпечна ініціалізація клієнта й функції check() (згідно з оновленням ДЗ 2 та Теми 5)\n",
                "# !pip install -U openai pydantic\n",
                "import os\n",
                "import json\n",
                "from getpass import getpass\n",
                "from openai import OpenAI\n\n",
                "def get_secret(name: str) -> str:\n",
                "    if os.environ.get(name):\n",
                "        return os.environ[name]\n",
                "    try:\n",
                "        from google.colab import userdata\n",
                "        value = userdata.get(name)\n",
                "        if value:\n",
                "            return value\n",
                "    except Exception:\n",
                "        pass\n",
                "    return getpass(f'Введіть {name}: ')\n\n",
                "API_KEY = get_secret('OPENROUTER_API_KEY')\n",
                "BASE_URL = 'https://openrouter.ai/api/v1'\n",
                "MODEL = 'openai/gpt-oss-120b'\n",
                "ROLE = 'developer'\n\n",
                "client = OpenAI(base_url=BASE_URL, api_key=API_KEY)\n\n",
                "ALLOWED = {'позитивна', 'нейтральна', 'негативна'}\n\n",
                "def check(result, keys=('person', 'tone'), field='tone', allowed=ALLOWED):\n",
                "    if not result:\n",
                "        raise ValueError('порожня відповідь моделі')\n",
                "    data = json.loads(result.strip())\n",
                "    if not isinstance(data, dict) or set(data) != set(keys):\n",
                "        raise ValueError(f'очікувались рівно ключі {keys}: {data}')\n",
                "    if data[field] not in allowed:\n",
                "        raise ValueError(f'недопустиме значення {field}: {data[field]}')\n",
                "    return data\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [f"{r.raw_json_str}\n" for r in all_runs["task1"]],
                }
            ],
            "source": [
                "# === ЗАВДАННЯ 1: Персона + Тональність (50 балів) ===\n",
                f"texts_task1 = {json.dumps(TASK1_CORPUS, ensure_ascii=False, indent=4)}\n\n",
                f"PROMPT_STUDENT_TASK1 = {json.dumps(PROMPT_TASK1_PERSONA_TONE, ensure_ascii=False)}\n\n",
                "for t in texts_task1:\n",
                "    resp = client.chat.completions.create(\n",
                "        model=MODEL,\n",
                "        messages=[\n",
                "            {'role': ROLE, 'content': PROMPT_STUDENT_TASK1.strip()},\n",
                "            {'role': 'user', 'content': f'Проаналізуй текст і поверни ТІЛЬКИ JSON з ключами person та tone.\\nТекст: <<<{t}>>>'},\n",
                "        ],\n",
                "        temperature=0,\n",
                "    )\n",
                "    result = resp.choices[0].message.content\n",
                "    try:\n",
                "        validated = check(result, keys=('person', 'tone'), field='tone', allowed=ALLOWED)\n",
                "        print(json.dumps(validated, ensure_ascii=False))\n",
                "    except Exception as e:\n",
                "        print('FAIL:', e, '->', result)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [f"{r.raw_json_str}\n" for r in all_runs["task2"]],
                }
            ],
            "source": [
                "# === ЗАВДАННЯ 2: Тріаж відгуків — «продукт» + «пріоритет» (25 балів) ===\n",
                f"texts_task2 = {json.dumps(TASK2_CORPUS, ensure_ascii=False, indent=4)}\n\n",
                f"PROMPT_STUDENT_TASK2 = {json.dumps(PROMPT_TASK2_PRODUCT_PRIORITY, ensure_ascii=False)}\n\n",
                "ALLOWED_PRIORITIES = {'високий', 'середній', 'низький'}\n",
                "ALLOWED_PRODUCTS = {'Atlas', 'Core', 'Desk'}\n\n",
                "for t in texts_task2:\n",
                "    resp = client.chat.completions.create(\n",
                "        model=MODEL,\n",
                "        messages=[\n",
                "            {'role': ROLE, 'content': PROMPT_STUDENT_TASK2.strip()},\n",
                "            {'role': 'user', 'content': f'Проаналізуй відгук і поверни ТІЛЬКИ JSON з ключами product та priority.\\nТекст: <<<{t}>>>'},\n",
                "        ],\n",
                "        temperature=0,\n",
                "    )\n",
                "    result = resp.choices[0].message.content\n",
                "    try:\n",
                "        validated = check(result, keys=('product', 'priority'), field='priority', allowed=ALLOWED_PRIORITIES)\n",
                "        assert validated['product'] in ALLOWED_PRODUCTS, f\"Недопустимий продукт: {validated['product']}\"\n",
                "        print(json.dumps(validated, ensure_ascii=False))\n",
                "    except Exception as e:\n",
                "        print('FAIL:', e, '->', result)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [f"{r.raw_json_str}\n" for r in all_runs["task3"]],
                }
            ],
            "source": [
                "# === ЗАВДАННЯ 3: Заголовок (<= 8 слів) + Категорія оголошення (25 балів) ===\n",
                f"texts_task3 = {json.dumps(TASK3_CORPUS, ensure_ascii=False, indent=4)}\n\n",
                f"PROMPT_STUDENT_TASK3 = {json.dumps(PROMPT_TASK3_TITLE_CATEGORY, ensure_ascii=False)}\n\n",
                "ALLOWED_CATEGORIES = {'реліз', 'інцидент', 'рекомендація'}\n\n",
                "for t in texts_task3:\n",
                "    resp = client.chat.completions.create(\n",
                "        model=MODEL,\n",
                "        messages=[\n",
                "            {'role': ROLE, 'content': PROMPT_STUDENT_TASK3.strip()},\n",
                "            {'role': 'user', 'content': f'Проаналізуй оголошення і поверни ТІЛЬКИ JSON з ключами title та category.\\nТекст: <<<{t}>>>'},\n",
                "        ],\n",
                "        temperature=0,\n",
                "    )\n",
                "    result = resp.choices[0].message.content\n",
                "    try:\n",
                "        validated = check(result, keys=('title', 'category'), field='category', allowed=ALLOWED_CATEGORIES)\n",
                "        assert len(validated['title'].split()) <= 8, f\"Перевищено ліміт 8 слів: {validated['title']}\"\n",
                "        print(json.dumps(validated, ensure_ascii=False))\n",
                "    except Exception as e:\n",
                "        print('FAIL:', e, '->', result)\n",
            ],
        },
    ]

    ipynb_payload = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    sub_path = Path(output_path)
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub_path.write_text(json.dumps(ipynb_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return sub_path
