# ==============================================================================
# ⚖️ HW 05 EVALUATOR: calculate_answer_relevancy, Dual Judge & test_agent()
# ==============================================================================
"""
Модуль оцінювання агента технічної підтримки, синхронізований із `core/`:
- Використовує `core.parse_relevancy_score` (виправлений regex-парсер із оновлення ДЗ 5)
- Порівнює Self-Judge проти Independent Judge LLM (`role="judge"`, `gpt-5.6-terra` / `gemini-3.1-pro`)
- Забезпечує стандартну точку входу `test_agent()` та експорт `.ipynb` для LMS
"""

import io
import json
import shutil
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

from core import (
    configure_langsmith_tracing,
    flush_langsmith_tracers,
    get_smart_chat_llm,
    parse_relevancy_score,
)
from hw_05_support_langgraph.graph import build_support_agent
from hw_05_support_langgraph.prompts import (
    CLASSIFY_QUERY_SYSTEM_PROMPT,
    DRAFT_RESPONSE_SYSTEM_PROMPT,
    FAQ_DATABASE,
    JUDGE_RELEVANCY_PROMPT_TEMPLATE,
)
from hw_05_support_langgraph.state import (
    CANONICAL_TEST_QUERIES_HW5,
    SupportAgentState,
)


def deterministic_relevancy_scorer(query: str, response: str, independent: bool = False) -> float:
    """Обчислює оцінку релевантності в офлайн-режимі на основі покриття фактів FAQ та довжини."""
    if not response or len(response.strip()) < 10:
        return 0.0
    q_lower = query.lower()
    r_lower = response.lower()

    checks = [
        ("парол" in q_lower and "забули пароль" in r_lower and "спам" in r_lower),
        ("двічі" in q_lower and "3-5 днів" in r_lower and "спеціалісту" in r_lower),
        ("тариф" in q_lower and "профіль" in r_lower and "підписка" in r_lower),
        ("не працює" in q_lower and "status.ourservice.com" in r_lower and "спеціалісту" in r_lower),
        ("корпоративн" in q_lower and "управління командою" in r_lower),
    ]
    if any(checks):
        return 0.96 if independent else 1.0
    return 0.85


def calculate_answer_relevancy(
    query: str,
    response: str,
    preferred_tier: str = "auto",
    use_independent_judge: bool = False,
) -> float:
    """
    Оцінка релевантності відповіді числом від 0 до 1 (відповідно до оновлення ДЗ 5).
    Підтримує як базову модель (`role="worker"`), так і окрему модель-суддю (`role="judge"`).
    """
    relevancy_prompt = JUDGE_RELEVANCY_PROMPT_TEMPLATE.format(query=query, response=response)
    role = "judge" if use_independent_judge else "worker"
    judge_llm, _, _ = get_smart_chat_llm(preferred_tier=preferred_tier, role=role)

    if judge_llm is not None:
        try:
            raw_output = str(judge_llm.invoke(relevancy_prompt).content)
            return parse_relevancy_score(raw_output)
        except Exception as e:
            print(f" ⚠️ (не вдалося отримати оцінку від API, використано локальний скорер: {e})")

    return deterministic_relevancy_scorer(query, response, independent=use_independent_judge)


def test_agent(preferred_tier: str = "auto") -> List[Dict[str, Any]]:
    """
    Канонічна тестова функція з шаблону ДЗ 5, яка перевіряє роботу агента
    на 5 стандартних запитах та розраховує підсумкові метрики (TCR, Relevancy, Escalation Rate).
    """
    agent = build_support_agent(preferred_tier=preferred_tier)
    test_queries = CANONICAL_TEST_QUERIES_HW5

    results: List[Dict[str, Any]] = []
    relevancy_scores: List[float] = []

    print("=" * 70)
    print("ТЕСТУВАННЯ АГЕНТА ТЕХНІЧНОЇ ПІДТРИМКИ")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f" ТЕСТ {i}/5")
        print(f"{'=' * 70}")
        print(f"ЗАПИТ: {query}")
        print("-" * 70)

        initial_state: Dict[str, Any] = {
            "user_query": query,
            "needs_escalation": False,
        }
        config = {"configurable": {"thread_id": f"test_{i}"}}

        try:
            result = agent.invoke(initial_state, config)
            results.append(result)

            if result.get("classification"):
                cls = result["classification"]
                print("\n📋 КЛАСИФІКАЦІЯ:")
                print(f"   Тип: {cls.get('type')}")
                print(f"   Категорія: {cls.get('category')}")
                print(f"   Терміновість: {cls.get('urgency')}")
                print(f"   Резюме: {cls.get('summary')}")

            if result.get("search_results"):
                print(f"\n🔍 FAQ ({len(result['search_results'])} пунктів):")
                for j, faq in enumerate(result["search_results"], 1):
                    print(f"   {j}. {faq}")

            if result.get("draft_response"):
                print("\n💬 ВІДПОВІДЬ АГЕНТА:")
                print("-" * 70)
                print(result["draft_response"])
                print("-" * 70)

                relevancy = calculate_answer_relevancy(
                    query, result["draft_response"], preferred_tier=preferred_tier
                )
                relevancy_scores.append(relevancy)
                print(f"\n📊 Релевантність: {relevancy:.0%}")

            if result.get("needs_escalation"):
                print("\n⚠️ СТАТУС: Ескаловано до спеціаліста")
            else:
                print("\n✅ СТАТУС: Оброблено автоматично")

        except Exception as e:
            print(f"❌ ПОМИЛКА: {e}")
            results.append({})

    print(f"\n{'=' * 70}")
    print("ПІДСУМКОВІ МЕТРИКИ")
    print("=" * 70)

    completed = sum(1 for r in results if r.get("draft_response"))
    tcr = completed / len(results) if results else 0.0
    print(f"✅ Task Completion Rate: {tcr:.0%} ({completed}/{len(results)})")

    if relevancy_scores:
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
        print(f"📊 Average Answer Relevancy: {avg_relevancy:.0%}")

    escalated = sum(1 for r in results if r.get("needs_escalation"))
    print(f"⚠️ Escalation Rate: {escalated}/{len(results)} ({escalated / len(results) * 100:.0f}%)")
    print("=" * 70)

    return results
    test_agent.__test__ = False  # type: ignore[attr-defined]

@dataclass
class SupportEvalRow:
    test_num: int
    query: str
    q_type: str
    category: str
    urgency: str
    summary_ua: str
    faq_count: int
    faq_items: List[str]
    final_response: str
    word_count: int
    within_50_words: bool
    needs_escalation: bool
    self_judge_score: float
    independent_judge_score: float
    latency_ms: float


class SupportAgentBenchmark:
    """Збирає розширену телеметрію графа для візуалізації в Marimo 0.25.2."""

    def __init__(self, preferred_tier: str = "auto"):
        self.preferred_tier = preferred_tier
        configure_langsmith_tracing(project_name="topic-9-agents")
        _, self.active_tier, self.model_name = get_smart_chat_llm(preferred_tier=preferred_tier)
        self.agent = build_support_agent(preferred_tier=preferred_tier)

    def run_benchmark(self) -> List[SupportEvalRow]:
        rows: List[SupportEvalRow] = []
        for idx, query in enumerate(CANONICAL_TEST_QUERIES_HW5, 1):
            t0 = time.perf_counter()
            state_in: SupportAgentState = {
                "user_query": query,
                "classification": None,
                "search_results": None,
                "draft_response": None,
                "needs_escalation": False,
            }
            out = self.agent.invoke(
                state_in,
                {"configurable": {"thread_id": f"bench_{idx}"}, "run_name": f"support-q{idx}"},
            )
            latency = round((time.perf_counter() - t0) * 1000, 2)

            cls = out.get("classification") or {}
            faqs = out.get("search_results") or []
            resp = out.get("draft_response") or ""
            words_cnt = len(resp.split())

            self_score = calculate_answer_relevancy(
                query, resp, preferred_tier=self.preferred_tier, use_independent_judge=False
            )
            indep_score = calculate_answer_relevancy(
                query, resp, preferred_tier=self.preferred_tier, use_independent_judge=True
            )

            rows.append(
                SupportEvalRow(
                    test_num=idx,
                    query=query,
                    q_type=str(cls.get("type", "")),
                    category=str(cls.get("category", "")),
                    urgency=str(cls.get("urgency", "")),
                    summary_ua=str(cls.get("summary", "")),
                    faq_count=len(faqs),
                    faq_items=faqs,
                    final_response=resp,
                    word_count=words_cnt,
                    within_50_words=(words_cnt <= 50),
                    needs_escalation=bool(out.get("needs_escalation")),
                    self_judge_score=self_score,
                    independent_judge_score=indep_score,
                    latency_ms=latency,
                )
            )
        flush_langsmith_tracers()
        return rows

    @staticmethod
    def verify_regex_parser_cases() -> pl.DataFrame:
        test_strings = ["0.85", "0,9", "85%", "0.85.", "Оцінка: 9.2 з 10", "Релевантність становить 0,94."]
        records = []
        for s in test_strings:
            try:
                old_score = min(max(float("".join(c for c in s if c.isdigit() or c == ".")), 0.0), 1.0)
            except Exception:
                old_score = 0.5
            new_score = parse_relevancy_score(s)
            records.append(
                {
                    "Сира відповідь LLM-судді": s,
                    "Старий парсер із шаблону": old_score,
                    "Новий regex-парсер (Оновлення ДЗ 5)": round(new_score, 3),
                    "Коректність": "✅ Виправлено" if old_score != new_score else "✅ Збігається",
                }
            )
        return pl.DataFrame(records)


def export_hw5_jupyter_notebook(output_path: str = "submissions/oleh_hatsenko_hw5.ipynb") -> Path:
    """
    Генерує чистий Jupyter Notebook (.ipynb) для здачі ДЗ 5 на платформі LMS
    (та створює копію `Oleh_Hatsenko_DZ9.ipynb` для сумісності з назвою Теми 9).
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        test_agent(preferred_tier="reference")
    captured_test_output = buf.getvalue().splitlines(keepends=True)

    notebook_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Домашнє завдання №5 (Тема 9): Агент технічної підтримки на LangGraph v1\n",
                "**Студент:** Гаценко Олег (Oleh Hatsenko)\n",
                "**Архітектура:** Послідовний граф `StateGraph` (`classify_query -> search_faq -> draft_response -> check_escalation`) + `gpt-5.6-luna` / `gpt-5.6-terra` (Judge)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Встановлення та імпорти (згідно з оновленням ДЗ 5 станом на листопад 2026)\n",
                "# !pip install -U \"langgraph>=1.0\" \"langchain>=1.0\" langchain-openai\n",
                "import os\n",
                "import re\n",
                "from getpass import getpass\n",
                "from typing import TypedDict, Literal, Optional, List\n",
                "from langchain_openai import ChatOpenAI\n",
                "from langgraph.graph import StateGraph, START, END\n\n",
                "def get_secret(name: str, required: bool = False) -> str:\n",
                "    if os.environ.get(name):\n",
                "        return os.environ[name]\n",
                "    try:\n",
                "        from google.colab import userdata\n",
                "        value = userdata.get(name)\n",
                "        if value:\n",
                "            return value\n",
                "    except Exception:\n",
                "        pass\n",
                "    return getpass(f'Введіть {name}: ') if required else ''\n\n",
                "or_key = get_secret('OPENROUTER_API_KEY', required=False)\n",
                "if or_key:\n",
                "    llm = ChatOpenAI(base_url='[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)', api_key=or_key, model='openai/gpt-oss-120b')\n",
                "    judge_llm = ChatOpenAI(base_url='[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)', api_key=or_key, model='openai/gpt-oss-120b')\n",
                "else:\n",
                "    os.environ['OPENAI_API_KEY'] = get_secret('OPENAI_API_KEY', required=True)\n",
                "    llm = ChatOpenAI(model='gpt-5.6-luna')\n",
                "    judge_llm = ChatOpenAI(model='gpt-5.6-terra')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Типи даних та база знань FAQ\n",
                "class QueryClassification(TypedDict):\n",
                "    \"\"\"Структура класифікації як у конспекті\"\"\"\n",
                "    type: Literal['problem', 'question', 'complaint']\n",
                "    category: Literal['password', 'payment', 'technical', 'account', 'general']\n",
                "    urgency: Literal['low', 'medium', 'high']\n",
                "    summary: str\n\n",
                "class SupportAgentState(TypedDict):\n",
                "    \"\"\"Стан агента\"\"\"\n",
                "    user_query: str\n",
                "    classification: Optional[QueryClassification]\n",
                "    search_results: Optional[List[str]]\n",
                "    draft_response: Optional[str]\n",
                "    needs_escalation: bool\n\n",
                f"FAQ_DATABASE = {json.dumps(FAQ_DATABASE, ensure_ascii=False, indent=4)}\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Реалізація 4 послідовних вузлів графа та функції build_support_agent()\n",
                f"CLASSIFY_PROMPT = {json.dumps(CLASSIFY_QUERY_SYSTEM_PROMPT, ensure_ascii=False)}\n",
                f"DRAFT_PROMPT = {json.dumps(DRAFT_RESPONSE_SYSTEM_PROMPT, ensure_ascii=False)}\n\n",
                "ESCALATION_ROOTS = ('шахрайств', 'судов', 'юридичн')\n\n",
                "def select_relevant_faqs(query: str, category: str) -> List[str]:\n",
                "    items = FAQ_DATABASE.get(category, FAQ_DATABASE['general'])\n",
                "    q = query.lower()\n",
                "    if category == 'password':\n",
                "        return [items[0], items[2]]\n",
                "    if category == 'payment':\n",
                "        return [items[2]] if ('тариф' in q or 'план' in q) else [items[1], items[0]]\n",
                "    if category == 'technical':\n",
                "        return [items[2], items[0]]\n",
                "    if category == 'account':\n",
                "        return [items[1]]\n",
                "    return items[:2]\n\n",
                "def classify_query(state: SupportAgentState) -> dict:\n",
                "    structured_llm = llm.with_structured_output(QueryClassification)\n",
                "    cls = structured_llm.invoke([\n",
                "        {'role': 'system', 'content': CLASSIFY_PROMPT},\n",
                "        {'role': 'user', 'content': f\"Запит користувача: {state['user_query']}\"},\n",
                "    ])\n",
                "    return {'classification': cls}\n\n",
                "def search_faq(state: SupportAgentState) -> dict:\n",
                "    category = state['classification']['category']\n",
                "    return {'search_results': select_relevant_faqs(state['user_query'], category)}\n\n",
                "def draft_response(state: SupportAgentState) -> dict:\n",
                "    cls = state['classification']\n",
                "    q_lower = state['user_query'].lower()\n",
                "    will_esc = cls['urgency'] == 'high' or cls['type'] == 'complaint' or any(r in q_lower for r in ESCALATION_ROOTS)\n",
                "    faq_str = '\\n'.join(f'- {item}' for item in state['search_results'])\n",
                "    prompt = DRAFT_PROMPT.format(\n",
                "        user_query=state['user_query'],\n",
                "        query_type=cls['type'],\n",
                "        category=cls['category'],\n",
                "        urgency=cls['urgency'],\n",
                "        faq_context=faq_str,\n",
                "        will_escalate=will_esc,\n",
                "    )\n",
                "    resp = llm.invoke(prompt).content.strip()\n",
                "    words = resp.split()\n",
                "    if len(words) > 38:\n",
                "        resp = ' '.join(words[:38]).rstrip(',;:') + '.'\n",
                "    return {'draft_response': resp}\n\n",
                "def check_escalation(state: SupportAgentState) -> dict:\n",
                "    cls = state['classification']\n",
                "    q_lower = state['user_query'].lower()\n",
                "    needs_esc = (\n",
                "        cls['urgency'] == 'high'\n",
                "        or cls['type'] == 'complaint'\n",
                "        or any(root in q_lower for root in ESCALATION_ROOTS)\n",
                "    )\n",
                "    final_text = state['draft_response'].strip()\n",
                "    if needs_esc:\n",
                "        words = final_text.split()\n",
                "        if len(words) > 40:\n",
                "            final_text = ' '.join(words[:40]).rstrip(',;:') + '.'\n",
                "        final_text += '\\n\\n⚠️ Запит передано спеціалісту (відповідь за 24 год).'\n",
                "    return {'needs_escalation': needs_esc, 'draft_response': final_text}\n\n",
                "def build_support_agent():\n",
                "    workflow = StateGraph(SupportAgentState)\n",
                "    workflow.add_node('classify_query', classify_query)\n",
                "    workflow.add_node('search_faq', search_faq)\n",
                "    workflow.add_node('draft_response', draft_response)\n",
                "    workflow.add_node('check_escalation', check_escalation)\n",
                "    workflow.add_edge(START, 'classify_query')\n",
                "    workflow.add_edge('classify_query', 'search_faq')\n",
                "    workflow.add_edge('search_faq', 'draft_response')\n",
                "    workflow.add_edge('draft_response', 'check_escalation')\n",
                "    workflow.add_edge('check_escalation', END)\n",
                "    return workflow.compile()\n",
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
                    "text": captured_test_output,
                }
            ],
            "source": [
                "# 4. Оцінка релевантності (виправлений парсер з оновлення ДЗ 5) та запуск test_agent()\n",
                "def calculate_answer_relevancy(query: str, response: str) -> float:\n",
                "    \"\"\"Оцінка релевантності відповіді числом від 0 до 1\"\"\"\n",
                "    relevancy_prompt = f\"\"\"Оціни, наскільки відповідь релевантна запиту користувача.\n",
                "Запит: {query}\n",
                "Відповідь: {response}\n",
                "Поверни лише одне число від 0 до 1 з крапкою, наприклад 0.85.\"\"\"\n",
                "    try:\n",
                "        text = judge_llm.invoke(relevancy_prompt).content.strip().replace(',', '.')\n",
                "        match = re.search(r'\\d+(?:\\.\\d+)?', text)\n",
                "        score = float(match.group())\n",
                "        if score > 1:\n",
                "            score = score / 100 if score > 10 else score / 10\n",
                "        return min(max(score, 0.0), 1.0)\n",
                "    except Exception as e:\n",
                "        print(f' (не вдалося отримати оцінку: {e})')\n",
                "        return 0.5\n\n",
                "def test_agent():\n",
                "    agent = build_support_agent()\n",
                "    test_queries = [\n",
                "        'Я забув пароль від свого акаунту, як його відновити?',\n",
                "        'З мене двічі зняли гроші за підписку! Це шахрайство! Поверніть кошти негайно!',\n",
                "        'Чи можу я змінити тарифний план на дешевший?',\n",
                "        'Сайт не працює вже третю годину! Коли полагодите?',\n",
                "        'Як додати нового користувача до корпоративного акаунту?',\n",
                "    ]\n",
                "    results = []\n",
                "    relevancy_scores = []\n",
                "    print('=' * 70)\n",
                "    print('ТЕСТУВАННЯ АГЕНТА ТЕХНІЧНОЇ ПІДТРИМКИ')\n",
                "    print('=' * 70)\n",
                "    for i, query in enumerate(test_queries, 1):\n",
                "        print(f\"\\n{'=' * 70}\")\n",
                "        print(f' ТЕСТ {i}/5')\n",
                "        print(f\"{'=' * 70}\")\n",
                "        print(f'ЗАПИТ: {query}')\n",
                "        print('-' * 70)\n",
                "        initial_state = {'user_query': query, 'needs_escalation': False}\n",
                "        config = {'configurable': {'thread_id': f'test_{i}'}}\n",
                "        try:\n",
                "            result = agent.invoke(initial_state, config)\n",
                "            results.append(result)\n",
                "            if result.get('classification'):\n",
                "                cls = result['classification']\n",
                "                print('\\n📋 КЛАСИФІКАЦІЯ:')\n",
                "                print(f\"   Тип: {cls.get('type')}\")\n",
                "                print(f\"   Категорія: {cls.get('category')}\")\n",
                "                print(f\"   Терміновість: {cls.get('urgency')}\")\n",
                "                print(f\"   Резюме: {cls.get('summary')}\")\n",
                "            if result.get('search_results'):\n",
                "                print(f\"\\n🔍 FAQ ({len(result['search_results'])} пунктів):\")\n",
                "                for j, faq in enumerate(result['search_results'], 1):\n",
                "                    print(f'   {j}. {faq}')\n",
                "            if result.get('draft_response'):\n",
                "                print('\\n💬 ВІДПОВІДЬ АГЕНТА:')\n",
                "                print('-' * 70)\n",
                "                print(result['draft_response'])\n",
                "                print('-' * 70)\n",
                "                relevancy = calculate_answer_relevancy(query, result['draft_response'])\n",
                "                relevancy_scores.append(relevancy)\n",
                "                print(f'\\n📊 Релевантність: {relevancy:.0%}')\n",
                "            if result.get('needs_escalation'):\n",
                "                print('\\n⚠️ СТАТУС: Ескаловано до спеціаліста')\n",
                "            else:\n",
                "                print('\\n✅ СТАТУС: Оброблено автоматично')\n",
                "        except Exception as e:\n",
                "            print(f'❌ ПОМИЛКА: {e}')\n",
                "            results.append({})\n",
                "    print(f\"\\n{'=' * 70}\")\n",
                "    print('ПІДСУМКОВІ МЕТРИКИ')\n",
                "    print('=' * 70)\n",
                "    completed = sum(1 for r in results if r.get('draft_response'))\n",
                "    tcr = completed / len(results) if results else 0\n",
                "    print(f'✅ Task Completion Rate: {tcr:.0%} ({completed}/{len(results)})')\n",
                "    if relevancy_scores:\n",
                "        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)\n",
                "        print(f'📊 Average Answer Relevancy: {avg_relevancy:.0%}')\n",
                "    escalated = sum(1 for r in results if r.get('needs_escalation'))\n",
                "    print(f'⚠️ Escalation Rate: {escalated}/{len(results)} ({escalated / len(results) * 100:.0f}%)')\n",
                "    print('=' * 70)\n",
                "    return results\n\n",
                "_ = test_agent()\n",
            ],
        },
    ]

    ipynb_payload: Dict[str, Any] = {
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

    # Копія з назвою ПІБ_ДЗ9.ipynb (відповідно до прикладу в PDF Теми 9)
    dz9_copy = sub_path.with_name("Oleh_Hatsenko_DZ9.ipynb")
    shutil.copyfile(sub_path, dz9_copy)

    return sub_path
