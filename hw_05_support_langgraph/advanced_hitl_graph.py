# ==============================================================================
# 🚦 HW 05 R&D EXTENSION: Topic 9 Command Routing, interrupt() & MemorySaver
# ==============================================================================
"""
Реалізація просунутого патерну з ноутбука Теми 9 (`AI_topic_9_lesson.ipynb`):
- Вузол `draft_response_with_command` повертає `Command[Literal["human_review", "send_reply"]]`
- Вузол `human_review` зупиняє виконання графа через `langgraph.types.interrupt(...)`
- Чекпоінтер `MemorySaver()` зберігає стан потоку (`thread_id`) між зупинкою та `Command(resume=...)`
"""

from typing import Any, Dict, Literal
import polars as pl
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from hw_05_support_langgraph.nodes import (
    classify_query_node,
    is_escalation_required,
    search_faq_node,
)
from hw_05_support_langgraph.state import SupportAgentState


def draft_and_route_command(
    state: SupportAgentState,
) -> Command[Literal["human_review", "auto_finalize"]]:
    """
    Формує чернетку відповіді та динамічно маршрутизує потік виконання через Command
    (патерн Частини 1 ноутбука Теми 9).
    """
    from hw_05_support_langgraph.nodes import draft_response_node

    draft_update = draft_response_node(state, preferred_tier="reference")
    draft_text = draft_update["draft_response"]
    needs_review = is_escalation_required(state["user_query"], state.get("classification"))
    goto_node: Literal["human_review", "auto_finalize"] = (
        "human_review" if needs_review else "auto_finalize"
    )
    return Command(
        update={"draft_response": draft_text, "needs_escalation": needs_review},
        goto=goto_node,
    )


def human_review_node(
    state: SupportAgentState,
) -> Command[Literal["auto_finalize", "__end__"]]:
    """
    Зупиняє виконання графа через `interrupt(...)` для перегляду та редагування
    відповіді людиною-оператором (патерн `human_review` із ноутбука Теми 9).
    """
    cls = state.get("classification") or {}
    human_decision = interrupt(
        {
            "original_query": state.get("user_query", ""),
            "draft_response": state.get("draft_response", ""),
            "urgency": cls.get("urgency"),
            "type": cls.get("type"),
            "action": "Будь ласка, перегляньте та схваліть або відредагуйте цю відповідь перед відправкою",
        }
    )

    if isinstance(human_decision, dict) and human_decision.get("approved"):
        edited = human_decision.get("edited_response", state.get("draft_response", ""))
        return Command(
            update={"draft_response": edited, "needs_escalation": True},
            goto="auto_finalize",
        )
    return Command(update={}, goto=END)


def auto_finalize_node(state: SupportAgentState) -> Dict[str, Any]:
    """Фіналізує схвалену або автоматичну відповідь."""
    return {"draft_response": state.get("draft_response", "")}


def build_hitl_command_support_graph() -> Any:
    """Компілює граф із `MemorySaver()` та підтримкою `interrupt` / `Command(resume=...)`."""
    workflow = StateGraph(SupportAgentState)
    workflow.add_node("classify_query", lambda s: classify_query_node(s, preferred_tier="reference"))
    workflow.add_node("search_faq", search_faq_node, retry_policy=RetryPolicy(max_attempts=3))
    workflow.add_node("draft_and_route", draft_and_route_command)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("auto_finalize", auto_finalize_node)

    workflow.add_edge(START, "classify_query")
    workflow.add_edge("classify_query", "search_faq")
    workflow.add_edge("search_faq", "draft_and_route")
    workflow.add_edge("auto_finalize", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def simulate_topic9_hitl_interrupt_resume() -> pl.DataFrame:
    """
    Виконує реальний двофазний цикл `invoke -> interrupt -> Command(resume=...)`
    із ноутбука Теми 9 на запиті №2 («З мене двічі зняли гроші за підписку! Це шахрайство!»).
    """
    app_hitl = build_hitl_command_support_graph()
    config = {"configurable": {"thread_id": "customer_123"}}

    initial_state: SupportAgentState = {
        "user_query": "З мене двічі зняли гроші за підписку! Це шахрайство! Поверніть кошти негайно!",
        "classification": None,
        "search_results": None,
        "draft_response": None,
        "needs_escalation": False,
    }

    # Фаза 1: Перший запуск зупиняється на вузлі `human_review` через interrupt()
    paused_state = app_hitl.invoke(initial_state, config)
    draft_before_human = paused_state.get("draft_response", "")

    # Фаза 2: Оператор схвалює та редагує відповідь через Command(resume=...)
    edited_by_operator = (
        "Щиро перепрошуємо за подвійне списання. Кошти повернуться автоматично за 3-5 днів, "
        "а я вже ініціював пріоритетну перевірку вашої транзакції фінансовим відділом."
    )
    resumed_state = app_hitl.invoke(
        Command(resume={"approved": True, "edited_response": edited_by_operator}),
        config,
    )

    return pl.DataFrame(
        [
            {
                "Етап життєвого циклу (Тема 9)": "1. Первинний виклик app.invoke(initial_state, config)",
                "Активний вузол": "classify_query ➔ search_faq ➔ draft_and_route ➔ human_review",
                "Стан потоку (MemorySaver)": "⏸️ Paused at interrupt() (thread_id='customer_123')",
                "Текст відповіді у стані": draft_before_human,
            },
            {
                "Етап життєвого циклу (Тема 9)": "2. Відновлення app.invoke(Command(resume={'approved': True, ...}))",
                "Активний вузол": "human_review ➔ auto_finalize ➔ END",
                "Стан потоку (MemorySaver)": "✅ Completed & Sent",
                "Текст відповіді у стані": resumed_state.get("draft_response", ""),
            },
        ]
    )
