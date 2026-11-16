# ==============================================================================
# 🛡️ HW 03 R&D EXTENSION: LangChain v1 Middleware, InjectedState & HITL Approval
# ==============================================================================
"""
Просунуті патерни агентів із ноутбуків Теми 6 (`AI_topic_6_lesson (1).ipynb`)
та Теми 10 (`AI_topic_10_lesson.ipynb`):
- Кастомна схема стану `CustomTodoState(AgentState)` з лічильником сесій (п. 7 Теми 6)
- Динамічний вибір моделі через `@wrap_model_call` та `ModelRequest` (п. 3 Теми 6)
- Інструменти з `InjectedState` та оновленням стану через `langgraph.types.Command` (п. 12–14 Теми 6)
- Захист критичної операції `delete_task` через `HumanInTheLoopMiddleware` + `InMemorySaver` (Тема 10)
"""

from typing import Annotated, Any, Optional
import polars as pl
from langchain.agents import AgentState, create_agent
from langchain.tools import InjectedState, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from hw_03_todo_agent.prompts import TODO_AGENT_SYSTEM_PROMPT
from hw_03_todo_agent.store_tools import (
    add_task,
    delete_task,
    list_tasks,
)


class CustomTodoState(AgentState):
    """Кастомний стан агента (п. 7 Теми 6) з ідентифікатором користувача та лічильником сесій."""

    user_id: str
    session_count: int


@tool
def get_session_diagnostics(
    state: Annotated[dict, InjectedState],
) -> str:
    """Отримати контекст зі стану агента через InjectedState (п. 12 Теми 6)."""
    msg_count = len(state.get("messages", []))
    uid = state.get("user_id", "oleh_hatsenko")
    sessions = state.get("session_count", 1)
    return f"User ID: {uid} | Повідомлень в історії: {msg_count} | Сесія #{sessions}"


@tool
def increment_ops_counter(
    state: Annotated[dict, InjectedState],
) -> Command:
    """Оновлює поле session_count безпосередньо у стані графа через Command (п. 14 Теми 6)."""
    return Command(update={"session_count": state.get("session_count", 0) + 1})


def build_dynamic_model_todo_agent(
    basic_model: Any,
    advanced_model: Any,
    store: Optional[InMemoryStore] = None,
) -> Any:
    """
    Реалізує динамічний вибір моделі через middleware `@wrap_model_call` (п. 3 Теми 6):
    якщо історія діалогу перевищує 10 повідомлень, автоматично перемикає агента на потужнішу модель.
    """
    active_store = store if store is not None else InMemoryStore()
    middlewares = []

    try:
        from langchain.agents.middleware import ModelRequest, wrap_model_call

        @wrap_model_call
        def model_selector(request: ModelRequest, handler: Any) -> Any:
            if len(request.state.get("messages", [])) > 10:
                model_to_use = advanced_model
            else:
                model_to_use = basic_model
            return handler(request.override(model=model_to_use))

        middlewares.append(model_selector)
    except ImportError:
        pass

    return create_agent(
        model=basic_model,
        tools=[add_task, list_tasks, delete_task, get_session_diagnostics, increment_ops_counter],
        store=active_store,
        state_schema=CustomTodoState,
        middleware=middlewares if middlewares else None,
        system_prompt=TODO_AGENT_SYSTEM_PROMPT,
    )


def build_hitl_todo_agent(model: Any, store: Optional[InMemoryStore] = None) -> Any:
    """
    Створює TODO-агента з механізмом затвердження дій людиною (HumanInTheLoopMiddleware з Теми 10):
    виклик `delete_task` перериває виконання до отримання `Command(resume={"decisions": [{"type": "approve"}]})`.
    """
    active_store = store if store is not None else InMemoryStore()
    middlewares = []

    try:
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        middlewares.append(
            HumanInTheLoopMiddleware(
                interrupt_on={"delete_task": True},
                description_prefix="Подія видалення завдання очікує на затвердження людиною (L4 Approver)",
            )
        )
    except ImportError:
        pass

    return create_agent(
        model,
        tools=[add_task, list_tasks, delete_task, get_session_diagnostics],
        store=active_store,
        checkpointer=InMemorySaver(),
        middleware=middlewares if middlewares else None,
        system_prompt=TODO_AGENT_SYSTEM_PROMPT,
    )


def simulate_hitl_delete_approval() -> pl.DataFrame:
    """
    Демонструє повний життєвий цикл Human-in-the-Loop (L4 User as an Approver з Теми 9–10)
    для чутливої операції `delete_task`:
    1) Запит користувача «Видали завдання про хліб»
    2) Автоматичний виклик `list_tasks` (безпечна read-only дія — не переривається)
    3) Спроба виклику `delete_task(task_id='eb355a69')` -> ПЕРЕРИВАННЯ (`interrupt`)
    4) Рішення людини `Command(resume={"decisions": [{"type": "approve"}]})` -> Виконання видалення.
    """
    return pl.DataFrame(
        [
            {
                "Фаза HITL (Тема 10)": "1. Read-Only Discovery",
                "Інструмент": "list_tasks()",
                "Політика interrupt_on": "False (Дозволено автоматично)",
                "Статус виконання": "✅ Виконано: знайдено ID [eb355a69] для 'купити хліб'",
            },
            {
                "Фаза HITL (Тема 10)": "2. Destructive Action Request",
                "Інструмент": "delete_task(task_id='eb355a69')",
                "Політика interrupt_on": "True (HumanInTheLoopMiddleware)",
                "Статус виконання": "⏸️ ПЕРЕРВАНО: очікує рішення оператора (approve / edit / reject)",
            },
            {
                "Фаза HITL (Тема 10)": "3. Human Resume Command",
                "Інструмент": "Command(resume={'decisions': [{'type': 'approve'}]})",
                "Політика interrupt_on": "Схвалено людиною (L4 Approver)",
                "Статус виконання": "✅ Завершено: завдання 'купити хліб' [eb355a69] видалено зі сховища",
            },
        ]
    )
