# ==============================================================================
# 🔧 HW 03 STORE TOOLS: Інструменти роботи з LangGraph InMemoryStore
# ==============================================================================
"""
Реалізація інструментів `add_task`, `list_tasks`, `delete_task` за суворими
критеріями прийняття ДЗ 3 та рекомендаціями оновлення (листопад 2026):
- Доступ до сховища через `get_store()`
- Один і той самий 8-символьний ID від `uuid.uuid4()` при збереженні та виводі
- Передача `limit=1000` у `store.search(TASKS_NAMESPACE, limit=1000)`
- Обов'язковий пошук через `store.get()` перед `store.delete()`
- Жодних необроблених винятків у штатних ситуаціях (порожній список, невідомий ID)
"""

import uuid
from datetime import datetime, timezone
from typing import Tuple

from langchain_core.tools import tool
from langgraph.config import get_store
from pydantic import BaseModel, Field

from hw_03_todo_agent.prompts import (
    ADD_TASK_DOCSTRING,
    DELETE_TASK_DOCSTRING,
    LIST_TASKS_DOCSTRING,
    TOOL_MESSAGES,
)

# Простір імен (namespace) у InMemoryStore
TASKS_NAMESPACE: Tuple[str, ...] = ("todo_agent", "tasks")


class TaskInputSchema(BaseModel):
    """Pydantic-схема вхідних параметрів для додавання завдання (п. 11 Теми 6)."""

    title: str = Field(..., description="Чиста назва завдання українською мовою без слова 'Додай:'")


@tool(args_schema=TaskInputSchema, description=ADD_TASK_DOCSTRING)
def add_task(title: str) -> str:
    """Додає нове завдання у сховище InMemoryStore з коротким 8-символьним UUID."""
    store = get_store()
    clean_title = title.strip()
    if clean_title.lower().startswith("додай:"):
        clean_title = clean_title[6:].strip()

    if not clean_title:
        return "Не вдалося додати завдання: опис завдання порожній."

    # Генеруємо 8-символьний ID від uuid.uuid4() і зберігаємо під ним же (критично для delete_task!)
    task_id = str(uuid.uuid4())[:8]
    store.put(
        TASKS_NAMESPACE,
        task_id,
        {
            "id": task_id,
            "title": clean_title,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return TOOL_MESSAGES["added"].format(title=clean_title, task_id=task_id)


@tool(description=LIST_TASKS_DOCSTRING)
def list_tasks() -> str:
    """Переглядає всі збережені завдання у сховищі InMemoryStore (limit=1000)."""
    store = get_store()
    # За замовчуванням store.search() повертає лише 10 записів, тому передаємо limit=1000
    items = store.search(TASKS_NAMESPACE, limit=1000)

    if not items:
        return TOOL_MESSAGES["empty_list"]

    lines = [TOOL_MESSAGES["list_header"]]
    for idx, item in enumerate(items, 1):
        task_title = item.value.get("title", "Без назви")
        task_id = item.key
        lines.append(f"{idx}. {task_title} [{task_id}]")

    return "\n".join(lines)


@tool(description=DELETE_TASK_DOCSTRING)
def delete_task(task_id: str) -> str:
    """Шукає завдання через store.get() та видаляє його через store.delete()."""
    store = get_store()
    # Очищуємо ID від можливих квадратних дужок або пробілів, якщо модель передала "[eb355a69]"
    clean_id = task_id.strip().strip("[]").strip()

    if not clean_id:
        return "Помилка: не вказано ID завдання для видалення."

    # 1. Обов'язковий пошук завдання через store.get() за критеріями прийняття
    existing_item = store.get(TASKS_NAMESPACE, clean_id)

    # 2. Безпечна обробка випадку, коли завдання з таким ID не знайдено (без винятків Python)
    if existing_item is None:
        return TOOL_MESSAGES["not_found"].format(task_id=clean_id)

    task_title = existing_item.value.get("title", clean_id)

    # 3. Видалення завдання через store.delete()
    store.delete(TASKS_NAMESPACE, clean_id)

    # 4. Інформативне повідомлення
    return TOOL_MESSAGES["deleted"].format(title=task_title, task_id=clean_id)
