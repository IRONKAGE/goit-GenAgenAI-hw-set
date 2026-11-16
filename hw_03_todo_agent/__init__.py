# ==============================================================================
# 📝 HW 03 PACKAGE: LangChain v1 TODO Agent, InMemoryStore & HITL Middleware
# ==============================================================================
"""Ініціалізація пакета ДЗ 3 для монорепозиторію goit-GenAgenAI-hw-set."""

from hw_03_todo_agent.agent import (
    CANONICAL_TESTS_HW3,
     AgentStepLog,
    DeterministicReActTodoChatModel,
    TodoAgentOrchestrator,
    build_todo_agent,
    export_hw3_jupyter_notebook,
)
from hw_03_todo_agent.middleware_hitl import (
    CustomTodoState,
    build_dynamic_model_todo_agent,
    build_hitl_todo_agent,
    get_session_diagnostics,
    increment_ops_counter,
    simulate_hitl_delete_approval,
)
from hw_03_todo_agent.prompts import (
    ADD_TASK_DOCSTRING,
    DELETE_TASK_DOCSTRING,
    LIST_TASKS_DOCSTRING,
    TODO_AGENT_SYSTEM_PROMPT,
    TOOL_MESSAGES,
)
from hw_03_todo_agent.store_tools import (
    TASKS_NAMESPACE,
    TaskInputSchema,
    add_task,
    delete_task,
    list_tasks,
)
from hw_03_todo_agent.report_builder import generate_hw3_google_docs_markdown
from hw_03_todo_agent.ui_labels import UI_LABELS

__all__ = [
    "TASKS_NAMESPACE",
    "TaskInputSchema",
    "add_task",
    "list_tasks",
    "delete_task",
    "DeterministicReActTodoChatModel",
    "AgentStepLog",
    "build_todo_agent",
    "TodoAgentOrchestrator",
    "export_hw3_jupyter_notebook",
    "CANONICAL_TESTS_HW3",
    "CustomTodoState",
    "get_session_diagnostics",
    "increment_ops_counter",
    "build_hitl_todo_agent",
    "build_dynamic_model_todo_agent",
    "simulate_hitl_delete_approval",
    "TODO_AGENT_SYSTEM_PROMPT",
    "ADD_TASK_DOCSTRING",
    "LIST_TASKS_DOCSTRING",
    "DELETE_TASK_DOCSTRING",
    "TOOL_MESSAGES",
    "generate_hw3_google_docs_markdown",
    "UI_LABELS",
]
