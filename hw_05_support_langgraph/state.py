# ==============================================================================
# 📋 HW 05 STATE DEFINITIONS: QueryClassification & SupportAgentState
# ==============================================================================
"""
Канонічні структури стану агента технічної підтримки на базі TypedDict
відповідно до шаблону ДЗ 5.
"""

from typing import List, Literal, Optional, TypedDict


class QueryClassification(TypedDict):
    """Структура класифікації запиту користувача (як у конспекті та шаблоні ДЗ 5)."""

    type: Literal["problem", "question", "complaint"]
    category: Literal["password", "payment", "technical", "account", "general"]
    urgency: Literal["low", "medium", "high"]
    summary: str


class SupportAgentState(TypedDict):
    """Стан агента технічної підтримки, що передається між 4 вузлами графа LangGraph."""

    user_query: str
    classification: Optional[QueryClassification]
    search_results: Optional[List[str]]
    draft_response: Optional[str]
    needs_escalation: bool


# 5 стандартних запитів для тестування з умови ДЗ 5
CANONICAL_TEST_QUERIES_HW5: List[str] = [
    "Я забув пароль від свого акаунту, як його відновити?",
    "З мене двічі зняли гроші за підписку! Це шахрайство! Поверніть кошти негайно!",
    "Чи можу я змінити тарифний план на дешевший?",
    "Сайт не працює вже третю годину! Коли полагодите?",
    "Як додати нового користувача до корпоративного акаунту?",
]
