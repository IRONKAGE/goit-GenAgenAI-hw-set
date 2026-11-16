# ==============================================================================
# 🕸️ HW 05 GRAPH BUILDER: Послідовний StateGraph (build_support_agent)
# ==============================================================================
"""
Точка входу `build_support_agent()`, яка компілює строго послідовний граф LangGraph v1
із чотирьох вузлів: classify_query -> search_faq -> draft_response -> check_escalation.
"""

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from hw_05_support_langgraph.nodes import (
    check_escalation_node,
    classify_query_node,
    draft_response_node,
    search_faq_node,
)
from hw_05_support_langgraph.state import SupportAgentState


def build_support_agent(preferred_tier: str = "auto") -> Any:
    """
    Створює та компілює граф агента технічної підтримки на базі StateGraph(SupportAgentState).
    Усі 4 вузли виконуються строго послідовно для кожного запиту.
    Додано RetryPolicy(max_attempts=3) на вузлах пошуку та LLM-викликів (патерн Теми 9).
    """
    workflow = StateGraph(SupportAgentState)
    retry_cfg = RetryPolicy(max_attempts=3)

    workflow.add_node(
        "classify_query",
        partial(classify_query_node, preferred_tier=preferred_tier),
        retry_policy=retry_cfg,
    )
    workflow.add_node(
        "search_faq",
        search_faq_node,
        retry_policy=retry_cfg,
    )
    workflow.add_node(
        "draft_response",
        partial(draft_response_node, preferred_tier=preferred_tier),
        retry_policy=retry_cfg,
    )
    workflow.add_node("check_escalation", check_escalation_node)

    workflow.add_edge(START, "classify_query")
    workflow.add_edge("classify_query", "search_faq")
    workflow.add_edge("search_faq", "draft_response")
    workflow.add_edge("draft_response", "check_escalation")
    workflow.add_edge("check_escalation", END)

    return workflow.compile()
