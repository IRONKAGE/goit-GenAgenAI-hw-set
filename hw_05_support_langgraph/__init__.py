# ==============================================================================
# 🛠️ HW 05 PACKAGE: LangGraph v1 Technical Support Agent & Dual LLM-as-Judge
# ==============================================================================
"""Ініціалізація пакета ДЗ 5 для монорепозиторію goit-GenAgenAI-hw-set."""

from hw_05_support_langgraph.advanced_hitl_graph import (
    build_hitl_command_support_graph,
    simulate_topic9_hitl_interrupt_resume,
)
from hw_05_support_langgraph.evaluator import (
    SupportAgentBenchmark,
    SupportEvalRow,
    calculate_answer_relevancy,
    export_hw5_jupyter_notebook,
    test_agent,
)
from hw_05_support_langgraph.graph import build_support_agent
from hw_05_support_langgraph.nodes import (
    check_escalation_node,
    classify_query_node,
    deterministic_classify,
    deterministic_draft_response,
    draft_response_node,
    is_escalation_required,
    search_faq_node,
    select_relevant_faq_items,
)
from hw_05_support_langgraph.prompts import (
    CLASSIFY_QUERY_SYSTEM_PROMPT,
    CLASSIFY_QUERY_USER_PROMPT,
    DRAFT_RESPONSE_SYSTEM_PROMPT,
    ESCALATION_KEYWORD_ROOTS,
    ESCALATION_NOTICE_SUFFIX,
    FAQ_DATABASE,
    JUDGE_RELEVANCY_PROMPT_TEMPLATE,
    SEARCH_FAQ_RANKER_PROMPT,
)
from hw_05_support_langgraph.state import (
    CANONICAL_TEST_QUERIES_HW5,
    QueryClassification,
    SupportAgentState,
)
from hw_05_support_langgraph.report_builder import generate_hw5_google_docs_markdown
from hw_05_support_langgraph.ui_labels import UI_LABELS

__all__ = [
    "QueryClassification",
    "SupportAgentState",
    "CANONICAL_TEST_QUERIES_HW5",
    "FAQ_DATABASE",
    "ESCALATION_KEYWORD_ROOTS",
    "ESCALATION_NOTICE_SUFFIX",
    "CLASSIFY_QUERY_SYSTEM_PROMPT",
    "CLASSIFY_QUERY_USER_PROMPT",
    "SEARCH_FAQ_RANKER_PROMPT",
    "DRAFT_RESPONSE_SYSTEM_PROMPT",
    "JUDGE_RELEVANCY_PROMPT_TEMPLATE",
    "is_escalation_required",
    "deterministic_classify",
    "select_relevant_faq_items",
    "deterministic_draft_response",
    "classify_query_node",
    "search_faq_node",
    "draft_response_node",
    "check_escalation_node",
    "build_support_agent",
    "calculate_answer_relevancy",
    "test_agent",
    "SupportEvalRow",
    "SupportAgentBenchmark",
    "export_hw5_jupyter_notebook",
    "build_hitl_command_support_graph",
    "simulate_topic9_hitl_interrupt_resume",
    "generate_hw5_google_docs_markdown",
    "UI_LABELS",
]
