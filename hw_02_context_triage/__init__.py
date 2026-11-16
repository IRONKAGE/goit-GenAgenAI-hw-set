# ==============================================================================
# 🎯 HW 02 PACKAGE: Context Engineering, JSON Triage & LCEL Structured Outputs
# ==============================================================================
"""Ініціалізація пакета ДЗ 2 для монорепозиторію goit-GenAgenAI-hw-set."""

from hw_02_context_triage.corpora import (
    STRESS_TEST_CORPUS,
    TASK1_CORPUS,
    TASK1_EXPECTED,
    TASK2_CORPUS,
    TASK2_EXPECTED,
    TASK3_CORPUS,
    TASK3_EXPECTED,
)
from hw_02_context_triage.extractor import (
    ContextTriageExtractor,
    ExtractionRecord,
    export_hw2_jupyter_notebook,
)
from hw_02_context_triage.lcel_structured import (
    StreamingTriageDict,
    UnifiedTriageOutput,
    build_parallel_summary_chain,
    build_triage_router_chain,
    simulate_typeddict_stream,
)
from hw_02_context_triage.prompts import (
    JSON_SELF_CORRECTION_PROMPT,
    PROMPT_ABLATION_REASONING_FIRST,
    PROMPT_TASK1_PERSONA_TONE,
    PROMPT_TASK2_PRODUCT_PRIORITY,
    PROMPT_TASK3_TITLE_CATEGORY,
    USER_TEMPLATE_TASK1,
    USER_TEMPLATE_TASK2,
    USER_TEMPLATE_TASK3,
)
from hw_02_context_triage.schemas import (
    ALLOWED,
    ALLOWED_CATEGORIES,
    ALLOWED_PRIORITIES,
    ALLOWED_PRODUCTS,
    ALLOWED_TONES,
    PersonaToneModel,
    ProductPriorityModel,
    TitleCategoryModel,
    check,
    check_task2,
    check_task3,
)
from hw_02_context_triage.report_builder import generate_hw2_google_docs_markdown
from hw_02_context_triage.ui_labels import UI_LABELS

__all__ = [
    "ContextTriageExtractor",
    "ExtractionRecord",
    "export_hw2_jupyter_notebook",
    "StreamingTriageDict",
    "UnifiedTriageOutput",
    "build_triage_router_chain",
    "build_parallel_summary_chain",
    "simulate_typeddict_stream",
    "PROMPT_TASK1_PERSONA_TONE",
    "PROMPT_TASK2_PRODUCT_PRIORITY",
    "PROMPT_TASK3_TITLE_CATEGORY",
    "USER_TEMPLATE_TASK1",
    "USER_TEMPLATE_TASK2",
    "USER_TEMPLATE_TASK3",
    "JSON_SELF_CORRECTION_PROMPT",
    "PROMPT_ABLATION_REASONING_FIRST",
    "TASK1_CORPUS",
    "TASK1_EXPECTED",
    "TASK2_CORPUS",
    "TASK2_EXPECTED",
    "TASK3_CORPUS",
    "TASK3_EXPECTED",
    "STRESS_TEST_CORPUS",
    "ALLOWED",
    "ALLOWED_TONES",
    "ALLOWED_PRODUCTS",
    "ALLOWED_PRIORITIES",
    "ALLOWED_CATEGORIES",
    "check",
    "check_task2",
    "check_task3",
    "PersonaToneModel",
    "ProductPriorityModel",
    "TitleCategoryModel",
    "generate_hw2_google_docs_markdown",
    "UI_LABELS",
]
