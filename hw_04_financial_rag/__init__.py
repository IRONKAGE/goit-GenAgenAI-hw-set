# ==============================================================================
# 📈 HW 04 PACKAGE: Financial Reports RAG, Advanced Query Transforms & RAGAS
# ==============================================================================
"""Ініціалізація пакета ДЗ 4 для монорепозиторію goit-GenAgenAI-hw-set."""

from hw_04_financial_rag.advanced_rag import (
    build_step_back_financial_prompt,
    format_qa_pairs,
    get_unique_union,
    reciprocal_rank_fusion,
    run_advanced_rag_comparison,
)
from hw_04_financial_rag.chunker import split_financial_reports
from hw_04_financial_rag.prompts import (
    DECOMPOSITION_FINANCIAL_PROMPT,
    FINANCIAL_RAG_SYSTEM_PROMPT,
    HYDE_DOCUMENT_GENERATOR_PROMPT,
    MULTI_QUERY_EXPANSION_PROMPT,
    RAG_FUSION_QUERY_PROMPT,
    RAW_FINANCIAL_REPORTS_TEXT,
    SELF_RAG_FAITHFULNESS_GRADER_PROMPT,
    TEST_QUESTIONS_HW4,
)
from hw_04_financial_rag.rag_chain import (
    DeterministicFinancialRAGChatModel,
    FinancialRAGOrchestrator,
    RAGExecutionRecord,
    build_rag_lcel_chain,
    export_hw4_jupyter_notebook,
    format_docs,
)
from hw_04_financial_rag.report_builder import generate_hw4_google_docs_markdown
from hw_04_financial_rag.ui_labels import UI_LABELS
from hw_04_financial_rag.vector_store import (
    COLLECTION_NAME_HW4,
    build_chroma_vectorstore,
    compute_cosine_similarity_matrix,
)

__all__ = [
    "RAW_FINANCIAL_REPORTS_TEXT",
    "TEST_QUESTIONS_HW4",
    "FINANCIAL_RAG_SYSTEM_PROMPT",
    "MULTI_QUERY_EXPANSION_PROMPT",
    "RAG_FUSION_QUERY_PROMPT",
    "DECOMPOSITION_FINANCIAL_PROMPT",
    "HYDE_DOCUMENT_GENERATOR_PROMPT",
    "SELF_RAG_FAITHFULNESS_GRADER_PROMPT",
    "split_financial_reports",
    "COLLECTION_NAME_HW4",
    "build_chroma_vectorstore",
    "compute_cosine_similarity_matrix",
    "DeterministicFinancialRAGChatModel",
    "RAGExecutionRecord",
    "format_docs",
    "build_rag_lcel_chain",
    "FinancialRAGOrchestrator",
    "export_hw4_jupyter_notebook",
    "get_unique_union",
    "reciprocal_rank_fusion",
    "format_qa_pairs",
    "build_step_back_financial_prompt",
    "run_advanced_rag_comparison",
    "generate_hw4_google_docs_markdown",
    "UI_LABELS",
]
