# ==============================================================================
# 🌍 HW 01 PACKAGE: Pre-training, Alignment, Sampling Math & Base vs Instruct
# ==============================================================================
"""Ініціалізація пакета ДЗ 1 для монорепозиторію goit-GenAgenAI-hw-set."""

from hw_01_model_alignment.benchmark import (
    AlignmentRecord,
    HW1BenchmarkRunner,
    PoemEvaluation,
    analyze_bpe_tokenization_fertility,
    compute_sampling_distribution,
    evaluate_ukrainian_poem,
)
from hw_01_model_alignment.prompts import (
    AlignmentPair,
    HW1_REPORT_SYNTHESIS_PROMPT,
    TASK1_KNOWLEDGE_CUTOFF_PROMPTS,
    TASK1_POEM_BASE_PROMPT,
    TASK1_POEM_REPROMPT,
    TASK1_POEM_STRICT_REPROMPT,
    TASK2_ALIGNMENT_MATRIX,
    TASK3_BASE_VS_INSTRUCT_SUITE,
)
from hw_01_model_alignment.report_builder import (
    export_hw1_jupyter_notebook,
    generate_google_docs_markdown,
)
from hw_01_model_alignment.ui_labels import UI_LABELS

__all__ = [
    "AlignmentPair",
    "PoemEvaluation",
    "AlignmentRecord",
    "HW1BenchmarkRunner",
    "evaluate_ukrainian_poem",
    "analyze_bpe_tokenization_fertility",
    "compute_sampling_distribution",
    "TASK1_POEM_BASE_PROMPT",
    "TASK1_POEM_REPROMPT",
    "TASK1_POEM_STRICT_REPROMPT",
    "TASK1_KNOWLEDGE_CUTOFF_PROMPTS",
    "TASK2_ALIGNMENT_MATRIX",
    "TASK3_BASE_VS_INSTRUCT_SUITE",
    "HW1_REPORT_SYNTHESIS_PROMPT",
    "generate_google_docs_markdown",
    "export_hw1_jupyter_notebook",
    "UI_LABELS",
]
