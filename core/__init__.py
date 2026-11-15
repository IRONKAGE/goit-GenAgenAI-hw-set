# ==============================================================================
# 🧠 CORE PACKAGE INITIALIZATION: goit-GenAgenAI-hw-set (Nov 15, 2026)
# ==============================================================================
"""
Єдина точка входу до інфраструктурного ядра монорепозиторію:
- Автоматичне підвантаження змінних із `.env`
- Ініціалізація системного логера та User-Agent для веб-скрапінгу/Wikipedia
- Експорт 3-рівневих маршрутизаторів (LLM, Embeddings, Web Search, Hardware, Eval)
"""

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

# Встановлюємо стандартний USER_AGENT (необхідно для WebBaseLoader та Wikipedia API з Теми 6-7)
os.environ.setdefault(
    "USER_AGENT",
    "MyLangChainAgent/1.0 (contact: oleh.hatsenko@neoversity.edu)",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from core.datasets import (
    ensure_directory,
    read_excel_safe,
    read_tabular_data_safe,
    save_excel_formatted,
    smart_read_csv,
)
from core.embeddings_router import (
    DeterministicSemanticEmbeddings,
    get_smart_embeddings,
)
from core.eval_core import (
    ALLOWED_TONES_DEFAULT,
    check,
    compute_ragas_metrics_offline,
    parse_relevancy_score,
    robust_structured_invoke,
)
from core.hardware import (
    clear_vram,
    get_hardware_config,
    log_system_info,
    set_global_seed,
)
from core.llm_router import (
    DEPRECATED_MODELS_NOV_2026,
    SQLiteResponseCache,
    configure_langsmith_tracing,
    flush_langsmith_tracers,
    get_secret,
    get_smart_chat_llm,
    validate_model_actualization,
)
from core.search_router import (
    SmartSearchRouter,
    compute_haversine_distance_km,
    fetch_open_meteo_coordinates,
)

__all__ = [
    "set_global_seed",
    "get_hardware_config",
    "clear_vram",
    "log_system_info",
    "get_secret",
    "get_smart_chat_llm",
    "validate_model_actualization",
    "configure_langsmith_tracing",
    "flush_langsmith_tracers",
    "SQLiteResponseCache",
    "DEPRECATED_MODELS_NOV_2026",
    "get_smart_embeddings",
    "DeterministicSemanticEmbeddings",
    "SmartSearchRouter",
    "compute_haversine_distance_km",
    "fetch_open_meteo_coordinates",
    "ensure_directory",
    "read_tabular_data_safe",
    "smart_read_csv",
    "read_excel_safe",
    "save_excel_formatted",
    "check",
    "robust_structured_invoke",
    "parse_relevancy_score",
    "compute_ragas_metrics_offline",
    "ALLOWED_TONES_DEFAULT",
]
