# ==============================================================================
# 🛡️ CORE LLM ROUTER: 3-Tier Smart AI Routing (Actualized: November 15, 2026)
# ==============================================================================
"""
Центральний маршрутизатор мовних моделей для монорепозиторію `goit-GenAgenAI-hw-set`.
Забезпечує:
1. Універсальне отримання секретів `get_secret()` (.env -> Colab Secrets -> getpass).
2. Автоматичну заміну вимкнених у 2026 році моделей (gpt-4, gpt-3.5-turbo, claude-opus-4-1).
3. Автоматичне видалення параметра `temperature` для міркувальних моделей сімейства `gpt-5`.
4. Трьохрівневу маршрутизацію: Cloud API (Gemini / OpenRouter / OpenAI) -> Local llama.cpp -> SQLite Cache.
5. Динамічне керування трейсингом LangSmith із очищенням кешу `ls_utils.get_env_var.cache_clear()`.
"""

import hashlib
import logging
import os
import sqlite3
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from langchain_openai import ChatOpenAI

logger = logging.getLogger("core.llm_router")

# Моделі, які офіційно вимкнені станом на 15 листопада 2026 року (згідно з оновленнями ДЗ 1–5)
DEPRECATED_MODELS_NOV_2026: Dict[str, str] = {
    "claude-opus-4-1-20250805": "claude-opus-5-5 (вимкнена 05.08.2026)",
    "gpt-3.5-turbo": "gpt-5.6-luna або openai/gpt-oss-120b (вимкнена 23.10.2026)",
    "gpt-4": "gpt-5.6-luna або openai/gpt-oss-120b (вимкнена 23.10.2026)",
    "gpt-4.5": "gpt-5.6-sol (вимкнена у 2026 році)",
}

# Попередження для моделей, що вимикаються у грудні 2026 — січні 2027
SUNSET_WARNING_MODELS: Dict[str, str] = {
    "gpt-5": "Вимикається 11.12.2026 -> рекомендовано gpt-5.6-luna",
    "gpt-5-mini": "Вимикається 11.12.2026 -> рекомендовано gpt-5.6-luna",
    "gpt-4o": "Вимикається 14.01.2027 -> рекомендовано gpt-5.6-luna або openai/gpt-oss-120b",
    "gpt-4o-mini": "Вимикається 14.01.2027 -> рекомендовано gpt-5.6-luna або openai/gpt-oss-120b",
}

# Міркувальні (reasoning) моделі, які не приймають параметр temperature
REASONING_NO_TEMP_PREFIXES: Tuple[str, ...] = ("gpt-5", "o1", "o3", "o4-mini")


def get_secret(name: str, required: bool = False, default: str = "") -> str:
    """
    Універсальний резолвер ключів API (патерн із практичних ноутбуків Тем 5, 6, 7):
    1) Змінна середовища / .env
    2) Google Colab Secrets (`google.colab.userdata.get`)
    3) Ручне введення через `getpass()`, якщо `required=True`
    """
    val = os.environ.get(name, "").strip()
    if val:
        return val

    try:
        from google.colab import userdata  # type: ignore

        colab_val = userdata.get(name)
        if colab_val:
            os.environ[name] = str(colab_val).strip()
            return os.environ[name]
    except Exception:
        pass

    if required:
        entered = getpass(f"Введіть {name}: ").strip()
        if entered:
            os.environ[name] = entered
            return entered

    return default


def validate_model_actualization(model_name: str) -> str:
    """
    Перевіряє актуальність назви моделі станом на 15 листопада 2026 року.
    Якщо передано вимкнену модель (напр. `gpt-4` або `gpt-3.5-turbo`), автоматично
    перемикає на актуальну `gpt-5.6-luna` або `claude-opus-5-5`.
    """
    clean = model_name.strip()
    if clean in DEPRECATED_MODELS_NOV_2026:
        replacement = "gpt-5.6-luna" if "gpt" in clean else "claude-opus-5-5"
        logger.warning(
            f"⚠️ Модель '{clean}' більше не працює ({DEPRECATED_MODELS_NOV_2026[clean]}). "
            f"Виконано автоматичне перемикання на '{replacement}'."
        )
        return replacement

    if clean in SUNSET_WARNING_MODELS:
        logger.info(f"ℹ️ Життєвий цикл моделі '{clean}': {SUNSET_WARNING_MODELS[clean]}")

    return clean


class SQLiteResponseCache:
    """Контентно-адресований SQLite-кеш відповідей LLM (Tier 3 Deterministic Cache)."""

    def __init__(self, db_path: str = "data/cache/llm_cache.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_responses (
                    prompt_hash TEXT PRIMARY KEY,
                    model_name TEXT,
                    response_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @staticmethod
    def compute_hash(model_name: str, prompt_payload: str) -> str:
        raw = f"{model_name}::{prompt_payload}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, model_name: str, prompt_payload: str) -> Optional[str]:
        h = self.compute_hash(model_name, prompt_payload)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response_text FROM llm_responses WHERE prompt_hash = ?", (h,)
            ).fetchone()
            return row[0] if row else None

    def put(self, model_name: str, prompt_payload: str, response_text: str) -> None:
        h = self.compute_hash(model_name, prompt_payload)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_responses (prompt_hash, model_name, response_text)
                VALUES (?, ?, ?)
                """,
                (h, model_name, response_text),
            )


def get_smart_chat_llm(
    preferred_tier: str = "auto",
    role: str = "worker",
    temperature: float = 0.0,
) -> Tuple[Optional[ChatOpenAI], str, str]:
    """
    Повертає ініціалізований клієнт `ChatOpenAI` за 3-рівневим пріоритетом:
    1) Cloud API: Gemini (Google AI Studio Free) -> OpenRouter (`openai/gpt-oss-120b`) -> OpenAI (`gpt-5.6-luna`)
    2) Local Autonomous: локальний сервер `llama.cpp` / `Ollama` (`LOCAL_LLM_URL`)
    3) Reference Offline: `None` (сигналізує модулю використати детермінований офлайн-рушій)
    """
    if preferred_tier == "reference":
        return None, "reference", "deterministic-offline-v1"

    # 🥇 Tier 1A: Google Gemini API (OpenAI-compatible endpoint)
    gemini_key = get_secret("GEMINI_API_KEY", required=False)
    if preferred_tier in {"auto", "gemini"} and gemini_key:
        default_gem = "gemini-3.1-pro" if role == "judge" else "gemini-2.5-flash"
        model_id = os.getenv("GEMINI_MODEL", default_gem)
        llm = ChatOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key,
            model=model_id,
            temperature=temperature,
        )
        return llm, "gemini", model_id

    # 🥇 Tier 1B: OpenRouter API (Рекомендовано в оновленнях ДЗ 2, 3, 4)
    or_key = get_secret("OPENROUTER_API_KEY", required=False)
    if preferred_tier in {"auto", "openrouter"} and or_key:
        model_id = validate_model_actualization(
            os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
        )
        kwargs: Dict[str, Any] = {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": or_key,
            "model": model_id,
        }
        if not model_id.startswith(REASONING_NO_TEMP_PREFIXES):
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs), "openrouter", model_id

    # 🥇 Tier 1C: Direct OpenAI API (`gpt-5.6-luna` для генерації, `gpt-5.6-terra` для судді в ДЗ 5)
    oa_key = get_secret("OPENAI_API_KEY", required=False)
    if preferred_tier in {"auto", "openai"} and oa_key:
        default_oa = (
            os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.6-terra")
            if role == "judge"
            else os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        )
        model_id = validate_model_actualization(default_oa)
        kwargs = {"model": model_id, "api_key": oa_key}
        # Моделі сімейства gpt-5 є міркувальними і не приймають довільну температуру
        if not model_id.startswith(REASONING_NO_TEMP_PREFIXES):
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs), "openai", model_id

    # 🥈 Tier 2: Autonomous Local Server (`llama.cpp` / `Ollama` / `vLLM`)
    local_url = os.getenv("LOCAL_LLM_URL", "").strip()
    if preferred_tier in {"auto", "local"} and local_url:
        model_id = os.getenv("LOCAL_LLM_MODEL", "qwen3.5-14b-instruct")
        llm = ChatOpenAI(
            base_url=local_url,
            api_key="local-no-key-needed",
            model=model_id,
            temperature=temperature,
        )
        return llm, "local", model_id

    # 🥉 Tier 3: Deterministic Offline Reference Mode
    return None, "reference", "deterministic-offline-v1"


def configure_langsmith_tracing(
    project_name: str = "neoversity-genagenai-2026",
    enable: Optional[bool] = None,
    use_eu_endpoint: bool = False,
) -> bool:
    """
    Вмикає трасування LangSmith ЛИШЕ за наявності валідного ключа `lsv2_...`.
    Якщо ключа немає або він порожній — примусово вимикає і LANGSMITH_TRACING,
    і LANGCHAIN_TRACING_V2, щоб уникнути помилок 401 Unauthorized та попереджень.
    """
    ls_key = get_secret("LANGSMITH_API_KEY", required=False)
    has_valid_key = bool(ls_key and ls_key.startswith("lsv2_") and len(ls_key) > 20)
    should_enable = has_valid_key if enable is None else (enable and has_valid_key)

    if should_enable:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = ls_key
        os.environ["LANGSMITH_PROJECT"] = project_name
        if use_eu_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
        else:
            os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGSMITH_API_KEY", None)
        os.environ.pop("LANGCHAIN_API_KEY", None)

    try:
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()
    except Exception:
        pass

    return should_enable


def flush_langsmith_tracers() -> None:
    """Чекає завершення фонових трейсів лише якщо трасування реально увімкнене."""
    if os.environ.get("LANGSMITH_TRACING") != "true":
        return
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception:
        pass
