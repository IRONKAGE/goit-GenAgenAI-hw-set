# ==============================================================================
# 🛡️ UNIT TESTS FOR CORE INFRASTRUCTURE (core/)
# ==============================================================================
"""
Перевірка спільного ядра `core/`:
- Актуалізація вимкнених моделей станом на 15 листопада 2026 року
- Контентно-адресований SQLiteResponseCache
- Нормалізовані L2-вектори DeterministicSemanticEmbeddings
- Кешування та формула Гаверсинуса у SmartSearchRouter
- Валідатор check() та виправлений regex-парсер parse_relevancy_score()
- Безпечний запис і читання Excel (save_excel_formatted / read_excel_safe)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from core import (
    DeterministicSemanticEmbeddings,
    SQLiteResponseCache,
    SmartSearchRouter,
    check,
    compute_haversine_distance_km,
    compute_ragas_metrics_offline,
    get_hardware_config,
    get_smart_chat_llm,
    get_smart_embeddings,
    parse_relevancy_score,
    read_excel_safe,
    save_excel_formatted,
    validate_model_actualization,
)


@pytest.mark.parametrize(
    ("deprecated_name", "expected_replacement"),
    [
        ("gpt-4", "gpt-5.6-luna"),
        ("gpt-3.5-turbo", "gpt-5.6-luna"),
        ("claude-opus-4-1-20250805", "claude-opus-5-5"),
        ("gpt-5.6-luna", "gpt-5.6-luna"),
        ("openai/gpt-oss-120b", "openai/gpt-oss-120b"),
    ],
)
def test_model_actualization_nov_2026(deprecated_name: str, expected_replacement: str) -> None:
    """Перевіряє автоматичну заміну вимкнених у 2026 році моделей на живі аналоги."""
    assert validate_model_actualization(deprecated_name) == expected_replacement


def test_sqlite_response_cache_roundtrip(isolated_cache_paths: dict[str, Path]) -> None:
    """Перевіряє детерміноване збереження та зчитування відповідей LLM за SHA-256."""
    cache = SQLiteResponseCache(db_path=str(isolated_cache_paths["sqlite_llm"]))
    assert cache.get("gpt-5.6-luna", "test prompt") is None

    cache.put("gpt-5.6-luna", "test prompt", '{"status": "ok"}')
    assert cache.get("gpt-5.6-luna", "test prompt") == '{"status": "ok"}'
    assert cache.get("other-model", "test prompt") is None


def test_deterministic_embeddings_l2_normalization() -> None:
    """Перевіряє розмірність (64) та L2-норму (=1.0) офлайн-ембедингів для ChromaDB."""
    emb = DeterministicSemanticEmbeddings()
    vec = np.array(emb.embed_query("Виручка компанії ТехНова Інк за 2024 рік"))
    assert vec.shape == (64,)
    assert pytest.approx(float(np.linalg.norm(vec)), abs=1e-6) == 1.0

    # Орфографічна близькість: "ТехНова Інк" та "Технова Інк" мають давати високу косинусну схожість
    v1 = np.array(emb.embed_query("ТехНова Інк"))
    v2 = np.array(emb.embed_query("Технова Інк"))
    assert float(np.dot(v1, v2)) > 0.90


def test_search_router_cache_and_haversine(isolated_cache_paths: dict[str, Path]) -> None:
    """Перевіряє роботу JSON-кешу пошуку та геодезичного розрахунку відстаней."""
    router = SmartSearchRouter(cache_path=str(isolated_cache_paths["json_search"]))
    text1, src1 = router.query_with_cache("test query", fallback_text="Paris to Berlin is 878 km")
    assert "878" in text1
    assert src1 in {"tavily_api", "offline_reference"}

    # Повторний запит має братися з локального кешу без мережевих витрат
    text2, src2 = router.query_with_cache("test query", fallback_text="ignored")
    assert text2 == text1
    assert src2 == "cache"
    assert router.cache_hits_count >= 1

    dist_london_rome = compute_haversine_distance_km("London", "Rome", allow_live_geocoding=False)
    assert dist_london_rome is not None
    assert 1400 <= dist_london_rome <= 1470


@pytest.mark.parametrize(
    ("raw_judge_output", "expected_score"),
    [
        ("0.85", 0.85),
        ("0,9", 0.9),
        ("85%", 0.85),
        ("0.85.", 0.85),
        ("Оцінка релевантності: 9.2 з 10", 0.92),
        ("1.0", 1.0),
    ],
)
def test_parse_relevancy_score_fixes_template_bugs(raw_judge_output: str, expected_score: float) -> None:
    """Перевіряє виправлений парсер оцінок релевантності з оновлення ДЗ 5."""
    assert pytest.approx(parse_relevancy_score(raw_judge_output), abs=1e-4) == expected_score


def test_json_contract_check_function() -> None:
    """Перевіряє канонічну функцію check() з оновлення ДЗ 2 на валідних та битих JSON."""
    valid = check('{"person": "Оксана Дрозд", "tone": "позитивна"}')
    assert valid == {"person": "Оксана Дрозд", "tone": "позитивна"}

    with pytest.raises(ValueError, match="порожня"):
        check("")

    with pytest.raises(ValueError, match="очікувались рівно ключі"):
        check('{"person": "Оксана Дрозд", "tone": "позитивна", "extra": 1}')

    with pytest.raises(ValueError, match="недопустиме значення"):
        check('{"person": "Оксана Дрозд", "tone": "супер"}')


def test_excel_formatted_io_and_hardware(tmp_path: Path) -> None:
    """Перевіряє безпечне збереження/читання .xlsx та детектор обладнання."""
    device, name = get_hardware_config(global_seed=42)
    assert device is not None and len(name) > 0

    llm, tier, _ = get_smart_chat_llm(preferred_tier="reference")
    assert llm is None and tier == "reference"

    _, emb_tier = get_smart_embeddings(preferred_tier="reference")
    assert emb_tier == "reference"

    df_in = pd.DataFrame([{"City": "Kyiv", "Country": "Ukraine", "Score": 100}])
    out_xlsx = save_excel_formatted(df_in, tmp_path / "sample.xlsx")
    df_out = read_excel_safe(out_xlsx)
    assert list(df_out.columns) == ["City", "Country", "Score"]
    assert df_out.loc[0, "City"] == "Kyiv"

    ragas = compute_ragas_metrics_offline(
        question="Яка виручка?",
        answer="Виручка становить 450 млн дол.",
        retrieved_contexts=["Технова Інк за 2024 рік: виручка 450 млн дол"],
        expected_fact_substring="450",
        expected_entity="Технова Інк",
    )
    assert ragas["faithfulness"] == 1.0
    assert ragas["context_precision"] == 1.0
