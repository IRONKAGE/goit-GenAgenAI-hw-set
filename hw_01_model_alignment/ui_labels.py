# ==============================================================================
# 🌐 HW 01 UI LABELS: Локалізація для Marimo 0.25.2 та Svelte 5 Control Panel
# ==============================================================================
"""Словники локалізації та константи візуалізації для ДЗ 1."""

UI_LABELS = {
    "title": "🌍 ДЗ 1: Дослідження Pre-training, Worldview Alignment та Base vs Instruct",
    "subtitle": "Трьохрівневий бенчмарк (Cloud API ⟷ Local llama.cpp ⟷ Reference Telemetry)",
    "tabs": {
        "task1": "📝 Завдання 1: Претрейн, Вірш (6 рядків) & Cutoff",
        "task2": "⚖️ Завдання 2: Картини Світу (Alignment & Полярність Так↔Ні)",
        "task3": "🧬 Завдання 3: Base vs Instruct (SFT/RLHF/DPO)",
        "sampling": "🎛️ R&D: Математика Семплінгу (Temperature, Top-P, Top-K)",
        "tokenization": "🔢 R&D: Економіка BPE-Токенізації",
        "gdocs_export": "📄 Готовий Звіт для Google Docs (ПІ_ДЗ1) & .ipynb",
    },
    "models": {
        "cn_small": "Qwen3.5-14B-Instruct (Китайська екосистема, менша за флагман)",
        "us_flagship": "Claude Opus 5 / Gemini 3.1 Pro (Американський флагман)",
        "cn_reasoning": "DeepSeek-V4-Thinking (Китайська reasoning-модель)",
        "base_model": "deepseek/deepseek-v3.1-base (Pre-training only)",
        "instruct_model": "deepseek/deepseek-v3.1 (SFT + DPO/RLHF)",
    },
}
