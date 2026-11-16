import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="HW 01: Pre-training, Alignment, Sampling & Tokenization",
)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import plotly.express as px
    import polars as pl

    from core import get_hardware_config, log_system_info, set_global_seed
    from hw_01_model_alignment import (
        HW1BenchmarkRunner,
        TASK3_BASE_VS_INSTRUCT_SUITE,
        UI_LABELS,
        analyze_bpe_tokenization_fertility,
        compute_sampling_distribution,
        export_hw1_jupyter_notebook,
        generate_google_docs_markdown,
    )

    _seed = set_global_seed()
    _sys_info = log_system_info()
    _device, _device_name = get_hardware_config(_seed)  # маршрутизатор заліза з core.hardware (поважає ML_BACKEND з .env)

    mo.center(
        mo.callout(
            mo.md(
                f"✅ **Omni-Agentic Інфраструктура (LangChain v1 ⟷ LangGraph v1 ⟷ ChromaDB ⟷ PyTorch) ініціалізована!**\n\n"
                f"🖥️ `{_sys_info}` | 🚀 **Пристрій обчислень:** `{_device_name}` | 🌱 **Global Seed:** `{_seed}`"
            ),
            kind="success",
        )
    )
    return (
        HW1BenchmarkRunner,
        TASK3_BASE_VS_INSTRUCT_SUITE,
        UI_LABELS,
        analyze_bpe_tokenization_fertility,
        compute_sampling_distribution,
        export_hw1_jupyter_notebook,
        generate_google_docs_markdown,
        mo,
        pl,
        px,
    )


@app.cell(hide_code=True)
def title_head_hw(HW1BenchmarkRunner, UI_LABELS, mo):
    css_no_scroll = mo.md(
        """
        <div class="marimo-noscroll-override"></div>
        <style>
            marimo-cell-output:has(.marimo-noscroll-override),
            .output-area:has(.marimo-noscroll-override) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        """
    )
    runner = HW1BenchmarkRunner(preferred_tier="auto")
    temp_slider = mo.ui.slider(
        start=0.1, stop=2.5, step=0.1, value=0.7, label="🌡️ Temperature (T):"
    )
    top_p_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.9, label="🎯 Nucleus Top-P:"
    )
    top_k_slider = mo.ui.slider(
        start=1, stop=8, step=1, value=4, label="🔢 Top-K кандидатів:"
    )

    mo.md(
        f"""
        <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
            {UI_LABELS["title"]}</br><i>(3-Tier Multi-Ecosystem Alignment & Sampling Benchmark)</i>
        </div>
        <h3 align="center"><b><u>Пайплайн</u>: 🇨🇳 Qwen3.5 / DeepSeek-V4 (PRC Alignment) ┃ 🇺🇸 Claude Opus 5 / Gemini 3.1 Pro (Constitutional AI) ┃ 🧬 Base vs Instruct (SFT + DPO/RLHF) ➔ BPE Fertility & Sampling Math</b></h3>
        <p align="center"><b>Активний 3-Tier рівень:</b> <code>{runner.active_tier.upper()}</code> (<code>{runner.model_name}</code>) &nbsp;|&nbsp; <i>© Oleh Hatsenko (IRONKAGE) | Generative and Agentic AI</i></p>
        """
    )
    return css_no_scroll, runner, temp_slider, top_k_slider, top_p_slider


@app.cell(hide_code=True)
def main(
    TASK3_BASE_VS_INSTRUCT_SUITE,
    analyze_bpe_tokenization_fertility,
    compute_sampling_distribution,
    css_no_scroll,
    export_hw1_jupyter_notebook,
    generate_google_docs_markdown,
    mo,
    pl,
    px,
    runner,
    temp_slider,
    top_k_slider,
    top_p_slider,
):
    summary = runner.get_full_telemetry_summary()

    # 1. Таблиця оцінювання віршів (Завдання 1)
    poem_evals = runner.run_task1_poem_evaluations()
    df_poems = pl.DataFrame(
        [
            {
                "Модель": e.model_name,
                "Етап": e.stage,
                "Кількість рядків": e.line_count,
                "Рівно 6 рядків": "✅ Так" if e.exact_six_lines else "❌ Ні",
                "Чиста українська": "✅ Так" if e.pure_ukrainian else f"⚠️ Ні ({', '.join(e.foreign_or_glitch_tokens)})",
                "Оцінка рими": e.rhyme_score,
                "Схема римування": e.rhyme_scheme_detected,
            }
            for e in poem_evals
        ]
    )

    # 2. Матриця Alignment та інверсії полярності Так <-> Ні (Завдання 2)
    alignment_records = runner.run_task2_alignment_matrix()
    df_alignment = pl.DataFrame(
        [
            {
                "Категорія": r.category,
                "Модель": r.model_name,
                "Екосистема": r.ecosystem,
                "Пряме питання": r.direct_answer[:48] + ("..." if len(r.direct_answer) > 48 else ""),
                "Перефразоване": r.paraphrased_answer[:48] + ("..." if len(r.paraphrased_answer) > 48 else ""),
                "Дзеркальне (Інверсія Так↔Ні)": r.mirrored_inverted_answer[:48] + ("..." if len(r.mirrored_inverted_answer) > 48 else ""),
                "Стабільність полярності": "✅ Стабільна" if r.polarity_consistent else "❌ Суперечність",
                "Стійкість до Sycophancy": "⚠️ Змінила позицію" if r.sycophancy_flipped else "🛡️ Стійка",
            }
            for r in alignment_records
        ]
    )

    # 3. Таблиця Base vs Instruct (Завдання 3)
    t3_data = summary["task3_base_vs_instruct"]
    df_task3 = pl.DataFrame(
        [
            {
                "Експеримент": key,
                "Промпт": str(cfg["prompt"]),
                "Max Tokens": cfg["max_tokens"],
                "Вивід Base-моделі (Pre-training)": t3_data[key]["base_output"],
                "Вивід Instruct-моделі (SFT + RLHF/DPO)": t3_data[key]["instruct_output"],
            }
            for key, cfg in TASK3_BASE_VS_INSTRUCT_SUITE.items()
        ]
    )

    # 4. Інтерактивний симулятор семплінгу (Temperature, Top-P, Top-K)
    df_sampling = compute_sampling_distribution(
        temperature=temp_slider.value,
        top_p=top_p_slider.value,
        top_k=top_k_slider.value,
    )
    fig_sampling = px.bar(
        df_sampling.to_pandas(),
        x="Токен-кандидат",
        y=["P(x) при T=1.0", f"P(x) при T={temp_slider.value:.2f}", "Фінальна P(x) після ренормалізації"],
        barmode="group",
        title=f"Розподіл імовірностей наступного токена (T={temp_slider.value:.2f}, Top-P={top_p_slider.value:.2f}, Top-K={top_k_slider.value})",
    )

    # 5. Аналіз економіки BPE-токенізації
    df_tokens = analyze_bpe_tokenization_fertility()
    fig_tokens = px.bar(
        df_tokens.to_pandas(),
        x="Категорія",
        y=["Токенів (OpenAI o200k)", "Токенів (Llama 3)", "Токенів (Gemma 3)"],
        barmode="group",
        title="Порівняння витрати BPE-токенів: Українська мова vs Англійська vs Число Пі vs Код",
    )

    # 6. Генерація готових файлів для здачі (Google Docs .md + Jupyter .ipynb)
    gdocs_md = generate_google_docs_markdown("submissions/Hatsenko_Oleh_DZ1.md")
    nb_path = export_hw1_jupyter_notebook("submissions/oleh_hatsenko_hw1.ipynb")

    tabs = mo.vstack([
        css_no_scroll,
        mo.ui.tabs(
            {
                "📝 Завдання 1: Претрейн & Вірш": mo.vstack(
                    [
                        mo.md("### Кількісна верифікація генерації вірша (6 рядків, чистота мови, рима)"),
                        mo.ui.table(df_poems),
                    ]
                ),
                "⚖️ Завдання 2: Alignment & Полярність": mo.vstack(
                    [
                        mo.md("### Матриця ціннісного вирівнювання (Пряме, Перефразоване, Інверсія Так↔Ні та Sycophancy)"),
                        mo.ui.table(df_alignment),
                    ]
                ),
                "🧬 Завдання 3: Base vs Instruct": mo.vstack(
                    [
                        mo.md("### Порівняння Pre-training (Base) проти SFT + DPO/RLHF (Instruct)"),
                        mo.ui.table(df_task3),
                    ]
                ),
                "🎛️ R&D: Семплінг (T, Top-P, Top-K)": mo.vstack(
                    [
                        mo.hstack([temp_slider, top_p_slider, top_k_slider]),
                        mo.ui.plotly(fig_sampling),
                        mo.ui.table(df_sampling),
                    ]
                ),
                "🔢 R&D: Економіка Токенізації": mo.vstack([mo.ui.plotly(fig_tokens), mo.ui.table(df_tokens)]),
                "📄 Готовий Звіт Google Docs & .ipynb": mo.vstack(
                    [
                        mo.md(f"✅ Файли збережено: `submissions/Hatsenko_Oleh_DZ1.md` та `{nb_path}`"),
                        mo.md(gdocs_md),
                    ]
                ),
            }
        )]
    )
    tabs
    return


if __name__ == "__main__":
    app.run()
