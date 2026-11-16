import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="HW 02: Context Engineering, JSON Triage & LCEL",
)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import polars as pl

    from core import get_hardware_config, log_system_info, set_global_seed
    from hw_02_context_triage import (
        ContextTriageExtractor,
        TASK1_CORPUS,
        TASK2_CORPUS,
        TASK3_CORPUS,
        UI_LABELS,
        build_parallel_summary_chain,
        export_hw2_jupyter_notebook,
        simulate_typeddict_stream,
        generate_hw2_google_docs_markdown,
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
        ContextTriageExtractor,
        TASK1_CORPUS,
        TASK2_CORPUS,
        TASK3_CORPUS,
        UI_LABELS,
        build_parallel_summary_chain,
        export_hw2_jupyter_notebook,
        generate_hw2_google_docs_markdown,
        mo,
        pl,
        simulate_typeddict_stream,
    )


@app.cell(hide_code=True)
def title_head_hw(ContextTriageExtractor, UI_LABELS, mo):
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

    extractor = ContextTriageExtractor(preferred_tier="auto")
    all_runs = extractor.run_all_tasks()

    mo.md(
            f"""
            <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
                {UI_LABELS["title"]}</br><i>(Strict JSON Contract & Self-Healing Context Pipeline)</i>
            </div>
            <h3 align="center"><b><u>Пайплайн</u>: 🛡️ XML Role/Data Isolation ┃ ⚙️ Pydantic v2 + check() Self-Healing ┃</br>🔗 LCEL (RunnableBranch ⟷ RunnableParallel ⟷ TypedDict Stream) ➔ CoT Key-Order Ablation</b></h3>
            <p align="center"><b>Активний 3-Tier рівень:</b> <code>{extractor.active_tier.upper()}</code> (<code>{extractor.model_name}</code>) &nbsp;|&nbsp; <i>© Oleh Hatsenko (IRONKAGE) | Generative and Agentic AI</i></p>
            """
        )
    return all_runs, css_no_scroll


@app.cell(hide_code=True)
def main(
    ContextTriageExtractor,
    TASK1_CORPUS,
    TASK2_CORPUS,
    TASK3_CORPUS,
    all_runs,
    build_parallel_summary_chain,
    css_no_scroll,
    export_hw2_jupyter_notebook,
    generate_hw2_google_docs_markdown,
    mo,
    pl,
    simulate_typeddict_stream,
):
    def _to_df(records):
        return pl.DataFrame(
            [
                {
                    "Вхідний текст": r.input_text,
                    "Чистий JSON вивід моделі": r.raw_json_str,
                    "Перевірка check()": "✅ Валідний" if r.is_valid_contract else "❌ Помилка",
                    "Збіг з еталоном": "✅ 100%" if r.is_exact_match else "⚠️ Відхилення",
                    "Спроб": r.attempts_used,
                    "Час (мс)": r.latency_ms,
                }
                for r in records
            ]
        )

    df_t1 = _to_df(all_runs["task1"])
    df_t2 = _to_df(all_runs["task2"])
    df_t3 = _to_df(all_runs["task3"])
    df_ablation = ContextTriageExtractor.run_key_order_ablation_study()

    parallel_chain = build_parallel_summary_chain()
    sample_inputs = [
        {"text": TASK1_CORPUS[0]},
        {"text": TASK2_CORPUS[0]},
        {"text": TASK3_CORPUS[0]},
    ]
    routed_rows = []
    for item in sample_inputs:
        out = parallel_chain.invoke(item)
        routed_rows.append(
            {
                "Вхідне повідомлення": item["text"],
                "Маршрут RunnableBranch": out["routed"]["branch"],
                "Слів (RunnableParallel)": out["stats"]["words"],
                "Символів (RunnableParallel)": out["stats"]["chars"],
            }
        )
    df_lcel_router = pl.DataFrame(routed_rows)
    df_stream = simulate_typeddict_stream(
        {"title": "Версія 2.0 з офлайн-режимом та новим дизайном", "category": "реліз"}
    )

    sub_path = export_hw2_jupyter_notebook(all_runs, "submissions/oleh_hatsenko_hw2.ipynb")
    gdocs_md = generate_hw2_google_docs_markdown("submissions/Hatsenko_Oleh_DZ2.md")

    tabs = mo.vstack([
        css_no_scroll,
        mo.ui.tabs(
            {
                "👤 Завдання 1 (50 б.)": mo.vstack(
                    [mo.md("### Персона + Тональність (`person`, `tone`)"), mo.ui.table(df_t1)]
                ),
                "🚨 Завдання 2 (25 б.)": mo.vstack(
                    [
                        mo.md("### Тріаж відгуків (`product`, `priority`, нормалізація `Atlas Pro -> Atlas`)"),
                        mo.ui.table(df_t2),
                    ]
                ),
                "📰 Завдання 3 (25 б.)": mo.vstack(
                    [
                        mo.md("### Заголовок (≤ 8 слів) + Категорія (`title`, `category`)"),
                        mo.ui.table(df_t3),
                    ]
                ),
                "🔗 LCEL & TypedDict Stream (Тема 5)": mo.vstack(
                    [
                        mo.md("### Маршрутизація `RunnableBranch` + `RunnableParallel`"),
                        mo.ui.table(df_lcel_router),
                        mo.md("### Почастковий стримінг `TypedDict` (`.stream()`)"),
                        mo.ui.table(df_stream),
                    ]
                ),
                "🔬 R&D: Порядок ключів JSON": mo.vstack(
                    [
                        mo.md("### Експеримент зі слайду 25 Теми 4: генерація `explanation` перед цільовим класом"),
                        mo.ui.table(df_ablation),
                    ]
                ),
                "📄 Звіт Google Docs & .ipynb": mo.vstack( # <--- Нова вкладка зі звітом
                    [
                        mo.md(f"✅ Файли збережено: `submissions/Hatsenko_Oleh_DZ2.md` та `{sub_path}`"),
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
