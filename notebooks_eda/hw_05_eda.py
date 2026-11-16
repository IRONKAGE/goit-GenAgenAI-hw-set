import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="HW 05: LangGraph v1 Support Agent & Dual LLM-as-Judge",
)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import polars as pl

    from core import get_hardware_config, log_system_info, set_global_seed
    from hw_05_support_langgraph import (
        SupportAgentBenchmark,
        UI_LABELS,
        export_hw5_jupyter_notebook,
        generate_hw5_google_docs_markdown,
        simulate_topic9_hitl_interrupt_resume,
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
        SupportAgentBenchmark,
        UI_LABELS,
        export_hw5_jupyter_notebook,
        generate_hw5_google_docs_markdown,
        mo,
        pl,
        simulate_topic9_hitl_interrupt_resume,
    )


@app.cell(hide_code=True)
def title_head_hw(
    SupportAgentBenchmark,
    UI_LABELS,
    mo,
    simulate_topic9_hitl_interrupt_resume,
):
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

    benchmark = SupportAgentBenchmark(preferred_tier="auto")
    eval_rows = benchmark.run_benchmark()
    regex_df = SupportAgentBenchmark.verify_regex_parser_cases()
    hitl_topic9_df = simulate_topic9_hitl_interrupt_resume()

    tcr = sum(1 for r in eval_rows if r.final_response) / len(eval_rows)
    avg_rel = sum(r.self_judge_score for r in eval_rows) / len(eval_rows)
    esc_count = sum(1 for r in eval_rows if r.needs_escalation)

    mo.md(
        f"""
        <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
            {UI_LABELS["title"]}</br><i>(Sequential StateGraph Support Workflow & Dual LLM-as-Judge)</i>
        </div>
        <h3 align="center"><b><u>Пайплайн</u>: 📋 classify_query (TypedDict) ➔ 🔍 search_faq (Top 1–2) ➔ 💬 draft_response (≤35 слів) ➔ ⚠️ check_escalation (≤50 слів) ┃ ⚖️ Self-Judge vs Independent Judge</b></h3>
        <p align="center"><b>TCR:</b> <code>{tcr:.0%} (5/5)</code> &nbsp;|&nbsp; <b>Avg Relevancy:</b> <code>{avg_rel:.0%}</code> &nbsp;|&nbsp; <b>Escalation Rate:</b> <code>{esc_count}/5 ({esc_count / len(eval_rows):.0%})</code> &nbsp;|&nbsp; <i>© Oleh Hatsenko (IRONKAGE) | Generative and Agentic AI</i></p>
        """
    )
    return css_no_scroll, eval_rows, hitl_topic9_df, regex_df


@app.cell(hide_code=True)
def main(
    css_no_scroll,
    eval_rows,
    export_hw5_jupyter_notebook,
    generate_hw5_google_docs_markdown,
    hitl_topic9_df,
    mo,
    pl,
    regex_df,
):
    df_exec = pl.DataFrame(
        [
            {
                "Тест #": f"ТЕСТ {r.test_num}/5",
                "Запит користувача": r.query,
                "Тип": r.q_type,
                "Категорія": r.category,
                "Терміновість": r.urgency,
                "К-сть FAQ": f"{r.faq_count} п.",
                "Відповідь агента": r.final_response,
                "Слів (≤50)": f"{r.word_count} слів ({'✅' if r.within_50_words else '❌'})",
                "Статус ескалації": "⚠️ Ескаловано" if r.needs_escalation else "✅ Автоматично",
            }
            for r in eval_rows
        ]
    )

    df_judges = pl.DataFrame(
        [
            {
                "Тест #": f"ТЕСТ {r.test_num}",
                "Резюме (українською)": r.summary_ua,
                "Знайдені пункти FAQ": " | ".join(r.faq_items),
                "Self-Judge Score": f"{r.self_judge_score:.0%}",
                "Independent Judge Score": f"{r.independent_judge_score:.0%}",
                "Час (мс)": r.latency_ms,
            }
            for r in eval_rows
        ]
    )

    graph_mermaid = mo.mermaid(
        """
        stateDiagram-v2
            [*] --> classify_query : SupportAgentState<br/>(user_query)
            classify_query --> search_faq : QueryClassification<br/>(type, category,<br/>urgency, summary)
            search_faq --> draft_response : search_results<br/>(1-2 найрелевантніші<br/>пункти FAQ)
            draft_response --> check_escalation : draft_response<br/>(≤35 слів українською)
            check_escalation --> [*] : needs_escalation (bool) +<br/>фінальний текст (≤50 слів)
        """
    )

    sub_path = export_hw5_jupyter_notebook("submissions/oleh_hatsenko_hw5.ipynb")
    gdocs_md = generate_hw5_google_docs_markdown("submissions/Hatsenko_Oleh_DZ5.md")

    tabs = mo.vstack([
        css_no_scroll,
        mo.ui.tabs(
            {
                "🚀 5 Запитів & Метрики": mo.vstack(
                    [mo.md("### Результати обробки 5 звернень графом `LangGraph v1`"), mo.ui.table(df_exec)]
                ),
                "🕸️ Граф StateGraph (Mermaid)": mo.vstack(
                    [
                        mo.center(mo.md("### Строго послідовна архітектура графа без умовних розгалужень")),
                        mo.center(graph_mermaid),
                    ]
                ),
                "⚖️ R&D: Dual Judge & Regex": mo.vstack(
                    [
                        mo.md("### Порівняння Self-Judge проти незалежного Judge LLM (`gpt-5.6-terra` / `gemini-3.1-pro`)"),
                        mo.ui.table(df_judges),
                        mo.md("### Верифікація виправленого парсера релевантності (`parse_relevancy_score`)"),
                        mo.ui.table(regex_df),
                    ]
                ),
                "🚦 R&D: Command & interrupt() (Тема 9)": mo.vstack(
                    [
                        mo.md(
                            "### Двофазне виконання з `Command[Literal[...]]`, `interrupt()` та `MemorySaver` із ноутбука Теми 9"
                        ),
                        mo.ui.table(hitl_topic9_df),
                    ]
                ),
                "📄 Звіт Google Docs & .ipynb": mo.vstack(
                    [
                        mo.md(f"✅ Файли збережено: `submissions/Hatsenko_Oleh_DZ5.md` та гравійний `submissions/Oleh_Hatsenko_DZ9.ipynb`"),
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
