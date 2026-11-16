import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="HW 04: Financial Reports RAG, ChromaDB & Advanced Query Transforms",
)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import plotly.express as px
    import polars as pl

    from core import get_hardware_config, log_system_info, set_global_seed
    from hw_04_financial_rag import (
        COLLECTION_NAME_HW4,
        FinancialRAGOrchestrator,
        TEST_QUESTIONS_HW4,
        UI_LABELS,
        compute_cosine_similarity_matrix,
        export_hw4_jupyter_notebook,
        generate_hw4_google_docs_markdown,
        run_advanced_rag_comparison,
        split_financial_reports,
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
        FinancialRAGOrchestrator,
        TEST_QUESTIONS_HW4,
        UI_LABELS,
        compute_cosine_similarity_matrix,
        export_hw4_jupyter_notebook,
        generate_hw4_google_docs_markdown,
        mo,
        pl,
        px,
        run_advanced_rag_comparison,
        split_financial_reports,
    )


@app.cell(hide_code=True)
def title_head_hw(
    FinancialRAGOrchestrator,
    UI_LABELS,
    mo,
    split_financial_reports,
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

    chunks = split_financial_reports()
    orchestrator = FinancialRAGOrchestrator(preferred_tier="auto")
    qa_records = orchestrator.evaluate_all_questions()

    mo.md(
        f"""
        <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
            {UI_LABELS["title"]}</br><i>(Anti-Hallucination Financial RAG & Advanced Query Transformations)</i>
        </div>
        <h3 align="center"><b><u>Пайплайн</u>: ✂️ Entity Chunker (4 чанки) ┃ 🗄️ ChromaDB (finansovi_zvity_kompaniy, k=2) ┃</br>🚀 Multi-Query + RAG-Fusion (RRF) + Step-Back + HyDE ➔ 🛡️ Entity Guardrail & RAGAS</b></h3>
        <p align="center"><b>Кількість чанків:</b> <code>{len(chunks)}</code> &nbsp;|&nbsp; <b>Активний стек:</b> <code>{orchestrator.active_tier}</code> &nbsp;|&nbsp; <i>© Oleh Hatsenko (IRONKAGE) | Generative and Agentic AI</i></p>
        """
    )
    return chunks, css_no_scroll, orchestrator, qa_records


@app.cell(hide_code=True)
def main(
    TEST_QUESTIONS_HW4,
    chunks,
    compute_cosine_similarity_matrix,
    css_no_scroll,
    export_hw4_jupyter_notebook,
    generate_hw4_google_docs_markdown,
    mo,
    orchestrator,
    pl,
    px,
    qa_records,
    run_advanced_rag_comparison,
):
    df_qa = pl.DataFrame(
        [
            {
                "Питання #": f"Питання {r.question_num}",
                "Запитання користувача": r.question,
                "Знайдені чанки (k=2)": " + ".join(r.retrieved_companies),
                "Відповідь RAG-системи": r.answer,
                "Захист від галюцинацій": "🛡️ Спрацював (Компанія відсутня)"
                if r.hallucination_guard_triggered
                else "✅ Точні цифри з контексту",
                "Час (мс)": r.latency_ms,
            }
            for r in qa_records
        ]
    )

    df_chunks = pl.DataFrame(
        [
            {
                "Чанк #": i + 1,
                "Компанія (metadata)": c.metadata.get("company", "N/A"),
                "Довжина (символів)": len(c.page_content),
                "Текст чанка (page_content)": c.page_content,
            }
            for i, c in enumerate(chunks)
        ]
    )

    df_sim = compute_cosine_similarity_matrix(TEST_QUESTIONS_HW4, chunks)
    company_cols = [c.metadata.get("company", f"Chunk {i}") for i, c in enumerate(chunks)]
    sim_matrix_values = df_sim.select(company_cols).to_numpy()
    fig_heatmap = px.imshow(
        sim_matrix_values,
        x=company_cols,
        y=[f"Q{i + 1}: {q[:32]}..." for i, q in enumerate(TEST_QUESTIONS_HW4)],
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="Матриця косинусної схожості (Чому Q6 «ЕкоПакування Солюшнс» притягує чанк «ГрінЕнерджі Солюшнс»)",
    )

    df_adv_rag = run_advanced_rag_comparison(orchestrator.retriever)

    df_ragas = pl.DataFrame(
        [
            {
                "Питання #": f"Q{r.question_num}",
                "Faithfulness": r.faithfulness,
                "Answer Relevancy": r.answer_relevancy,
                "Context Precision (k=2)": r.context_precision,
                "Context Recall": r.context_recall,
            }
            for r in qa_records
        ]
    )

    sub_path = export_hw4_jupyter_notebook(chunks, qa_records, "submissions/oleh_hatsenko_hw4.ipynb")
    gdocs_md = generate_hw4_google_docs_markdown("submissions/Hatsenko_Oleh_DZ4.md")  # <--- Генерація звіту

    tabs = mo.vstack([
            css_no_scroll,
            mo.ui.tabs(
            {
                "💬 6 Питань & Відповіді": mo.vstack(
                    [mo.md("### Результати роботи фінансової RAG-системи (`k=2`)"), mo.ui.table(df_qa)]
                ),
                "✂️ 4 Чанки (Верифікація)": mo.vstack(
                    [
                        mo.md(
                            "### Перевірка розбиття `RecursiveCharacterTextSplitter(separators=['\\n\\n'], chunk_size=300)`"
                        ),
                        mo.ui.table(df_chunks),
                    ]
                ),
                "🔥 Матриця Косинусної Схожості": mo.vstack([mo.ui.plotly(fig_heatmap), mo.ui.table(df_sim)]),
                "🚀 R&D: Multi-Query, RRF, Step-Back & HyDE": mo.vstack(
                    [
                        mo.md("### Просунуті трансформації запитів із живого ноутбука Теми 7"),
                        mo.ui.table(df_adv_rag),
                    ]
                ),
                "📐 Метрики RAGAS": mo.vstack(
                    [mo.md("### Оцінка якості RAG за 4 метриками RAGAS (Тема 9)"), mo.ui.table(df_ragas)]
                ),
                "📄 Звіт Google Docs & .ipynb": mo.vstack(
                    [
                        mo.md(f"✅ Файли збережено: `submissions/Hatsenko_Oleh_DZ4.md` та `{sub_path}`"),
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
