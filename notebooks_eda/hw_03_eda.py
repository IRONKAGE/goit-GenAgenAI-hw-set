import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="HW 03: LangChain v1 TODO Agent, InMemoryStore & HITL",
)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import polars as pl

    from core import get_hardware_config, log_system_info, set_global_seed
    from hw_03_todo_agent import (
        TodoAgentOrchestrator,
        UI_LABELS,
        export_hw3_jupyter_notebook,
        simulate_hitl_delete_approval,
        generate_hw3_google_docs_markdown,
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
        TodoAgentOrchestrator,
        UI_LABELS,
        export_hw3_jupyter_notebook,
        generate_hw3_google_docs_markdown,
        mo,
        pl,
        simulate_hitl_delete_approval,
    )


@app.cell(hide_code=True)
def title_head_hw(
    TodoAgentOrchestrator,
    UI_LABELS,
    mo,
    simulate_hitl_delete_approval,
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

    orchestrator = TodoAgentOrchestrator(preferred_tier="auto")
    canonical_logs = orchestrator.run_canonical_suite()
    edge_df = TodoAgentOrchestrator.run_edge_case_verification()
    hitl_df = simulate_hitl_delete_approval()

    mo.md(
        f"""
        <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
            {UI_LABELS["title"]}</br><i>(Stateful ReAct Tool-Calling & Long-Term Memory Architecture)</i>
        </div>
        <h3 align="center"><b><u>Пайплайн</u>: 🤖 LangChain v1 (create_agent) ┃ 🗄️ LangGraph InMemoryStore (UUID-8 + limit=1000) ┃</br>🔄 Two-Step Delete Protocol (list_tasks ➔ store.get ➔ delete) ➔ ⏸️ HITL Middleware</b></h3>
        <p align="center"><b>Активний 3-Tier рівень:</b> <code>{orchestrator.active_tier.upper()}</code> (<code>{orchestrator.model_name}</code>) &nbsp;|&nbsp; <i>© Oleh Hatsenko (IRONKAGE) | Generative and Agentic AI</i></p>
        """
    )
    return canonical_logs, css_no_scroll, edge_df, hitl_df


@app.cell(hide_code=True)
def main(
    canonical_logs,
    css_no_scroll,
    edge_df,
    export_hw3_jupyter_notebook,
    generate_hw3_google_docs_markdown,
    hitl_df,
    mo,
    pl,
):
    df_canonical = pl.DataFrame(
        [
            {
                "Тест #": f"TEST {log.test_index}",
                "Запит користувача": log.user_query,
                "Викликані інструменти (ReAct Trace)": " ➔ ".join(log.tool_calls_chain),
                "Відповідь агента": log.final_response,
                "Записів у InMemoryStore": log.store_snapshot_count,
                "Вміст InMemoryStore після кроку": "; ".join(log.store_snapshot_items)
                if log.store_snapshot_items
                else "(порожньо)",
                "Час (мс)": log.latency_ms,
            }
            for log in canonical_logs
        ]
    )

    mermaid_diagram = mo.mermaid(
        """
        flowchart LR
            U[👤 Користувач] -->|Новий діалог| A[🤖 create_agent\nLangChain v1]
            A -->|Tool Call| T1[➕ add_task\nstore.put + UUID-8]
            A -->|Tool Call| T2[📋 list_tasks\nstore.search limit=1000]
            A -->|Tool Call| T3[🗑️ delete_task\nstore.get ➔ store.delete]
            T1 <--> S[(🗄️ InMemoryStore\nget_store)]
            T2 <--> S
            T3 <--> S
            T2 -.->|Крок 1: Знайти ID хліба| A
            A -.->|Крок 2: Передати 8-симв. ID| T3
        """
    )

    sub_path = export_hw3_jupyter_notebook(canonical_logs, "submissions/oleh_hatsenko_hw3.ipynb")
    gdocs_md = generate_hw3_google_docs_markdown("submissions/Hatsenko_Oleh_DZ3.md")

    tabs = mo.vstack([
        css_no_scroll,
        mo.ui.tabs(
            {
                "🚀 5 Тестових Сценаріїв": mo.vstack(
                    [
                        mo.md("### Трасування виконання 5 сценаріїв та стан `InMemoryStore`"),
                        mo.ui.table(df_canonical),
                    ]
                ),
                "🛡️ R&D: Перевірка Edge-Cases": mo.vstack(
                    [
                        mo.md("### Верифікація стійкості інструментів (`limit=1000`, `store.get()`, порожній список)"),
                        mo.ui.table(edge_df),
                    ]
                ),
                "⏸️ R&D: HITL Middleware (Теми 6, 10)": mo.vstack(
                    [
                        mo.md("### Захист видалення завдань через `HumanInTheLoopMiddleware(interrupt_on={'delete_task': True})`"),
                        mo.ui.table(hitl_df),
                    ]
                ),
                "🕸️ Архітектура Агента": mo.vstack(
                    [
                        mo.md("### Діаграма взаємодії `create_agent` та `InMemoryStore`"),
                        mermaid_diagram,
                    ]
                ),
                "📄 Звіт Google Docs & .ipynb": mo.vstack(
                    [
                        mo.md(f"✅ Файли збережено: `submissions/Hatsenko_Oleh_DZ3.md` та `{sub_path}`"),
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
