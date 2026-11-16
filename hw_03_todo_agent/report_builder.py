# ==============================================================================
# 📄 HW 03 REPORT BUILDER: Генератор професійного звіту для ДЗ 3 (Google Docs & Marimo)
# ==============================================================================
"""
Формує детальний текстовий звіт `submissions/Hatsenko_Oleh_DZ3.md` з результатами
виконання 5 канонічних сценаріїв TODO-агента, станом InMemoryStore та верифікацією edge-cases.
"""

from pathlib import Path
from hw_03_todo_agent.agent import TodoAgentOrchestrator


def generate_hw3_google_docs_markdown(output_path: str = "submissions/Hatsenko_Oleh_DZ3.md") -> str:
    orchestrator = TodoAgentOrchestrator(preferred_tier="reference")
    logs = orchestrator.run_canonical_suite()
    edge_df = TodoAgentOrchestrator.run_edge_case_verification()
    hitl_df = orchestrator.simulate_hitl_delete_approval()

    canonical_lines = []
    for log in logs:
        canonical_lines.append(
            f"**Тест #{log.test_index}:** `{log.user_query}`\n"
            f"- *Викликані інструменти (ReAct Trace):* `{' ➔ '.join(log.tool_calls_chain)}`\n"
            f"- *Відповідь агента:* {log.final_response}\n"
            f"- *Стан InMemoryStore (записів: {log.store_snapshot_count}):* `{'; '.join(log.store_snapshot_items) if log.store_snapshot_items else '(порожньо)'}`\n"
            f"- *Латентність:* `{log.latency_ms} мс`\n"
        )

    edge_lines = []
    for row in edge_df.iter_rows(named=True):
        # Безпечний доступ до ключів рядка незалежно від їх назви у DataFrame
        values = list(row.values())
        edge_lines.append(f"- **Перевірка:** `{' | '.join(str(v) for v in values)}`\n")

    hitl_lines = []
    for row in hitl_df.iter_rows(named=True):
        values = list(row.values())
        hitl_lines.append(f"- **HITL етап:** `{' | '.join(str(v) for v in values)}`\n")

    report = f"""# Гаценко Олег — ДЗ 3 (Generative and Agentic AI)

---

## Загальний огляд архітектури
У цьому домашньому завданні реалізовано інструментального **ReAct-агента** на базі фреймворку `LangChain v1` (`create_agent`) із довготривалою пам'яттю `InMemoryStore` (`get_store()`), яка забезпечує персистентність стану між ізольованими сесіями діалогу.

Основні інженерні особливості реалізації:
1. **Короткі 8-символьні UUID (`uuid4().hex[:8]`):** забезпечують зручну адресацію завдань без перевантаження контекстного вікна LLM.
2. **Двокроковий протокол безпечного видалення за змістом:** оскільки тести надходять у нових сесіях без історії, системний промпт примушує агента спершу викликати `list_tasks(limit=1000)`, знайти ID за назвою, перевірити через `store.get()` і лише потім викликати `store.delete()`.
3. **Обхід обмежень пакування:** використання параметру `limit=1000` гарантує відсутність пропуску завдань при масштабуванні сховища.
4. **Контур захисту HITL (Human-in-the-Loop):** перехоплення деструктивних дій через `HumanInTheLoopMiddleware` з можливістю підтвердження або відхилення оператором.

---

## 🚀 Результати виконання 5 канонічних тестів

{"\n".join(canonical_lines)}

---

## 🛡️ Верифікація стійкості інструментів (Edge-Cases)

{"\n".join(edge_lines)}

---

## ⏸️ Симуляція HITL Middleware (Теми 6, 10)

{"\n".join(hitl_lines)}

---
*Звіт згенеровано автоматично інфраструктурним ядром goit-GenAgenAI-hw-set (листопад 2026 року).*
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return report
