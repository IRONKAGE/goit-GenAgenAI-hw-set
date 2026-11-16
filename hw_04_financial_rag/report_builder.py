# ==============================================================================
# 📄 HW 04 REPORT BUILDER: Генератор професійного звіту для ДЗ 4 (Google Docs & Marimo)
# ==============================================================================
"""
Формує детальний текстовий звіт `submissions/Hatsenko_Oleh_DZ4.md` з результатами
оцінювання 6 тестових питань, верифікацією 4 чанків, R&D трансформаціями (RRF, HyDE) та RAGAS.
"""

from pathlib import Path
from hw_04_financial_rag.rag_chain import FinancialRAGOrchestrator


def generate_hw4_google_docs_markdown(output_path: str = "submissions/Hatsenko_Oleh_DZ4.md") -> str:
    orchestrator = FinancialRAGOrchestrator(preferred_tier="reference")
    chunks = orchestrator.chunks
    records = orchestrator.evaluate_all_questions()

    qa_lines = []
    for r in records:
        guard_status = "🛡️ Спрацював (Компанія відсутня)" if r.hallucination_guard_triggered else "✅ Точні цифри з контексту"
        qa_lines.append(
            f"**Питання #{r.question_num}:** `{r.question}`\n"
            f"- *Знайдені чанки (k=2):* `{' + '.join(r.retrieved_companies)}`\n"
            f"- *Відповідь RAG-системи:* {r.answer}\n"
            f"- *Захист від галюцинацій:* `{guard_status}` | Латентність: `{r.latency_ms} мс`\n"
        )

    chunk_lines = []
    for idx, c in enumerate(chunks, 1):
        chunk_lines.append(
            f"**Чанк #{idx}** (Компанія: `{c.metadata.get('company', 'N/A')}`, символів: `{c.metadata.get('char_length', len(c.page_content))}`):\n"
            f"```text\n{c.page_content}\n```\n"
        )

    ragas_lines = []
    for r in records:
        ragas_lines.append(
            f"- **Q{r.question_num}** | Faithfulness: `{r.faithfulness}` | Answer Relevancy: `{r.answer_relevancy}` | "
            f"Context Precision (k=2): `{r.context_precision}` | Context Recall: `{r.context_recall}`\n"
        )

    report = f"""# Гаценко Олег — ДЗ 4 (Generative and Agentic AI)

---

## Загальний огляд архітектури
У цьому домашньому завданні реалізована повноцінна **RAG-система** для аналізу фінансових звітів українських компаній за 2024 рік.
Основні інженерні особливості реалізації:
1. **Детерміноване чанкування (`RecursiveCharacterTextSplitter`):** розбиття вихідного тексту на рівно 4 логічні чанки за розмежувачем `separators=["\\n\\n"]` (`chunk_size=300`, `chunk_overlap=0`).
2. **Векторне сховище ChromaDB (`finansovi_zvity_kompaniy`):** індексація документів та пошук із параметром `k=2` для витягування двох найрелевантніших контекстів.
3. **Двошаровий антигалюцинаційний Entity Guardrail:** система дозволяє гнучку орфографічну нормалізацію («ТехНова Інк ≡ Технова Інк»), але жорстко блокує перенесення числових показників на запити про неіснуючі компанії (наприклад, для «ЕкоПакування Солюшнс»).
4. **Просунуті трансформації запитів та метрики RAGAS:** реалізовано Multi-Query, RAG-Fusion (Reciprocal Rank Fusion), Decomposition, Step-Back, HyDE та оцінювання за 4 класичними метриками RAGAS (*Faithfulness*, *Answer Relevancy*, *Context Precision*, *Context Recall*).

---

## 💬 Результати обробки 6 тестових питань (`k=2`)

{"\n".join(qa_lines)}

---

## ✂️ Верифікація 4 чанків даних

{"\n".join(chunk_lines)}

---

## 📐 Оцінка якості RAG за 4 метриками RAGAS (Тема 9)

{"\n".join(ragas_lines)}

---
*Звіт згенеровано автоматично інфраструктурним ядром goit-GenAgenAI-hw-set (листопад 2026 року).*
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return report
