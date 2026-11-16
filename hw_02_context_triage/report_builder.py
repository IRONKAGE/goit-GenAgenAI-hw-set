# ==============================================================================
# 📄 HW 02 REPORT BUILDER: Генератор професійного звіту для ДЗ 2 (Google Docs & Marimo)
# ==============================================================================
"""
Формує детальний текстовий звіт `submissions/Hatsenko_Oleh_DZ2.md` з результатами
виконання всіх 3 завдань, роботою функції check(), Self-Healing циклу та R&D абляції ключів.
"""

from pathlib import Path
from hw_02_context_triage.extractor import ContextTriageExtractor, ExtractionRecord
from typing import List


def generate_hw2_google_docs_markdown(output_path: str = "submissions/Hatsenko_Oleh_DZ2.md") -> str:
    extractor = ContextTriageExtractor(preferred_tier="reference")
    all_runs = extractor.run_all_tasks()
    ablation_df = ContextTriageExtractor.run_key_order_ablation_study()

    def format_records(records: List[ExtractionRecord]) -> str:
        lines = []
        for idx, r in enumerate(records, 1):
            status = "✅ Пройдено" if r.is_exact_match else "⚠️ Відхилення"
            lines.append(
                f"**Приклад {idx}:**\n"
                f"- *Вхідний текст:* `{r.input_text}`\n"
                f"- *JSON-відповідь моделі:* `{r.raw_json_str}`\n"
                f"- *Статус валідації check():* `{status}` (Спроб: `{r.attempts_used}`, Латентність: `{r.latency_ms} мс`)\n"
            )
        return "\n".join(lines)

    # Використовуємо звичайний багаторядковий рядок ("""), щоб уникнути конфліктів фігурних дужок з {text}
    report = """# Гаценко Олег — ДЗ 2 (Generative and Agentic AI)

---

## Загальний огляд архітектури
У цьому домашньому завданні реалізовано надійний конвеєр **Context Engineering** для екстракції структурованих даних у суворому форматі JSON.
Основні інженерні компоненти:
1. **XML Role/Data Isolation:** ізоляція інструкцій та вхідних даних за допомогою тегів `<role>`, `<task>`, `<output_contract>` та `<<<text>>>`.
2. **Функція суворої валідації `check()`:** забезпечує перевірку наявності ключів, їх відповідність схемі та допустимим значенням.
3. **Контур самовідновлення (Self-Correction Loop):** у разі помилки валідації модель отримує детальну підказку зі звітом про помилку та виправляє свій вивід.

---

## 👤 Завдання 1. Персона + Тональність (50 балів)
Екстракція імені персони з відсіканням посад (напр., «Керівник відділу Андрій Король ➔ Андрій Король») та визначення тональності (*позитивна*, *нейтральна*, *негативна*).

""" + format_records(all_runs["task1"]) + """

---

## 🚨 Завдання 2. Тріаж відгуків: Продукт + Пріоритет (25 балів)
Нормалізація назв продуктів (`Atlas Pro ➔ Atlas`, `Core 3.1 ➔ Core`) та класифікація пріоритету за рівнем впливу на роботу бізнесу.

""" + format_records(all_runs["task2"]) + """

---

## 📰 Завдання 3. Заголовок (≤ 8 слів) + Категорія оголошення (25 балів)
Формування стислого інформативного заголовка українською мовою з жорстким лімітом не більше 8 слів та класифікація за категоріями (*реліз*, *інцидент*, *рекомендація*).

""" + format_records(all_runs["task3"]) + """

---

## 🔬 R&D Дослідження: Вплив порядку ключів у JSON (Тема 4, Слайд 25 — Explanation First)
Досліджено, що генерація пояснення (`explanation`) *перед* цільовими полями класу дозволяє механізму авторегресійної уваги моделі ефективно сформувати ланцюжок міркувань (*Chain-of-Thought*), суттєво підвищуючи точність на складних кейсах із сарказмом чи змішаними сигналами.

| Кейс # | Завдання | Тип ускладнення | Очікуваний клас | Промпт 1 (Class First) | Промпт 2 (Explanation First — CoT) |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for row in ablation_df.iter_rows(named=True):
        report += (
            f"| {row['Кейс #']} | {row['Завдання']} | {row['Тип ускладнення']} | "
            f"{row['Очікуваний клас']} | {row['Промпт 1 (Class First)']} | {row['Промпт 2 (Explanation First — CoT)']} |\n"
        )

    report += """
---
*Звіт згенеровано автоматично інфраструктурним ядром goit-GenAgenAI-hw-set (листопад 2026 року).*
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return report
