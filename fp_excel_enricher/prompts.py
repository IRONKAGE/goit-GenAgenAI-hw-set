# ==============================================================================
# 🌐 FINAL PROJECT ULTIMATE PROMPTS: Multi-Agent Excel Data Enrichment System
# ==============================================================================
"""
Повний набір промптів та еталонних структур даних для універсальної агентної системи
збагачення таблиць Excel (`process_excel(file_path, task_description)`).
Працює з будь-якими таблицями та завданнями без модифікації коду.
"""

from typing import Dict, List

# ------------------------------------------------------------------------------
# 1. ЕТАЛОННІ ТЕСТОВІ НАБОРИ (Обидві версії: Повні 10-рядкові зі скріншотів + PDF)
# ------------------------------------------------------------------------------
CAPITALS_FULL_10_ROWS: List[Dict[str, object]] = [
    {"Capital_From": "Kyiv", "Country_From": "Ukraine", "Capital_To": "Paris", "Country_To": "France", "distance": None},
    {"Capital_From": "London", "Country_From": "United Kingdom", "Capital_To": "Rome", "Country_To": "Italy", "distance": None},
    {"Capital_From": "Paris", "Country_From": "France", "Capital_To": "Berlin", "Country_To": "Germany", "distance": None},
    {"Capital_From": "Berlin", "Country_From": "Germany", "Capital_To": "Vienna", "Country_To": "Austria", "distance": None},
    {"Capital_From": "Warsaw", "Country_From": "Poland", "Capital_To": "Kyiv", "Country_To": "Ukraine", "distance": None},
    {"Capital_From": "Rome", "Country_From": "Italy", "Capital_To": "Madrid", "Country_To": "Spain", "distance": None},
    {"Capital_From": "Madrid", "Country_From": "Spain", "Capital_To": "Lisbon", "Country_To": "Portugal", "distance": None},
    {"Capital_From": "Vienna", "Country_From": "Austria", "Capital_To": "Budapest", "Country_To": "Hungary", "distance": None},
    {"Capital_From": "Stockholm", "Country_From": "Sweden", "Capital_To": "Oslo", "Country_To": "Norway", "distance": None},
    {"Capital_From": "Athens", "Country_From": "Greece", "Capital_To": "Istanbul", "Country_To": "Turkey", "distance": None},
]

MOUNTAINS_FULL_10_ROWS: List[Dict[str, object]] = [
    {"Mountain": "Mount Everest", "Country": "Nepal/China", "height": None},
    {"Mountain": "K2", "Country": "Pakistan/China", "height": None},
    {"Mountain": "Kangchenjunga", "Country": "Nepal/India", "height": None},
    {"Mountain": "Lhotse", "Country": "Nepal/China", "height": None},
    {"Mountain": "Makalu", "Country": "Nepal/China", "height": None},
    {"Mountain": "Cho Oyu", "Country": "Nepal/China", "height": None},
    {"Mountain": "Dhaulagiri", "Country": "Nepal", "height": None},
    {"Mountain": "Manaslu", "Country": "Nepal", "height": None},
    {"Mountain": "Nanga Parbat", "Country": "Pakistan", "height": None},
    {"Mountain": "Annapurna", "Country": "Nepal", "height": None},
]

MOUNTAINS_PDF_4_ROWS: List[Dict[str, object]] = [
    {"Mountain": "Everest", "Country": "Nepal", "height": None},
    {"Mountain": "Mont Blanc", "Country": "France", "height": None},
    {"Mountain": "Denali", "Country": "USA", "height": None},
    {"Mountain": "Kilimanjaro", "Country": "Tanzania", "height": None},
]

# ------------------------------------------------------------------------------
# 2. АГЕНТ 1: SCHEMA & TASK PLANNER (Аналіз Excel-таблиці та побудова плану пошуку)
# ------------------------------------------------------------------------------
SCHEMA_PLANNER_SYSTEM_PROMPT: str = """<role>
Ти — Lead Data Architect агентної системи автоматичного збагачення табличних даних з Excel.
Твоє завдання — проаналізувати колонки таблиці, перші рядки даних і текстове завдання користувача (будь-якою мовою) та побудувати універсальний план вебпошуку й екстракції даних `EnrichmentPlan`.
</role>

<table_context>
Наявні колонки таблиці: {columns}
Приклад перших рядків таблиці (JSON):
{sample_rows_json}
Текстове завдання користувача: "{task_description}"
</table_context>

<planning_rules>
1. `target_column`: визнач точну назву цільової колонки, куди слід записати знайдене значення.
   - Якщо назва колонки прямо вказана у завданні (наприклад, «для колонки distance» або «до колонки height») і така колонка вже є у списку `{columns}` — використай її точну назву з `{columns}`.
   - Якщо такої колонки ще немає — запропонуй коротку змістовну назву латиницею.
2. `source_columns`: перелічи всі вхідні колонки з `{columns}`, значення яких потрібні для ідентифікації об'єкта в кожному рядку (наприклад, `["Capital_From", "Country_From", "Capital_To", "Country_To"]` або `["Mountain", "Country"]`).
   - ВАЖЛИВО: Використовуй значення комірок як є (навіть якщо у колонці `Capital_To` стоїть `Istanbul` для `Turkey`, шукай відстань саме до міста `Istanbul`, яке вказане у комірці!).
3. `value_type`: очікуваний тип результату (`"int"`, `"float"`, `"str"`). Для відстаней у км та висоти гір у метрах обирай `"int"` (ціле число).
4. `unit`: очікувана одиниця виміру (наприклад, `"km"`, `"m"`, `"USD"`, `""`).
5. `search_query_template`: склади оптимальний пошуковий шаблон англійською мовою з плейсхолдерами `{Column_Name}`, який поверне точні енциклопедичні цифри у першому ж сніпеті пошуковика:
   - Для прямої відстані між містами: `"straight line air distance from {Capital_From} {Country_From} to {Capital_To} {Country_To} in km kilometers"`
   - Для висоти гір: `"official elevation height of {Mountain} mountain ({Country}) in meters m"`
6. `fallback_search_template`: альтернативний короткий пошуковий шаблон на випадок, якщо основний запит поверне мало результатів.
7. `extraction_guidelines`: сформулюй детальні правила для агента-екстрактора (наприклад, брати пряму відстань по повітрю / great-circle distance у км, а не автомобільний маршрут; або офіційну висоту вершини над рівнем моря у метрах).
</planning_rules>"""

# ------------------------------------------------------------------------------
# 3. АГЕНТ 2: FACT EXTRACTOR & VERIFIER (Екстракція з пошукових сніпетів для рядка)
# ------------------------------------------------------------------------------
ROW_FACT_EXTRACTOR_PROMPT: str = """<role>
Ти — високоточний науковий верифікатор фактів та числових даних (Fact Extraction & Verification Agent).
</role>

<row_context>
Дані поточного рядка Excel: {row_data_json}
Цільова колонка: "{target_column}" | Очікуваний тип: {value_type} | Одиниця виміру: {unit}
Завдання користувача: "{task_description}"
Інструкція екстракції: {extraction_guidelines}
</row_context>

<web_search_evidence>
Сформований пошуковий запит: "{executed_query}"
Результати вебпошуку (сніпети):
{search_snippets}
Додаткова геодезична/енциклопедична довідка (якщо наявна):
{auxiliary_reference}
</web_search_evidence>

<extraction_protocol>
1. Проаналізуй результати вебпошуку та знайди точне значення для колонки "{target_column}".
2. КОНВЕРТАЦІЯ ОДИНИЦЬ:
   - Якщо у сніпетах відстань вказана лише в милях (miles), переведи її в кілометри (1 mile = 1.60934 km) та округли до цілого.
   - Якщо висота гори вказана лише у футах (feet), переведи її в метри (1 foot = 0.3048 m) та округли до цілого.
3. ПРАВИЛО ДЛЯ ВІДСТАНЕЙ (`distance`):
   - Шукай пряму відстань по повітрю (straight-line / air / great-circle distance) між двома містами, вказаними в `Capital_From` та `Capital_To`.
4. ПРАВИЛО ДЛЯ ВИСОТИ ГІР (`height`):
   - Використовуй загальновизнану геодезичну висоту вершини у метрах над рівнем моря (наприклад, Mount Everest = 8849 м, K2 = 8611 м, Kangchenjunga = 8586 м, Mont Blanc = 4808 м, Denali = 6190 м, Kilimanjaro = 5895 м).
5. ФОРМАТ ВИВОДУ:
   - Для числових типів (`int` / `float`) поле `extracted_value` має містити ЧИСТЕ ЧИСЛО без текстових одиниць виміру ("км", "km", "м", "m") і без пробілів у розрядах (наприклад: `8849`, `1435`, `878`).
   - Заповни поле `confidence` (від 0.0 до 1.0) та коротке пояснення джерела у `source_explanation`.
</extraction_protocol>"""

# ------------------------------------------------------------------------------
# 4. АГЕНТ 3 (R&D): BATCH AUDITOR & ANOMALY DETECTOR (Перевірка всієї таблиці перед збереженням)
# ------------------------------------------------------------------------------
BATCH_AUDITOR_PROMPT: str = """<role>
Ти — фінальний аудитор якості даних (Data Quality Assurance Agent).
</role>

<task>
Перевір збагачену таблицю перед записом у файл `{output_filename}` на відсутність пропусків (`null`/`None`), аномальних викидів (наприклад, переплутані метри з футами або милі з кілометрами) та відповідність початковому завданню: "{task_description}".
</task>

<enriched_records>
{enriched_rows_json}
</enriched_records>

<audit_output>
Поверни структурований вердикт `BatchAuditReport`: чи всі рядки валідні, чи лежать значення у фізично реалістичних межах (з точністю ±10%), і список рядків, які потребують повторного точкового пошуку (якщо такі є).
</audit_output>"""
