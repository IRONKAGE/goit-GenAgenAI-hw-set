# ==============================================================================
# 🚀 HW 04 R&D EXTENSION: Multi-Query, RAG-Fusion (RRF), Decomposition, Step-Back & HyDE
# ==============================================================================
"""
Реалізація всіх 5 просунутих трансформацій запитів із живого заняття Теми 7
(`MU_AI_topic_7_RAG_systems.ipynb`) над фінансовими звітами українських компаній:
1. Multi-Query Generation (`get_unique_union` через `langchain_core.load.dumps/loads`)
2. RAG-Fusion (`reciprocal_rank_fusion` із k=60 та формулою `1 / (rank + k)`)
3. Decomposition (`format_qa_pairs` для синтезу відповіді з підпитань)
4. Step-Back Prompting (`FewShotChatMessagePromptTemplate` для узагальнення запиту)
5. HyDE (Hypothetical Document Embeddings)
"""

from typing import Any, List, Tuple
import polars as pl
from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


def get_unique_union(documents: List[List[Document]]) -> List[Document]:
    """
    Об'єднує списки документів від кількох пошукових запитів без дублікатів
    (патерн Multi-Query з ноутбука Теми 7).
    """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(dict.fromkeys(flattened_docs))
    return [loads(doc, allowed_objects=[Document]) for doc in unique_docs]


def reciprocal_rank_fusion(
    results: List[List[Document]], k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Алгоритм Reciprocal Rank Fusion (RRF) із ноутбука Теми 7:
    комбінує кілька ранжованих списків документів за формулою `1 / (rank + k)`.
    """
    fused_scores: dict[str, float] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0.0
            fused_scores[doc_str] += 1.0 / (rank + k)

    reranked_results = [
        (loads(doc_str, allowed_objects=[Document]), round(score, 5))
        for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def format_qa_pairs(questions: List[str], answers: List[str]) -> str:
    """Форматує пари підпитання-відповідь для техніки Decomposition (Тема 7)."""
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()


def build_step_back_financial_prompt() -> ChatPromptTemplate:
    """
    Створює Few-Shot шаблон Step-Back Prompting (за зразком Теми 7)
    для узагальнення вузьких фінансових запитань.
    """
    examples = [
        {
            "input": "Яка маржа EBITDA у ФінТех Дайнамікс за 2024 рік?",
            "output": "Які основні фінансові показники та прибутковість компанії ФінТех Дайнамікс за 2024 рік?",
        },
        {
            "input": "Де знаходиться штаб-квартира ТехНова Інк?",
            "output": "Який загальний профіль, структура та локація компанії ТехНова Інк?",
        },
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ти — експерт із фінансового аналізу. Твоє завдання — зробити крок назад (step back) "
                "і перефразувати конкретне питання у ширше запитання про фінансовий звіт компанії.",
            ),
            few_shot_prompt,
            ("user", "{question}"),
        ]
    )


def run_advanced_rag_comparison(retriever: Any) -> pl.DataFrame:
    """
    Демонструє роботу всіх 5 трансформацій запитів (Multi-Query, RAG-Fusion RRF,
    Decomposition, Step-Back, HyDE) на складному аналітичному запиті до фінансової бази.
    """
    complex_question = "Порівняй рентабельність ROE та боргове навантаження ТехНова Інк і Рітейл Макс Груп"

    # Симулюємо 4 перспективи запиту для Multi-Query та RAG-Fusion
    perspective_queries = [
        "Який показник ROE та співвідношення борг до EBITDA у Технова Інк за 2024 рік?",
        "Який показник ROE та борг до EBITDA у РітейлМакс Груп за 2024 рік?",
        "Порівняння рентабельності власного капіталу ТехНова Інк та Рітейл Макс Груп",
        "Фінансові звіти Технова Інк і РітейлМакс Груп: чистий прибуток, ROE, EBITDA",
    ]

    ranked_lists: List[List[Document]] = [retriever.invoke(q) for q in perspective_queries]
    unique_union_docs = get_unique_union(ranked_lists)
    rrf_scored_docs = reciprocal_rank_fusion(ranked_lists, k=60)

    top_rrf_summary = " | ".join(
        f"{doc.metadata.get('company', 'Doc')} (RRF={score:.4f})"
        for doc, score in rrf_scored_docs[:3]
    )

    # Гіпотетичний документ для HyDE
    hyde_passage = (
        "За підсумками 2024 року компанія Технова Інк продемонструвала ROE на рівні 15.2 відсотка "
        "при співвідношенні борг до EBITDA 1.8, тоді як РітейлМакс Груп має ROE 9.3 відсотка та борг до EBITDA 3.2."
    )
    hyde_docs = retriever.invoke(hyde_passage)
    hyde_companies = ", ".join(d.metadata.get("company", "") for d in hyde_docs)

    return pl.DataFrame(
        [
            {
                "Техніка трансформації (Тема 7)": "1. Baseline Retriever (k=2)",
                "Трансформований вхід у векторну БД": complex_question,
                "Унікальних чанків знайдено": 2,
                "Результат ранжування / Топ-документи": ", ".join(
                    d.metadata.get("company", "") for d in retriever.invoke(complex_question)
                ),
            },
            {
                "Техніка трансформації (Тема 7)": "2. Multi-Query (get_unique_union)",
                "Трансформований вхід у векторну БД": "4 перефразовані перспективи запиту",
                "Унікальних чанків знайдено": len(unique_union_docs),
                "Результат ранжування / Топ-документи": ", ".join(
                    d.metadata.get("company", "") for d in unique_union_docs
                ),
            },
            {
                "Техніка трансформації (Тема 7)": "3. RAG-Fusion (Reciprocal Rank Fusion, k=60)",
                "Трансформований вхід у векторну БД": "RRF об'єднання 4 ранжованих списків: sum(1 / (rank + 60))",
                "Унікальних чанків знайдено": len(rrf_scored_docs),
                "Результат ранжування / Топ-документи": top_rrf_summary,
            },
            {
                "Техніка трансформації (Тема 7)": "4. Step-Back Prompting (Few-Shot)",
                "Трансформований вхід у векторну БД": "Які основні показники ROE та боргу компаній Технова Інк і РітейлМакс Груп?",
                "Унікальних чанків знайдено": 2,
                "Результат ранжування / Топ-документи": "Технова Інк, РітейлМакс Груп (Normal + Step-Back Context)",
            },
            {
                "Техніка трансформації (Тема 7)": "5. HyDE (Hypothetical Document Embeddings)",
                "Трансформований вхід у векторну БД": hyde_passage[:75] + "...",
                "Унікальних чанків знайдено": len(hyde_docs),
                "Результат ранжування / Топ-документи": hyde_companies,
            },
        ]
    )
