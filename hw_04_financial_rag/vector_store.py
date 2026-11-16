# ==============================================================================
# 🗄️ HW 04 VECTOR STORE: ChromaDB (finansovi_zvity_kompaniy) & Cosine Matrix
# ==============================================================================
"""
Ініціалізація векторного сховища `langchain_chroma.Chroma`, синхронізована з
`core.get_smart_embeddings` та `core.DeterministicSemanticEmbeddings`,
плюс розрахунок матриці косинусної схожості між запитами й чанками.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core import DeterministicSemanticEmbeddings, get_smart_embeddings
from hw_04_financial_rag.chunker import split_financial_reports

logger = logging.getLogger("hw_04.vector_store")

# Латинська назва колекції згідно з правилами Chroma та критеріями ДЗ 4
COLLECTION_NAME_HW4: str = "finansovi_zvity_kompaniy"


def build_chroma_vectorstore(
    chunks: Optional[List[Document]] = None,
    preferred_tier: str = "auto",
    collection_name: str = COLLECTION_NAME_HW4,
) -> Tuple[Chroma, Embeddings, str]:
    """
    Індексує 4 чанки фінансових звітів у колекцію Chroma `finansovi_zvity_kompaniy`.
    Використовує детерміновані ID `chunk_0..chunk_3`, щоб уникнути дублювання при повторних викликах.
    """
    docs = chunks if chunks is not None else split_financial_reports()
    doc_ids = [f"chunk_{i}" for i in range(len(docs))]
    embeddings, tier_name = get_smart_embeddings(preferred_tier=preferred_tier)

    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            ids=doc_ids,
            collection_name=collection_name,
        )
        return vectorstore, embeddings, tier_name
    except Exception as err:
        logger.warning(f"Фолбек індексації Chroma на детерміновані ембединги через збій API: {err}")
        fallback_emb = DeterministicSemanticEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=fallback_emb,
            ids=doc_ids,
            collection_name=f"{collection_name}_offline",
        )
        return vectorstore, fallback_emb, f"{tier_name}->reference"


def compute_cosine_similarity_matrix(
    questions: List[str],
    chunks: List[Document],
    embeddings: Optional[Embeddings] = None,
) -> pl.DataFrame:
    """
    R&D-аналіз (Тема 7): обчислює повну матрицю косинусної схожості між 6 тестовими
    питаннями та 4 чанками компаній, наочно демонструючи Top-2 вибір ретрівера (k=2).
    """
    emb_model = embeddings if embeddings is not None else DeterministicSemanticEmbeddings()
    try:
        q_vecs = np.array(emb_model.embed_documents(questions))
        c_vecs = np.array(emb_model.embed_documents([c.page_content for c in chunks]))
    except Exception:
        fallback = DeterministicSemanticEmbeddings()
        q_vecs = np.array(fallback.embed_documents(questions))
        c_vecs = np.array(fallback.embed_documents([c.page_content for c in chunks]))

    company_labels = [c.metadata.get("company", f"Chunk {i}") for i, c in enumerate(chunks)]
    rows = []
    for q_idx, q_text in enumerate(questions):
        q_vec = q_vecs[q_idx]
        q_norm = max(1e-9, float(np.linalg.norm(q_vec)))
        sims = {}
        for c_idx, comp_name in enumerate(company_labels):
            c_vec = c_vecs[c_idx]
            c_norm = max(1e-9, float(np.linalg.norm(c_vec)))
            cos_sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            sims[comp_name] = round(cos_sim, 3)

        top2 = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)[:2]
        rows.append(
            {
                "Питання #": f"Q{q_idx + 1}",
                "Запитання користувача": q_text,
                **sims,
                "Top-2 Retrieved (k=2)": f"1) {top2[0][0]} ({top2[0][1]:.2f}) | 2) {top2[1][0]} ({top2[1][1]:.2f})",
            }
        )
    return pl.DataFrame(rows)
