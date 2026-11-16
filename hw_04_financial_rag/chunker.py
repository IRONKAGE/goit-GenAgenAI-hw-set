# ==============================================================================
# ✂️ HW 04 CHUNKER: Детермінований сплітер фінансових звітів (рівно 4 чанки)
# ==============================================================================
"""
Розбиває фінансовий текст на логічні чанки по компаніях за допомогою
RecursiveCharacterTextSplitter(separators=["\\n\\n"], chunk_size=300, chunk_overlap=0).
Гарантує виконання вимоги: print(len(chunks)) == 4.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hw_04_financial_rag.prompts import RAW_FINANCIAL_REPORTS_TEXT


def split_financial_reports(
    raw_text: str = RAW_FINANCIAL_REPORTS_TEXT,
    chunk_size: int = 300,
    chunk_overlap: int = 0,
) -> List[Document]:
    """
    Розбиває текст фінансових звітів на 4 окремі документи (по одній компанії в чанку).
    Використовує параметр `separators=["\\n\\n"]` у множині згідно з оновленням ДЗ 4.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.create_documents([raw_text.strip()])

    for idx, doc in enumerate(chunks):
        first_part = doc.page_content.split(" за 2024 рік:")[0].strip()
        doc.metadata = {
            "chunk_id": idx,
            "company": first_part,
            "char_length": len(doc.page_content),
            "collection_title_ua": "Фінансові звіти українських компаній за 2024 рік",
        }

    if len(chunks) != 4:
        raise ValueError(f"Критерій ДЗ 4 порушено: очікувалось рівно 4 чанки, отримано {len(chunks)}")

    return chunks
