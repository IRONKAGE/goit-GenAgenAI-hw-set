# ==============================================================================
# 🛡️ HW 02 SCHEMAS & VALIDATORS: Функція check() за стандартом курсу + Pydantic v2
# ==============================================================================
"""
Реалізує:
1. Канонічну функцію `check()` з офіційного документа оновлень ДЗ 2 (листопад 2026).
2. Спеціалізовані функції `check_task2()` та `check_task3()` (з перевіркою ліміту <= 8 слів).
3. Моделі Pydantic v2 з автоматичним очищенням посад, нормалізацією продуктів та лімітом слів.
"""

import re
from typing import Any, Dict, Literal, Set
from pydantic import BaseModel, Field, field_validator

from core import ALLOWED_TONES_DEFAULT, check

ALLOWED: Set[str] = ALLOWED_TONES_DEFAULT
ALLOWED_TONES: Set[str] = ALLOWED_TONES_DEFAULT
ALLOWED_PRODUCTS: Set[str] = {"Atlas", "Core", "Desk"}
ALLOWED_PRIORITIES: Set[str] = {"високий", "середній", "низький"}
ALLOWED_CATEGORIES: Set[str] = {"реліз", "інцидент", "рекомендація"}


def check_task2(result: str) -> Dict[str, Any]:
    """Повна перевірка для Завдання 2: рівно ключі ('product', 'priority') та обидві множини значень."""
    data = check(
        result,
        keys=("product", "priority"),
        field="priority",
        allowed=ALLOWED_PRIORITIES,
    )
    if data["product"] not in ALLOWED_PRODUCTS:
        raise ValueError(f"недопустиме значення product: {data['product']} (очікувалось з {ALLOWED_PRODUCTS})")
    return data


def check_task3(result: str) -> Dict[str, Any]:
    """Повна перевірка для Завдання 3: ключі ('title', 'category'), категорія та ліміт <= 8 слів."""
    data = check(
        result,
        keys=("title", "category"),
        field="category",
        allowed=ALLOWED_CATEGORIES,
    )
    title_text = str(data.get("title", "")).strip()
    words = [w for w in title_text.split() if w]
    if not (1 <= len(words) <= 8):
        raise ValueError(f"заголовок має містити від 1 до 8 слів (отримано {len(words)}): '{title_text}'")
    return data


class PersonaToneModel(BaseModel):
    person: str = Field(..., min_length=3, description="Точне Ім'я та Прізвище з тексту без посад")
    tone: Literal["позитивна", "нейтральна", "негативна"]

    @field_validator("person")
    @classmethod
    def clean_person_titles(cls, v: str) -> str:
        prefixes = ("Керівник відділу ", "Сьогодні ", "Менеджер ", "Директор ")
        cleaned = v.strip()
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned


class ProductPriorityModel(BaseModel):
    product: Literal["Atlas", "Core", "Desk"]
    priority: Literal["високий", "середній", "низький"]

    @field_validator("product", mode="before")
    @classmethod
    def normalize_product_name(cls, v: Any) -> str:
        raw = str(v).strip()
        for base in ("Atlas", "Core", "Desk"):
            if base.lower() in raw.lower():
                return base
        return raw


class TitleCategoryModel(BaseModel):
    title: str = Field(..., description="Стислий заголовок українською мовою до 8 слів")
    category: Literal["реліз", "інцидент", "рекомендація"]

    @field_validator("title")
    @classmethod
    def validate_word_limit(cls, v: str) -> str:
        cleaned = re.sub(
            r"^(Оголошення|Новина|Увага|Повідомлення)\s*:\s*",
            "",
            v.strip(),
            flags=re.IGNORECASE,
        )
        words = cleaned.split()
        if len(words) > 8:
            cleaned = " ".join(words[:8])
        return cleaned
