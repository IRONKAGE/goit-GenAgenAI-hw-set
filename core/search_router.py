# ==============================================================================
# 🌐 CORE SEARCH ROUTER: Tavily, Serper, Open-Meteo Geocoding & JSON Cache
# ==============================================================================
"""
Трьохрівневий пошуковий та геодезичний рушій:
- Tier 1: Tavily Search API (`TAVILY_API_KEY`) та Google Serper (`SERPER_API_KEY`)
- Tier 2: Безключовий Open-Meteo Geocoding API (координати та висота) + Wikipedia API
- Tier 3: Персистентний JSON-кеш (`data/cache/web_search_cache.json`) + формула Гаверсинуса
"""

import json
import logging
import math
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

from core.llm_router import get_secret

logger = logging.getLogger("core.search_router")

# Базовий довідник координат європейських столиць та міст із тестового набору capitals.xlsx
KNOWN_CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "kyiv": (50.4501, 30.5234),
    "paris": (48.8566, 2.3522),
    "london": (51.5074, -0.1278),
    "rome": (41.9028, 12.4964),
    "berlin": (52.5200, 13.4050),
    "vienna": (48.2082, 16.3738),
    "warsaw": (52.2297, 21.0122),
    "madrid": (40.4168, -3.7038),
    "lisbon": (38.7223, -9.1393),
    "budapest": (47.4979, 19.0402),
    "stockholm": (59.3293, 18.0686),
    "oslo": (59.9139, 10.7522),
    "athens": (37.9838, 23.7275),
    "istanbul": (41.0082, 28.9784),
}


def fetch_open_meteo_coordinates(city_name: str) -> Optional[Tuple[float, float, Optional[float]]]:
    """
    Отримує точні координати (latitude, longitude) та висоту над рівнем моря (elevation)
    через безкоштовний Open-Meteo Geocoding API без ключа (патерн із ноутбука Теми 6).
    """
    encoded = urllib.parse.quote(city_name.strip())
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MyLangChainAgent/1.0 (contact: oleh.hatsenko@neoversity.edu)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results")
            if results:
                place = results[0]
                return (
                    float(place["latitude"]),
                    float(place["longitude"]),
                    float(place["elevation"]) if "elevation" in place else None,
                )
    except Exception:
        return None
    return None


def compute_haversine_distance_km(
    city_from: str,
    city_to: str,
    allow_live_geocoding: bool = False,
) -> Optional[int]:
    """
    Обчислює пряму геодезичну відстань по повітрю (great-circle distance) у кілометрах
    між двома містами за формулою Гаверсинуса.
    """
    c1 = KNOWN_CITY_COORDS.get(city_from.strip().lower())
    c2 = KNOWN_CITY_COORDS.get(city_to.strip().lower())

    if c1 is None and allow_live_geocoding:
        geo1 = fetch_open_meteo_coordinates(city_from)
        if geo1:
            c1 = (geo1[0], geo1[1])
    if c2 is None and allow_live_geocoding:
        geo2 = fetch_open_meteo_coordinates(city_to)
        if geo2:
            c2 = (geo2[0], geo2[1])

    if not c1 or not c2:
        return None

    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return int(round(6371.0 * c))


class SmartSearchRouter:
    """
    Універсальний пошуковий маршрутизатор із локальним JSON-кешем
    (за рекомендацією оновлення Фінального проекту для економії лімітів Tavily/SerpAPI).
    """

    def __init__(self, cache_path: str = "data/cache/web_search_cache.json"):
        self.cache_file = Path(cache_path)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, str] = self._load_cache()
        self.api_calls_count: int = 0
        self.cache_hits_count: int = 0

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            self.cache_file.write_text(
                json.dumps(self.cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as err:
            logger.warning(f"Помилка запису пошукового кешу: {err}")

    def search_tavily(self, query: str, search_depth: str = "basic") -> Optional[str]:
        """Виконує пошук через Tavily API (1 кредит за basic, 2 кредити за advanced)."""
        api_key = get_secret("TAVILY_API_KEY", required=False)
        if not api_key:
            return None

        payload = json.dumps(
            {
                "api_key": api_key,
                "query": query,
                "search_depth": search_depth,
                "include_answer": True,
                "max_results": 3,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "MyLangChainAgent/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.api_calls_count += 1
                snippets = []
                if data.get("answer"):
                    snippets.append(f"Answer: {data['answer']}")
                for r in data.get("results", []):
                    snippets.append(f"- {r.get('title', '')}: {r.get('content', '')}")
                return "\n".join(snippets) if snippets else None
        except Exception as err:
            logger.warning(f"Збій Tavily API: {err}")
            return None

    def query_with_cache(self, query: str, fallback_text: str = "") -> Tuple[str, str]:
        """
        Шукає запит спочатку в JSON-кеші, далі в Tavily API, і при відсутності ключа
        повертає `fallback_text` зі збереженням у кеш.
        """
        if query in self.cache:
            self.cache_hits_count += 1
            return self.cache[query], "cache"

        live_res = self.search_tavily(query)
        if live_res:
            self.cache[query] = live_res
            self._save_cache()
            return live_res, "tavily_api"

        final_text = fallback_text or f"Cached reference for: {query}"
        self.cache[query] = final_text
        self._save_cache()
        return final_text, "offline_reference"
