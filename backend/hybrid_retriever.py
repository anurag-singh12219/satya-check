from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

try:
    from realtime_evidence import get_domain_weight
except ModuleNotFoundError:
    from .realtime_evidence import get_domain_weight

try:
    import numpy as np
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "and",
    "or",
    "that",
    "this",
    "it",
    "as",
    "all",
    "new",
    "now",
    "breaking",
    "government",
    "india",
}

DOMAIN_ANCHORS = {
    "upi",
    "tax",
    "income",
    "repo",
    "rbi",
    "gst",
    "aadhaar",
    "kisan",
    "covid",
    "vaccine",
    "whatsapp",
    "chandrayaan",
    "moon",
    "t20",
    "world",
    "cup",
    "petrol",
    "diesel",
    "gold",
}


def _normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s₹.%/-\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_terms(text: str) -> set[str]:
    return {t for t in _normalize_text(text).split() if len(t) > 1 and t not in STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


def _safe_year(value: str) -> int | None:
    if not value:
        return None
    m = re.search(r"(19\d{2}|20\d{2})", value)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


@dataclass
class _FactFeatures:
    fact: dict[str, Any]
    terms: set[str]
    year: int | None
    trust: float
    semantic_embedding: Any | None


class HybridFactRetriever:
    """Metadata-aware hybrid retriever for fact records.

    Signals:
    - Lexical similarity (Jaccard over normalized terms)
    - Optional semantic similarity (SentenceTransformer cosine)
    - Freshness score from year metadata
    - Trust score from source/source_url domain
    """

    def __init__(
        self,
        facts: list[dict[str, Any]],
        enable_semantic: bool = False,
        semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.facts = facts
        self.enable_semantic = bool(enable_semantic)
        self.semantic_model_name = semantic_model_name

        self._model = None
        self._years: list[int] = []
        self._features: list[_FactFeatures] = []

        if self.enable_semantic and SentenceTransformer is not None and np is not None:
            try:
                self._model = SentenceTransformer(self.semantic_model_name)
            except Exception:
                self._model = None

        for fact in self.facts:
            source_url = str(fact.get("source_url", "")).strip()
            source = str(fact.get("source", "")).strip().lower()
            domain = urlparse(source_url).netloc.lower() if source_url else source
            trust = get_domain_weight(domain)

            text = f"{fact.get('topic', '')} {fact.get('claim', '')} {' '.join(fact.get('tags', []))}"
            terms = _to_terms(text)

            year = _safe_year(str(fact.get("date", ""))) or _safe_year(str(fact.get("claim", "")))
            if year is not None:
                self._years.append(year)

            embedding = None
            if self._model is not None:
                try:
                    embedding = self._model.encode(text)
                except Exception:
                    embedding = None

            self._features.append(
                _FactFeatures(
                    fact=fact,
                    terms=terms,
                    year=year,
                    trust=trust,
                    semantic_embedding=embedding,
                )
            )

        self._min_year = min(self._years) if self._years else None
        self._max_year = max(self._years) if self._years else None

    def _freshness(self, year: int | None) -> float:
        if year is None:
            return 0.45
        if self._min_year is None or self._max_year is None or self._min_year == self._max_year:
            return 0.65
        return (year - self._min_year) / max(1, self._max_year - self._min_year)

    def _semantic_similarity(self, query_embedding: Any, fact_embedding: Any) -> float:
        if np is None or query_embedding is None or fact_embedding is None:
            return 0.0
        try:
            a = np.array(query_embedding)
            b = np.array(fact_embedding)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            sim = float(np.dot(a, b) / denom)
            return max(0.0, (sim + 1.0) / 2.0)
        except Exception:
            return 0.0

    def retrieve(self, query: str, top_k: int = 50, include_inactive: bool = False) -> list[dict[str, Any]]:
        query_terms = _to_terms(query)

        query_embedding = None
        if self._model is not None:
            try:
                query_embedding = self._model.encode(query)
            except Exception:
                query_embedding = None

        scored: list[tuple[float, str, dict[str, Any]]] = []

        for f in self._features:
            status = str(f.fact.get("status", "active")).lower()
            if not include_inactive and status != "active":
                continue

            lexical = _jaccard(query_terms, f.terms)
            semantic = self._semantic_similarity(query_embedding, f.semantic_embedding)
            freshness = self._freshness(f.year)
            trust = max(0.0, min(1.0, f.trust))

            overlap = query_terms & f.terms
            anchor_overlap = overlap & DOMAIN_ANCHORS

            # Require at least lexical or semantic grounding before metadata boosts.
            if lexical <= 0.0 and semantic < 0.15:
                continue
            if not anchor_overlap and len(overlap) < 2 and semantic < 0.22:
                continue

            if query_embedding is not None and f.semantic_embedding is not None:
                score = (0.55 * lexical) + (0.25 * semantic) + (0.10 * freshness) + (0.10 * trust)
            else:
                score = (0.75 * lexical) + (0.15 * freshness) + (0.10 * trust)

            if score <= 0:
                continue

            date = str(f.fact.get("date", ""))
            scored.append((score, date, f.fact))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if scored:
            return [fact for _, _, fact in scored[:top_k]]

        fallback = [f.fact for f in self._features if include_inactive or str(f.fact.get("status", "active")).lower() == "active"]
        return fallback[:top_k]
