from __future__ import annotations
from .vector_store import VectorStore


def retrieve(store: VectorStore, query: str, k: int = 5, min_score: float = 0.02) -> list[dict]:
    return [h for h in store.search(query, k) if h["score"] >= min_score]
