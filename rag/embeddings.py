"""Local, offline sparse embeddings (TF-IDF). No network and no API key are needed.

This is lexical retrieval, not neural semantic search; docs/RAG_SOURCES.md states this limitation.
"""
from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
