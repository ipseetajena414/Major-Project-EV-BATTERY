"""Persistent local vector store (TF-IDF matrix + chunk records saved with joblib)."""
from __future__ import annotations
from pathlib import Path
import joblib

from .embeddings import make_vectorizer
from m5integration.paths import RAG_INDEX_DIR


class VectorStore:
    def __init__(self, index_dir: Path | str = RAG_INDEX_DIR):
        self.dir = Path(index_dir)
        self.chunks: list[dict] = []
        self.vectorizer = None
        self.matrix = None
        self._load()

    @property
    def _file(self) -> Path:
        return self.dir / "index.joblib"

    def _load(self):
        if self._file.exists():
            data = joblib.load(self._file)
            self.chunks = data["chunks"]
            self._fit()

    def _fit(self):
        if self.chunks:
            self.vectorizer = make_vectorizer()
            self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
        else:
            self.vectorizer, self.matrix = None, None

    def add(self, chunks: list[dict]):
        self.chunks.extend(chunks)
        self._fit()
        self.dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"chunks": self.chunks}, self._file)

    def remove_source(self, title: str):
        self.chunks = [c for c in self.chunks if c["source"]["title"] != title]
        self._fit()
        self.dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"chunks": self.chunks}, self._file)

    def sources(self) -> list[dict]:
        seen = {}
        for c in self.chunks:
            seen.setdefault(c["source"]["title"], dict(c["source"], chunks=0))["chunks"] += 1
        return list(seen.values())

    def search(self, query: str, k: int = 5):
        if not self.chunks or self.vectorizer is None:
            return []
        from sklearn.metrics.pairwise import linear_kernel
        sims = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [dict(self.chunks[i], score=float(sims[i])) for i in order if sims[i] > 0]
