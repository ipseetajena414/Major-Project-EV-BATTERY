"""Word-window chunking that keeps page numbers and source metadata on every chunk."""
from __future__ import annotations
from .document_loader import LoadedDoc


def chunk_document(doc: LoadedDoc, size_words: int = 180, overlap_words: int = 40) -> list[dict]:
    if overlap_words >= size_words:
        raise ValueError("overlap must be smaller than chunk size")
    chunks = []
    step = size_words - overlap_words
    for page, text in doc.pages:
        words = text.split()
        for start in range(0, max(len(words), 1), step):
            piece = words[start:start + size_words]
            if len(piece) < 15 and start > 0:
                break  # drop tiny tails
            if piece:
                chunks.append({"text": " ".join(piece), "page": page, "source": doc.meta.to_dict()})
            if start + size_words >= len(words):
                break
    return chunks
