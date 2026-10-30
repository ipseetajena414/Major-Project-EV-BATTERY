"""Retrieval-first QA. The retrieved context and any generated interpretation are returned separately.

Without GOOGLE_API_KEY the answer is retrieval-only (source excerpts); nothing is 'generated'.
Optional Gemini interpretation is only attempted when a key is configured, is instructed to use only the
supplied excerpts, and is labelled as model-generated. (This optional path is not exercised in the tests.)
"""
from __future__ import annotations
import os
from .retriever import retrieve
from .vector_store import VectorStore

EMPTY_MSG = ("The literature corpus is empty, so there is nothing to retrieve. Upload permitted research PDFs "
             "(open-access papers or files you are allowed to use) on this page to build the local index.")
NO_HIT_MSG = "No indexed passage matched this question. Try different keywords or add more sources."


def answer(store: VectorStore, question: str, k: int = 5) -> dict:
    if not store.chunks:
        return {"status": "empty_corpus", "message": EMPTY_MSG, "passages": [], "interpretation": None}
    hits = retrieve(store, question, k)
    if not hits:
        return {"status": "no_match", "message": NO_HIT_MSG, "passages": [], "interpretation": None}
    passages = [{"rank": i + 1, "score": round(h["score"], 3), "page": h["page"], "excerpt": h["text"],
                 "source": h["source"]} for i, h in enumerate(hits)]
    return {"status": "retrieved", "message": "Retrieval-only result: excerpts below are verbatim from indexed sources.",
            "passages": passages, "interpretation": _interpret(question, passages)}


def _interpret(question: str, passages: list[dict]) -> str | None:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        ctx = "\n\n".join(f"[{p['rank']}] {p['excerpt']}" for p in passages)
        llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), google_api_key=key)
        res = llm.invoke("Answer ONLY from the numbered excerpts; cite them as [n]; say so if they are insufficient.\n\n"
                         f"Excerpts:\n{ctx}\n\nQuestion: {question}")
        return str(getattr(res, "content", res))
    except Exception as exc:
        return f"(LLM interpretation unavailable: {type(exc).__name__}: {exc})"
