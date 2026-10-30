"""Load permitted documents (PDF / txt / md) into page-level text with user-supplied metadata.

Bibliographic metadata is NEVER guessed: fields the user did not provide stay empty and are shown as such.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO


@dataclass
class SourceMeta:
    title: str
    authors: str = ""
    year: str = ""
    identifier: str = ""   # DOI or arXiv id, as supplied
    url: str = ""
    access: str = "full_text"  # full_text | abstract_only (only what was actually loaded)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class LoadedDoc:
    meta: SourceMeta
    pages: list[tuple[int, str]] = field(default_factory=list)  # (page_number, text)


def load_pdf(stream: IO[bytes] | str | Path, meta: SourceMeta) -> LoadedDoc:
    from pypdf import PdfReader
    reader = PdfReader(stream)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return LoadedDoc(meta, pages)


def load_text(text: str, meta: SourceMeta) -> LoadedDoc:
    return LoadedDoc(meta, [(1, text)] if text.strip() else [])


def load_path(path: str | Path, meta: SourceMeta) -> LoadedDoc:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return load_pdf(p, meta)
    return load_text(p.read_text(errors="ignore"), meta)
