"""Golden question set: questions with verified evidence, independent of how documents are chunked.

Evidence is a verbatim span of the source page text, not a chunk id. A retrieved chunk counts as
relevant when it contains an acceptable span, so the labels stay valid when chunk size or the
chunking strategy changes. Several spans may be listed when the same text appears in more than
one document.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import Page

TYPES = ("factoid_de", "english", "cross_lingual", "keyword", "out_of_scope")
_WS = re.compile(r"\s+")
_LINE_HYPHEN = re.compile(r"(?<=\w)- (?=[a-zäöüß])")


def norm(text: str) -> str:
    """Normalise for matching: drop soft hyphens, unify whitespace, case-fold, and join words
    that PDF extraction split across lines ("verlaen- gern")."""
    text = text.replace("\u00ad ", "").replace("\u00ad", "")
    return _LINE_HYPHEN.sub("", _WS.sub(" ", text).strip().casefold())


@dataclass(frozen=True)
class Evidence:
    source: str
    span: str

    def matches(self, source: str, text: str) -> bool:
        return source == self.source and norm(self.span) in norm(text)


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    lang: str
    question: str
    answerable: bool
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)

    def is_relevant(self, source: str, text: str) -> bool:
        return any(e.matches(source, text) for e in self.evidence)


def load_golden(path: str | Path) -> list[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[Question] = []
    for raw in data["questions"]:
        ev = tuple(Evidence(e["source"], e["span"]) for e in raw.get("evidence", []))
        q = Question(raw["id"], raw["type"], raw["lang"], raw["question"], bool(raw["answerable"]), ev)
        if q.type not in TYPES:
            raise ValueError(f"{q.id}: unknown type {q.type!r}")
        if q.answerable != bool(ev):
            raise ValueError(f"{q.id}: answerable questions need evidence, out-of-scope ones must have none")
        out.append(q)
    ids = [q.id for q in out]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids in golden set")
    return out


def validate_against_pages(questions: list[Question], pages: list[Page]) -> list[str]:
    """Return problems: evidence spans that do not occur in the corpus, and out-of-scope
    questions whose key words do occur (a hint that they may not be out of scope)."""
    by_source: dict[str, list[str]] = {}
    for p in pages:
        by_source.setdefault(p.source, []).append(norm(p.text))
    problems = []
    for q in questions:
        for e in q.evidence:
            if not any(norm(e.span) in t for t in by_source.get(e.source, [])):
                problems.append(f"{q.id}: span not found in {e.source}: {e.span!r}")
    return problems
