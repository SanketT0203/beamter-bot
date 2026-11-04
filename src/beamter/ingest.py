"""Document loading and structure-aware chunking.

Differences from the fixed-size character windows in the v1 bot:

* Chunks are packed from whole sentences, so a chunk never starts or ends in the
  middle of a sentence. Overlap is measured in sentences, not characters.
* Headings are detected heuristically (ALL-CAPS lines, numbered lines) and carried
  as ``section`` metadata. A chunk is embedded together with its document title and
  section ("contextual chunk"), which helps a short passage stay findable.
* Chunks never cross a page boundary, so every citation names a real page.
* German abbreviations ("z. B.", "Abs.") and dates ("1. Januar") are protected so
  they do not trigger false sentence breaks.

Cloud version: this module only runs in the offline ingest job. The deployed API never
reads PDFs; it gets its chunks back from Qdrant through ``Chunk.from_payload``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
# Notes that live beside the documents but are not content to answer from.
SKIPPED_NAMES = {"readme.md", "sources.md"}


@dataclass(frozen=True)
class Page:
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: int
    source: str
    page: int
    section: str
    ordinal: int
    text: str

    @property
    def embed_text(self) -> str:
        """Text that gets embedded / indexed: title and section give the chunk context."""
        title = Path(self.source).stem.replace("_", " ")
        header = f"{title} - {self.section}" if self.section else title
        return f"{header}\n{self.text}"

    @property
    def content_hash(self) -> str:
        """Identity of what gets embedded. Same hash + same embedding model = the same vector."""
        return hashlib.sha1(self.embed_text.encode("utf-8")).hexdigest()[:16]

    def to_payload(self) -> dict:
        """What gets stored next to the vector in Qdrant (the id is the point id, not payload)."""
        return {
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "ordinal": self.ordinal,
            "text": self.text,
        }

    @classmethod
    def from_payload(cls, chunk_id: int, payload: dict) -> "Chunk":
        """Inverse of ``to_payload``: the API rebuilds its chunks from Qdrant, not from PDFs."""
        return cls(
            id=int(chunk_id),
            source=payload["source"],
            page=int(payload["page"]),
            section=payload.get("section", ""),
            ordinal=int(payload["ordinal"]),
            text=payload["text"],
        )


# ── loading ──────────────────────────────────────────────────────────────

def _read_pdf(path: Path, name: str) -> list[Page]:
    """All pages of one PDF. Raises on unreadable files; the caller decides what to do."""
    from pypdf import PdfReader

    out: list[Page] = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            log.warning("%s page %d has no extractable text (scanned?). Skipped.", name, number)
            continue
        out.append(Page(name, number, text))
    return out


def load_pages(folder: str | Path) -> list[Page]:
    """Read every supported file below ``folder``, one Page per PDF page.

    ``.txt`` / ``.md`` files are split into pages on form-feed characters (or kept as
    a single page). Pages without extractable text (scanned PDFs) are skipped with a
    warning rather than silently indexed as empty.
    """
    folder = Path(folder)
    pages: list[Page] = []
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and p.name.lower() not in SKIPPED_NAMES)
    for path in files:
        name = path.relative_to(folder).as_posix()
        if path.suffix.lower() == ".pdf":
            try:
                pages.extend(_read_pdf(path, name))
            except Exception as exc:  # one broken file must not stop the whole ingest job
                log.warning("%s could not be read (%s: %s). Skipped.", name, type(exc).__name__, exc)
        else:
            raw = path.read_text(encoding="utf-8")
            for number, part in enumerate(raw.split("\f"), start=1):
                if part.strip():
                    pages.append(Page(name, number, part))
    return pages


# ── cleaning ─────────────────────────────────────────────────────────────

_HYPHEN_BREAK = re.compile(r"(?<=[a-zäöüß])-\n(?=[a-zäöüß])")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub("", text)  # "Melde-\ngesetz" -> "Meldegesetz"
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── headings, blocks, sentences ──────────────────────────────────────────

# "1.2 Title" or "IV. Title" (plain "1." is a list item). Segments are 1-2 digits so amounts such as
# "2.500 Euro) oder" are not mistaken for section numbers.
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(\.\d{1,2}){1,4}\.?|[IVX]+\.)\s+[A-ZÄÖÜ]")
CHUNKER_VERSION = 2  # bump when chunking logic changes, so stored indexes are rebuilt
MIN_CHUNK_CHARS_KEPT = 60  # shorter chunks are merged into their neighbour or dropped
_BULLET = re.compile(r"^([•·▪\-–*]|\d+[.)])\s+")


# Callout labels and units that are printed in capitals but are not section titles.
_NOT_HEADINGS = {
    "hinweis", "beispiel", "tipp", "tipps", "tip", "tips", "note", "example", "wichtig", "achtung",
    "bitte beachten sie", "bitte beachten", "eur", "euro", "zum beispiel",
    "oder", "und", "bzw", "sowie", "bgb", "sgb",
}


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80 or s[-1] in ".,;:!?":
        return False
    if s.lower() in _NOT_HEADINGS:
        return False
    if len(s.split()) > 9:
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3 or s.count(")") > s.count("("):
        return False
    if _NUMBERED_HEADING.match(s):
        return True
    return s.isupper() and len(letters) >= 4


_ABBREVIATIONS = [
    "z. B.", "z.B.", "d. h.", "d.h.", "u. a.", "u.a.", "i. d. R.", "i.d.R.", "z. T.", "e. V.",
    "ca.", "Nr.", "Abs.", "Art.", "bzw.", "evtl.", "ggf.", "vgl.", "inkl.", "max.", "min.",
    "Std.", "Mio.", "Mrd.", "Tel.", "Str.", "gem.", "sog.", "usw.", "etc.", "bspw.", "Dr.", "Prof.",
]
_DOT = "․"  # one-dot leader: stands in for protected dots while splitting
_MONTHS = "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
_DATE = re.compile(rf"(\d{{1,2}})\.(\s+)(?=(?:{_MONTHS})\b)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9„\"(•])")


def split_sentences(text: str) -> list[str]:
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _DOT))
    protected = _DATE.sub(lambda m: f"{m.group(1)}{_DOT}{m.group(2)}", protected)
    parts = _SENT_SPLIT.split(protected)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


def extract_units(
    text: str, carry_section: str = "", ignore: frozenset[str] | set[str] = frozenset()
) -> tuple[list[tuple[str, str]], str]:
    """Turn page text into [(section, sentence), ...] and the section in effect at the end.

    Lines are re-joined into blocks (PDF extraction wraps lines mid-sentence); a heading
    or a bullet starts a new block, and headings update the current section. Lines in
    ``ignore`` (running page headers and footers) are dropped.
    """
    section = carry_section
    units: list[tuple[str, str]] = []
    block: list[str] = []

    def flush() -> None:
        if block:
            for sentence in split_sentences(" ".join(block)):
                units.append((section, sentence))
            block.clear()

    for raw in clean_text(text).split("\n"):
        line = raw.strip()
        if line in ignore:
            continue
        if not line:
            flush()
        elif is_heading(line):
            flush()
            section = line
        elif _BULLET.match(line):
            flush()
            block.append(line)
        else:
            block.append(line)
    flush()
    return units, section


# ── packing ──────────────────────────────────────────────────────────────

def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Split one over-long sentence on word boundaries."""
    pieces, cur = [], ""
    for word in sentence.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            pieces.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def pack_units(
    units: list[tuple[str, str]], max_chars: int, overlap_sentences: int, min_chars: int
) -> list[tuple[str, str]]:
    """Pack (section, sentence) units into [(section, chunk_text), ...].

    A new chunk starts when adding the next sentence would exceed ``max_chars``, or when
    the section changes and the current chunk already has ``min_chars`` characters. The
    last ``overlap_sentences`` sentences carry into the next chunk when the section stays
    the same and they fit.
    """
    out: list[tuple[str, str]] = []
    cur: list[str] = []
    cur_section = ""

    def size() -> int:
        return len(" ".join(cur))

    def emit() -> None:
        if cur:
            out.append((cur_section, " ".join(cur)))

    for section, sentence in units:
        for piece in _hard_split(sentence, max_chars) if len(sentence) > max_chars else [sentence]:
            section_break = bool(cur) and section != cur_section and size() >= min_chars
            overflow = bool(cur) and size() + 1 + len(piece) > max_chars
            if section_break or overflow:
                emit()
                keep = cur[-overlap_sentences:] if (overlap_sentences and not section_break) else []
                cur = list(keep)
                if cur and size() + 1 + len(piece) > max_chars:
                    cur = []
            if not cur:
                cur_section = section
            cur.append(piece)
    emit()
    return out


def _chunk_id(source: str, page: int, ordinal: int) -> int:
    digest = hashlib.sha1(f"{source}|{page}|{ordinal}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16)  # fits an unsigned 64-bit Qdrant point id


def running_lines(pages: list[Page], min_pages: int = 3) -> dict[str, set[str]]:
    """Heading-like lines that repeat on several pages of one document are running headers or
    footers ("IN GERMANY", a chapter tab), not headings. Returns them per source."""
    seen: dict[str, dict[str, set[int]]] = {}
    for page in pages:
        for raw in clean_text(page.text).split("\n"):
            line = raw.strip()
            if line and is_heading(line):
                seen.setdefault(page.source, {}).setdefault(line, set()).add(page.page)
    return {
        src: {line for line, on in lines.items() if len(on) >= min_pages} for src, lines in seen.items()
    }


def _tidy(packed: list[tuple[str, str]], max_chars: int) -> list[tuple[str, str]]:
    """Merge a very short chunk into the one before it (when it still fits), else drop it if it
    is too short to carry meaning (page numbers, index fragments)."""
    out: list[tuple[str, str]] = []
    for section, text in packed:
        if len(text) < MIN_CHUNK_CHARS_KEPT * 2 and out and len(out[-1][1]) + 1 + len(text) <= max_chars:
            out[-1] = (out[-1][0], f"{out[-1][1]} {text}")
        elif len(text) >= MIN_CHUNK_CHARS_KEPT:
            out.append((section, text))
    return out or packed  # a page that is short in total is kept as it is


def chunk_pages(
    pages: list[Page],
    max_chars: int = 700,
    overlap_sentences: int = 1,
    min_chars: int = 250,
    carry_sections: bool = False,
) -> list[Chunk]:
    """Split pages into chunks. A section label only comes from a heading on the same page,
    unless ``carry_sections`` is set: carrying the last heading across pages mislabelled whole
    documents whenever a later heading was not recognised."""
    chunks: list[Chunk] = []
    ignore = running_lines(pages)
    carry: dict[str, str] = {}
    for page in pages:
        units, last = extract_units(
            page.text, carry.get(page.source, "") if carry_sections else "", ignore.get(page.source, set())
        )
        carry[page.source] = last
        packed = _tidy(pack_units(units, max_chars, overlap_sentences, min_chars), max_chars)
        for ordinal, (section, text) in enumerate(packed):
            chunks.append(
                Chunk(_chunk_id(page.source, page.page, ordinal), page.source, page.page, section, ordinal, text)
            )
    return chunks


def fingerprint(pages: list[Page], params: dict) -> str:
    """Hash of everything that determines the index content (documents + settings)."""
    h = hashlib.sha256()
    h.update(json.dumps(params, sort_keys=True).encode("utf-8"))
    for p in pages:
        h.update(f"\n{p.source}|{p.page}\n".encode("utf-8"))
        h.update(p.text.encode("utf-8"))
    return h.hexdigest()
