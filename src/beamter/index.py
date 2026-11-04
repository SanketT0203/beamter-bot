"""Vector index in Qdrant Cloud, in two halves that never run in the same process.

WRITE side (offline ingest job): ``build_index``
    chunks -> embeddings -> a NEW versioned collection ``<name>__<fingerprint>`` ->
    switch the alias ``<name>`` to it -> prune old versions.
    The API never sees a half-written index, because the alias only moves after the
    new collection is complete. Rolling back is pointing the alias at the previous one.

Incremental: chunks whose text and embedding model are unchanged reuse the vector stored in the
previous version (matched by content hash), so updating the corpus embeds only what changed.

READ side (the API): ``KnowledgeBase``
    resolve the alias, scroll the chunk payloads back out of Qdrant, rebuild the
    small BM25 index in memory, and answer dense/sparse searches. No PDFs, no local files.

The fingerprint (documents + chunking settings + embedding model) is part of the
collection name, so "is this index up to date?" is just "does that collection exist?".
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from .config import Settings
from .ingest import CHUNKER_VERSION, Chunk, Page, chunk_pages, fingerprint

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1  # bump when the payload layout or the way embed_text is built changes

_STOPWORDS = set(
    """der die das den dem des und oder ist sind war ein eine einer eines einem einen zu zur zum für fuer mit von
    im in am an auf aus bei nach über ueber unter nicht auch wie was wo wer wann ich sie es wird werden kann
    können koennen muss müssen muessen hat haben sich als dass the a an and or is are was to of for with on at
    what how do does i my you your need can must it be by from that this""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1 and t not in _STOPWORDS]


# ── embeddings ───────────────────────────────────────────────────────────

class TextEmbedder(Protocol):
    """What the rest of the code needs from an embedding model. Swappable (local model, hosted API, test fake)."""

    model_name: str
    dim: int

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers model, loaded lazily. Adds the query/passage prefixes E5 models need."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def _e5(self) -> bool:
        return "e5" in self.model_name.lower()

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode([f"query: {text}" if self._e5 else text], normalize_embeddings=True)[0]


# ── client + naming ──────────────────────────────────────────────────────

def make_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=30)


def index_params(settings: Settings, embed_model: str | None = None) -> dict:
    """Everything besides the documents that changes what the index contains."""
    return {
        "schema": SCHEMA_VERSION,
        "chunker": CHUNKER_VERSION,
        "embed_model": embed_model or settings.embed_model,
        "chunk_chars": settings.chunk_chars,
        "min_chunk_chars": settings.min_chunk_chars,
        "overlap_sentences": settings.overlap_sentences,
    }


def embed_key(embedder: TextEmbedder) -> str:
    """Vectors may only be reused between points that carry the same key (same model, same layout)."""
    return hashlib.sha1(f"{embedder.model_name}|schema{SCHEMA_VERSION}".encode("utf-8")).hexdigest()[:12]


def versioned_name(alias: str, fp: str) -> str:
    return f"{alias}__{fp[:12]}"


def alias_target(client: QdrantClient, alias: str) -> str | None:
    for a in client.get_aliases().aliases:
        if a.alias_name == alias:
            return a.collection_name
    return None


# ── write side ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BuildReport:
    collection: str
    chunks: int
    documents: int
    action: str  # "up_to_date" | "switched" | "built"
    embedded: int  # chunks that went through the embedding model in this run
    reused: int  # chunks whose vectors were copied from the previous index version
    pruned: list[str]


def build_index(
    settings: Settings,
    pages: list[Page],
    embedder: TextEmbedder,
    client: QdrantClient,
    *,
    force: bool = False,
    full: bool = False,
    keep_previous: int = 1,
    batch_size: int = 64,
) -> BuildReport:
    """Make the alias point at an index that matches ``pages`` and the settings.

    Incremental: a new version is still built as a fresh collection (so the live one is never
    touched), but any chunk whose text and embedding model are unchanged gets its vector copied
    from the previous version instead of being embedded again.

    ``force`` rebuilds even when nothing changed; ``full`` additionally ignores previous vectors
    and re-embeds everything.
    """
    force = force or full
    chunks = chunk_pages(pages, settings.chunk_chars, settings.overlap_sentences, settings.min_chunk_chars)
    if not chunks:
        raise ValueError("No indexable text found. Fetch the corpus first (scripts/fetch_docs.py).")
    alias = settings.collection
    fp = fingerprint(pages, index_params(settings, embedder.model_name))
    target = versioned_name(alias, fp)
    if force and client.collection_exists(target):
        target = f"{target}_{int(time.time())}"  # never overwrite a collection the alias may be serving
    previous = alias_target(client, alias)
    documents = len({c.source for c in chunks})

    if not force and previous == target and _is_complete(client, target, len(chunks)):
        log.info("Index %s is up to date (%d chunks). Nothing to do.", target, len(chunks))
        return BuildReport(target, len(chunks), documents, "up_to_date", 0, len(chunks), [])

    if not force and _is_complete(client, target, len(chunks)):
        log.info("Collection %s is already complete; only switching the alias.", target)
        action, embedded, reused = "switched", 0, len(chunks)
    else:
        key = embed_key(embedder)
        reusable = {} if (full or previous is None) else _load_reusable(client, previous, key)
        embedded, reused = _write_collection(client, target, chunks, embedder, key, reusable, batch_size)
        action = "built"

    _switch_alias(client, alias, target, previous)
    pruned = _prune(client, alias, keep={target, previous} if keep_previous else {target})
    return BuildReport(target, len(chunks), documents, action, embedded, reused, pruned)


def _is_complete(client: QdrantClient, name: str, expected: int) -> bool:
    return client.collection_exists(name) and client.count(name, exact=True).count == expected


def _load_reusable(client: QdrantClient, collection: str, key: str) -> dict[str, list[float]]:
    """content_hash -> vector for every point of ``collection`` that was embedded with the same model."""
    if not client.collection_exists(collection):
        return {}
    found: dict[str, list[float]] = {}
    offset = None
    while True:
        records, offset = client.scroll(
            collection, limit=256, offset=offset, with_payload=["content_hash", "embed_key"], with_vectors=True
        )
        for r in records:
            payload = r.payload or {}
            if payload.get("embed_key") == key and payload.get("content_hash") and r.vector is not None:
                found[payload["content_hash"]] = list(r.vector)
        if offset is None:
            return found


def _write_collection(
    client: QdrantClient,
    name: str,
    chunks: list[Chunk],
    embedder: TextEmbedder,
    key: str,
    reusable: dict[str, list[float]],
    batch_size: int,
) -> tuple[int, int]:
    """Create ``name`` and fill it. Returns (embedded, reused) chunk counts."""
    if client.collection_exists(name):
        client.delete_collection(name)  # an earlier run died half-way: start this version clean
    client.create_collection(name, vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE))
    client.create_payload_index(name, field_name="source", field_schema=PayloadSchemaType.KEYWORD)

    def point(chunk: Chunk, vector: list[float]) -> PointStruct:
        payload = {**chunk.to_payload(), "content_hash": chunk.content_hash, "embed_key": key}
        return PointStruct(id=chunk.id, vector=vector, payload=payload)

    old = [c for c in chunks if c.content_hash in reusable]
    new = [c for c in chunks if c.content_hash not in reusable]
    log.info("%d chunks: reusing %d stored vectors, embedding %d new ones into %s", len(chunks), len(old), len(new), name)

    for start in range(0, len(old), batch_size):
        batch = old[start : start + batch_size]
        client.upsert(name, points=[point(c, reusable[c.content_hash]) for c in batch], wait=True)
    for start in range(0, len(new), batch_size):
        batch = new[start : start + batch_size]
        vectors = embedder.encode_passages([c.embed_text for c in batch])
        client.upsert(name, points=[point(c, v.tolist()) for c, v in zip(batch, vectors)], wait=True)
        log.info("  embedded %d / %d", min(start + batch_size, len(new)), len(new))
    return len(new), len(old)


def _switch_alias(client: QdrantClient, alias: str, target: str, previous: str | None) -> None:
    """Move the alias in one atomic request, so readers see the old or the new index, never neither."""
    ops = []
    if previous is not None:
        ops.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias)))
    ops.append(CreateAliasOperation(create_alias=CreateAlias(collection_name=target, alias_name=alias)))
    client.update_collection_aliases(change_aliases_operations=ops)


def _prune(client: QdrantClient, alias: str, keep: set[str | None]) -> list[str]:
    prefix = f"{alias}__"
    old = [c.name for c in client.get_collections().collections if c.name.startswith(prefix) and c.name not in keep]
    for name in old:
        client.delete_collection(name)
        log.info("Pruned old index version %s", name)
    return old


# ── read side ────────────────────────────────────────────────────────────

class IndexNotReady(RuntimeError):
    """The alias does not exist yet: the ingest job has not run."""


class IndexMismatch(RuntimeError):
    """The index was built with a different embedding model than this process would query with."""


class KnowledgeBase:
    def __init__(self, settings: Settings, embedder: TextEmbedder, client: QdrantClient):
        self.s = settings
        self.embedder = embedder
        self.client = client
        self.collection: str | None = None
        self.chunks: list[Chunk] = []
        self.by_id: dict[int, Chunk] = {}
        self._bm25 = None

    def load(self) -> dict:
        """Pull all chunks out of Qdrant and build BM25. Called once at API startup."""
        from rank_bm25 import BM25Okapi

        target = alias_target(self.client, self.s.collection)
        if target is None:
            raise IndexNotReady(f"No index named {self.s.collection!r} in Qdrant. Run the ingest job first.")
        chunks: list[Chunk] = []
        built_with: set[str] = set()
        offset = None
        while True:
            records, offset = self.client.scroll(
                target, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            chunks.extend(Chunk.from_payload(r.id, r.payload) for r in records)
            built_with.update(r.payload.get("embed_key", "") for r in records)
            if offset is None:
                break
        if not chunks:
            raise IndexNotReady(f"Index {target!r} is empty.")
        expected = embed_key(self.embedder)
        if built_with != {expected}:
            # Vectors from different models are not comparable: searching would silently return garbage.
            raise IndexMismatch(
                f"Index {target!r} was built with a different embedding model than "
                f"{self.embedder.model_name!r}. Re-run the ingest job with --full, or fix BEAMTER_EMBED_MODEL."
            )
        chunks.sort(key=lambda c: (c.source, c.page, c.ordinal))  # scroll order is arbitrary; keep it stable
        self.collection, self.chunks = target, chunks
        self.by_id = {c.id: c for c in chunks}
        self._bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])
        log.info("Loaded %d chunks from %s", len(chunks), target)
        return {"collection": target, "chunks": len(chunks), "documents": len(self.sources())}

    def ping(self) -> bool:
        """Cheap liveness check of the Qdrant connection for the readiness endpoint."""
        try:
            return self.collection is not None and self.client.collection_exists(self.collection)
        except Exception:  # noqa: BLE001 - any failure means "not ready"
            return False

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    def dense_search(self, question: str, limit: int, source: str | None = None) -> list[int]:
        flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]) if source else None
        points = self.client.query_points(
            self.collection, query=self.embedder.encode_query(question).tolist(), query_filter=flt, limit=limit
        ).points
        return [int(p.id) for p in points]

    def sparse_search(self, question: str, limit: int, source: str | None = None) -> list[int]:
        scores = self._bm25.get_scores(tokenize(question))
        order = np.argsort(-scores, kind="stable")
        ids = []
        for idx in order:
            if scores[idx] <= 0:
                break  # no query term matched: not a real candidate
            chunk = self.chunks[idx]
            if source and chunk.source != source:
                continue
            ids.append(chunk.id)
            if len(ids) == limit:
                break
        return ids
