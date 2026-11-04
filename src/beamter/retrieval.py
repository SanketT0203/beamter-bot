"""Retrieval: dense + BM25 fused with Reciprocal Rank Fusion, then optional cross-encoder rerank.

Four modes so each stage can be measured on its own (see evaluation.py):

    dense          vector search only
    hybrid         dense + BM25 fused with RRF
    dense_rerank   dense candidates re-scored by the cross-encoder
    hybrid_rerank  hybrid candidates re-scored by the cross-encoder  (default)

``Retriever.retrieve_traced`` also returns per-stage timings, so latency can be reported and
tuned from numbers instead of guesses.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .index import KnowledgeBase
from .ingest import Chunk

MODES = ("dense", "hybrid", "dense_rerank", "hybrid_rerank")
RRF_K = 60  # constant from the original RRF paper (Cormack et al., 2009)


@dataclass
class Hit:
    chunk: Chunk
    score: float  # the score the final ordering is based on
    dense_rank: int | None = None  # 1-based rank in the dense list, None if absent
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


def rrf_fuse(rank_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Combines ranked lists by position, not raw score, because a
    BM25 score and a cosine similarity are on incomparable scales. Ties break on id so the
    result is deterministic."""
    scores: dict[int, float] = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Reranker(Protocol):
    def score(self, question: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Lazy wrapper so the model only loads when a rerank mode is used. Loading is thread-safe
    because the API serves requests from a thread pool."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, question: str, texts: list[str]) -> list[float]:
        model = self._model or self._load()
        return [float(s) for s in model.predict([(question, t) for t in texts])]


class Retriever:
    def __init__(self, kb: KnowledgeBase, settings: Settings, reranker: Reranker | None = None):
        self.kb = kb
        self.s = settings
        self.reranker = reranker or CrossEncoderReranker(settings.rerank_model)

    def warm_up(self) -> None:
        """Load the models now, so the first real question does not pay the loading time."""
        self.kb.embedder.encode_query("warm up")
        if self.s.retrieval_mode.endswith("rerank"):
            self.reranker.score("warm up", ["warm up"])

    def retrieve(
        self, question: str, mode: str | None = None, k: int | None = None, source: str | None = None
    ) -> list[Hit]:
        return self.retrieve_traced(question, mode, k, source)[0]

    def retrieve_traced(
        self, question: str, mode: str | None = None, k: int | None = None, source: str | None = None
    ) -> tuple[list[Hit], dict[str, float]]:
        """Like ``retrieve`` but also returns milliseconds spent per stage."""
        mode = mode or self.s.retrieval_mode
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; choose from {MODES}")
        k = k or self.s.top_k
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        dense_ids = self.kb.dense_search(question, self.s.dense_pool, source)
        timings["dense_ms"] = _ms(t0)
        dense_rank = {i: r for r, i in enumerate(dense_ids, start=1)}
        sparse_rank: dict[int, int] = {}
        fused: dict[int, float] = {}

        if mode.startswith("hybrid"):
            t0 = time.perf_counter()
            sparse_ids = self.kb.sparse_search(question, self.s.sparse_pool, source)
            timings["sparse_ms"] = _ms(t0)
            sparse_rank = {i: r for r, i in enumerate(sparse_ids, start=1)}
            fused_list = rrf_fuse([dense_ids, sparse_ids])
            fused = dict(fused_list)
            candidates = [doc_id for doc_id, _ in fused_list]
        else:
            candidates = list(dense_ids)

        hits = [
            Hit(
                chunk=self.kb.by_id[i],
                score=fused.get(i, 1.0 / dense_rank[i] if i in dense_rank else 0.0),
                dense_rank=dense_rank.get(i),
                sparse_rank=sparse_rank.get(i),
                fused_score=fused.get(i),
            )
            for i in candidates
        ]

        if mode.endswith("rerank") and hits:
            t0 = time.perf_counter()
            pool = hits[: self.s.rerank_pool]
            scores = self.reranker.score(question, [h.chunk.text for h in pool])
            for h, sc in zip(pool, scores):
                h.rerank_score = sc
                h.score = sc
            hits = sorted(pool, key=lambda h: -h.score)
            timings["rerank_ms"] = _ms(t0)
        timings["total_ms"] = round(sum(timings.values()), 1)
        return hits[:k], timings


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
