"""Retrieval evaluation against the golden set.

    python -m beamter.evaluation                       # all four modes, current chunk size
    python -m beamter.evaluation --chunk-sizes 500 700 1000 --modes hybrid_rerank
    python -m beamter.evaluation --details --modes dense

Runs on an in-memory Qdrant, so nothing is written to the cloud cluster. A chunk is relevant when
it contains an evidence span of the question (see golden.py).

Metrics (answerable questions only): hit@k = share with a relevant chunk in the top k, MRR =
mean of 1/rank of the first relevant chunk, recall = share with a relevant chunk anywhere in the
candidate list (up to 30; rerank modes only return their rerank pool), indexed = share whose
evidence exists in some chunk at all (a ceiling set by chunking). For out-of-scope questions, ``calibrate_threshold`` looks at
the top rerank score: answerable questions should score above it, out-of-scope ones below.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .golden import Question, load_golden, validate_against_pages
from .ingest import Chunk, load_pages
from .pipeline import search
from .retrieval import MODES, Hit, Retriever

KS = (1, 3, 5)
CANDIDATES = 30  # how many results to inspect per question (rerank modes return at most the rerank pool)


@dataclass
class QResult:
    question: Question
    rank: int | None  # 1-based rank of the first relevant chunk within the retrieved list
    top_score: float | None  # score of the best hit (rerank score in rerank modes)
    indexed: bool  # some chunk in the index holds the evidence
    top: tuple = ()  # (source, page, section, snippet) of the best hits, for the miss report
    ms: float = 0.0  # retrieval time for this question


def haystack(chunk: Chunk) -> str:
    """Text a span is looked up in. The section heading is part of it because the chunker moves
    heading lines out of ``text`` into ``section``."""
    return f"{chunk.section} {chunk.text}"


def first_relevant_rank(question: Question, hits: list[Hit]) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if question.is_relevant(h.chunk.source, haystack(h.chunk)):
            return rank
    return None


def evidence_indexed(question: Question, chunks: list[Chunk]) -> bool:
    return any(question.is_relevant(c.source, haystack(c)) for c in chunks)


def run(
    retriever: Retriever,
    questions: list[Question],
    mode: str,
    k: int = CANDIDATES,
    queries: dict[str, str] | None = None,
    merge_original: bool = True,
) -> list[QResult]:
    """``queries`` maps question id -> rewritten search query. With ``merge_original`` the
    original question is searched too and the results are merged (what the pipeline does)."""
    chunks = list(retriever.kb.by_id.values())
    out = []
    for q in questions:
        if queries and q.id in queries:
            hits, timings = search(
                retriever, q.question, queries[q.id], mode=mode, k=k, merge_original=merge_original
            )
        else:
            hits, timings = retriever.retrieve_traced(q.question, mode=mode, k=k)
        out.append(
            QResult(
                q,
                first_relevant_rank(q, hits) if q.answerable else None,
                hits[0].score if hits else None,
                evidence_indexed(q, chunks) if q.answerable else False,
                tuple(
                    (h.chunk.source, h.chunk.page, h.chunk.section, " ".join(h.chunk.text.split())[:170])
                    for h in hits[:3]
                ),
                timings.get("total_ms", 0.0),
            )
        )
    return out


def summarize(results: list[QResult]) -> dict:
    """Overall and per-type metrics for the answerable questions."""

    def block(rs: list[QResult]) -> dict:
        n = len(rs)
        if not n:
            return {"n": 0}
        m = {
            "n": n,
            "mrr": sum(1 / r.rank for r in rs if r.rank) / n,
            "recall": sum(1 for r in rs if r.rank) / n,  # evidence anywhere in the candidate list
            "indexed": sum(r.indexed for r in rs) / n,
        }
        for k in KS:
            m[f"hit@{k}"] = sum(1 for r in rs if r.rank and r.rank <= k) / n
        return m

    ans = [r for r in results if r.question.answerable]
    by_type: dict[str, list[QResult]] = defaultdict(list)
    for r in ans:
        by_type[r.question.type].append(r)
    return {"overall": block(ans), "by_type": {t: block(rs) for t, rs in sorted(by_type.items())}}


def calibrate_threshold(results: list[QResult]) -> dict:
    """Best score threshold separating answerable (should be above) from out-of-scope questions.
    Only meaningful for rerank modes, where the score is comparable across questions."""
    pos = [r.top_score for r in results if r.question.answerable and r.top_score is not None]
    neg = [r.top_score for r in results if not r.question.answerable and r.top_score is not None]
    if not pos or not neg:
        return {}
    best = None
    for t in sorted(set(pos + neg)):
        correct = sum(s >= t for s in pos) + sum(s < t for s in neg)
        acc = correct / (len(pos) + len(neg))
        if best is None or acc > best["accuracy"]:
            best = {"threshold": t, "accuracy": acc}
    best["answerable_min"] = min(pos)
    best["out_of_scope_max"] = max(neg)
    return best


def format_summary(label: str, summary: dict) -> str:
    cols = ["n", "hit@1", "hit@3", "hit@5", "mrr", "recall", "indexed"]
    lines = [f"{label}", "  " + f"{'':<15}" + "".join(f"{c:>8}" for c in cols)]
    rows = [("ALL", summary["overall"])] + list(summary["by_type"].items())
    for name, m in rows:
        cells = [str(m["n"])] + [f"{m[c]:.2f}" for c in cols[1:]] if m["n"] else ["0"]
        lines.append("  " + f"{name:<15}" + "".join(f"{c:>8}" for c in cells))
    return "\n".join(lines)


def format_misses(results: list[QResult], k: int = 3, show_hits: bool = False) -> str:
    lines = []
    for r in results:
        if r.question.answerable and not (r.rank and r.rank <= k):
            note = f"rank {r.rank}" if r.rank else "not in candidates" if r.indexed else "EVIDENCE NOT IN ANY CHUNK"
            lines.append(f"  [{r.question.type}] {r.question.id}: {note}  -- {r.question.question}")
            if show_hits:
                want = "; ".join(f"{e.source}: {e.span!r}" for e in r.question.evidence)
                lines.append(f"      wanted: {want}")
                for i, (src, page, section, snippet) in enumerate(r.top, start=1):
                    lines.append(f"      got {i}: {src} p.{page} [{section}] {snippet}")
    return "\n".join(lines) or "  (none)"


class CachedEmbedder:
    """Remembers passage embeddings on disk, keyed by model and text, so repeating an experiment
    only embeds chunks it has not seen. Queries are cheap and are not cached."""

    def __init__(self, inner, path: str | Path):
        self.inner = inner
        self.path = Path(path)
        self.cache: dict[str, object] = {}
        if self.path.exists():
            with self.path.open("rb") as f:
                self.cache = pickle.load(f)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.inner.model_name}\n{text}".encode("utf-8")).hexdigest()

    def encode_passages(self, texts, batch_size: int = 32):
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        if missing:
            print(f"    embedding {len(missing)} new passages ({len(texts) - len(missing)} cached) ...", flush=True)
            vectors = self.inner.encode_passages([texts[i] for i in missing], batch_size)
            for i, v in zip(missing, vectors):
                self.cache[keys[i]] = v
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb") as f:
                pickle.dump(self.cache, f)
        return np.stack([self.cache[k] for k in keys])


def rewrite_questions(llm, model: str, questions: list[Question], path: str | Path) -> dict[str, str]:
    """Search query per question id. Rewrites are cached on disk, so repeating an experiment
    costs no API calls."""
    from .generation import REWRITE_SYSTEM, rewrite_query

    tag = hashlib.sha256(REWRITE_SYSTEM.encode()).hexdigest()[:8]  # a changed prompt must not reuse old rewrites
    cache: dict[str, str] = {}
    p = Path(path)
    if p.exists():
        cache = json.loads(p.read_text(encoding="utf-8"))
    changed = False
    out = {}
    for q in questions:
        key = f"{model}|{tag}|{q.question}"
        if key not in cache:
            cache[key] = rewrite_query(llm, q.question)
            changed = True
        out[q.id] = cache[key]
    if changed:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def build_memory_index(settings: Settings, pages, embedder):
    from qdrant_client import QdrantClient

    from .index import KnowledgeBase, build_index

    client = QdrantClient(":memory:")
    build_index(settings, pages, embedder, client)
    kb = KnowledgeBase(settings, embedder, client)
    kb.load()
    return kb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate retrieval against the golden question set.")
    ap.add_argument("--golden", default="data/golden.json")
    ap.add_argument("--docs", default=None, help="Corpus folder (default: BEAMTER_DOCS_DIR / corpus)")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=None)
    ap.add_argument("--rerank-pool", type=int, default=None)
    ap.add_argument("--rerank-model", default=None, help="Cross-encoder to use instead of the configured one")
    ap.add_argument("--dense-pool", type=int, default=None, help="Candidates taken from vector search")
    ap.add_argument("--sparse-pool", type=int, default=None, help="Candidates taken from BM25")
    ap.add_argument("--cache", default=".cache/eval_embeddings.pkl", help="Embedding cache file ('' disables)")
    ap.add_argument(
        "--rewrite", choices=("off", "only", "merge"), default="off",
        help="LLM query rewriting: off, only (search the rewrite), merge (rewrite + original, as the pipeline does)",
    )
    ap.add_argument("--details", action="store_true", help="List questions without a relevant chunk in the top 3")
    ap.add_argument("--show-misses", action="store_true", help="With the miss list, print wanted evidence and the top 3 hits")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):  # PDFs contain characters Windows' default codepage rejects
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    base = Settings.from_env()
    pages = load_pages(args.docs or base.docs_dir)
    questions = load_golden(args.golden)
    problems = validate_against_pages(questions, pages)
    if problems:
        print("Golden set does not match the corpus:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2

    from .index import SentenceTransformerEmbedder
    from .retrieval import CrossEncoderReranker

    embedder = SentenceTransformerEmbedder(base.embed_model)
    if args.cache:
        embedder = CachedEmbedder(embedder, args.cache)
    reranker = CrossEncoderReranker(args.rerank_model or base.rerank_model)
    print(f"reranker: {reranker.model_name}", flush=True)
    queries = None
    if args.rewrite != "off":
        from .generation import AnthropicLLM

        base.validate()
        queries = rewrite_questions(AnthropicLLM(base), base.llm_model, questions, ".cache/rewrites.json")
        for q in questions:
            if queries[q.id] != q.question:
                print(f"  rewrite {q.id}: {q.question!r} -> {queries[q.id]!r}")
    for size in args.chunk_sizes or [base.chunk_chars]:
        s = dataclasses.replace(
            base,
            chunk_chars=size,
            min_chunk_chars=int(size * base.min_chunk_chars / base.chunk_chars),
            rerank_pool=args.rerank_pool or base.rerank_pool,
            dense_pool=args.dense_pool or base.dense_pool,
            sparse_pool=args.sparse_pool or base.sparse_pool,
        )
        print(f"\n=== chunk size {size}: building in-memory index ...", flush=True)
        kb = build_memory_index(s, pages, embedder)
        retriever = Retriever(kb, s, reranker)
        print(f"    {len(kb.by_id)} chunks")
        for mode in args.modes:
            results = run(retriever, questions, mode, queries=queries, merge_original=args.rewrite == "merge")
            print(f"    mean retrieval time {sum(r.ms for r in results) / len(results):.0f} ms/question")
            print(format_summary(f"\n--- chunk {size} / {mode} / rewrite {args.rewrite} / dense {s.dense_pool} sparse {s.sparse_pool} rerank {s.rerank_pool}", summarize(results)))
            cal = calibrate_threshold(results) if mode.endswith("rerank") else {}
            if cal:
                print(
                    f"  threshold {cal['threshold']:.2f} separates answerable/out-of-scope at {cal['accuracy']:.0%} "
                    f"(lowest answerable top score {cal['answerable_min']:.2f}, highest out-of-scope {cal['out_of_scope_max']:.2f})"
                )
            if args.details or args.show_misses:
                print(format_misses(results, show_hits=args.show_misses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
