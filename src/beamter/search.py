"""Try retrieval from the terminal, against the real index in Qdrant Cloud.

    python -m beamter.search "Wie lange habe ich Zeit für die Anmeldung?"
    python -m beamter.search "blue card salary threshold" --mode hybrid --k 5
    python -m beamter.search "Rundfunkbeitrag Befreiung" --compare       # all four modes side by side
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import ConfigError, Settings
from .index import IndexNotReady, KnowledgeBase, SentenceTransformerEmbedder, make_client
from .retrieval import MODES, Hit, Retriever


def format_hits(hits: list[Hit], snippet_chars: int = 160) -> str:
    lines = []
    for n, h in enumerate(hits, start=1):
        c = h.chunk
        ranks = f"dense#{h.dense_rank or '-'} bm25#{h.sparse_rank or '-'}"
        rerank = f" rerank={h.rerank_score:.2f}" if h.rerank_score is not None else ""
        where = f"{c.source} p.{c.page}" + (f" | {c.section}" if c.section else "")
        snippet = " ".join(c.text.split())[:snippet_chars]
        lines.append(f"{n}. {where}\n   {ranks}{rerank}  score={h.score:.4f}\n   {snippet}...")
    return "\n".join(lines) if lines else "(no results)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m beamter.search", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, help="retrieval mode (default: BEAMTER_RETRIEVAL_MODE)")
    parser.add_argument("--k", type=int, help="number of results (default: BEAMTER_TOP_K)")
    parser.add_argument("--source", help="restrict to one document, e.g. anmeldung_merkblatt.pdf")
    parser.add_argument("--compare", action="store_true", help="run all four modes and print each")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    settings = Settings.from_env()
    try:
        settings.validate(require_llm=False)
        kb = KnowledgeBase(settings, SentenceTransformerEmbedder(settings.embed_model), make_client(settings))
        info = kb.load()
    except (ConfigError, IndexNotReady) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"index {info['collection']}: {info['chunks']} chunks, {info['documents']} documents\n")

    retriever = Retriever(kb, settings)
    modes = MODES if args.compare else [args.mode or settings.retrieval_mode]
    kb.embedder.encode_query("warm up")  # load models first, so the timings below exclude loading
    if any(m.endswith("rerank") for m in modes):
        retriever.reranker.score("warm up", ["warm up"])
    for mode in modes:
        hits, timings = retriever.retrieve_traced(args.question, mode, args.k, args.source)
        print(f"== {mode}  ({', '.join(f'{k}={v}' for k, v in timings.items())})")
        print(format_hits(hits) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
