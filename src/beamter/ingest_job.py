"""Offline ingest job: documents -> chunks -> embeddings -> Qdrant Cloud.

    python -m beamter.ingest_job --dry-run     # parse + chunk only: no models, no network
    python -m beamter.ingest_job               # build (or reuse) the index and switch the alias
    python -m beamter.ingest_job --force       # rebuild even if nothing changed (still reuses vectors)
    python -m beamter.ingest_job --full        # re-embed everything from scratch

The same command runs on a laptop, in Cloud Shell, or as a Cloud Run Job.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from .config import ConfigError, Settings
from .index import SentenceTransformerEmbedder, build_index, index_params, make_client
from .ingest import chunk_pages, fingerprint, load_pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m beamter.ingest_job", description=__doc__)
    parser.add_argument("--docs", help="folder with the source documents (default: BEAMTER_DOCS_DIR or ./corpus)")
    parser.add_argument("--dry-run", action="store_true", help="only parse and chunk; print statistics")
    parser.add_argument("--force", action="store_true", help="rebuild even when the index is up to date")
    parser.add_argument("--full", action="store_true", help="ignore stored vectors and re-embed every chunk")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font warnings drown the useful output

    settings = Settings.from_env()
    if args.docs:
        settings = dataclasses.replace(settings, docs_dir=Path(args.docs))

    pages = load_pages(settings.docs_dir)
    chunks = chunk_pages(pages, settings.chunk_chars, settings.overlap_sentences, settings.min_chunk_chars)
    print(f"{len({p.source for p in pages})} documents, {len(pages)} pages, {len(chunks)} chunks "
          f"from {settings.docs_dir}")
    if args.dry_run:
        print("fingerprint:", fingerprint(pages, index_params(settings))[:12])
        return 0 if chunks else 1

    try:
        settings.validate(require_llm=False)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    client = make_client(settings)
    try:  # fail fast, before a 1 GB model download, if Qdrant cannot be reached or the key is wrong
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot reach Qdrant at {settings.qdrant_url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    report = build_index(settings, pages, SentenceTransformerEmbedder(settings.embed_model), client, force=args.force, full=args.full)
    print(f"{report.action}: collection {report.collection} ({report.chunks} chunks, "
          f"{report.documents} documents); alias {settings.collection!r} now points to it")
    print(f"embedded {report.embedded} chunks, reused {report.reused} stored vectors")
    if report.pruned:
        print("pruned old versions:", ", ".join(report.pruned))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
