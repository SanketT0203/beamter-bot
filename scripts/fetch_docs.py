"""Download the corpus listed in docs_manifest.csv into ./corpus (stdlib only).

    python scripts/fetch_docs.py                # download everything missing
    python scripts/fetch_docs.py --force        # re-download all
    python scripts/fetch_docs.py --only bamf_bildung_und_beruf
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beamter import corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(ROOT / "docs_manifest.csv"))
    parser.add_argument("--out", default=str(ROOT / "corpus"))
    parser.add_argument("--only", help="download a single manifest id")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    entries = corpus.load_manifest(args.manifest)
    if args.only:
        entries = [e for e in entries if e.id == args.only]
        if not entries:
            print(f"no manifest entry with id {args.only!r}")
            return 2
    results = corpus.fetch_all(entries, args.out, force=args.force, delay=args.delay)
    corpus.write_sources(results, Path(args.out) / "SOURCES.md")
    print("\n" + corpus.summarize(results))
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
