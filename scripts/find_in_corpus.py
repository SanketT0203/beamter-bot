"""Search the downloaded corpus with regular expressions (case-insensitive). Dev tool for writing eval questions.

    python scripts/find_in_corpus.py "Mietkaution" "drei Monatsmieten" --context 160 --max 5
"""

import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beamter.ingest import clean_text, load_pages  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--docs", default=str(ROOT / "corpus"))
    parser.add_argument("--context", type=int, default=140, help="characters shown around each match")
    parser.add_argument("--max", type=int, default=4, help="matches shown per pattern")
    parser.add_argument("--source", help="only search documents whose name contains this text")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    pages = [(p.source, p.page, " ".join(clean_text(p.text).split())) for p in load_pages(args.docs)]
    for pattern in args.patterns:
        rx = re.compile(pattern, re.IGNORECASE)
        shown = total = 0
        print(f"\n### {pattern}")
        for source, page, text in pages:
            if args.source and args.source not in source:
                continue
            for m in rx.finditer(text):
                total += 1
                if shown < args.max:
                    a, b = max(0, m.start() - args.context), min(len(text), m.end() + args.context)
                    print(f"- {source} p.{page}: ...{text[a:b]}...")
                    shown += 1
        print(f"  ({total} matches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
