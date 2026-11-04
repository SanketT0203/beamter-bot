"""Ask questions from the terminal, with the same pipeline the API will use.

    python -m beamter.ask "Wie hoch ist das Kindergeld?"      one question
    python -m beamter.ask                                      interactive chat with memory
    python -m beamter.ask --debug --no-rewrite "..."           show search query and timings
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from .config import ConfigError, Settings
from .generation import Answer, Delta, GenerationError
from .pipeline import Pipeline, build_pipeline


def format_sources(answer: Answer) -> str:
    if not answer.citations:
        return ""
    lines = ["", "Quellen / Sources:"]
    for c in answer.citations:
        where = f"{c.source}, S. {c.page}" + (f" ({c.section})" if c.section else "")
        lines.append(f"  [{c.n}] {where}")
    return "\n".join(lines)


def format_debug(answer: Answer) -> str:
    t = ", ".join(f"{k.removesuffix('_ms')} {v:.0f}" for k, v in answer.timings.items())
    flags = f"covered={answer.covered} grounded={answer.grounded}" + (f" reason={answer.reason}" if answer.reason else "")
    return f"\n[search query: {answer.search_query!r}] [{flags}] [ms: {t}]"


def answer_question(pipeline: Pipeline, question: str, history: list[dict], debug: bool, out=sys.stdout) -> Answer:
    final = None
    for event in pipeline.stream(question, history):
        if isinstance(event, Delta):
            print(event.text, end="", flush=True, file=out)
        else:
            final = event
    print(format_sources(final), file=out)
    if debug:
        print(format_debug(final), file=out)
    return final


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ask the Beamter-Bot from the terminal.")
    ap.add_argument("question", nargs="*", help="Leave empty for an interactive chat")
    ap.add_argument("--debug", action="store_true", help="Show the search query, decision and timings")
    ap.add_argument("--no-rewrite", action="store_true", help="Search with the question as typed")
    ap.add_argument("--mode", default=None, help="Retrieval mode (default from config)")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):  # answers contain umlauts; Windows' codepage rejects them
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    settings = Settings.from_env()
    if args.no_rewrite:
        settings = dataclasses.replace(settings, rewrite_queries=False)
    if args.mode:
        settings = dataclasses.replace(settings, retrieval_mode=args.mode)
    try:
        pipeline = build_pipeline(settings)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    history: list[dict] = []
    questions = [" ".join(args.question)] if args.question else None
    try:
        while True:
            if questions is not None:
                if not questions:
                    return 0
                question = questions.pop(0)
            else:
                question = input("\nFrage / Question (Ctrl+C to quit): ").strip()
                if not question:
                    continue
            try:
                answer = answer_question(pipeline, question, history, args.debug)
            except GenerationError as exc:
                print(f"\nThe language model is not reachable right now ({exc}).", file=sys.stderr)
                return 3
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer.text}]
    except (KeyboardInterrupt, EOFError):
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
