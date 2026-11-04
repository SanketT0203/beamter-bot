"""Corpus manifest + downloader (stdlib only).

The PDFs themselves are NOT committed: copyright and reuse terms differ per publisher.
What is committed is ``docs_manifest.csv`` (what we use, from whom, where). This module
reads the manifest, downloads each file once, checks it really is a PDF, and writes a
``SOURCES.md`` with provenance and retrieval dates.

It runs on a person's machine (or CI) as a one-off job, never inside the deployed API.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("id", "title", "publisher", "lang", "topic", "url", "note")
_ID = re.compile(r"^[a-z0-9_]+$")
USER_AGENT = "Mozilla/5.0 (compatible; beamter-bot-corpus-fetcher/0.3; portfolio project)"


class ManifestError(ValueError):
    pass


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    id: str
    title: str
    publisher: str
    lang: str
    topic: str
    url: str
    note: str


@dataclass(frozen=True)
class Result:
    entry: Entry
    status: str  # "ok" | "skipped" | "failed"
    detail: str = ""
    size: int = 0


def load_manifest(path: str | Path) -> list[Entry]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"manifest is missing columns: {', '.join(missing)}")
        entries, seen = [], set()
        for line, row in enumerate(reader, start=2):
            e = Entry(**{c: (row[c] or "").strip() for c in REQUIRED_COLUMNS})
            if not _ID.match(e.id):
                raise ManifestError(f"line {line}: id {e.id!r} must match [a-z0-9_]+")
            if e.id in seen:
                raise ManifestError(f"line {line}: duplicate id {e.id!r}")
            if not e.url.startswith("https://"):
                raise ManifestError(f"line {line}: url for {e.id!r} must start with https://")
            if not e.title or not e.publisher:
                raise ManifestError(f"line {line}: {e.id!r} needs a title and a publisher")
            seen.add(e.id)
            entries.append(e)
    return entries


def looks_like_pdf(head: bytes) -> bool:
    return head.lstrip()[:5] == b"%PDF-"


def _ssl_context() -> ssl.SSLContext:
    """TLS context that trusts what the operating system trusts.

    Python's own certificate list is smaller than the OS one, and many German government sites
    do not send their intermediate certificate (browsers repair that via the OS; plain Python
    cannot). ``truststore`` hands verification to the OS. Verification is never switched off.
    """
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


def download(url: str, *, timeout: float = 30.0, max_bytes: int = 40 * 1024 * 1024) -> bytes:
    """GET ``url`` and return the body, or raise FetchError with a human-readable reason."""
    if not url.startswith(("http://", "https://")):
        raise FetchError("only http(s) URLs are allowed")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
            body = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise FetchError(
                "TLS certificate could not be verified (is `truststore` installed? pip install truststore)"
            ) from exc
        raise FetchError(f"network error: {exc}") from exc
    if len(body) > max_bytes:
        raise FetchError(f"larger than {max_bytes // (1024 * 1024)} MB")
    if not looks_like_pdf(body[:1024]):
        raise FetchError("response is not a PDF (probably an HTML page)")
    return body


def fetch_all(
    entries: list[Entry],
    out_dir: str | Path,
    *,
    force: bool = False,
    delay: float = 1.0,
    max_bytes: int = 40 * 1024 * 1024,
    sleep: Callable[[float], None] = time.sleep,
    fetcher: Callable[..., bytes] = download,
) -> list[Result]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    for i, entry in enumerate(entries, start=1):
        target = out / f"{entry.id}.pdf"
        if target.exists() and not force:
            results.append(Result(entry, "skipped", "already downloaded", target.stat().st_size))
            continue
        try:
            body = fetcher(entry.url, max_bytes=max_bytes)
        except FetchError as exc:
            log.warning("[%d/%d] %s failed: %s", i, len(entries), entry.id, exc)
            results.append(Result(entry, "failed", str(exc)))
        else:
            tmp = target.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.replace(target)  # atomic: never leaves a half-written PDF behind
            log.info("[%d/%d] %s ok (%d KB)", i, len(entries), entry.id, len(body) // 1024)
            results.append(Result(entry, "ok", size=len(body)))
        sleep(delay)  # be polite to the publishers' servers
    return results


def write_sources(results: list[Result], path: str | Path, today: dt.date | None = None) -> None:
    """Provenance list for the files that are actually present. Skipped by the loader."""
    today = today or dt.date.today()
    lines = [
        "# Sources",
        "",
        "Documents used in this corpus. Files are not redistributed; fetch them with the manifest.",
        "Check each publisher's terms of use before reusing them anywhere else.",
        "",
        "| File | Title | Publisher | Lang | Retrieved | URL |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.status == "failed":
            continue
        e = r.entry
        title = e.title.replace("|", "/")
        lines.append(f"| {e.id}.pdf | {title} | {e.publisher} | {e.lang} | {today.isoformat()} | {e.url} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(results: list[Result]) -> str:
    counts = {s: sum(1 for r in results if r.status == s) for s in ("ok", "skipped", "failed")}
    text = f"{counts['ok']} downloaded, {counts['skipped']} already present, {counts['failed']} failed"
    failed = [r for r in results if r.status == "failed"]
    if failed:
        text += "\n\nFailed:\n" + "\n".join(f"  - {r.entry.id}: {r.detail}" for r in failed)
    return text
