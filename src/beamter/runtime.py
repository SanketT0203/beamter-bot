"""Keeps the pipeline loaded and up to date while the API runs.

Why this exists: loading takes a minute (models, index), so the web server must start
listening first and load in the background, and report "not ready" until it is done. The
ingest job also publishes new index versions while the API is running; ``Runtime`` notices
that the alias moved, loads the new version next to the old one and swaps it in with a single
assignment, so requests never see a half-loaded index.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .config import Settings
from .index import KnowledgeBase, alias_target
from .pipeline import Pipeline, build_pipeline
from .retrieval import Retriever

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, settings: Settings, build: Callable[[Settings], Pipeline] = build_pipeline):
        self.settings = settings
        self._build = build
        self.pipeline: Pipeline | None = None
        self.state = "loading"  # loading | ready | failed
        self.error = ""
        self.loaded_at = 0.0
        self._lock = threading.Lock()  # one load or reload at a time
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ── loading ──────────────────────────────────────────────────────────

    def load(self) -> bool:
        with self._lock:
            try:
                self.pipeline = self._build(self.settings)
            except Exception as exc:  # noqa: BLE001 - report any startup failure through /readyz
                self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"
                log.error("Startup load failed: %s", self.error)
                return False
            self.state, self.error, self.loaded_at = "ready", "", time.time()
            log.info("Pipeline ready: %s", self.info())
            return True

    def maybe_reload(self) -> bool:
        """Swap in a newer index version if the alias moved. Keeps the old one on any error."""
        with self._lock:
            if self.pipeline is None:
                return False
            old = self.pipeline.retriever
            try:
                current = alias_target(old.kb.client, old.kb.s.collection)
                if current is None or current == old.kb.collection:
                    return False
                log.info("Index alias moved %s -> %s; loading it", old.kb.collection, current)
                kb = KnowledgeBase(old.kb.s, old.kb.embedder, old.kb.client)
                kb.load()
                self.pipeline.retriever = Retriever(kb, old.s, old.reranker)  # the atomic swap
                self.loaded_at = time.time()
                return True
            except Exception as exc:  # noqa: BLE001
                log.error("Reload failed, still serving %s: %s", old.kb.collection, exc)
                return False

    # ── background work ──────────────────────────────────────────────────

    def start(self) -> None:
        """Load in the background and, if configured, keep watching for new index versions."""
        self._spawn(self._startup)

    def _startup(self) -> None:
        self.load()
        interval = self.settings.reload_check_s
        while interval and not self._stop.wait(interval):
            if self.state != "ready":
                self.load()  # e.g. Qdrant was unreachable or the first ingest had not run yet
            else:
                self.maybe_reload()

    def _spawn(self, target) -> None:
        t = threading.Thread(target=target, daemon=True, name="beamter-runtime")
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()

    # ── status ───────────────────────────────────────────────────────────

    def info(self) -> dict:
        if self.pipeline is None:
            return {}
        kb = self.pipeline.retriever.kb
        return {"collection": kb.collection, "chunks": len(kb.chunks), "documents": len(kb.sources())}

    def sources(self) -> dict[str, int]:
        return self.pipeline.retriever.kb.sources() if self.pipeline else {}
