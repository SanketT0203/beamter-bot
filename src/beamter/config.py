"""Central configuration for the cloud version of Beamter-Bot.

Every value comes from an environment variable: locally from a ``.env`` file, in the cloud
from the platform's secret store. Nothing is read from, or written to, local state.

Two rules shape this file:

* Secrets never appear in ``repr()``, so a stray ``log.info(settings)`` cannot leak them.
* ``validate()`` runs once at startup and reports EVERY problem together, so a bad deploy
  fails immediately with a clear message instead of failing on the first user request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:  # a local .env is a development convenience; in the cloud it simply doesn't exist
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # never overrides variables that are already set
except ImportError:  # pragma: no cover
    pass

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

RETRIEVAL_MODES = ("dense", "hybrid", "dense_rerank", "hybrid_rerank")


class ConfigError(RuntimeError):
    """Required configuration is missing or malformed."""


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    # ── secrets: repr=False keeps them out of logs and tracebacks ────────
    qdrant_api_key: str = field(repr=False)
    anthropic_api_key: str = field(repr=False)
    service_api_key: str = field(repr=False)  # clients must send it; empty = auth off (dev only)

    # ── vector store (Qdrant Cloud) ──────────────────────────────────────
    qdrant_url: str
    collection: str

    # ── documents: only the offline ingest command reads these ──────────
    docs_dir: Path

    # ── models ───────────────────────────────────────────────────────────
    embed_model: str
    rerank_model: str

    # ── chunking ─────────────────────────────────────────────────────────
    chunk_chars: int  # target maximum characters per chunk
    min_chunk_chars: int  # a section change starts a new chunk only past this size
    overlap_sentences: int

    # ── retrieval ────────────────────────────────────────────────────────
    dense_pool: int  # candidates from the vector search
    sparse_pool: int  # candidates from BM25
    rerank_pool: int  # candidates the cross-encoder re-scores
    top_k: int  # passages that reach the LLM
    retrieval_mode: str
    min_rerank_score: float  # below this best score the bot answers "not covered" without calling the LLM (clearly unrelated input only; the model abstains in the grey zone)

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_model: str
    max_answer_tokens: int
    history_turns: int
    llm_timeout_s: int
    llm_max_retries: int
    rewrite_queries: bool  # let the LLM restate the question as a German search query first

    # ── API ──────────────────────────────────────────────────────────────
    max_concurrent_requests: int  # questions processed at once per instance (protects RAM and CPU)
    rate_limit_per_min: int  # per client and instance; 0 = unlimited
    reload_check_s: int  # how often to look for a new index version; 0 = never
    cors_origins: str  # comma-separated browser origins allowed to call the API; empty = none

    @classmethod
    def from_env(cls) -> "Settings":
        """Read the environment. Does not check required values: call validate() for that."""
        return cls(
            qdrant_api_key=_str("QDRANT_API_KEY"),
            anthropic_api_key=_str("ANTHROPIC_API_KEY"),
            service_api_key=_str("SERVICE_API_KEY"),
            qdrant_url=_str("QDRANT_URL"),
            collection=_str("BEAMTER_COLLECTION", "beamter_docs"),
            docs_dir=Path(_str("BEAMTER_DOCS_DIR", str(ROOT / "corpus"))),
            embed_model=_str("BEAMTER_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            rerank_model=_str("BEAMTER_RERANK_MODEL", DEFAULT_RERANK_MODEL),
            chunk_chars=_int("BEAMTER_CHUNK_CHARS", 700),
            min_chunk_chars=_int("BEAMTER_MIN_CHUNK_CHARS", 250),
            overlap_sentences=_int("BEAMTER_OVERLAP_SENTENCES", 1),
            dense_pool=_int("BEAMTER_DENSE_POOL", 40),
            sparse_pool=_int("BEAMTER_SPARSE_POOL", 20),
            rerank_pool=_int("BEAMTER_RERANK_POOL", 30),
            top_k=_int("BEAMTER_TOP_K", 4),
            retrieval_mode=_str("BEAMTER_RETRIEVAL_MODE", "hybrid_rerank"),
            min_rerank_score=_float("BEAMTER_MIN_RERANK_SCORE", -2.5),
            llm_model=_str("BEAMTER_LLM_MODEL", "claude-haiku-4-5"),
            max_answer_tokens=_int("BEAMTER_MAX_ANSWER_TOKENS", 700),
            history_turns=_int("BEAMTER_HISTORY_TURNS", 3),
            llm_timeout_s=_int("BEAMTER_LLM_TIMEOUT_S", 30),
            llm_max_retries=_int("BEAMTER_LLM_MAX_RETRIES", 2),
            rewrite_queries=_int("BEAMTER_REWRITE_QUERIES", 1) != 0,
            max_concurrent_requests=_int("BEAMTER_MAX_CONCURRENCY", 4),
            rate_limit_per_min=_int("BEAMTER_RATE_LIMIT_PER_MIN", 30),
            reload_check_s=_int("BEAMTER_RELOAD_CHECK_S", 60),
            cors_origins=_str("BEAMTER_CORS_ORIGINS"),
        )

    def validate(self, require_llm: bool = True) -> None:
        """Raise ConfigError listing every problem at once.

        The offline ingest job passes ``require_llm=False``: it never calls Claude.
        """
        problems: list[str] = []
        if not self.qdrant_url.startswith(("http://", "https://")):
            problems.append("QDRANT_URL must be the cluster URL from the Qdrant Cloud dashboard (https://...)")
        elif self.qdrant_url.startswith("https://") and not self.qdrant_api_key:
            problems.append("QDRANT_API_KEY is required when QDRANT_URL is an https Qdrant Cloud URL")
        if require_llm and not self.anthropic_api_key:
            problems.append("ANTHROPIC_API_KEY is not set")
        if self.retrieval_mode not in RETRIEVAL_MODES:
            problems.append(f"BEAMTER_RETRIEVAL_MODE must be one of {RETRIEVAL_MODES}, got {self.retrieval_mode!r}")
        if self.chunk_chars < 100:
            problems.append("BEAMTER_CHUNK_CHARS must be at least 100")
        if self.min_chunk_chars >= self.chunk_chars:
            problems.append("BEAMTER_MIN_CHUNK_CHARS must be smaller than BEAMTER_CHUNK_CHARS")
        if self.rerank_pool < self.top_k:
            problems.append("BEAMTER_RERANK_POOL must be at least BEAMTER_TOP_K")
        if min(self.dense_pool, self.sparse_pool, self.top_k, self.history_turns) < 1:
            problems.append("pool sizes, TOP_K and HISTORY_TURNS must each be at least 1")
        if self.max_concurrent_requests < 1 or self.rate_limit_per_min < 0 or self.reload_check_s < 0:
            problems.append("MAX_CONCURRENCY must be >= 1; RATE_LIMIT_PER_MIN and RELOAD_CHECK_S must be >= 0")
        if self.llm_timeout_s < 1 or self.llm_max_retries < 0:
            problems.append("LLM timeout must be >= 1 second and retries >= 0")
        if problems:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))
