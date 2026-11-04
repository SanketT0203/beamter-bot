"""HTTP API for the Beamter-Bot.

    uvicorn beamter.api:create_app --factory --host 0.0.0.0 --port 8080

Endpoints
    GET  /                the chat page (static, no data; talks to the endpoints below)
    GET  /healthz        liveness: the process is up (never touches Qdrant or models)
    GET  /readyz         readiness: models and index loaded, 503 until then
    GET  /sources        documents in the index                      (needs API key)
    POST /ask            question -> cited answer as JSON            (needs API key)
    POST /ask/stream     same, streamed as server-sent events        (needs API key)

The server starts listening at once and loads models in the background (see runtime.py), so
a platform health check does not time out while the models load.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings
from .generation import Answer, Delta, GenerationError
from .runtime import Runtime

log = logging.getLogger("beamter.api")


# ── request and response shapes ──────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Message] = Field(default_factory=list, max_length=20)
    source: str | None = Field(default=None, max_length=200, description="Limit the search to one document")


class CitationOut(BaseModel):
    n: int
    source: str
    page: int
    section: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    covered: bool = Field(description="False when the documents do not answer the question")
    grounded: bool = Field(description="False when the answer carries no valid citation")
    reason: str
    search_query: str
    interpreted_question: str = Field(default="", description="The question as the bot understood it")
    top_score: float | None = Field(default=None, description="Rerank score of the best passage")
    timings_ms: dict[str, float]


def to_response(a: Answer) -> AskResponse:
    return AskResponse(
        answer=a.text,
        citations=[CitationOut(**vars(c)) for c in a.citations],
        covered=a.covered,
        grounded=a.grounded,
        reason=a.reason,
        search_query=a.search_query,
        interpreted_question=a.interpreted,
        top_score=a.top_score,
        timings_ms=a.timings,
    )


# ── the chat page ────────────────────────────────────────────────────────

PAGE = Path(__file__).parent / "static" / "index.html"


def load_page() -> tuple[str, dict[str, str]]:
    """The page and its security headers. The Content-Security-Policy allows only the page's own
    inline script and style (by hash) and requests to this origin, so text that ends up in the
    page cannot load or run anything else."""
    html = PAGE.read_text(encoding="utf-8")

    def digest(kind: str) -> str:
        blocks = re.findall(rf"<{kind}[^>]*>(.*?)</{kind}>", html, flags=re.S)
        return " ".join(
            "'sha256-" + base64.b64encode(hashlib.sha256(b.encode("utf-8")).digest()).decode() + "'" for b in blocks
        )

    csp = (
        f"default-src 'none'; script-src {digest('script')}; style-src {digest('style')}; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    return html, {
        "Content-Security-Policy": csp, "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer", "Cache-Control": "no-cache",
    }


def client_ip(request: Request) -> str:
    """The caller's address for rate limiting. Behind Cloud Run the proxy *appends* the real
    address to ``X-Forwarded-For``, so the last entry is trustworthy; earlier entries are
    whatever the client wrote and could be faked to dodge the limit."""
    forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    if forwarded:
        return forwarded[-1]
    return request.client.host if request.client else "unknown"


def setup_logging() -> None:
    """Send our log lines (one per request, plus errors) to stdout, where Cloud Logging collects
    them. Only does anything when no logging has been configured yet."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("beamter").setLevel(logging.INFO)


# ── rate limiting ────────────────────────────────────────────────────────

class RateLimiter:
    """Sliding one-minute window per client, kept in memory. With several Cloud Run instances
    each one counts separately, so this is a cost brake, not an exact quota."""

    def __init__(self, per_minute: int, clock=time.monotonic):
        self.per_minute, self.clock = per_minute, clock
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, client: str) -> float:
        """0 if the request may proceed, otherwise the seconds to wait."""
        if self.per_minute <= 0:
            return 0.0
        now = self.clock()
        with self.lock:
            q = self.hits[client]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(1.0, 60 - (now - q[0]))
            q.append(now)
            if len(self.hits) > 10_000:  # forget idle clients
                for key in [k for k, v in self.hits.items() if not v or now - v[-1] >= 60]:
                    del self.hits[key]
            return 0.0


# ── the app ──────────────────────────────────────────────────────────────

def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    autostart = runtime is None
    if autostart:
        setup_logging()
        settings.validate()  # fail the deploy immediately on bad configuration
    rt = runtime or Runtime(settings)
    limiter = RateLimiter(settings.rate_limit_per_min)
    slots = threading.BoundedSemaphore(settings.max_concurrent_requests)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if autostart:
            if not settings.service_api_key:
                log.warning("SERVICE_API_KEY is not set: the API is open to anyone who can reach it")
            rt.start()
        yield
        rt.stop()

    app = FastAPI(title="Beamter-Bot", version="0.3.0", lifespan=lifespan,
                  description="Multilingual RAG over German official documents, with cited answers.")
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                           allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])

    # ── guards ───────────────────────────────────────────────────────────

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        wanted = settings.service_api_key
        if wanted and not (x_api_key and secrets.compare_digest(x_api_key.encode(), wanted.encode())):
            raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})

    def throttle(request: Request, x_api_key: str | None = Header(default=None)) -> None:
        client = x_api_key or client_ip(request)
        wait = limiter.check(client)
        if wait:
            raise HTTPException(429, "Too many requests", headers={"Retry-After": str(int(wait) + 1)})

    def ready_pipeline():
        if rt.state == "ready" and rt.pipeline is not None:
            return rt.pipeline
        detail = "The index is still loading" if rt.state == "loading" else "The service is not ready"
        raise HTTPException(503, detail, headers={"Retry-After": "10"})

    def check_source(source: str | None) -> None:
        if source and source not in rt.sources():
            raise HTTPException(400, f"Unknown source {source!r}; see GET /sources")

    @contextmanager
    def slot():
        if not slots.acquire(timeout=20):
            raise HTTPException(503, "Busy, please retry", headers={"Retry-After": "5"})
        try:
            yield
        finally:
            slots.release()

    def recover_if_index_vanished() -> bool:
        """After a failure: if the collection this instance serves was pruned by a newer ingest,
        load the new version now instead of waiting for the next periodic check."""
        pipeline = rt.pipeline
        if pipeline is not None and not pipeline.retriever.kb.ping():
            return rt.maybe_reload()
        return False

    # ── routes ───────────────────────────────────────────────────────────

    page_html, page_headers = load_page()  # read once; a missing file fails the start

    @app.get("/", include_in_schema=False)
    def index():
        return HTMLResponse(page_html, headers=page_headers)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        body = {"status": rt.state, **rt.info()}
        if rt.state == "failed":
            body["error"] = rt.error
        return JSONResponse(body, status_code=200 if rt.state == "ready" else 503)

    @app.get("/sources", dependencies=[Depends(require_key)])
    def sources():
        ready_pipeline()
        return {"documents": rt.sources()}

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key), Depends(throttle)])
    def ask(req: AskRequest):
        pipeline = ready_pipeline()
        check_source(req.source)
        history = [m.model_dump() for m in req.history]
        started = time.perf_counter()
        with slot():
            try:
                try:
                    answer = pipeline.ask(req.question, history, req.source)
                except GenerationError:
                    raise
                except Exception:
                    if not recover_if_index_vanished():
                        raise
                    answer = pipeline.ask(req.question, history, req.source)
            except GenerationError as exc:
                log.error("LLM failure: %s", exc)
                raise HTTPException(503, "The language model is unavailable, please retry", headers={"Retry-After": "10"})
            except Exception:
                log.exception("Unexpected failure while answering")
                raise HTTPException(500, "Internal error")
        log_request("ask", req, answer, started)
        return to_response(answer)

    @app.post("/ask/stream", dependencies=[Depends(require_key), Depends(throttle)])
    def ask_stream(req: AskRequest):
        pipeline = ready_pipeline()
        check_source(req.source)
        history = [m.model_dump() for m in req.history]

        def events():
            started = time.perf_counter()
            try:
                with slot():
                    for event in pipeline.stream(req.question, history, req.source):
                        if isinstance(event, Delta):
                            yield sse("delta", {"text": event.text})
                        else:
                            log_request("stream", req, event, started)
                            yield sse("final", to_response(event).model_dump())
            except HTTPException as exc:
                yield sse("error", {"detail": exc.detail})
            except GenerationError as exc:
                log.error("LLM failure: %s", exc)
                yield sse("error", {"detail": "The language model is unavailable, please retry"})
            except Exception:
                log.exception("Unexpected failure while streaming")
                recover_if_index_vanished()
                yield sse("error", {"detail": "Internal error"})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def log_request(kind: str, req: AskRequest, a: Answer, started: float) -> None:
    """One line per request. The question text is deliberately not logged (it may be personal)."""
    log.info(
        "%s id=%s q_chars=%d history=%d covered=%s grounded=%s reason=%s cites=%d total_ms=%.0f %s",
        kind, uuid.uuid4().hex[:8], len(req.question), len(req.history), a.covered, a.grounded,
        a.reason or "-", len(a.citations), (time.perf_counter() - started) * 1000, a.timings,
    )
