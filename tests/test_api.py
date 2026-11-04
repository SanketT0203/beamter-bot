import dataclasses
import json

import pytest
from conftest import FakeReranker
from fastapi.testclient import TestClient

from beamter.api import RateLimiter, create_app
from beamter.generation import REWRITE_SYSTEM, GenerationError
from beamter.index import IndexNotReady, KnowledgeBase, build_index
from beamter.ingest import Page
from beamter.pipeline import Pipeline
from beamter.retrieval import Retriever
from beamter.runtime import Runtime

PAGES = [
    Page("anmeldung.txt", 1, "ANMELDUNG\nSie muessen sich innerhalb von 14 Tagen nach dem Einzug anmelden."),
    Page("rundfunk.txt", 1, "RUNDFUNKBEITRAG\nJede Wohnung zahlt einen Rundfunkbeitrag von 18,36 Euro im Monat."),
]
PAGES_V2 = PAGES + [Page("miete.txt", 1, "MIETSICHERHEIT\nDie Mietsicherheit darf hoechstens drei Monatsmieten betragen.")]


class ScriptedLLM:
    def __init__(self, answer="Sie haben 14 Tage Zeit [1].", error=None):
        self.answer, self.error = answer, error

    def complete(self, system, messages, max_tokens):
        if system == REWRITE_SYSTEM:
            return "Anmeldung Frist Einzug"
        if self.error:
            raise self.error
        return self.answer

    def stream(self, system, messages, max_tokens):
        if self.error:
            raise self.error
        yield from ["Sie haben ", "14 Tage Zeit [1]."]


def make_settings(settings, **kw):
    base = dict(collection="api", chunk_chars=300, min_chunk_chars=100, rerank_pool=5, dense_pool=5,
                sparse_pool=5, top_k=2, retrieval_mode="hybrid_rerank", min_rerank_score=-0.8,
                service_api_key="", rate_limit_per_min=0, reload_check_s=0)
    return dataclasses.replace(settings, **{**base, **kw})


def make_pipeline(s, embedder, client, pages=PAGES, llm=None):
    build_index(s, pages, embedder, client)
    kb = KnowledgeBase(s, embedder, client)
    kb.load()
    return Pipeline(s, Retriever(kb, s, FakeReranker()), llm or ScriptedLLM())


def make_client(settings, embedder, client, llm=None, ready=True, **kw):
    s = make_settings(settings, **kw)
    rt = Runtime(s, build=lambda _s: make_pipeline(s, embedder, client, llm=llm))
    if ready:
        assert rt.load()
    return TestClient(create_app(s, rt)), rt, s


# ── health and readiness ─────────────────────────────────────────────────

def test_liveness_answers_even_while_loading(settings, embedder, client):
    http, rt, _ = make_client(settings, embedder, client, ready=False)
    assert http.get("/healthz").json() == {"status": "ok"}
    r = http.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "loading"


def test_ready_reports_the_loaded_index(settings, embedder, client):
    http, rt, _ = make_client(settings, embedder, client)
    body = http.get("/readyz").json()
    assert http.get("/readyz").status_code == 200
    assert body["status"] == "ready" and body["chunks"] >= 2 and body["documents"] == 2


def test_failed_startup_is_visible_and_can_recover(settings, embedder, client):
    s = make_settings(settings)
    attempts = []

    def build(_s):
        attempts.append(1)
        if len(attempts) == 1:
            raise IndexNotReady("No index yet. Run the ingest job first.")
        return make_pipeline(s, embedder, client)

    rt = Runtime(s, build=build)
    http = TestClient(create_app(s, rt))
    assert rt.load() is False
    r = http.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "failed" and "ingest job" in r.json()["error"]
    assert rt.load() is True and http.get("/readyz").status_code == 200


# ── asking ───────────────────────────────────────────────────────────────

def test_ask_returns_a_cited_answer(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client)
    r = http.post("/ask", json={"question": "Wie lange habe ich für die Anmeldung?"})
    assert r.status_code == 200
    body = r.json()
    assert body["covered"] and body["grounded"] and body["answer"].endswith("[1].")
    assert body["citations"][0]["source"] in {"anmeldung.txt", "rundfunk.txt"}
    assert {"rewrite_ms", "llm_ms", "total_ms"} <= set(body["timings_ms"])


def test_ask_while_loading_is_a_retryable_503(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client, ready=False)
    r = http.post("/ask", json={"question": "Anmeldung?"})
    assert r.status_code == 503 and r.headers["retry-after"] == "10"


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "x" * 1001}, {},
    {"question": "ok", "history": [{"role": "system", "content": "evil"}]},
    {"question": "ok", "history": [{"role": "user", "content": "a"}] * 21},
])
def test_invalid_requests_are_rejected_before_any_work(settings, embedder, client, payload):
    http, _, _ = make_client(settings, embedder, client)
    assert http.post("/ask", json=payload).status_code == 422


def test_unknown_source_is_a_400(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client)
    assert http.post("/ask", json={"question": "x", "source": "nope.pdf"}).status_code == 400
    assert http.post("/ask", json={"question": "Beitrag", "source": "rundfunk.txt"}).status_code == 200


def test_sources_lists_documents(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client)
    assert set(http.get("/sources").json()["documents"]) == {"anmeldung.txt", "rundfunk.txt"}


# ── auth and limits ──────────────────────────────────────────────────────

def test_api_key_is_required_when_configured(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client, service_api_key="s3cret")
    body = {"question": "Anmeldung?"}
    assert http.post("/ask", json=body).status_code == 401
    assert http.post("/ask", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert http.post("/ask", json=body, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert http.get("/sources").status_code == 401
    assert http.get("/healthz").status_code == 200 and http.get("/readyz").status_code == 200  # probes stay open


def test_rate_limit_returns_429_with_retry_after(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client, rate_limit_per_min=2)
    for _ in range(2):
        assert http.post("/ask", json={"question": "Anmeldung?"}).status_code == 200
    r = http.post("/ask", json={"question": "Anmeldung?"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_rate_limiter_window_slides():
    now = [0.0]
    lim = RateLimiter(2, clock=lambda: now[0])
    assert lim.check("a") == 0 and lim.check("a") == 0
    assert lim.check("a") > 0 and lim.check("b") == 0  # other clients are unaffected
    now[0] = 61
    assert lim.check("a") == 0
    assert RateLimiter(0).check("a") == 0  # 0 = unlimited


# ── failures ─────────────────────────────────────────────────────────────

def test_llm_outage_is_a_503_not_a_crash(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client, llm=ScriptedLLM(error=GenerationError("down")))
    r = http.post("/ask", json={"question": "Anmeldung?"})
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]


def test_unexpected_errors_do_not_leak_details(settings, embedder, client):
    http, rt, _ = make_client(settings, embedder, client)
    rt.pipeline.retriever.reranker = type("Boom", (), {"score": lambda *a: 1 / 0})()
    r = http.post("/ask", json={"question": "Anmeldung?"})
    assert r.status_code == 500 and r.json() == {"detail": "Internal error"}


# ── streaming ────────────────────────────────────────────────────────────

def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_stream_sends_text_pieces_then_the_checked_answer(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client)
    r = http.post("/ask/stream", json={"question": "Anmeldung?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [n for n, _ in events][-1] == "final" and all(n == "delta" for n, _ in events[:-1])
    assert "".join(d["text"] for n, d in events if n == "delta") == "Sie haben 14 Tage Zeit [1]."
    assert events[-1][1]["citations"] and events[-1][1]["grounded"]


def test_stream_reports_llm_failure_as_an_error_event(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client, llm=ScriptedLLM(error=GenerationError("down")))
    events = parse_sse(http.post("/ask/stream", json={"question": "Anmeldung?"}).text)
    assert events[-1][0] == "error" and "unavailable" in events[-1][1]["detail"]


# ── a new index version appears while the API is running ────────────────

def test_runtime_swaps_to_a_new_index_version(settings, embedder, client):
    http, rt, s = make_client(settings, embedder, client)
    assert rt.maybe_reload() is False  # nothing changed
    assert http.post("/ask", json={"question": "Mietsicherheit", "source": "miete.txt"}).status_code == 400
    build_index(s, PAGES_V2, embedder, client)  # what the ingest job does
    assert rt.maybe_reload() is True
    assert rt.info()["documents"] == 3
    assert http.post("/ask", json={"question": "Mietsicherheit", "source": "miete.txt"}).status_code == 200


def test_request_recovers_when_the_old_index_was_pruned(settings, embedder, client):
    http, rt, s = make_client(settings, embedder, client)
    old = rt.pipeline.retriever.kb.collection
    build_index(s, PAGES_V2, embedder, client, keep_previous=0)  # the new version replaces and deletes the old one
    assert not client.collection_exists(old)
    r = http.post("/ask", json={"question": "Anmeldung Frist Einzug"})
    assert r.status_code == 200  # first attempt failed, the API reloaded and retried
    assert rt.pipeline.retriever.kb.collection != old


def test_a_failed_reload_keeps_serving_the_old_index(settings, embedder, client, monkeypatch):
    http, rt, s = make_client(settings, embedder, client)
    before = rt.pipeline.retriever.kb.collection
    build_index(s, PAGES_V2, embedder, client)
    monkeypatch.setattr(KnowledgeBase, "load", lambda self: (_ for _ in ()).throw(RuntimeError("qdrant down")))
    assert rt.maybe_reload() is False
    assert rt.pipeline.retriever.kb.collection == before
    assert http.post("/ask", json={"question": "Anmeldung?"}).status_code == 200


def test_bad_configuration_stops_the_app_from_starting(settings):
    with pytest.raises(Exception, match="Invalid configuration"):
        create_app(dataclasses.replace(settings, qdrant_url="not-a-url"))


# ── the chat page ────────────────────────────────────────────────────────

def test_chat_page_is_served_without_a_key_and_with_a_strict_policy(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client, service_api_key="s3cret")
    r = http.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "Beamter-Bot" in r.text and "/ask/stream" in r.text
    csp = r.headers["content-security-policy"]
    assert "'unsafe-inline'" not in csp and "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert r.headers["x-content-type-options"] == "nosniff"


def test_policy_hashes_match_the_inline_script_and_style(settings, embedder, client):
    import base64
    import hashlib
    import re

    http, _, _ = make_client(settings, embedder, client)
    r = http.get("/")
    for kind in ("script", "style"):
        body = re.search(rf"<{kind}[^>]*>(.*?)</{kind}>", r.text, flags=re.S).group(1)
        digest = "sha256-" + base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
        assert f"'{digest}'" in r.headers["content-security-policy"]


def test_page_never_inserts_model_text_as_html():
    from beamter.api import PAGE

    html = PAGE.read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert banned not in html
    assert "<script src" not in html and "http://" not in html and "https://" not in html  # nothing external


def test_answers_tell_the_client_how_the_question_was_understood(settings, embedder, client):
    http, _, _ = make_client(settings, embedder, client)
    body = http.post("/ask", json={"question": "Anmeldung?"}).json()
    assert body["interpreted_question"] and isinstance(body["top_score"], float)


# ── who counts as "the same client" for the rate limit ───────────────────

def test_rate_limit_ignores_a_forged_forwarded_address(settings, embedder, client):
    """A client can write anything at the start of X-Forwarded-For; only the last entry (added by
    the platform's proxy) identifies them. Changing the forged part must not reset the limit."""
    http, _, _ = make_client(settings, embedder, client, rate_limit_per_min=2)
    for i in range(2):
        r = http.post("/ask", json={"question": "Anmeldung?"}, headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.9"})
        assert r.status_code == 200
    r = http.post("/ask", json={"question": "Anmeldung?"}, headers={"X-Forwarded-For": "10.0.0.99, 203.0.113.9"})
    assert r.status_code == 429
    other = http.post("/ask", json={"question": "Anmeldung?"}, headers={"X-Forwarded-For": "10.0.0.1, 198.51.100.4"})
    assert other.status_code == 200  # a different real client is counted separately


def test_client_ip_falls_back_to_the_socket_address():
    from types import SimpleNamespace

    from beamter.api import client_ip

    req = SimpleNamespace(headers={}, client=SimpleNamespace(host="192.0.2.7"))
    assert client_ip(req) == "192.0.2.7"
    assert client_ip(SimpleNamespace(headers={"x-forwarded-for": " , "}, client=None)) == "unknown"


def test_logging_setup_does_not_override_an_existing_configuration():
    import logging

    from beamter.api import setup_logging

    before = list(logging.getLogger().handlers)
    setup_logging()
    assert logging.getLogger().handlers == before  # pytest already installed its handlers
    assert logging.getLogger("beamter").level == logging.INFO
