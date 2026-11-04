import datetime as dt
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from beamter import corpus
from beamter.corpus import Entry, FetchError, ManifestError

ROOT = Path(__file__).resolve().parents[1]
HEADER = "id,title,publisher,lang,topic,url,note\n"


def test_real_manifest_is_valid_and_has_enough_entries():
    entries = corpus.load_manifest(ROOT / "docs_manifest.csv")
    assert len(entries) >= 40
    assert {e.lang for e in entries} <= {"de", "en"}


@pytest.mark.parametrize(
    "row, message",
    [
        ("Bad-Id,T,P,de,x,https://a.de/x.pdf,", "must match"),
        ("ok_id,T,P,de,x,http://a.de/x.pdf,", "https://"),
        ("ok_id,,P,de,x,https://a.de/x.pdf,", "title"),
    ],
)
def test_manifest_rejects_bad_rows(tmp_path, row, message):
    path = tmp_path / "m.csv"
    path.write_text(HEADER + row + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match=message):
        corpus.load_manifest(path)


def test_manifest_rejects_duplicate_ids_and_missing_columns(tmp_path):
    dup = tmp_path / "dup.csv"
    dup.write_text(HEADER + "a,T,P,de,x,https://a.de/1.pdf,\n" + "a,T,P,de,x,https://a.de/2.pdf,\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="duplicate"):
        corpus.load_manifest(dup)
    short = tmp_path / "short.csv"
    short.write_text("id,url\na,https://a.de\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="missing columns"):
        corpus.load_manifest(short)


def test_looks_like_pdf():
    assert corpus.looks_like_pdf(b"%PDF-1.7 ...")
    assert not corpus.looks_like_pdf(b"<!DOCTYPE html>")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/good.pdf":
            self._send(200, b"%PDF-1.4\n" + b"x" * 100, "application/pdf")
        elif self.path == "/html":
            self._send(200, b"<html>Cookie banner</html>", "text/html")
        elif self.path == "/big.pdf":
            self._send(200, b"%PDF-1.4\n" + b"x" * 5000, "application/pdf")
        else:
            self._send(404, b"nope", "text/plain")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence test output
        pass


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_download_accepts_pdf_and_rejects_html_404_and_oversize(server):
    assert corpus.download(f"{server}/good.pdf").startswith(b"%PDF-")
    with pytest.raises(FetchError, match="not a PDF"):
        corpus.download(f"{server}/html")
    with pytest.raises(FetchError, match="HTTP 404"):
        corpus.download(f"{server}/missing.pdf")
    with pytest.raises(FetchError, match="larger than"):
        corpus.download(f"{server}/big.pdf", max_bytes=1000)
    with pytest.raises(FetchError, match="only http"):
        corpus.download("file:///etc/passwd")


def test_fetch_all_is_resumable_and_reports_failures(tmp_path):
    entries = [
        Entry("one", "Eins", "Pub", "de", "t", "https://a.de/1.pdf", ""),
        Entry("two", "Zwei", "Pub", "de", "t", "https://a.de/2.pdf", ""),
    ]
    calls = []

    def fake(url, **_):
        calls.append(url)
        if url.endswith("2.pdf"):
            raise FetchError("HTTP 403")
        return b"%PDF-1.4 data"

    first = corpus.fetch_all(entries, tmp_path, sleep=lambda _: None, fetcher=fake)
    assert [r.status for r in first] == ["ok", "failed"]
    assert (tmp_path / "one.pdf").exists() and not (tmp_path / "two.pdf").exists()
    assert not list(tmp_path.glob("*.part"))

    calls.clear()
    second = corpus.fetch_all(entries, tmp_path, sleep=lambda _: None, fetcher=fake)
    assert [r.status for r in second] == ["skipped", "failed"]
    assert calls == ["https://a.de/2.pdf"]  # the finished file was not fetched again
    assert "1 already present, 1 failed" in corpus.summarize(second)
    assert "two: HTTP 403" in corpus.summarize(second)


def test_sources_file_lists_only_present_documents_and_is_not_indexed(tmp_path):
    ok = Entry("one", "Eins | Test", "Pub", "de", "t", "https://a.de/1.pdf", "")
    bad = Entry("two", "Zwei", "Pub", "de", "t", "https://a.de/2.pdf", "")
    results = [corpus.Result(ok, "ok"), corpus.Result(bad, "failed", "HTTP 403")]
    path = tmp_path / "SOURCES.md"
    corpus.write_sources(results, path, today=dt.date(2026, 10, 18))
    text = path.read_text(encoding="utf-8")
    assert "one.pdf" in text and "2026-10-18" in text and "two.pdf" not in text
    assert "Eins / Test" in text  # a pipe would break the markdown table

    from beamter.ingest import load_pages

    assert load_pages(tmp_path) == []  # SOURCES.md is skipped by the loader


def test_certificate_errors_get_an_actionable_message(monkeypatch):
    import ssl
    import urllib.error

    def boom(*args, **kwargs):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("unable to get local issuer certificate"))

    monkeypatch.setattr(corpus.urllib.request, "urlopen", boom)
    with pytest.raises(FetchError, match="truststore"):
        corpus.download("https://example.invalid/x.pdf")


def test_ssl_context_is_a_verifying_context():
    import ssl

    ctx = corpus._ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED  # never switched off
