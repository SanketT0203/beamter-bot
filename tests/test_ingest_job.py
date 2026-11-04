import pytest

from beamter import ingest_job


@pytest.fixture()
def docs(tmp_path):
    (tmp_path / "a.txt").write_text("ANMELDUNG\nSie haben 14 Tage Zeit. Bringen Sie den Pass mit.", encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("QDRANT_URL", "QDRANT_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_dry_run_needs_no_keys_and_no_network(docs, capsys):
    assert ingest_job.main(["--docs", str(docs), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "1 documents, 1 pages" in out and "fingerprint:" in out


def test_missing_qdrant_config_is_reported_clearly(docs, capsys):
    assert ingest_job.main(["--docs", str(docs)]) == 2  # env is empty (see clean_env fixture)
    assert "QDRANT_URL" in capsys.readouterr().err


def test_unreachable_qdrant_fails_fast_with_a_readable_message(docs, capsys, monkeypatch):
    class Dead:
        def get_collections(self):
            raise ConnectionError("name resolution failed")

    monkeypatch.setenv("QDRANT_URL", "https://example.invalid:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-key-value")
    monkeypatch.setattr(ingest_job, "make_client", lambda settings: Dead())
    assert ingest_job.main(["--docs", str(docs)]) == 3
    err = capsys.readouterr().err
    assert "Cannot reach Qdrant" in err and "secret-key-value" not in err
