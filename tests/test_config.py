import dataclasses

import pytest

from beamter.config import ConfigError, Settings

ENV_NAMES = [
    "QDRANT_URL", "QDRANT_API_KEY", "ANTHROPIC_API_KEY", "SERVICE_API_KEY",
    "BEAMTER_COLLECTION", "BEAMTER_DOCS_DIR", "BEAMTER_EMBED_MODEL", "BEAMTER_RERANK_MODEL",
    "BEAMTER_CHUNK_CHARS", "BEAMTER_MIN_CHUNK_CHARS", "BEAMTER_OVERLAP_SENTENCES",
    "BEAMTER_DENSE_POOL", "BEAMTER_SPARSE_POOL", "BEAMTER_RERANK_POOL", "BEAMTER_TOP_K",
    "BEAMTER_RETRIEVAL_MODE", "BEAMTER_LLM_MODEL", "BEAMTER_MAX_ANSWER_TOKENS",
    "BEAMTER_HISTORY_TURNS", "BEAMTER_LLM_TIMEOUT_S", "BEAMTER_LLM_MAX_RETRIES",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts from an empty environment, whatever is in the developer's shell or .env."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def cloud_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://abc.eu-central.aws.cloud.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")


def test_defaults_with_required_values_set(cloud_env):
    s = Settings.from_env()
    s.validate()  # must not raise
    assert s.collection == "beamter_docs"
    assert (s.chunk_chars, s.top_k, s.rerank_pool) == (700, 4, 30)
    assert s.retrieval_mode == "hybrid_rerank"
    assert s.service_api_key == ""  # auth is off unless configured


def test_missing_required_values_are_all_reported_together():
    with pytest.raises(ConfigError) as err:
        Settings.from_env().validate()
    message = str(err.value)
    assert "QDRANT_URL" in message and "ANTHROPIC_API_KEY" in message


def test_https_url_without_api_key_is_rejected(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://abc.cloud.qdrant.io:6333")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(ConfigError, match="QDRANT_API_KEY"):
        Settings.from_env().validate()


def test_secrets_never_appear_in_repr(cloud_env, monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "service-secret")
    text = repr(Settings.from_env())
    for secret in ("qdrant-secret", "anthropic-secret", "service-secret"):
        assert secret not in text
    assert "qdrant_url" in text  # non-secret fields are still shown


def test_environment_overrides_defaults(cloud_env, monkeypatch):
    monkeypatch.setenv("BEAMTER_TOP_K", "6")
    monkeypatch.setenv("BEAMTER_RERANK_POOL", "15")
    monkeypatch.setenv("BEAMTER_RETRIEVAL_MODE", "hybrid")
    s = Settings.from_env()
    s.validate()
    assert (s.top_k, s.rerank_pool, s.retrieval_mode) == (6, 15, "hybrid")


def test_non_integer_value_fails_fast(monkeypatch):
    monkeypatch.setenv("BEAMTER_TOP_K", "four")
    with pytest.raises(ConfigError, match="BEAMTER_TOP_K"):
        Settings.from_env()


def test_inconsistent_retrieval_settings_are_rejected(cloud_env, monkeypatch):
    monkeypatch.setenv("BEAMTER_RERANK_POOL", "2")  # smaller than TOP_K=4
    monkeypatch.setenv("BEAMTER_RETRIEVAL_MODE", "magic")
    with pytest.raises(ConfigError) as err:
        Settings.from_env().validate()
    assert "RERANK_POOL" in str(err.value) and "RETRIEVAL_MODE" in str(err.value)


def test_settings_are_immutable(cloud_env):
    s = Settings.from_env()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.top_k = 99


def test_ingest_job_does_not_need_an_anthropic_key(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://x.cloud.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "k")
    s = Settings.from_env()
    s.validate(require_llm=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        s.validate()


def test_generation_settings_have_safe_defaults_and_validate_numbers(cloud_env, monkeypatch):
    s = Settings.from_env()
    assert s.min_rerank_score == -2.5 and s.rewrite_queries is True
    monkeypatch.setenv("BEAMTER_REWRITE_QUERIES", "0")
    monkeypatch.setenv("BEAMTER_MIN_RERANK_SCORE", "-1.5")
    s = Settings.from_env()
    assert s.rewrite_queries is False and s.min_rerank_score == -1.5
    monkeypatch.setenv("BEAMTER_MIN_RERANK_SCORE", "abc")
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_default_model_names_are_shared_constants():
    from beamter.config import DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MODEL

    s = Settings.from_env()
    assert s.embed_model == DEFAULT_EMBED_MODEL and s.rerank_model == DEFAULT_RERANK_MODEL
