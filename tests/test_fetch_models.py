import pytest
from conftest import FakeReranker

from beamter.fetch_models import fetch


def test_fetch_uses_both_models_once(embedder):
    out = fetch(embedder, FakeReranker())
    assert out["embed_dim"] == embedder.dim


def test_a_model_that_returns_nothing_fails_the_build():
    class Empty:
        def encode_query(self, text):
            return []

    with pytest.raises(RuntimeError, match="nothing usable"):
        fetch(Empty(), FakeReranker())


def test_a_missing_model_fails_loudly():
    class Missing:
        def encode_query(self, text):
            raise OSError("not in cache and offline")

    with pytest.raises(OSError):
        fetch(Missing(), FakeReranker())
