import zlib

import numpy as np
import pytest
from qdrant_client import QdrantClient

from beamter.config import Settings
from beamter.index import tokenize


class FakeEmbedder:
    """Deterministic bag-of-words embedder: similar words -> similar vectors. No model download."""

    model_name = "fake-embedder"
    dim = 64

    def __init__(self):
        self.calls = 0  # number of passages embedded, to prove when work is skipped

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts, batch_size=32):
        self.calls += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)


@pytest.fixture()
def embedder():
    return FakeEmbedder()


@pytest.fixture()
def client():
    return QdrantClient(":memory:")


@pytest.fixture()
def settings():
    return Settings.from_env()


class FakeReranker:
    """Scores a passage by how many question words it contains. Deterministic, no model."""

    def score(self, question, texts):
        q = set(tokenize(question))
        return [float(len(q & set(tokenize(t)))) for t in texts]
