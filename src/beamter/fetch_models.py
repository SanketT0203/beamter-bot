"""Download the two models into the local Hugging Face cache and prove they load.

    python -m beamter.fetch_models

The container image runs this once while it is built, so at runtime the models are already on
disk and nothing is downloaded (the service runs with HF_HUB_OFFLINE=1). The build runs it a
second time offline: if the first run left anything out, that second run fails the build instead
of failing the first request in production.
"""

from __future__ import annotations

import sys
import time

from .config import Settings


def fetch(embedder, reranker) -> dict:
    """Load both models by using them once. ``embedder`` and ``reranker`` are the same classes the
    service uses, so this exercises exactly the files the service will need."""
    t0 = time.perf_counter()
    vector = embedder.encode_query("Wie melde ich meinen Wohnsitz an?")
    scores = reranker.score("Anmeldung", ["Frist für die Anmeldung des Wohnsitzes"])
    if len(vector) == 0 or len(scores) != 1:
        raise RuntimeError("A model loaded but returned nothing usable")
    return {"embed_dim": len(vector), "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    from .index import SentenceTransformerEmbedder
    from .retrieval import CrossEncoderReranker

    s = Settings.from_env()  # model names only; no keys are needed
    print(f"embedding model: {s.embed_model}\nreranker:        {s.rerank_model}", flush=True)
    result = fetch(SentenceTransformerEmbedder(s.embed_model), CrossEncoderReranker(s.rerank_model))
    print(f"both models load (embedding dimension {result['embed_dim']}, {result['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
