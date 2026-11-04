import dataclasses

import pytest
from conftest import FakeReranker

from beamter.index import KnowledgeBase, build_index
from beamter.ingest import Page
from beamter.retrieval import MODES, Retriever, rrf_fuse
from beamter.search import format_hits

PAGES = [
    Page("anmeldung.txt", 1, "ANMELDUNG\nSie muessen sich innerhalb von 14 Tagen nach dem Einzug anmelden. "
                              "Bringen Sie Ihren Reisepass und die Wohnungsgeberbestaetigung mit."),
    Page("rundfunk.txt", 1, "RUNDFUNKBEITRAG\nJede Wohnung zahlt einen Rundfunkbeitrag von 18,36 Euro. "
                            "Studierende mit BAfoeG koennen eine Befreiung beantragen."),
    Page("steuer.txt", 1, "STEUER-ID\nDie Steuer-Identifikationsnummer erhalten Sie automatisch per Post."),
]
QUESTION = "Wie viele Tage Zeit habe ich fuer die Anmeldung nach dem Einzug?"


@pytest.fixture()
def retriever(settings, embedder, client):
    s = dataclasses.replace(settings, collection="t", chunk_chars=300, min_chunk_chars=100,
                            rerank_pool=5, top_k=2, dense_pool=5, sparse_pool=5)
    build_index(s, PAGES, embedder, client)
    kb = KnowledgeBase(s, embedder, client)
    kb.load()
    return Retriever(kb, s, reranker=FakeReranker())


def test_rrf_rewards_agreement_and_is_deterministic():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    ids = [i for i, _ in fused]
    assert ids[:2] == [1, 3]  # appear in both lists
    assert set(ids) == {1, 2, 3, 4}
    assert fused == rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    assert dict(fused)[1] == pytest.approx(1 / 61 + 1 / 62)  # rank 1 in list A, rank 2 in list B


def test_all_modes_return_valid_ranked_hits(retriever):
    for mode in MODES:
        hits = retriever.retrieve(QUESTION, mode=mode, k=2)
        assert 1 <= len(hits) <= 2
        assert all(h.chunk.id in retriever.kb.by_id for h in hits)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
    top = retriever.retrieve(QUESTION, mode="hybrid_rerank", k=1)[0]
    assert top.chunk.source == "anmeldung.txt"
    assert top.rerank_score is not None and top.sparse_rank is not None and top.dense_rank is not None


def test_dense_mode_has_no_sparse_or_rerank_information(retriever):
    hit = retriever.retrieve(QUESTION, mode="dense", k=1)[0]
    assert hit.sparse_rank is None and hit.rerank_score is None and hit.fused_score is None


def test_timings_cover_only_the_stages_that_ran(retriever):
    _, dense = retriever.retrieve_traced(QUESTION, mode="dense")
    assert set(dense) == {"dense_ms", "total_ms"}
    _, full = retriever.retrieve_traced(QUESTION, mode="hybrid_rerank")
    assert {"dense_ms", "sparse_ms", "rerank_ms", "total_ms"} <= set(full)
    assert full["total_ms"] == pytest.approx(full["dense_ms"] + full["sparse_ms"] + full["rerank_ms"], abs=0.2)


def test_source_filter_restricts_all_stages(retriever):
    hits = retriever.retrieve("Wohnung Euro Frist", mode="hybrid_rerank", source="rundfunk.txt")
    assert hits and {h.chunk.source for h in hits} == {"rundfunk.txt"}


def test_unknown_source_gives_no_hits_not_an_error(retriever):
    assert retriever.retrieve(QUESTION, mode="hybrid_rerank", source="gibt-es-nicht.pdf") == []


def test_unknown_mode_rejected(retriever):
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        retriever.retrieve("x", mode="magic")


def test_rerank_can_overrule_the_first_stage(retriever):
    class Reverse:
        def score(self, question, texts):  # prefers the LAST candidate
            return [float(i) for i in range(len(texts))]

    retriever.reranker = Reverse()
    plain = retriever.retrieve(QUESTION, mode="hybrid", k=3)
    reranked = retriever.retrieve(QUESTION, mode="hybrid_rerank", k=3)
    assert reranked[0].chunk.id == plain[-1].chunk.id


def test_warm_up_touches_both_models(retriever, embedder):
    seen = []

    class Spy:
        def score(self, q, texts):
            seen.append(q)
            return [0.0] * len(texts)

    retriever.reranker = Spy()
    retriever.warm_up()
    assert seen == ["warm up"]


def test_format_hits_is_readable(retriever):
    text = format_hits(retriever.retrieve(QUESTION, mode="hybrid_rerank", k=2))
    assert "anmeldung.txt p.1" in text and "dense#" in text and "rerank=" in text
    assert format_hits([]) == "(no results)"
