import dataclasses

import pytest

from beamter.index import IndexNotReady, KnowledgeBase, alias_target, build_index
from beamter.ingest import Page

PAGES = [
    Page("anmeldung.txt", 1, "ANMELDUNG\nSie muessen sich innerhalb von 14 Tagen nach dem Einzug anmelden. "
                              "Bringen Sie Ihren Reisepass und die Wohnungsgeberbestaetigung mit."),
    Page("rundfunk.txt", 1, "RUNDFUNKBEITRAG\nJede Wohnung zahlt einen Rundfunkbeitrag. "
                            "Studierende mit BAfoeG koennen eine Befreiung beantragen."),
    Page("steuer.txt", 1, "STEUER-ID\nDie Steuer-Identifikationsnummer erhalten Sie automatisch per Post vom Bundeszentralamt."),
]


@pytest.fixture()
def small(settings):
    return dataclasses.replace(settings, collection="test_docs", chunk_chars=300, min_chunk_chars=100)


def test_build_creates_versioned_collection_and_alias(small, embedder, client):
    report = build_index(small, PAGES, embedder, client)
    assert report.action == "built"
    assert report.collection.startswith("test_docs__")
    assert alias_target(client, "test_docs") == report.collection
    assert client.count(report.collection, exact=True).count == report.chunks == 3


def test_rerun_without_changes_does_no_embedding(small, embedder, client):
    build_index(small, PAGES, embedder, client)
    embedded = embedder.calls
    again = build_index(small, PAGES, embedder, client)
    assert again.action == "up_to_date"
    assert embedder.calls == embedded


def test_changed_document_builds_new_version_then_switches_and_prunes(small, embedder, client):
    first = build_index(small, PAGES, embedder, client)
    edited = PAGES[:2] + [Page("steuer.txt", 1, "STEUER-ID\nGeaenderter Text zur Steuer-ID.")]
    second = build_index(small, edited, embedder, client)
    assert second.collection != first.collection
    assert alias_target(client, "test_docs") == second.collection
    assert client.collection_exists(first.collection)  # kept one version back for rollback
    third = build_index(small, PAGES[:2], embedder, client)
    assert not client.collection_exists(first.collection)  # now two generations old: pruned
    assert third.pruned == [first.collection]


def test_changed_chunk_settings_change_the_fingerprint(small, embedder, client):
    a = build_index(small, PAGES, embedder, client)
    b = build_index(dataclasses.replace(small, chunk_chars=200), PAGES, embedder, client)
    assert a.collection != b.collection


def test_force_rebuilds_without_touching_the_live_collection(small, embedder, client):
    first = build_index(small, PAGES, embedder, client)
    forced = build_index(small, PAGES, embedder, client, force=True)
    assert forced.action == "built"
    assert forced.collection != first.collection
    assert alias_target(client, "test_docs") == forced.collection


def test_half_written_collection_is_rebuilt(small, embedder, client):
    ok = build_index(small, PAGES, embedder, client)
    client.delete(ok.collection, points_selector=[client.scroll(ok.collection, limit=1)[0][0].id])
    again = build_index(small, PAGES, embedder, client)
    assert again.action == "built"
    assert client.count(again.collection, exact=True).count == 3


def test_empty_corpus_is_an_error(small, embedder, client):
    with pytest.raises(ValueError, match="No indexable text"):
        build_index(small, [], embedder, client)


def test_knowledge_base_loads_chunks_back_from_qdrant(small, embedder, client):
    build_index(small, PAGES, embedder, client)
    kb = KnowledgeBase(small, embedder, client)
    info = kb.load()
    assert info["chunks"] == 3 and info["documents"] == 3
    assert kb.sources() == {"anmeldung.txt": 1, "rundfunk.txt": 1, "steuer.txt": 1}
    assert kb.ping()


def test_dense_and_sparse_search_find_the_right_chunk(small, embedder, client):
    build_index(small, PAGES, embedder, client)
    kb = KnowledgeBase(small, embedder, client)
    kb.load()
    q = "Wie lange habe ich Zeit fuer die Anmeldung nach dem Einzug?"
    assert kb.by_id[kb.dense_search(q, 1)[0]].source == "anmeldung.txt"
    assert kb.by_id[kb.sparse_search(q, 1)[0]].source == "anmeldung.txt"
    assert kb.sparse_search("xyzzy voellig unbekannt", 3) == []  # no term matched: no candidates


def test_source_filter_restricts_both_searches(small, embedder, client):
    build_index(small, PAGES, embedder, client)
    kb = KnowledgeBase(small, embedder, client)
    kb.load()
    q = "Befreiung Rundfunkbeitrag Anmeldung Steuer"
    for ids in (kb.dense_search(q, 5, source="steuer.txt"), kb.sparse_search(q, 5, source="steuer.txt")):
        assert ids and {kb.by_id[i].source for i in ids} == {"steuer.txt"}


def test_loading_before_ingest_gives_a_clear_error(small, embedder, client):
    with pytest.raises(IndexNotReady, match="Run the ingest job"):
        KnowledgeBase(small, embedder, client).load()
    assert not KnowledgeBase(small, embedder, client).ping()


# ── incremental ingest ───────────────────────────────────────────────────

def _vectors(client, collection):
    records, _ = client.scroll(collection, limit=1000, with_payload=False, with_vectors=True)
    return {r.id: r.vector for r in records}


def test_first_build_embeds_everything(small, embedder, client):
    report = build_index(small, PAGES, embedder, client)
    assert (report.embedded, report.reused) == (3, 0)
    assert embedder.calls == 3


def test_editing_one_document_embeds_only_that_document(small, embedder, client):
    build_index(small, PAGES, embedder, client)
    before = embedder.calls
    edited = PAGES[:2] + [Page("steuer.txt", 1, "STEUER-ID\nGeaenderter Text zur Steuer-ID.")]
    report = build_index(small, edited, embedder, client)
    assert (report.embedded, report.reused) == (1, 2)
    assert embedder.calls - before == 1  # the model really was called for one chunk only


def test_adding_a_document_embeds_only_the_new_one(small, embedder, client):
    build_index(small, PAGES[:2], embedder, client)
    before = embedder.calls
    report = build_index(small, PAGES, embedder, client)
    assert (report.embedded, report.reused) == (1, 2)
    assert embedder.calls - before == 1


def test_removing_a_document_embeds_nothing(small, embedder, client):
    build_index(small, PAGES, embedder, client)
    before = embedder.calls
    report = build_index(small, PAGES[:2], embedder, client)
    assert (report.embedded, report.reused) == (0, 2)
    assert embedder.calls == before
    assert client.count(report.collection, exact=True).count == 2


def test_incremental_result_equals_a_build_from_scratch(small, embedder, client):
    import numpy as np
    from qdrant_client import QdrantClient

    from conftest import FakeEmbedder

    build_index(small, PAGES, embedder, client)
    edited = PAGES[:2] + [Page("steuer.txt", 1, "STEUER-ID\nGeaenderter Text zur Steuer-ID.")]
    incremental = build_index(small, edited, embedder, client)

    fresh_client = QdrantClient(":memory:")
    scratch = build_index(small, edited, FakeEmbedder(), fresh_client)

    a, b = _vectors(client, incremental.collection), _vectors(fresh_client, scratch.collection)
    assert a.keys() == b.keys()
    assert all(np.allclose(a[k], b[k]) for k in a)


def test_a_different_embedding_model_never_reuses_old_vectors(small, embedder, client):
    from conftest import FakeEmbedder

    build_index(small, PAGES, embedder, client)
    other = FakeEmbedder()
    other.model_name = "another-model"
    report = build_index(small, PAGES, other, client)
    assert (report.embedded, report.reused) == (3, 0)


def test_full_flag_re_embeds_everything(small, embedder, client):
    build_index(small, PAGES, embedder, client)
    report = build_index(small, PAGES, embedder, client, full=True)
    assert (report.action, report.embedded, report.reused) == ("built", 3, 0)


def test_force_still_reuses_vectors(small, embedder, client):
    build_index(small, PAGES, embedder, client)
    report = build_index(small, PAGES, embedder, client, force=True)
    assert (report.action, report.embedded, report.reused) == ("built", 0, 3)


def test_old_index_without_hashes_is_not_reused(small, embedder, client):
    build_index(small, PAGES, embedder, client)
    # simulate an index written by an older version that stored no content_hash / embed_key
    name = alias_target(client, "test_docs")
    ids = [r.id for r in client.scroll(name, limit=10)[0]]
    client.overwrite_payload(name, payload={"source": "x", "page": 1, "ordinal": 0, "text": "t"}, points=ids)
    report = build_index(small, PAGES, embedder, client, force=True)
    assert (report.embedded, report.reused) == (3, 0)


def test_loading_an_index_built_with_another_model_is_refused(small, embedder, client):
    from conftest import FakeEmbedder

    from beamter.index import IndexMismatch

    build_index(small, PAGES, embedder, client)
    other = FakeEmbedder()
    other.model_name = "another-model"
    with pytest.raises(IndexMismatch, match="different embedding model"):
        KnowledgeBase(small, other, client).load()
