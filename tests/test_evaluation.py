import dataclasses
import json

import pytest
from conftest import FakeReranker

from beamter.evaluation import calibrate_threshold, format_misses, run, summarize
from beamter.golden import Evidence, Question, load_golden, norm, validate_against_pages
from beamter.index import KnowledgeBase, build_index
from beamter.ingest import Page
from beamter.retrieval import Retriever

PAGES = [
    Page("anmeldung.txt", 1, "ANMELDUNG\nSie muessen sich innerhalb von 14 Tagen nach dem Einzug anmelden. "
                              "Bringen Sie Ihren Reisepass mit."),
    Page("rundfunk.txt", 1, "RUNDFUNKBEITRAG\nJede Wohnung zahlt einen Rundfunkbeitrag von 18,36 Euro im Monat."),
    Page("steuer.txt", 1, "STEUER-ID\nDie Steuer-Identifikationsnummer erhalten Sie automatisch per Post."),
]


def q(id, question, source=None, span=None, type="factoid_de"):
    ev = (Evidence(source, span),) if source else ()
    return Question(id, type, "de", question, bool(ev), ev)


QUESTIONS = [
    q("anm", "Wie viele Tage habe ich fuer die Anmeldung nach dem Einzug?", "anmeldung.txt", "14 Tagen"),
    q("rf", "Wie hoch ist der Rundfunkbeitrag?", "rundfunk.txt", "18,36 Euro"),
    q("oos", "Wie koche ich Maultaschen?", type="out_of_scope"),
]


@pytest.fixture()
def retriever(settings, embedder, client):
    s = dataclasses.replace(settings, collection="t", chunk_chars=300, min_chunk_chars=100,
                            rerank_pool=5, dense_pool=5, sparse_pool=5)
    build_index(s, PAGES, embedder, client)
    kb = KnowledgeBase(s, embedder, client)
    kb.load()
    return Retriever(kb, s, reranker=FakeReranker())


def test_norm_ignores_soft_hyphens_whitespace_case_and_line_hyphenation():
    assert norm("Ver­län- gern\n  JETZT") == "verlängern jetzt"
    assert norm("Netto-Einkommen") == "netto-einkommen"  # real hyphens stay


def test_relevance_requires_source_and_span():
    e = QUESTIONS[0]
    assert e.is_relevant("anmeldung.txt", "innerhalb von  14\nTagen")
    assert not e.is_relevant("other.txt", "innerhalb von 14 Tagen")
    assert not e.is_relevant("anmeldung.txt", "innerhalb von 7 Tagen")


def test_load_golden_rejects_inconsistent_entries(tmp_path):
    bad = {"questions": [{"id": "x", "type": "factoid_de", "lang": "de", "question": "?",
                          "answerable": True, "evidence": []}]}
    p = tmp_path / "g.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="evidence"):
        load_golden(p)
    bad["questions"][0].update(answerable=False, type="nonsense")
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="type"):
        load_golden(p)


def test_validate_reports_spans_missing_from_corpus():
    ok = validate_against_pages(QUESTIONS, PAGES)
    assert ok == []
    broken = [q("b", "?", "anmeldung.txt", "30 Tagen")]
    assert "span not found" in validate_against_pages(broken, PAGES)[0]


def test_run_and_summarize_find_relevant_chunks(retriever):
    results = run(retriever, QUESTIONS, "dense_rerank")
    by_id = {r.question.id: r for r in results}
    assert by_id["anm"].rank == 1 and by_id["rf"].rank == 1
    assert by_id["anm"].indexed
    assert by_id["oos"].rank is None
    s = summarize(results)
    assert s["overall"]["n"] == 2 and s["overall"]["hit@1"] == 1.0 and s["overall"]["mrr"] == 1.0
    assert s["by_type"]["factoid_de"]["n"] == 2


def test_unindexed_evidence_is_flagged(retriever):
    missing = [q("m", "Anmeldung Einzug", "anmeldung.txt", "nicht vorhanden")]
    r = run(retriever, missing, "dense")[0]
    assert r.rank is None and not r.indexed
    assert summarize([r])["overall"]["indexed"] == 0.0


def test_threshold_separates_answerable_from_out_of_scope(retriever):
    cal = calibrate_threshold(run(retriever, QUESTIONS, "dense_rerank"))
    assert cal["accuracy"] == 1.0
    assert cal["out_of_scope_max"] < cal["threshold"] <= cal["answerable_min"]
    assert calibrate_threshold(run(retriever, QUESTIONS[:2], "dense_rerank")) == {}


def test_heading_text_counts_as_evidence_and_recall_is_reported(settings, embedder, client):
    pages = [Page("w.txt", 1, "ANTRAG AUF WOHNGELD\nBringen Sie den Mietvertrag mit."),
             Page("x.txt", 1, "STEUER\nDie Steuer-ID kommt per Post.")]
    s = dataclasses.replace(settings, collection="h", chunk_chars=300, min_chunk_chars=100,
                            rerank_pool=5, dense_pool=5, sparse_pool=5)
    build_index(s, pages, embedder, client)
    kb = KnowledgeBase(s, embedder, client)
    kb.load()
    r = Retriever(kb, s, reranker=FakeReranker())
    heading_q = q("h", "Antrag auf Wohngeld", "w.txt", "Antrag auf Wohngeld")
    res = run(r, [heading_q], "dense")[0]
    assert res.indexed and res.rank == 1
    assert summarize([res])["overall"]["recall"] == 1.0


def test_miss_report_shows_wanted_and_retrieved(retriever):
    miss = [q("m", "Rundfunkbeitrag Hoehe", "anmeldung.txt", "steht nirgends")]
    out = format_misses(run(retriever, miss, "dense"), show_hits=True)
    assert "wanted: anmeldung.txt" in out and "got 1:" in out


def test_cached_embedder_only_embeds_new_passages(tmp_path, embedder):
    from beamter.evaluation import CachedEmbedder

    path = tmp_path / "c.pkl"
    c1 = CachedEmbedder(embedder, path)
    first = c1.encode_passages(["eins", "zwei"])
    assert embedder.calls == 2
    c2 = CachedEmbedder(embedder, path)  # new object, same file
    again = c2.encode_passages(["zwei", "eins", "drei"])
    assert embedder.calls == 3  # only "drei" was embedded
    assert (again[0] == first[1]).all() and (again[1] == first[0]).all()
    assert c2.model_name == "fake-embedder"
