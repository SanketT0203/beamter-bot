import dataclasses

import pytest
from conftest import FakeReranker

from beamter.answer_eval import (
    JUDGE_SYSTEM, evidence_pages, format_problems, format_summary, parse_verdict, run, save, summarize,
)
from beamter.generation import ANSWER_SYSTEM, REWRITE_SYSTEM, GenerationError
from beamter.golden import Evidence, Question
from beamter.index import KnowledgeBase, build_index
from beamter.ingest import Page
from beamter.pipeline import Pipeline
from beamter.retrieval import Retriever

PAGES = [
    Page("anmeldung.txt", 1, "ANMELDUNG\nSie muessen sich innerhalb von 14 Tagen nach dem Einzug anmelden."),
    Page("rundfunk.txt", 1, "RUNDFUNKBEITRAG\nJede Wohnung zahlt einen Rundfunkbeitrag von 18,36 Euro im Monat."),
]
Q_REG = Question("reg", "factoid_de", "de", "Wie lange habe ich für die Anmeldung nach dem Einzug?", True,
                 (Evidence("anmeldung.txt", "innerhalb von 14 Tagen nach dem Einzug"),))
Q_FEE = Question("fee", "factoid_de", "de", "Wie hoch ist der Rundfunkbeitrag pro Monat?", True,
                 (Evidence("rundfunk.txt", "Rundfunkbeitrag von 18,36 Euro"),))
Q_OOS = Question("oos", "out_of_scope", "en", "What is the weather like on Mars?", False, ())


class Brain:
    """Answers and judges by looking at the question text, so each test controls the outcome."""

    def __init__(self, answers, verdicts=None, judge_error=None):
        self.answers, self.verdicts, self.judge_error = answers, verdicts or {}, judge_error
        self.judge_calls = []

    def complete(self, system, messages, max_tokens):
        text = messages[-1]["content"]
        if system == REWRITE_SYSTEM:
            return text.split("Question:")[-1].strip()[:80] if "Question:" in text else text[:80]
        if system == JUDGE_SYSTEM:
            self.judge_calls.append(text)
            if self.judge_error:
                raise self.judge_error
            for key, verdict in self.verdicts.items():
                if key in text:
                    return verdict
            return '{"correct": "yes", "faithful": "yes", "language_ok": "yes", "note": ""}'
        assert system == ANSWER_SYSTEM
        for key, answer in self.answers.items():
            if key in text:
                return answer
        return "NOT_COVERED"

    def stream(self, system, messages, max_tokens):
        yield self.complete(system, messages, max_tokens)


def make_pipeline(settings, embedder, client, brain):
    s = dataclasses.replace(settings, collection="ae", chunk_chars=300, min_chunk_chars=100, rerank_pool=5,
                            dense_pool=5, sparse_pool=5, top_k=2, retrieval_mode="hybrid_rerank",
                            min_rerank_score=-0.8, rewrite_queries=False)
    build_index(s, PAGES, embedder, client)
    kb = KnowledgeBase(s, embedder, client)
    kb.load()
    return Pipeline(s, Retriever(kb, s, FakeReranker()), brain)


def test_parse_verdict_accepts_fenced_json_and_rejects_garbage():
    assert parse_verdict('```json\n{"correct": "Yes", "faithful": "yes"}\n```')["correct"] == "Yes"
    assert parse_verdict('Sure! {"correct": "partial", "note": "x"} done')["correct"] == "partial"
    assert parse_verdict('{"correct": "maybe"}') is None
    assert parse_verdict("no json here") is None


def test_evidence_pages_are_found_independent_of_chunking():
    assert evidence_pages(Q_REG, PAGES) == {("anmeldung.txt", 1)}
    assert evidence_pages(Q_OOS, PAGES) == set()


def test_good_answers_score_as_correct_with_checked_citations(settings, embedder, client):
    brain = Brain({"Wie lange": "Sie haben 14 Tage Zeit [1].", "Wie hoch": "Es sind 18,36 Euro [1]."})
    res = run(make_pipeline(settings, embedder, client, brain), brain, [Q_REG, Q_FEE], PAGES)
    assert [r.correct for r in res] == ["yes", "yes"]
    assert all(r.cite_ok and r.faithful and r.lang_ok and r.ctx_has_ev for r in res)
    assert "14 Tagen nach dem Einzug" in brain.judge_calls[0]  # the judge sees the reference evidence
    assert "Anmeldung" in brain.judge_calls[0]  # and the full cited passage


def test_false_refusal_and_out_of_scope_behaviour_are_counted_separately(settings, embedder, client):
    brain = Brain({"Wie lange": "Sie haben 14 Tage Zeit [1].", "Mars": "Hot [1]."})  # fee question -> NOT_COVERED
    res = run(make_pipeline(settings, embedder, client, brain), brain, [Q_REG, Q_FEE, Q_OOS], PAGES)
    s = summarize(res)
    assert s["rows"]["ALL answerable"]["answered"] == 0.5
    assert s["rows"]["ALL answerable"]["correct"] == 0.5  # refusals count as not correct
    assert s["false_refusals"] == {"model": 1}
    assert s["out_of_scope"]["refused"] == 0 and s["out_of_scope"]["rate"] == 0  # it answered about Mars
    text = format_summary("t", s) + format_problems(res)
    assert "FALSE REFUSAL" in text and "ANSWERED AN OUT-OF-SCOPE QUESTION" in text


def test_judge_sees_passages_under_the_numbers_the_answer_uses():
    from beamter.answer_eval import judge_messages

    text = judge_messages(Q_REG, "Antwort [3][4]", [(3, "Passage drei"), (4, "Passage vier")])[0]["content"]
    assert "[3] Passage drei" in text and "[4] Passage vier" in text and "[1]" not in text


def test_citation_to_the_wrong_page_is_caught_without_a_model(settings, embedder, client):
    brain = Brain({"Wie lange": "Rundfunk steht dort [1]."})
    pipe = make_pipeline(settings, embedder, client, brain)
    pipe.retriever.reranker = type("R", (), {"score": lambda self, q, texts: [1.0 if "Rundfunk" in t else 0.0 for t in texts]})()
    [r] = run(pipe, brain, [Q_REG], PAGES)
    assert r.cite_ok is False
    assert "citation does not point at the evidence page" in format_problems([r])


def test_judge_verdicts_feed_the_metrics(settings, embedder, client):
    brain = Brain(
        {"Wie lange": "Sie haben 14 Tage Zeit [1].", "Wie hoch": "Es sind 20 Euro [1]."},
        {"20 Euro": '{"correct": "no", "faithful": "no", "language_ok": "yes", "note": "wrong amount"}'},
    )
    res = run(make_pipeline(settings, embedder, client, brain), brain, [Q_REG, Q_FEE], PAGES)
    row = summarize(res)["rows"]["ALL answerable"]
    assert row["correct"] == 0.5 and row["faithful"] == 0.5
    assert "wrong amount" in format_problems(res)


def test_judge_failure_is_visible_and_not_scored_as_correct(settings, embedder, client):
    brain = Brain({"Wie lange": "Sie haben 14 Tage Zeit [1]."}, judge_error=GenerationError("judge down"))
    [r] = run(make_pipeline(settings, embedder, client, brain), brain, [Q_REG], PAGES)
    assert r.error.startswith("judge:") and r.correct is None
    s = summarize([r])
    assert s["rows"]["ALL answerable"]["correct"] == 0 and s["rows"]["ALL answerable"]["errors"] == 1
    assert "WARNING" in format_summary("t", s)


def test_unusable_judge_output_is_reported(settings, embedder, client):
    brain = Brain({"Wie lange": "Sie haben 14 Tage Zeit [1]."}, {"Wie lange": "I think it is fine"})
    [r] = run(make_pipeline(settings, embedder, client, brain), brain, [Q_REG], PAGES)
    assert r.error == "judge returned no usable verdict"


def test_answer_failure_is_recorded_not_raised(settings, embedder, client):
    class Down(Brain):
        def complete(self, system, messages, max_tokens):
            if system == ANSWER_SYSTEM:
                raise GenerationError("LLM call failed")
            return super().complete(system, messages, max_tokens)

    brain = Down({})
    [r] = run(make_pipeline(settings, embedder, client, brain), brain, [Q_REG], PAGES)
    assert "LLM call failed" in r.error


def test_results_are_saved_for_later_inspection(settings, embedder, client, tmp_path):
    brain = Brain({"Wie lange": "Sie haben 14 Tage Zeit [1]."})
    res = run(make_pipeline(settings, embedder, client, brain), brain, [Q_REG], PAGES)
    out = tmp_path / "x" / "r.json"
    save(res, summarize(res), out, {"judge_model": "j"})
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["meta"]["judge_model"] == "j" and data["results"][0]["correct"] == "yes"
