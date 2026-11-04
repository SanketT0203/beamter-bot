import dataclasses

import pytest
from conftest import FakeReranker

from beamter.ask import format_debug, format_sources
from beamter.evaluation import rewrite_questions, run
from beamter.generation import ANSWER_SYSTEM, REWRITE_SYSTEM, Answer, Delta, GenerationError
from beamter.golden import Evidence, Question
from beamter.index import KnowledgeBase, build_index
from beamter.ingest import Chunk, Page
from beamter.pipeline import Pipeline, merge_hits, search
from beamter.retrieval import Hit, Retriever

PAGES = [
    Page("anmeldung.txt", 1, "ANMELDUNG\nSie muessen sich innerhalb von 14 Tagen nach dem Einzug anmelden."),
    Page("rundfunk.txt", 1, "RUNDFUNKBEITRAG\nJede Wohnung zahlt einen Rundfunkbeitrag von 18,36 Euro im Monat."),
    Page("miete.txt", 1, "MIETSICHERHEIT\nDie Mietsicherheit darf hoechstens drei Monatsmieten betragen."),
]


class ScriptedLLM:
    def __init__(self, rewrite="", answer="", pieces=None, rewrite_error=None, answer_error=None):
        self.rewrite, self.answer, self.pieces = rewrite, answer, pieces
        self.rewrite_error, self.answer_error = rewrite_error, answer_error
        self.calls = []

    def complete(self, system, messages, max_tokens):
        self.calls.append(("rewrite" if system == REWRITE_SYSTEM else "answer", messages))
        if system == REWRITE_SYSTEM:
            if self.rewrite_error:
                raise self.rewrite_error
            return self.rewrite
        if self.answer_error:
            raise self.answer_error
        return self.answer

    def stream(self, system, messages, max_tokens):
        self.calls.append(("stream", messages))
        yield from (self.pieces or [self.answer])

    def kinds(self):
        return [k for k, _ in self.calls]


@pytest.fixture()
def retriever(settings, embedder, client):
    s = dataclasses.replace(settings, collection="p", chunk_chars=300, min_chunk_chars=100, rerank_pool=5,
                            dense_pool=5, sparse_pool=5, top_k=2, retrieval_mode="hybrid_rerank",
                            min_rerank_score=-0.8, rewrite_queries=True)
    build_index(s, PAGES, embedder, client)
    kb = KnowledgeBase(s, embedder, client)
    kb.load()
    return Retriever(kb, s, reranker=FakeReranker())


def pipeline(retriever, llm, **overrides):
    return Pipeline(dataclasses.replace(retriever.s, **overrides), retriever, llm)


def h(i, score):
    return Hit(Chunk(i, "a.txt", 1, "", 0, f"t{i}"), score, rerank_score=score)


# ── merging ──────────────────────────────────────────────────────────────

def test_merge_dedupes_keeps_best_score_and_limits():
    merged = merge_hits([[h(1, 0.2), h(2, 0.9)], [h(1, 0.8), h(3, 0.5)]], k=2)
    assert [x.chunk.id for x in merged] == [2, 1]
    assert merged[1].rerank_score == 0.8


# ── the pipeline ─────────────────────────────────────────────────────────

def test_ask_rewrites_searches_and_answers_with_checked_citations(retriever):
    llm = ScriptedLLM(rewrite="Mietsicherheit Hoechstbetrag", answer="Hoechstens drei Monatsmieten [1][9].")
    a = pipeline(retriever, llm).ask("How high can the deposit be?")
    assert llm.kinds() == ["rewrite", "answer"]
    assert a.search_query == "Mietsicherheit Hoechstbetrag"
    assert a.covered and a.grounded and [c.n for c in a.citations] == [1] and "[9]" not in a.text
    assert a.citations[0].source == "miete.txt"
    assert set(a.timings) >= {"rewrite_ms", "dense_ms", "llm_ms", "total_ms"}


def test_rewriting_can_be_switched_off(retriever):
    llm = ScriptedLLM(rewrite="should not be asked", answer="14 Tage [1].")
    a = pipeline(retriever, llm, rewrite_queries=False).ask("Anmeldung Frist Einzug")
    assert llm.kinds() == ["answer"] and a.search_query == "Anmeldung Frist Einzug"


def test_failed_rewrite_still_answers_with_the_original_question(retriever):
    llm = ScriptedLLM(rewrite_error=GenerationError("down"), answer="14 Tage [1].")
    a = pipeline(retriever, llm).ask("Anmeldung Frist Einzug")
    assert a.covered and a.search_query == "Anmeldung Frist Einzug"


def test_unrelated_question_is_declined_without_an_answer_call(retriever):
    llm = ScriptedLLM(rewrite="Maultaschen Rezept", answer="never used")
    a = pipeline(retriever, llm, min_rerank_score=50.0).ask("Wie koche ich Maultaschen?")
    assert not a.covered and a.reason == "low_score" and "answer" not in llm.kinds()


def test_answer_failure_surfaces_as_generation_error(retriever):
    llm = ScriptedLLM(rewrite="x", answer_error=GenerationError("timeout"))
    with pytest.raises(GenerationError):
        pipeline(retriever, llm).ask("Anmeldung")


def test_history_reaches_both_llm_calls(retriever):
    llm = ScriptedLLM(rewrite="Rundfunkbeitrag Hoehe", answer="18,36 Euro [1].")
    history = [{"role": "user", "content": "Was ist der Rundfunkbeitrag?"},
               {"role": "assistant", "content": "Eine Abgabe."}]
    pipeline(retriever, llm).ask("Und wie hoch?", history)
    rewrite_msgs = llm.calls[0][1]
    answer_msgs = llm.calls[1][1]
    assert [m["role"] for m in rewrite_msgs] == ["user", "assistant", "user"]
    assert [m["role"] for m in answer_msgs] == ["user", "assistant", "user"]


def test_stream_yields_text_then_the_final_answer_with_timings(retriever):
    llm = ScriptedLLM(rewrite="Anmeldung Frist", pieces=["14 Tage ", "Zeit [1]."])
    events = list(pipeline(retriever, llm).stream("Anmeldung?"))
    assert isinstance(events[-1], Answer) and all(isinstance(e, Delta) for e in events[:-1])
    assert events[-1].text == "14 Tage Zeit [1]." and "llm_ms" in events[-1].timings


def test_search_filters_by_source(retriever):
    hits, _ = search(retriever, "Beitrag", "Beitrag", source="rundfunk.txt", k=2)
    assert hits and {h.chunk.source for h in hits} == {"rundfunk.txt"}


def test_merging_original_adds_candidates_a_bad_rewrite_would_miss(retriever):
    both, _ = search(retriever, "Rundfunkbeitrag Euro", "Anmeldung Einzug", k=2)
    only, _ = search(retriever, "Rundfunkbeitrag Euro", "Anmeldung Einzug", k=2, merge_original=False)
    assert "rundfunk.txt" in {h.chunk.source for h in both}
    assert "rundfunk.txt" not in {h.chunk.source for h in only}


# ── terminal output and evaluation helpers ───────────────────────────────

def test_terminal_formatting(retriever):
    a = pipeline(retriever, ScriptedLLM(rewrite="Anmeldung", answer="14 Tage [1].")).ask("Anmeldung")
    assert "[1] anmeldung.txt, S. 1" in format_sources(a)
    assert "search query: 'Anmeldung'" in format_debug(a) and "total" in format_debug(a)


def test_rewrites_are_cached_between_runs(tmp_path):
    q = [Question("a", "factoid_de", "en", "How high is the deposit?", True, (Evidence("m.txt", "x"),))]
    llm = ScriptedLLM(rewrite="Mietsicherheit Hoechstbetrag")
    path = tmp_path / "r.json"
    assert rewrite_questions(llm, "m1", q, path) == {"a": "Mietsicherheit Hoechstbetrag"}
    assert rewrite_questions(llm, "m1", q, path) == {"a": "Mietsicherheit Hoechstbetrag"}
    assert llm.kinds() == ["rewrite"]  # the second call came from the cache


def test_evaluation_can_search_with_rewritten_queries(retriever):
    q = Question("m", "english", "en", "How high can the deposit be?", True, (Evidence("miete.txt", "drei Monatsmieten"),))
    plain = run(retriever, [q], "dense")[0]
    better = run(retriever, [q], "dense", queries={"m": "Mietsicherheit drei Monatsmieten hoechstens"})[0]
    assert better.rank == 1 and (plain.rank is None or plain.rank >= better.rank)


# ── understanding the message: follow-ups, keywords, small talk, language ─

def plan_json(**kw):
    import json

    return json.dumps({"intent": "search", "language": "de", "question": "Wie hoch ist die Mietsicherheit?",
                       "search_query": "Wie hoch darf die Mietsicherheit höchstens sein?", **kw}, ensure_ascii=False)


def test_analysis_reads_a_json_plan():
    from beamter.generation import analyze_query

    llm = ScriptedLLM(rewrite=plan_json(language="en", question="What is the maximum deposit?"))
    p = analyze_query(llm, "deposit")
    assert (p.intent, p.language, p.question) == ("search", "en", "What is the maximum deposit?")
    assert p.search_query.startswith("Wie hoch darf die Mietsicherheit")


def test_analysis_falls_back_when_the_reply_is_unusable():
    from beamter.generation import analyze_query

    assert analyze_query(ScriptedLLM(rewrite="Mietsicherheit Hoechstbetrag"), "How high can the deposit be?").search_query == "Mietsicherheit Hoechstbetrag"
    for reply in ("", "{not json", "{\"intent\": \"search\"}", "explains\nover two lines"):
        p = analyze_query(ScriptedLLM(rewrite=reply), "How high can the deposit be?")
        assert p.question == p.search_query == "How high can the deposit be?" and p.language == "en"
    p = analyze_query(ScriptedLLM(rewrite_error=GenerationError("down")), "Wie hoch ist die Kaution?")
    assert p.search_query == "Wie hoch ist die Kaution?" and p.language == "de"


def test_analysis_ignores_made_up_values():
    from beamter.generation import analyze_query

    p = analyze_query(ScriptedLLM(rewrite=plan_json(intent="delete everything", language="fr")), "Wie hoch ist die Kaution?")
    assert p.intent == "search" and p.language == "de"  # unknown language -> guessed from the question


def test_follow_up_that_asks_for_another_language_answers_the_previous_question(retriever):
    history = [{"role": "user", "content": "How high can the deposit be?"},
               {"role": "assistant", "content": "At most three months' rent [1]."}]
    llm = ScriptedLLM(rewrite=plan_json(language="de", question="Wie hoch darf die Mietsicherheit höchstens sein?"),
                      answer="Höchstens drei Monatsmieten [1].")
    a = pipeline(retriever, llm).ask("tell it in german", history)
    final = llm.calls[-1][1][-1]["content"]
    assert "Question: Wie hoch darf die Mietsicherheit höchstens sein?" in final  # not "tell it in german"
    assert "Antworte auf Deutsch" in final and "tell it in german" not in final
    assert a.covered and a.interpreted == "Wie hoch darf die Mietsicherheit höchstens sein?"


def test_stated_language_beats_the_guess_from_the_wording(retriever):
    llm = ScriptedLLM(rewrite=plan_json(language="en", question="Wie hoch ist die Mietsicherheit?"), answer="Three months [1].")
    pipeline(retriever, llm).ask("Wie hoch ist die Mietsicherheit? auf Englisch")
    assert "Respond in English" in llm.calls[-1][1][-1]["content"]


def test_greeting_gets_a_fixed_reply_without_search_or_answer_call(retriever):
    llm = ScriptedLLM(rewrite=plan_json(intent="chat", language="en", question="Hello"), answer="never used")
    a = pipeline(retriever, llm).ask("hi there")
    assert llm.kinds() == ["rewrite"] and a.reason == "chat" and not a.covered
    assert "Hello!" in a.text and a.citations == [] and "dense_ms" not in a.timings
    events = list(pipeline(retriever, ScriptedLLM(rewrite=plan_json(intent="chat", language="de", question="Hallo"))).stream("hallo"))
    assert events[0].text.startswith("Hallo!") and events[-1].reason == "chat"


def test_answers_report_confidence_and_how_the_question_was_understood(retriever):
    llm = ScriptedLLM(rewrite=plan_json(question="Wie lange habe ich für die Anmeldung?"), answer="14 Tage [1].")
    a = pipeline(retriever, llm).ask("anmeldung")
    assert a.interpreted == "Wie lange habe ich für die Anmeldung?" and isinstance(a.top_score, float)


def test_grey_zone_scores_reach_the_model_which_may_abstain(retriever):
    """Short keyword queries score lower than full questions. Only clearly unrelated input is
    stopped by the score; between the two limits the model decides."""
    llm = ScriptedLLM(rewrite=plan_json(), answer="NOT_COVERED")
    scores = lambda q, texts: [-1.5] * len(texts)  # noqa: E731
    retriever.reranker = type("R", (), {"score": staticmethod(scores)})()
    a = pipeline(retriever, llm, min_rerank_score=-2.5).ask("blocked account")
    assert "answer" in llm.kinds() and a.reason == "model"  # the model was asked and declined
    b = pipeline(retriever, ScriptedLLM(rewrite=plan_json(), answer="x"), min_rerank_score=-1.0).ask("blocked account")
    assert b.reason == "low_score"
