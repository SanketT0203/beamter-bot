import dataclasses

import pytest
from test_pipeline import PAGES, ScriptedLLM, pipeline, retriever  # noqa: F401  (fixtures and fakes)

from beamter.cache import AnswerCache, normalize
from beamter.generation import Answer


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def answer(text="ok", **kw):
    return Answer(text, **kw)


# ── the cache on its own ─────────────────────────────────────────────────

def test_normalize_ignores_case_spacing_and_trailing_punctuation():
    assert normalize("  Wie  melde ich mich AN?? ") == normalize("wie melde ich mich an")
    assert normalize("Kindergeld") != normalize("Kinderzuschlag")


def test_hit_returns_an_independent_copy():
    c = AnswerCache(4, 60)
    c.put("k", answer("a"))
    first = c.get("k")
    first.text = "changed"
    assert c.get("k").text == "a"
    assert c.stats()["hits"] == 2


def test_only_covered_grounded_answers_are_stored():
    c = AnswerCache(4, 60)
    c.put("declined", answer(covered=False))
    c.put("ungrounded", answer(grounded=False))
    assert c.get("declined") is None and c.get("ungrounded") is None and len(c) == 0


def test_entries_expire_after_the_ttl():
    clock = Clock()
    c = AnswerCache(4, 60, clock=clock)
    c.put("k", answer())
    clock.now = 59
    assert c.get("k") is not None
    clock.now = 61
    assert c.get("k") is None and len(c) == 0


def test_least_recently_used_entry_is_dropped():
    c = AnswerCache(2, 60)
    c.put("a", answer("a"))
    c.put("b", answer("b"))
    c.get("a")  # a is now the most recent
    c.put("c", answer("c"))
    assert c.get("b") is None and c.get("a") is not None and c.get("c") is not None


@pytest.mark.parametrize("max_entries, ttl", [(0, 60), (4, 0)])
def test_switched_off(max_entries, ttl):
    c = AnswerCache(max_entries, ttl)
    c.put("k", answer())
    assert not c.enabled and c.get("k") is None


# ── in the pipeline ──────────────────────────────────────────────────────

def test_same_question_twice_costs_one_set_of_llm_calls(retriever):  # noqa: F811
    llm = ScriptedLLM(rewrite="Mietsicherheit Hoechstbetrag", answer="Drei Monatsmieten [1].")
    p = pipeline(retriever, llm)
    first = p.ask("How high can the deposit be?")
    second = p.ask("how high can the deposit be?")
    assert llm.kinds() == ["rewrite", "answer"]  # the second call used nothing
    assert not first.cached and second.cached
    assert second.text == first.text and second.citations == first.citations
    assert set(second.timings) == {"cache_ms", "total_ms"}


def test_a_different_wording_with_the_same_rewrite_skips_search_and_answer(retriever):  # noqa: F811
    plan = ('{"intent": "search", "language": "en", "question": "How high can the rental deposit be?", '
            '"search_query": "Wie hoch darf die Mietsicherheit sein?"}')
    llm = ScriptedLLM(rewrite=plan, answer="Drei Monatsmieten [1].")
    p = pipeline(retriever, llm)
    p.ask("How high can the deposit be?")
    again = p.ask("What is the maximum rental deposit?")
    assert llm.kinds() == ["rewrite", "answer", "rewrite"]  # rewrite ran, answer did not
    assert again.cached and "rewrite_ms" in again.timings


def test_refusals_are_never_cached(retriever):  # noqa: F811
    llm = ScriptedLLM(rewrite="Maultaschen Rezept", answer="never used")
    p = pipeline(retriever, llm, min_rerank_score=50.0)
    assert not p.ask("Wie koche ich Maultaschen?").covered
    assert not p.ask("Wie koche ich Maultaschen?").covered
    assert llm.kinds().count("rewrite") == 2  # asked again from scratch
    assert len(p.cache) == 0


def test_not_covered_reply_from_the_model_is_not_cached(retriever):  # noqa: F811
    llm = ScriptedLLM(rewrite="Mietsicherheit", answer="NOT_COVERED")
    p = pipeline(retriever, llm)
    p.ask("deposit")
    p.ask("deposit")
    assert llm.kinds().count("answer") == 2 and len(p.cache) == 0


def test_document_filter_is_part_of_the_key(retriever):  # noqa: F811
    llm = ScriptedLLM(rewrite="Mietsicherheit", answer="Drei Monatsmieten [1].")
    p = pipeline(retriever, llm)
    p.ask("deposit", source="miete.txt")
    other = p.ask("deposit", source=None)
    assert not other.cached


def test_follow_up_with_history_does_not_use_the_raw_lookup_or_write_it(retriever):  # noqa: F811
    llm = ScriptedLLM(rewrite="Mietsicherheit Hoechstbetrag", answer="Drei Monatsmieten [1].")
    p = pipeline(retriever, llm)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    p.ask("and the deposit?", history)
    assert len(p.cache) == 1  # only the standalone-question entry
    assert not p.ask("and the deposit?", history).cached or llm.kinds().count("rewrite") == 2


def test_a_new_index_never_serves_old_answers(retriever):  # noqa: F811
    llm = ScriptedLLM(rewrite="Mietsicherheit", answer="Drei Monatsmieten [1].")
    p = pipeline(retriever, llm)
    p.ask("deposit")
    assert p._lookup_raw("deposit", None, None) is not None
    p.retriever.kb.collection = "p__newer"  # what a reload to a new ingest looks like
    assert p._lookup_raw("deposit", None, None) is None


def test_evaluation_path_bypasses_the_cache(retriever):  # noqa: F811
    llm = ScriptedLLM(rewrite="Mietsicherheit", answer="Drei Monatsmieten [1].")
    p = pipeline(retriever, llm)
    p.ask("deposit")
    a, _ = p.ask_with_hits("deposit")
    assert not a.cached and llm.kinds().count("answer") == 2


def test_stream_replays_a_cached_answer(retriever):  # noqa: F811
    llm = ScriptedLLM(rewrite="Mietsicherheit", answer="Drei Monatsmieten [1].", pieces=["Drei Monats", "mieten [1]."])
    p = pipeline(retriever, llm)
    first = list(p.stream("deposit"))
    second = list(p.stream("deposit"))
    assert first[-1].cached is False and second[-1].cached is True
    assert second[0].text == second[-1].text and len(second) == 2
    assert second[-1].text == first[-1].text


def test_cache_can_be_switched_off(retriever):  # noqa: F811
    llm = ScriptedLLM(rewrite="Mietsicherheit", answer="Drei Monatsmieten [1].")
    p = pipeline(retriever, llm, cache_max_entries=0)
    p.ask("deposit")
    assert not p.ask("deposit").cached and llm.kinds().count("answer") == 2
