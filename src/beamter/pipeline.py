"""The whole question-to-answer path, in one place.

    question (+ chat history)
      -> understand it: standalone question, German search query, answer language,
         small talk?                                 (optional, one LLM call)
      -> retrieve: vector + BM25 + cross-encoder rerank
      -> decide: is the best passage good enough?   (no LLM call if not)
      -> answer with checked citations              (LLM)

The API and the terminal chat both call ``Pipeline`` and nothing else, so behaviour and
timings are identical everywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from .cache import AnswerCache, normalize
from .config import Settings
from .generation import (
    LLM, Answer, Delta, GenerationError, QueryPlan, analyze_query, chat_answer, detect_language,
    generate_answer, stream_answer,
)
from .retrieval import Hit, Retriever


def merge_hits(lists: list[list[Hit]], k: int) -> list[Hit]:
    """Union of several result lists, de-duplicated by chunk. A chunk found by more than one
    query keeps its best score. Rerank scores are on one scale, so they sort across queries."""
    best: dict[int, Hit] = {}
    for hits in lists:
        for h in hits:
            old = best.get(h.chunk.id)
            if old is None or _key(h) > _key(old):
                best[h.chunk.id] = h
    return sorted(best.values(), key=_key, reverse=True)[:k]


def _key(h: Hit) -> float:
    return h.rerank_score if h.rerank_score is not None else h.score


def search(
    retriever: Retriever,
    question: str,
    search_query: str,
    *,
    mode: str | None = None,
    k: int | None = None,
    source: str | None = None,
    merge_original: bool = True,
) -> tuple[list[Hit], dict[str, float]]:
    """Retrieve for the rewritten query and, when it differs, also for the original question,
    then merge. A bad rewrite can then only add candidates, never remove the original ones."""
    k = k or retriever.s.top_k
    hits, timings = retriever.retrieve_traced(search_query, mode, k, source)
    if merge_original and search_query != question:
        other, t2 = retriever.retrieve_traced(question, mode, k, source)
        hits = merge_hits([hits, other], k)
        timings = {name: round(timings.get(name, 0) + t2.get(name, 0), 1) for name in {*timings, *t2}}
    return hits, timings


@dataclass
class Pipeline:
    settings: Settings
    retriever: Retriever
    llm: LLM
    cache: AnswerCache | None = None

    def __post_init__(self):
        if self.cache is None:
            self.cache = AnswerCache(self.settings.cache_max_entries, self.settings.cache_ttl_s)

    # ── the steps ────────────────────────────────────────────────────────

    def _plan(self, question: str, history: list[dict] | None) -> tuple[QueryPlan, float]:
        t0 = time.perf_counter()
        if self.settings.rewrite_queries:
            plan = analyze_query(self.llm, question, history)
        else:
            plan = QueryPlan(question, question, language=detect_language(question))
        return plan, _ms(t0)

    def _prepare(self, question: str, history: list[dict] | None, source: str | None,
                 plan: QueryPlan | None = None, analyze_ms: float = 0.0):
        if plan is None:
            plan, analyze_ms = self._plan(question, history)
        if plan.intent == "chat":
            return plan, [], {"rewrite_ms": analyze_ms}
        hits, timings = search(self.retriever, question, plan.search_query, source=source)
        return plan, hits, {**timings, "rewrite_ms": analyze_ms}

    # ── the cache ────────────────────────────────────────────────────────
    # Two lookups. 1) The raw question of a new conversation: skips even the rewrite call.
    # 2) The standalone question the rewrite produced: skips search, rerank and the answer call,
    # and also catches different wordings of the same question and follow-ups. The index name is
    # part of every key, so a new index never serves answers built from the old one.

    def _raw_key(self, question: str, source: str | None):
        return (self.retriever.kb.collection, source or "", "raw", normalize(question))

    def _plan_key(self, plan: QueryPlan, source: str | None):
        return (self.retriever.kb.collection, source or "", plan.language, normalize(plan.question))

    def _served(self, answer: Answer, started: float, analyze_ms: float | None = None) -> Answer:
        timings = {"cache_ms": _ms(started)}
        if analyze_ms is not None:
            timings["rewrite_ms"] = analyze_ms
        answer.cached = True
        answer.timings = {**timings, "total_ms": round(sum(timings.values()), 1)}
        return answer

    def _lookup_raw(self, question: str, history, source) -> Answer | None:
        if history:  # a follow-up depends on the conversation, so only the second lookup applies
            return None
        t0 = time.perf_counter()
        hit = self.cache.get(self._raw_key(question, source))
        return self._served(hit, t0) if hit else None

    def _lookup_plan(self, plan: QueryPlan, source, analyze_ms: float) -> Answer | None:
        if plan.intent == "chat":
            return None
        t0 = time.perf_counter()
        hit = self.cache.get(self._plan_key(plan, source))
        return self._served(hit, t0, analyze_ms) if hit else None

    def _remember(self, question: str, history, source, plan: QueryPlan, answer: Answer) -> None:
        if plan.intent == "chat" or answer.cached:
            return
        self.cache.put(self._plan_key(plan, source), answer)
        if not history:
            self.cache.put(self._raw_key(question, source), answer)

    # ── the public entry points ──────────────────────────────────────────

    def ask(self, question: str, history: list[dict] | None = None, source: str | None = None) -> Answer:
        question = question.strip()
        hit = self._lookup_raw(question, history, source)
        if hit:
            return hit
        plan, analyze_ms = self._plan(question, history)
        hit = self._lookup_plan(plan, source, analyze_ms)
        if hit:
            return hit
        answer, _ = self._answer(question, history, source, plan, analyze_ms)
        self._remember(question, history, source, plan, answer)
        return answer

    def ask_with_hits(
        self, question: str, history: list[dict] | None = None, source: str | None = None
    ) -> tuple[Answer, list[Hit]]:
        """Like ``ask``, but never cached and also returns the passages the model was given
        (used by the evaluation, which must measure the real pipeline every time)."""
        question = question.strip()
        plan, analyze_ms = self._plan(question, history)
        return self._answer(question, history, source, plan, analyze_ms)

    def _answer(self, question, history, source, plan: QueryPlan, analyze_ms: float) -> tuple[Answer, list[Hit]]:
        plan, hits, timings = self._prepare(question, history, source, plan, analyze_ms)
        if plan.intent == "chat":
            answer = chat_answer(plan)
            answer.timings = {**timings, "total_ms": round(sum(timings.values()), 1)}
            return answer, []
        t0 = time.perf_counter()
        answer = generate_answer(
            self.llm, self.settings, plan.question, hits, history, plan.search_query, plan.language
        )
        timings["llm_ms"] = _ms(t0)
        answer.timings = {**timings, "total_ms": round(sum(timings.values()), 1)}
        return answer, hits

    def stream(
        self, question: str, history: list[dict] | None = None, source: str | None = None
    ) -> Iterator[Delta | Answer]:
        """Text pieces as they arrive, then the final checked ``Answer``. A cached answer is
        replayed as one piece followed by the same final event."""
        question = question.strip()
        hit = self._lookup_raw(question, history, source)
        if hit:
            yield Delta(hit.text)
            yield hit
            return
        plan, analyze_ms = self._plan(question, history)
        hit = self._lookup_plan(plan, source, analyze_ms)
        if hit:
            yield Delta(hit.text)
            yield hit
            return
        plan, hits, timings = self._prepare(question, history, source, plan, analyze_ms)
        if plan.intent == "chat":
            answer = chat_answer(plan)
            answer.timings = {**timings, "total_ms": round(sum(timings.values()), 1)}
            yield Delta(answer.text)
            yield answer
            return
        t0 = time.perf_counter()
        for event in stream_answer(
            self.llm, self.settings, plan.question, hits, history, plan.search_query, plan.language
        ):
            if isinstance(event, Answer):
                timings["llm_ms"] = _ms(t0)
                event.timings = {**timings, "total_ms": round(sum(timings.values()), 1)}
                self._remember(question, history, source, plan, event)
            yield event


def build_pipeline(settings: Settings) -> Pipeline:
    """Connect to Qdrant, load the models and the index. Slow; call once at startup."""
    from .generation import AnthropicLLM
    from .index import KnowledgeBase, SentenceTransformerEmbedder, make_client

    settings.validate()
    kb = KnowledgeBase(settings, SentenceTransformerEmbedder(settings.embed_model), make_client(settings))
    kb.load()
    retriever = Retriever(kb, settings)
    retriever.warm_up()
    return Pipeline(settings, retriever, AnthropicLLM(settings))


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
