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

    def _prepare(self, question: str, history: list[dict] | None, source: str | None):
        t0 = time.perf_counter()
        if self.settings.rewrite_queries:
            plan = analyze_query(self.llm, question, history)
        else:
            plan = QueryPlan(question, question, language=detect_language(question))
        analyze_ms = _ms(t0)
        if plan.intent == "chat":
            return plan, [], {"rewrite_ms": analyze_ms}
        hits, timings = search(self.retriever, question, plan.search_query, source=source)
        return plan, hits, {**timings, "rewrite_ms": analyze_ms}

    def ask(self, question: str, history: list[dict] | None = None, source: str | None = None) -> Answer:
        return self.ask_with_hits(question, history, source)[0]

    def ask_with_hits(
        self, question: str, history: list[dict] | None = None, source: str | None = None
    ) -> tuple[Answer, list[Hit]]:
        """Like ``ask``, but also returns the passages the model was given (used by the evaluation)."""
        question = question.strip()
        plan, hits, timings = self._prepare(question, history, source)
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
        """Text pieces as they arrive, then the final checked ``Answer``."""
        question = question.strip()
        plan, hits, timings = self._prepare(question, history, source)
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
