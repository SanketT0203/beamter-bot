"""A small in-memory cache of finished answers.

Why: answering costs one cross-encoder pass over the candidate passages plus two LLM calls, about
10 seconds on a small CPU instance. Asking the same thing twice should not pay that twice.

What is stored: only answers that are covered and grounded. A refusal is never stored, because a
refusal can be a retrieval miss (the same question once failed and then worked after a different
rewrite), and storing it would pin the miss for every later visitor.

The key contains the index name, so a re-ingest (new collection) makes every old entry unreachable.
Entries expire after a time to live and the least recently used one is dropped when the cache is
full. The cache lives in one instance's memory: it is lost when the instance scales to zero and not
shared between instances. That is deliberate (no extra service to run); a shared store such as
Redis would be the production step.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from typing import Hashable

from .generation import Answer


def normalize(text: str) -> str:
    """Case, spacing and trailing punctuation do not make a different question."""
    return " ".join(text.casefold().split()).rstrip("?!. ")


class AnswerCache:
    def __init__(self, max_entries: int, ttl_s: float, clock=time.monotonic):
        self.max_entries = max(0, max_entries)
        self.ttl_s = ttl_s
        self._clock = clock
        self._items: OrderedDict[Hashable, tuple[float, Answer]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_s > 0

    def get(self, key: Hashable) -> Answer | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._items.get(key)
            if entry is None or self._clock() - entry[0] > self.ttl_s:
                self._items.pop(key, None)
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(entry[1])  # callers may change the copy (timings) freely

    def put(self, key: Hashable, answer: Answer) -> None:
        """Store an answer. Only a covered, grounded answer is worth keeping."""
        if not self.enabled or not (answer.covered and answer.grounded):
            return
        with self._lock:
            self._items[key] = (self._clock(), copy.deepcopy(answer))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)

    def stats(self) -> dict:
        return {"entries": len(self._items), "hits": self.hits, "misses": self.misses}
