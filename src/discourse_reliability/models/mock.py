"""Offline mock adapter -- for pipeline tests ONLY.

It is a keyword heuristic, not a language model.  Any number produced with
``--model mock`` is a test artefact and must never enter a paper.
"""

from __future__ import annotations

import hashlib
import random
import json
import re
import time
from typing import Sequence

from .base import Generation

_CONN = re.compile(r"连接词「(.+?)」")
_FALLBACKS = [
    "EXPANSION:Conjunction",
    "CONTINGENCY:Cause:Result",
    "COMPARISON:Contrast",
    "TEMPORAL:Asynchronous:Precedence",
]

_LEXICON = {
    "但是": "COMPARISON:Contrast",
    "然而": "COMPARISON:Contrast",
    "不过": "COMPARISON:Contrast",
    "可是": "COMPARISON:Contrast",
    "虽然": "COMPARISON:Concession:Arg1-as-denier",
    "尽管": "COMPARISON:Concession:Arg1-as-denier",
    "因为": "CONTINGENCY:Cause:Reason",
    "由于": "CONTINGENCY:Cause:Reason",
    "所以": "CONTINGENCY:Cause:Result",
    "因此": "CONTINGENCY:Cause:Result",
    "如果": "CONTINGENCY:Condition:Arg1-as-cond",
    "只要": "CONTINGENCY:Condition:Arg1-as-cond",
    "只有": "CONTINGENCY:Condition:Arg2-as-cond",
    "然后": "TEMPORAL:Asynchronous:Succession",
    "接着": "TEMPORAL:Asynchronous:Succession",
    "同时": "TEMPORAL:Synchronous",
    "而且": "EXPANSION:Conjunction",
    "并且": "EXPANSION:Conjunction",
    "例如": "EXPANSION:Instantiation",
    "也就是说": "EXPANSION:Restatement",
    "或者": "EXPANSION:Alternative",
}

_IMPLICIT_CUES = [
    ("因为", "CONTINGENCY:Cause:Reason"),
    ("所以", "CONTINGENCY:Cause:Result"),
    ("如果", "CONTINGENCY:Condition:Arg1-as-cond"),
    ("但是", "COMPARISON:Contrast"),
    ("却", "COMPARISON:Contrast"),
    ("虽然", "COMPARISON:Concession:Arg1-as-denier"),
    ("例如", "EXPANSION:Instantiation"),
]


class MockAdapter:
    name = "mock"

    def __init__(self, error_rate: float = 0.25, seed: int = 0) -> None:
        self.error_rate = error_rate
        self.seed = seed

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> list[Generation]:
        t0 = time.perf_counter()
        prompt = messages[-1]["content"] if messages else ""
        m = _CONN.search(prompt)
        if m:
            sense = _LEXICON.get(m.group(1))
            if sense is None:
                sense = next((s for c, s in _LEXICON.items() if c in prompt), _FALLBACKS[0])
        else:
            joined = prompt
            sense = next((s for c, s in _IMPLICIT_CUES if c in joined), _FALLBACKS[0])

        h = int(hashlib.sha256(f"{self.seed}|{prompt}".encode()).hexdigest(), 16)
        if (h % 1000) / 1000 < self.error_rate:
            sense = _FALLBACKS[h % len(_FALLBACKS)]

        out = []
        for i in range(n):
            s = sense
            if temperature > 0:
                # Simulated sampling variance so the confidence path is exercised.
                # This is mock noise, not model behaviour.
                r = random.Random(f"{self.seed}|{prompt}|{i}|{temperature}")
                if r.random() < 0.35:
                    s = r.choice(_FALLBACKS)
            out.append(
                Generation(
                    text=json.dumps({"sense": s}, ensure_ascii=False),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            )
        return out
