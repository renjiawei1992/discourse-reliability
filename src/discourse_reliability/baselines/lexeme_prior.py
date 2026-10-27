"""Lexeme-prior baseline.

Answers a single question: given the connective's identity but NO context,
how well can you do?  The gap between a model and this baseline is the
"contextual gain" used in Block 3.

The prior is estimated out-of-fold so a test item's own label never leaks
into its own prediction.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Sequence

from ..labels import sense_levels
from ..schema import Item, Prediction

LEVEL_ATTR = {"l1": "sense_l1", "l2": "sense_l2", "l3": "sense_l3"}


def _fold(item_id: str, k: int) -> int:
    return int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % k


def build_prior(items: Sequence[Item], level: str = "l3") -> tuple[dict[str, Counter], Counter]:
    per_conn: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    attr = LEVEL_ATTR[level]
    for it in items:
        lab = getattr(it, attr) or it.gold
        if not lab:
            continue
        per_conn[it.connective or "<implicit>"][lab] += 1
        global_counts[lab] += 1
    return per_conn, global_counts


def prior_distribution(connective: str, per_conn, global_counts: Counter, alpha: float = 1.0) -> dict[str, float]:
    """Smoothed P(sense | connective), backing off to the global distribution."""
    counts = per_conn.get(connective or "<implicit>", Counter())
    labels = set(global_counts) | set(counts)
    if not labels:
        return {}
    total = sum(counts.values())
    gtotal = sum(global_counts.values())
    probs = {
        lab: (counts.get(lab, 0) + alpha * (global_counts.get(lab, 0) / gtotal))
        / (total + alpha)
        for lab in labels
    }
    return probs


def entropy(probs: dict[str, float]) -> float:
    return -sum(p * math.log(p + 1e-12) for p in probs.values() if p > 0)


def prior_predictions(items: Sequence[Item], k: int = 5, level: str = "l3") -> list[Prediction]:
    """Out-of-fold prior predictions for every item."""
    items = list(items)
    preds: list[Prediction] = []
    for fold in range(k):
        train = [it for it in items if _fold(it.id, k) != fold]
        test = [it for it in items if _fold(it.id, k) == fold]
        per_conn, global_counts = build_prior(train, level=level)
        for it in test:
            probs = prior_distribution(it.connective, per_conn, global_counts)
            if not probs:
                preds.append(Prediction(item_id=it.id, model="lexeme_prior", regime=f"{k}fold", error="no prior"))
                continue
            top = max(probs, key=probs.get)
            l1, l2, l3 = sense_levels(top)
            preds.append(
                Prediction(
                    item_id=it.id,
                    model="lexeme_prior",
                    regime=f"{k}fold",
                    label_l1=l1,
                    label_l2=l2,
                    label_l3=l3,
                    confidence=probs[top],
                    meta={"entropy": entropy(probs), "fold": fold},
                )
            )
    preds.sort(key=lambda p: p.item_id)
    return preds


def attach_prior_entropy(
    items: Sequence[Item], k: int = 5, level: str = "l3"
) -> dict[str, float]:
    """Out-of-fold prior entropy per item id -- used as a triage feature."""
    items = list(items)
    out: dict[str, float] = {}
    for fold in range(k):
        train = [it for it in items if _fold(it.id, k) != fold]
        test = [it for it in items if _fold(it.id, k) == fold]
        per_conn, global_counts = build_prior(train, level=level)
        for it in test:
            out[it.id] = entropy(prior_distribution(it.connective, per_conn, global_counts))
    return out
