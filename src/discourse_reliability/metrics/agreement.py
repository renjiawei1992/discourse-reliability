"""Agreement and classification metrics (no third-party dependencies)."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    if not gold:
        return float("nan")
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa for two raters.

    Returns ``None`` when the expected agreement is 1 (undefined), which
    happens when both raters used a single category throughout.
    """
    if len(a) != len(b) or not a:
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if abs(1 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def fleiss_kappa(table: Sequence[Sequence[int]]) -> float | None:
    """Fleiss' kappa.  ``table`` rows are items, columns are categories."""
    rows = [r for r in table if sum(r) > 0]
    if not rows:
        return None
    n_items = len(rows)
    n_raters = sum(rows[0])
    if any(sum(r) != n_raters for r in rows) or n_raters < 2:
        raise ValueError("all rows must have the same, >=2 number of ratings")
    p_i = [(sum(x * x for x in r) - n_raters) / (n_raters * (n_raters - 1)) for r in rows]
    p_bar = sum(p_i) / n_items
    col = [sum(r[j] for r in rows) / (n_items * n_raters) for j in range(len(rows[0]))]
    p_e = sum(x * x for x in col)
    if abs(1 - p_e) < 1e-12:
        return None
    return (p_bar - p_e) / (1 - p_e)


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    labels = sorted(set(gold))
    if not labels:
        return float("nan")
    scores = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        if tp == 0:
            scores.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        scores.append(2 * prec * rec / (prec + rec))
    return sum(scores) / len(scores)


def confusion(gold: Sequence[str], pred: Sequence[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for g, p in zip(gold, pred):
        out.setdefault(g, {}).setdefault(p, 0)
        out[g][p] += 1
    return out
