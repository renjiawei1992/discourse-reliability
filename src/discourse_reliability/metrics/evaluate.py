"""Evaluation: per-level agreement, strata, and bootstrap CIs."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from ..schema import Item, Prediction
from .agreement import accuracy, cohen_kappa, confusion, macro_f1
from .bootstrap import bootstrap_ci

LEVEL_ATTR = {"l1": "sense_l1", "l2": "sense_l2", "l3": "sense_l3"}
PRED_ATTR = {"l1": "label_l1", "l2": "label_l2", "l3": "label_l3"}


def _pair(gold_items: Sequence[Item], preds: Sequence[Prediction], level: str):
    by_id = {p.item_id: p for p in preds if p.error is None and getattr(p, PRED_ATTR[level])}
    pairs = []
    for it in gold_items:
        g = getattr(it, LEVEL_ATTR[level])
        p = by_id.get(it.id)
        if g and p is not None:
            pairs.append((it, g, getattr(p, PRED_ATTR[level])))
    return pairs


def _summarise(gold: list[str], pred: list[str], n_boot: int, seed: int) -> dict:
    def acc_metric(idx):
        return accuracy([gold[i] for i in idx], [pred[i] for i in idx])

    def kap_metric(idx):
        k = cohen_kappa([gold[i] for i in idx], [pred[i] for i in idx])
        return float("nan") if k is None else k

    ap, alo, ahi = bootstrap_ci(len(gold), acc_metric, n_boot=n_boot, seed=seed)
    kp, klo, khi = bootstrap_ci(len(gold), kap_metric, n_boot=n_boot, seed=seed + 1)
    return {
        "n": len(gold),
        "accuracy": ap,
        "accuracy_ci": [alo, ahi],
        "kappa": None if kp != kp else kp,
        "kappa_ci": [None if x != x else x for x in (klo, khi)],
        "macro_f1": macro_f1(gold, pred),
    }


def evaluate(
    gold_items: Sequence[Item],
    preds: Sequence[Prediction],
    groups: dict[str, Sequence[str]] | None = None,
    levels: tuple[str, ...] = ("l1", "l2", "l3"),
    n_boot: int = 1000,
    seed: int = 13,
) -> dict:
    """``groups`` maps a model/regime label to the prediction set to score."""
    out: dict = {"levels": {}, "strata": {}, "coverage": {}}
    for level in levels:
        pairs = _pair(gold_items, preds, level)
        gold = [g for _, g, _ in pairs]
        pred = [p for _, _, p in pairs]
        out["levels"][level] = _summarise(gold, pred, n_boot, seed) if pairs else {"n": 0}
        out["coverage"][level] = round(len(pairs) / len(gold_items), 4) if gold_items else 0.0

        # Per-stratum, using the same pairs.
        strata: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for it, g, p in pairs:
            for axis, val in it.stratum().items():
                strata[f"{axis}={val}"].append((g, p))
        out["strata"][level] = {
            key: _summarise([g for g, _ in vals], [p for _, p in vals], n_boot, seed)
            for key, vals in sorted(strata.items())
        }

    if levels:
        lvl = "l3" if any(_pair(gold_items, preds, "l3")) else levels[-1]
        pairs = _pair(gold_items, preds, lvl)
        out["confusion"] = confusion([g for _, g, _ in pairs], [p for _, _, p in pairs])
    out["n_gold"] = len(gold_items)
    out["n_predictions"] = len(preds)
    return out


def evaluate_by_model(
    gold_items: Sequence[Item],
    preds: Sequence[Prediction],
    n_boot: int = 1000,
    seed: int = 13,
    levels: tuple[str, ...] = ("l1", "l2", "l3"),
) -> dict:
    """Score every ``(model, regime)`` combination found in ``preds``."""
    buckets: dict[tuple[str, str], list[Prediction]] = defaultdict(list)
    for p in preds:
        buckets[(p.model, p.regime)].append(p)
    out: dict[str, dict] = {}
    for (model, regime), rows in sorted(buckets.items()):
        out[f"{model}/{regime}"] = evaluate(
            gold_items, rows, levels=levels, n_boot=n_boot, seed=seed
        )
    return out
