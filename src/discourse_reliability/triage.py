"""Human-review triage: score items, simulate budgets, compare policies.

Quality at budget b is computed the way an annotation workflow would work:
the top-b share of items is reviewed and replaced by the gold label, the rest
keeps the model's label.  Kappa is then recomputed against gold.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np

from .labels import sense_levels
from .metrics.agreement import accuracy, cohen_kappa
from .schema import Item, Prediction

LEVEL = "l3"


def _item_features(it: Item, pred: Prediction, prior_entropy: float, consensus_entropy: float) -> list[float]:
    return [
        float(pred.confidence if pred.confidence is not None else 1.0),
        prior_entropy,
        consensus_entropy,
        float(it.n_lexicon_senses),
        1.0 if it.explicit else 0.0,
        1.0 if it.paired else 0.0,
        1.0 if it.cross_sentence else 0.0,
        min(len(it.text), 300) / 300.0,
    ]


def _entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * np.log(c / total + 1e-12) for c in counts.values() if c > 0)


def _gold_at(item: Item, level: str) -> str:
    return getattr(item, f"sense_{level}") or item.gold


def _pred_at(pred: Prediction, level: str) -> str:
    return getattr(pred, f"label_{level}") or ""


def _label_distribution(pred: Prediction, level: str) -> dict[str, float]:
    """This model's vote distribution over labels, from its repeated samples.

    Each model contributes equal mass regardless of how many samples it was
    run with.  Averaging raw counts instead would make one model with n=5
    outvote another with n=1 purely by sampling budget.
    """
    labels: list[str] = []
    for raw in pred.samples or ():
        l1, l2, l3 = sense_levels(raw)
        lab = {"l1": l1, "l2": l2, "l3": l3}.get(level, "")
        if lab:
            labels.append(lab)
    if not labels:
        lab = _pred_at(pred, level)
        if lab:
            labels.append(lab)
    counts: dict[str, int] = defaultdict(int)
    for lab in labels:
        counts[lab] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def consensus_entropy(
    preds_by_item: dict[str, dict[str, Prediction]], level: str = "l3"
) -> dict[str, float]:
    """Disagreement across models: high entropy = models do not agree.

    Computes the entropy of the per-model label distributions averaged across
    models.  With a single model this reduces to that model's own sampling
    entropy, so the score is defined for any number of models.
    """
    out: dict[str, float] = {}
    for item_id, by_model in preds_by_item.items():
        dists = [d for d in (_label_distribution(p, level) for p in by_model.values()) if d]
        if not dists:
            out[item_id] = 0.0
            continue
        keys: set[str] = set().union(*dists)
        mean = {k: sum(d.get(k, 0.0) for d in dists) / len(dists) for k in keys}
        out[item_id] = _entropy_from_counts(
            {k: v for k, v in mean.items() if v > 0}
        )
    return out


class _LogReg:
    def __init__(self, l2: float = 1.0, lr: float = 0.1, epochs: int = 1500) -> None:
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.mu = self.sd = self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogReg":
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        w = np.zeros(Z.shape[1])
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-(Z @ w)))
            grad = Z.T @ (p - y) / len(y) + self.l2 * w / len(y)
            w -= self.lr * grad
        self.w = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        return 1 / (1 + np.exp(-(Z @ self.w)))


def _fold(item_id: str, k: int) -> int:
    return int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % k


def build_scores(
    items: Sequence[Item],
    preds: Sequence[Prediction],
    prior_entropy: dict[str, float],
    n_folds: int = 5,
    level: str = "l3",
    base_model: str | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Return out-of-fold error scores per policy, plus an error AUC per policy.

    Higher score = review sooner.  Policies: ``random``, ``confidence``,
    ``consensus``, ``feature``, ``hybrid``.

    ``base_model`` names the model whose labels are being triaged.  Predictions
    from every other model stay in the pool and feed ``consensus`` only.  Without
    it, a file holding several models would silently score whichever prediction
    happened to come last for each item while still reporting a consensus score.
    """
    gold = {it.id: _gold_at(it, level) for it in items}
    by_item_model: dict[str, dict[str, Prediction]] = defaultdict(dict)
    for p in preds:
        by_item_model[p.item_id][p.model] = p
    scored = [p for p in preds if p.model == base_model] if base_model else list(preds)
    by_item: dict[str, Prediction] = {p.item_id: p for p in scored if p.error is None}
    cons = consensus_entropy(by_item_model, level=level)

    ids = [it.id for it in items if it.id in by_item]
    if not ids:
        return {}, {}
    items_by_id = {it.id: it for it in items}
    X, y = [], []
    for i in ids:
        it, p = items_by_id[i], by_item[i]
        X.append(_item_features(it, p, prior_entropy.get(i, 0.0), cons.get(i, 0.0)))
        y.append(0.0 if _pred_at(p, level) == gold[i] else 1.0)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    scores: dict[str, dict[str, float]] = {"random": {}, "confidence": {}, "consensus": {}, "feature": {}}
    rng = np.random.default_rng(13)
    for i in ids:
        scores["random"][i] = float(rng.random())
    for i in ids:
        p = by_item[i]
        scores["confidence"][i] = 1.0 - float(p.confidence if p.confidence is not None else 1.0)
        scores["consensus"][i] = cons.get(i, 0.0)
        it = items_by_id[i]
        scores["feature"][i] = (
            prior_entropy.get(i, 0.0)
            + 0.3 * it.n_lexicon_senses
            + (0.5 if it.paired else 0.0)
            + (0.5 if it.cross_sentence else 0.0)
        )

    # Out-of-fold hybrid scores: an item is never scored by a model that saw its label.
    hybrid: dict[str, float] = {}
    for fold in range(n_folds):
        tr = [k for k, i in enumerate(ids) if _fold(i, n_folds) != fold]
        te = [k for k, i in enumerate(ids) if _fold(i, n_folds) == fold]
        if not tr or not te:
            continue
        lr = _LogReg().fit(X[tr], y[tr])
        prob = lr.predict_proba(X[te])
        for k, p in zip(te, prob):
            hybrid[ids[k]] = float(p)
    if len(hybrid) < len(ids):
        for i in ids:
            hybrid.setdefault(i, 0.0)
    scores["hybrid"] = hybrid
    scores["oracle"] = {i: (1.0 if _pred_at(by_item[i], level) != gold[i] else 0.0) for i in ids}

    auc = {name: _auc(list(y), [scores[name][i] for i in ids]) for name in scores}
    return scores, auc


def _auc(y_true: list[float], score: list[float]) -> float:
    pos = [s for t, s in zip(y_true, score) if t == 1]
    neg = [s for t, s in zip(y_true, score) if t == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def quality_curve(
    items: Sequence[Item],
    preds: Sequence[Prediction],
    scores: dict[str, float],
    budgets: Sequence[float] = tuple(i / 20 for i in range(21)),
    level: str = "l3",
) -> list[dict]:
    gold = {it.id: _gold_at(it, level) for it in items}
    by_item = {p.item_id: p for p in preds if p.error is None}
    ids = [i for i in scores if i in by_item and i in gold]
    ranked = sorted(ids, key=lambda i: (-scores[i], i))
    out = []
    for b in budgets:
        k = int(round(b * len(ranked)))
        reviewed = set(ranked[:k])
        g = [gold[i] for i in ids]
        p = [gold[i] if i in reviewed else _pred_at(by_item[i], level) for i in ids]
        kap = cohen_kappa(g, p)
        out.append({"budget": b, "n_reviewed": k, "kappa": kap, "accuracy": accuracy(g, p)})
    return out


def compare_policies(
    items: Sequence[Item],
    preds: Sequence[Prediction],
    prior_entropy: dict[str, float],
    n_folds: int = 5,
    level: str = "l3",
    base_model: str | None = None,
) -> dict:
    scores, auc = build_scores(
        items, preds, prior_entropy, n_folds=n_folds, level=level, base_model=base_model
    )
    scored = [p for p in preds if p.model == base_model] if base_model else list(preds)
    curves = {
        name: quality_curve(items, scored, s, level=level) for name, s in scores.items()
    }
    return {"auc": auc, "curves": curves}
