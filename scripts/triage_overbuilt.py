"""R151: is the triage scorer over-built?

The paper claims a simple hybrid (logistic regression on eight features) is
enough to rank items for human review.  A reviewer will ask the obvious
question: would a stronger learner on more features do better?  If it does not,
the extra component is rejected explicitly -- that is the point of this run.

Compared systems, all scored out-of-fold by **document**:

    confidence      model's own vote agreement (n=5 sampling)
    self_assess     the model's verbal confidence, asked in a separate call
    consensus       cross-model disagreement
    feature         static item properties only
    lr_base         current hybrid: logistic regression on 8 features
    lr_all          logistic regression on every feature
    gbdt_all        gradient boosting on every feature
    oracle          perfect ranking (upper bound)

No sklearn here: the GBDT is a small NumPy implementation, kept deliberately
plain because the comparison is about features, not about tuning a library.

    python scripts/triage_overbuilt.py \
        --gold data/processed/l1/hit_balanced.jsonl \
        --base outputs/l1_hit_balanced_chat_n5_t07.jsonl \
        --self-assess outputs/l1_selfassess_chat.jsonl \
        --pool "outputs/l1_hit_balanced_*_zs*.jsonl" \
        --out outputs/l1_triage_overbuilt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.baselines.lexeme_prior import attach_prior_entropy  # noqa: E402
from discourse_reliability.metrics.agreement import cohen_kappa  # noqa: E402
from discourse_reliability.schema import load_items, load_predictions  # noqa: E402
from discourse_reliability.triage import consensus_entropy  # noqa: E402

# ---------------------------------------------------------------- GBDT


class _Node:
    __slots__ = ("feat", "thr", "left", "right", "leaf")

    def __init__(self) -> None:
        self.feat = None
        self.thr = None
        self.left = None
        self.right = None
        self.leaf = None


def _build(X, g, h, idx, depth, l2, min_leaf):
    node = _Node()
    G, H = g[idx].sum(), h[idx].sum()
    if depth == 0 or len(idx) < 2 * min_leaf:
        node.leaf = -G / (H + l2)
        return node
    best = None
    for f in range(X.shape[1]):
        vals = X[idx, f]
        cuts = np.unique(np.quantile(vals, np.linspace(0.1, 0.9, 9)))
        for t in cuts:
            mask = vals <= t
            li, ri = idx[mask], idx[~mask]
            if len(li) < min_leaf or len(ri) < min_leaf:
                continue
            GL, HL = g[li].sum(), h[li].sum()
            GR, HR = g[ri].sum(), h[ri].sum()
            gain = GL * GL / (HL + l2) + GR * GR / (HR + l2) - G * G / (H + l2)
            if best is None or gain > best[0]:
                best = (gain, f, t, li, ri)
    if best is None:
        node.leaf = -G / (H + l2)
        return node
    _, f, t, li, ri = best
    node.feat, node.thr = f, t
    node.left = _build(X, g, h, li, depth - 1, l2, min_leaf)
    node.right = _build(X, g, h, ri, depth - 1, l2, min_leaf)
    return node


def _predict_tree(node, row):
    while node.leaf is None:
        node = node.left if row[node.feat] <= node.thr else node.right
    return node.leaf


class GBDT:
    def __init__(self, n_estimators=60, depth=3, lr=0.08, l2=2.0, min_leaf=15):
        self.n_estimators, self.depth, self.lr = n_estimators, depth, lr
        self.l2, self.min_leaf = l2, min_leaf
        self.trees: list[_Node] = []
        self.f0 = 0.0

    def fit(self, X, y):
        n = len(X)
        p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.f0 = float(np.log(p0 / (1 - p0)))
        F = np.full(n, self.f0)
        idx_all = np.arange(n)
        self.trees = []
        for _ in range(self.n_estimators):
            p = 1 / (1 + np.exp(-F))
            g, h = p - y, p * (1 - p)
            tree = _build(X, g, h, idx_all, self.depth, self.l2, self.min_leaf)
            self.trees.append(tree)
            F += self.lr * np.array([_predict_tree(tree, x) for x in X])
        return self

    def predict_proba(self, X):
        F = np.full(len(X), self.f0)
        for tree in self.trees:
            F += self.lr * np.array([_predict_tree(tree, x) for x in X])
        return 1 / (1 + np.exp(-F))


class LogReg:
    def __init__(self, l2=1.0, lr=0.1, epochs=1500):
        self.l2, self.lr, self.epochs = l2, lr, epochs

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        w = np.zeros(Z.shape[1])
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-(Z @ w)))
            w -= self.lr * (Z.T @ (p - y) / len(y) + self.l2 * w / len(y))
        self.w = w
        return self

    def predict_proba(self, X):
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        return 1 / (1 + np.exp(-(Z @ self.w)))


# ---------------------------------------------------------------- data


def auc(y_true, score) -> float:
    y = np.asarray(y_true)
    s = np.asarray(score)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def quality_at(gold, pred, scores, ids, budgets=(0.1, 0.2, 0.3)):
    ranked = sorted(ids, key=lambda i: (-scores[i], i))
    out = {}
    for b in budgets:
        k = int(round(b * len(ranked)))
        reviewed = set(ranked[:k])
        g = [gold[i] for i in ids]
        p = [gold[i] if i in reviewed else pred[i] for i in ids]
        out[f"{b:.2f}"] = round(cohen_kappa(g, p) or 0.0, 4)
    return out


def doc_folds(docs: list[str], k: int, seed: int = 13) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(docs))
    rng.shuffle(uniq)
    sizes = Counter(docs)
    folds: dict[str, int] = {}
    load = np.zeros(k)
    for d in sorted(uniq, key=lambda x: -sizes[x]):
        f = int(np.argmin(load))
        folds[d] = f
        load[f] += sizes[d]
    return folds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--base", required=True, help="system being triaged (n=5 sampling run)")
    ap.add_argument("--self-assess", required=True)
    ap.add_argument("--pool", nargs="+", required=True, help="all model files for consensus")
    ap.add_argument("--level", default="l1", choices=["l1", "l2", "l3"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    items = {it.id: it for it in load_items(args.gold)}
    base = {p.item_id: p for p in load_predictions(args.base) if p.error is None}
    selfassess = {p.item_id: p for p in load_predictions(args.self_assess) if p.error is None}
    pool: dict[str, dict] = defaultdict(dict)
    for spec in args.pool:
        for path in sorted(ROOT.glob(spec)):
            for p in load_predictions(path):
                if p.error is None:
                    pool[p.item_id].setdefault(p.model, p)
    cons = consensus_entropy(pool, level=args.level)
    prior = attach_prior_entropy([items[i] for i in items], k=args.folds, level=args.level)

    ids = [i for i in items if i in base]
    docs = [str(items[i].doc_id or i) for i in ids]
    fold_of = doc_folds(docs, args.folds)

    gold = {i: getattr(items[i], f"sense_{args.level}") or items[i].gold for i in ids}
    pred = {i: getattr(base[i], f"label_{args.level}") or "" for i in ids}

    def text_stats(i):
        t = items[i].text or ""
        n = max(1, len(t))
        digits = sum(c.isdigit() for c in t)
        punct = sum(c in "，。；：、！？" for c in t)
        return [digits / n, punct / n, min(len(t), 300) / 300.0]

    def feats(i, extra: bool):
        it = items[i]
        base_f = [
            float(base[i].confidence if base[i].confidence is not None else 1.0),
            prior.get(i, 0.0),
            cons.get(i, 0.0),
            float(it.n_lexicon_senses),
            1.0 if it.explicit else 0.0,
            1.0 if it.paired else 0.0,
            1.0 if it.cross_sentence else 0.0,
            min(len(it.text or ""), 300) / 300.0,
        ]
        if not extra:
            return base_f
        sc = selfassess.get(i)
        self_conf = float(sc.confidence) if sc is not None and sc.confidence is not None else -1.0
        label = pred[i]
        dummies = [1.0 if label == c else 0.0 for c in
                   ("TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION")[:3]]
        genre = it.genre or "unknown"
        gd = [1.0 if genre == g else 0.0 for g in ("ontonotes_mz", "ontonotes_wb", "ontonotes_nw")[:2]]
        return base_f + [self_conf] + dummies + gd + text_stats(i)

    Xb = np.asarray([feats(i, False) for i in ids], dtype=float)
    Xa = np.asarray([feats(i, True) for i in ids], dtype=float)
    self_col = np.asarray([
        [float(selfassess[i].confidence) if i in selfassess
         and selfassess[i].confidence is not None else -1.0] for i in ids
    ], dtype=float)
    Xs = np.hstack([Xb, self_col])
    y = np.asarray([0.0 if pred[i] == gold[i] else 1.0 for i in ids])

    scores: dict[str, dict[str, float]] = {
        "confidence": {i: 1.0 - float(base[i].confidence or 1.0) for i in ids},
        "self_assess": {i: 1.0 - float(selfassess[i].confidence or 1.0) if i in selfassess else 0.5
                        for i in ids},
        "consensus": {i: cons.get(i, 0.0) for i in ids},
        "feature": {i: prior.get(i, 0.0) + 0.3 * items[i].n_lexicon_senses
                    + (0.5 if items[i].paired else 0.0)
                    + (0.5 if items[i].cross_sentence else 0.0) for i in ids},
    }

    def oof(make_model, X):
        out = np.zeros(len(ids))
        for f in range(args.folds):
            tr = np.array([fold_of[d] != f for d in docs])
            te = ~tr
            if tr.sum() < 20 or te.sum() == 0:
                continue
            m = make_model().fit(X[tr], y[tr])
            out[te] = m.predict_proba(X[te])
        return {i: float(v) for i, v in zip(ids, out)}

    scores["lr_base"] = oof(lambda: LogReg(), Xb)
    scores["lr_plus_self"] = oof(lambda: LogReg(), Xs)
    scores["lr_all"] = oof(lambda: LogReg(), Xa)
    scores["gbdt_all"] = oof(lambda: GBDT(), Xa)

    # A model-free combination: rank-normalise three cheap signals and average.
    def rank01(vals):
        order = np.argsort(np.argsort(vals))
        return order / max(1, len(order) - 1)

    sig_conf = rank01(np.asarray([1.0 - float(base[i].confidence or 1.0) for i in ids]))
    sig_self = rank01(np.asarray([1.0 - (float(selfassess[i].confidence)
                                        if i in selfassess and selfassess[i].confidence is not None
                                        else 1.0) for i in ids]))
    sig_cons = rank01(np.asarray([cons.get(i, 0.0) for i in ids]))
    simple = (sig_conf + sig_self + sig_cons) / 3.0
    scores["simple_mean"] = {i: float(v) for i, v in zip(ids, simple)}
    scores["oracle"] = {i: (1.0 if pred[i] != gold[i] else 0.0) for i in ids}

    aucs = {k: round(auc(y, [scores[k][i] for i in ids]), 4) for k in scores}
    curves = {k: quality_at(gold, pred, scores[k], ids) for k in scores}

    # Paired bootstrap over documents for the two decisions that matter.
    def delta_auc(a: str, b: str, n_boot: int = 1000, seed: int = 13):
        rng = np.random.default_rng(seed)
        by_doc: dict[str, list[int]] = defaultdict(list)
        for k, d in enumerate(docs):
            by_doc[d].append(k)
        keys = sorted(by_doc)
        deltas = []
        for _ in range(n_boot):
            pick = rng.choice(keys, size=len(keys), replace=True)
            idx = [k for d in pick for k in by_doc[d]]
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            deltas.append(auc(yy, [scores[a][ids[k]] for k in idx])
                          - auc(yy, [scores[b][ids[k]] for k in idx]))
        deltas.sort()
        return {
            "delta": round(float(np.mean(deltas)), 4),
            "ci95": [round(float(deltas[int(0.025 * len(deltas))]), 4),
                     round(float(deltas[int(0.975 * len(deltas))]), 4)],
            "p_sign": round(2 * min(sum(d <= 0 for d in deltas),
                                    sum(d >= 0 for d in deltas)) / len(deltas), 4),
        }

    result = {
        "n": len(ids),
        "n_docs": len(set(docs)),
        "fold_sizes": dict(sorted(Counter(fold_of[d] for d in docs).items())),
        "base_error_rate": round(float(y.mean()), 4),
        "auc": aucs,
        "quality_at_budget": curves,
        "gbdt_vs_lr_base": delta_auc("gbdt_all", "lr_base"),
        "gbdt_vs_lr_all": delta_auc("gbdt_all", "lr_all"),
        "lr_all_vs_lr_base": delta_auc("lr_all", "lr_base"),
        "lr_plus_self_vs_lr_base": delta_auc("lr_plus_self", "lr_base"),
        "gbdt_vs_lr_plus_self": delta_auc("gbdt_all", "lr_plus_self"),
        "simple_mean_vs_gbdt": delta_auc("simple_mean", "gbdt_all"),
        "self_vs_confidence": delta_auc("self_assess", "confidence"),
        "consensus_vs_confidence": delta_auc("consensus", "confidence"),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
