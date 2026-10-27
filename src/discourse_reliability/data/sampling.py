"""Stratified sampling for the 600-item main set.

The strata are the four axes fixed in the experiment plan: explicitness,
lexicon ambiguity, connective type (single vs paired) and span
(intra- vs cross-sentence).  CDTB is roughly 75% implicit, so the plan
deliberately oversamples explicit relations and reweights at analysis time.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from ..schema import Item


def cap_per_connective(items: list[Item], cap: int, seed: int = 13) -> list[Item]:
    """Prevent one high-frequency connective from dominating the sample."""
    if cap <= 0:
        return list(items)
    rng = random.Random(seed)
    buckets: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        buckets[it.connective or "<implicit>"].append(it)
    out: list[Item] = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda x: x.id)
        rng.shuffle(bucket)
        out.extend(bucket[:cap])
    return out


def stratified_sample(
    items: list[Item],
    n: int,
    seed: int = 13,
    explicit_fraction: float | None = 1 / 3,
    min_per_stratum: int = 5,
) -> list[Item]:
    """Sample ``n`` items across the four axes.

    ``explicit_fraction`` fixes the share of explicit items in the sample
    (``None`` = keep the corpus proportion).  All other axes are allocated
    proportionally within each explicitness level, with ``min_per_stratum``
    reserved where the stratum has enough items.
    """
    pool = [it for it in items if it.gold and it.task]
    if not pool:
        raise ValueError("no labelled items to sample")
    rng = random.Random(seed)

    groups: dict[str, list[Item]] = defaultdict(list)
    for it in pool:
        groups[it.stratum_key()].append(it)
    for g in groups.values():
        g.sort(key=lambda x: x.id)
        rng.shuffle(g)

    keys = sorted(groups)
    explicit_keys = [k for k in keys if "explicitness=explicit" in k]
    implicit_keys = [k for k in keys if "explicitness=implicit" in k]

    if explicit_fraction is None:
        quota = {k: len(groups[k]) / len(pool) * n for k in keys}
    else:
        n_exp = 0 if not explicit_keys else round(n * explicit_fraction)
        if not implicit_keys:
            n_exp = n
        n_imp = n - n_exp
        quota = {}
        for k in keys:
            share = n_exp if "explicitness=explicit" in k else n_imp
            denom = sum(len(groups[x]) for x in (explicit_keys if k in explicit_keys else implicit_keys))
            quota[k] = share * len(groups[k]) / denom if denom else 0.0

    # Reserve a floor, never above what the stratum holds.
    alloc = {k: min(len(groups[k]), max(min_per_stratum, int(quota[k]))) for k in keys}
    total = sum(alloc.values())
    # Trim or grow proportionally.
    order = sorted(keys, key=lambda k: (-len(groups[k]), k))
    while total > n:
        for k in reversed(order):
            if total <= n:
                break
            if alloc[k] > min(len(groups[k]), min_per_stratum):
                alloc[k] -= 1
                total -= 1
    while total < n:
        progressed = False
        for k in order:
            if total >= n:
                break
            if alloc[k] < len(groups[k]):
                alloc[k] += 1
                total += 1
                progressed = True
        if not progressed:
            break

    out: list[Item] = []
    for k in keys:
        out.extend(groups[k][: alloc[k]])
    out.sort(key=lambda x: x.id)
    return out


def describe(items: list[Item]) -> dict:
    """Counts per stratum plus the label distribution."""
    strata: Counter[str] = Counter(it.stratum_key() for it in items)
    return {
        "n": len(items),
        "strata": dict(sorted(strata.items())),
        "label_l2": dict(Counter(it.sense_l2 for it in items).most_common()),
        "connectives": len({it.connective for it in items if it.connective}),
        "corpora": dict(Counter(it.corpus for it in items)),
    }


def assign_splits(
    items: list[Item],
    seed: int = 7,
    train: float = 0.7,
    dev: float = 0.15,
    group_by: str | None = None,
) -> list[Item]:
    """Deterministic split, optionally keeping whole groups on one side.

    ``group_by="doc_id"`` is required for HIT-CDTB and TED-CDB.  Both corpora
    annotate several relations per document (HIT additionally annotates the
    same document at three granularities), so an item-level split puts sibling
    relations from the same document in train and test -- a leak that inflates
    every score.  Groups are assigned whole, in shuffled order, until each
    split's item quota is met.
    """
    rng = random.Random(seed)
    if group_by:
        groups: dict[str, list[Item]] = defaultdict(list)
        for it in items:
            groups[str(getattr(it, group_by) or it.id)].append(it)
        keys = sorted(groups)
        rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        cut_train = total * train
        cut_dev = total * (train + dev)
        seen = 0
        for k in keys:
            split = "train" if seen < cut_train else ("dev" if seen < cut_dev else "test")
            for it in groups[k]:
                it.split = split
            seen += len(groups[k])
        return items

    buckets: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        buckets[it.sense_l2 or it.sense_l1].append(it)
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda x: x.id)
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = round(n * train)
        n_dev = round(n * dev)
        for i, it in enumerate(bucket):
            it.split = "train" if i < n_train else ("dev" if i < n_train + n_dev else "test")
    return items


def assign_folds(
    items: list[Item], k: int = 5, seed: int = 23, group_by: str | None = "doc_id"
) -> list[Item]:
    """Assign ``meta["fold"]`` for document-level cross-validation.

    Folds are drawn over documents, not items, for the reason in
    :func:`assign_splits`.  Returns the same list, mutated.
    """
    rng = random.Random(seed)
    if not group_by:
        raise ValueError("assign_folds needs group_by to avoid document leakage")
    groups: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        groups[str(getattr(it, group_by) or it.id)].append(it)
    keys = sorted(groups)
    rng.shuffle(keys)
    # Greedy largest-first: documents differ by two orders of magnitude in
    # relation count (HIT: 6 vs 1,400), so round-robin gives folds of 455 and
    # 1,068 items.  Placing each document in the currently smallest fold keeps
    # fold sizes and per-class coverage close.
    keys.sort(key=lambda key: -len(groups[key]))
    sizes = [0] * k
    for key in keys:
        fold = min(range(k), key=lambda f: (sizes[f], f))
        sizes[fold] += len(groups[key])
        for it in groups[key]:
            it.meta["fold"] = fold
    return items
