"""Item-level bootstrap confidence intervals.

Resampling units are ITEMS, not API calls: repeated samples of the same item
are not independent observations, and treating them as such would produce
intervals that are too narrow.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

Metric = Callable[[Sequence[int]], float]


def bootstrap_ci(
    n_items: int,
    metric: Metric,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float, float] | tuple[None, None, None]:
    """Return ``(point, lo, hi)``; ``point`` uses all items.

    ``metric`` receives a list of resampled indices and must return a float
    (may return ``float('nan')`` for degenerate resamples, which are dropped).
    """
    if n_items == 0:
        return None, None, None
    idx = list(range(n_items))
    point = metric(idx)
    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        sample = [rng.randrange(n_items) for _ in idx]
        v = metric(sample)
        if v == v:  # not NaN
            vals.append(v)
    if not vals:
        return point, None, None
    vals.sort()
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return point, lo, hi
