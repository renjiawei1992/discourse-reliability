"""Draw a stratified evaluation subset, reproducibly.

R171 (prompt-wordings × temperature) is designed on a 200-item subset so the
full 3 × 2 × models grid stays cheap.  Sampling has to be *within* each fold and
stratified by explicitness, otherwise a subset can drift in class balance and
the kappa movements it is meant to measure get mixed with a change of task mix.

    python scripts/subsample.py \
        --input data/processed/l1/hit_balanced.jsonl \
        --out data/processed/l1/hit_balanced_sub200.jsonl \
        --n 200 --seed 13
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items, save_items  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    items = list(load_items(ROOT / args.input))
    cells: dict[tuple[str, bool], list] = defaultdict(list)
    for it in items:
        cells[(str(it.meta.get("fold")), it.explicit)].append(it)

    # Proportional allocation with largest-remainder rounding, so the subset
    # matches the parent on both fold and explicitness without hand-tuning.
    total = len(items)
    raw = {k: args.n * len(v) / total for k, v in cells.items()}
    alloc = {k: int(x) for k, x in raw.items()}
    short = args.n - sum(alloc.values())
    for k, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:short]:
        alloc[k] += 1

    rng = random.Random(args.seed)
    picked = []
    for k, group in sorted(cells.items(), key=lambda kv: str(kv[0])):
        take = alloc[k]
        picked.extend(rng.sample(group, min(take, len(group))))
    picked.sort(key=lambda it: it.id)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    save_items(out, picked)
    print(f"{len(picked)} items -> {args.out}")
    by_explicit = defaultdict(int)
    for it in picked:
        by_explicit[it.explicit] += 1
    print(f"explicit={by_explicit[True]} implicit={by_explicit[False]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
