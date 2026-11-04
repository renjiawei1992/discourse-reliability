"""R173: materialise the ``.p1`` (sentence-group) level for the appendix.

The L1 blocks exclude ``.p1`` on purpose — it is a granularity PDTB does not
have, and the same text is already counted at p2/p3.  This script builds it as
a standalone set so the level can be reported descriptively, with the class
distribution (90% EXPANSION) and the resulting caveat printed alongside.

    python scripts/build_l1_p1.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.data.hitcdtb import load_hitcdtb  # noqa: E402
from discourse_reliability.data.sampling import assign_folds  # noqa: E402
from discourse_reliability.schema import save_items  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/hit-cdtb/extracted")
    ap.add_argument("--out", default="data/processed/l1/hit_p1.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    items = load_hitcdtb(ROOT / args.raw, levels=("p1",))
    if not items:
        raise SystemExit("no .p1 relations loaded — check --raw")

    # Document-level folds so a future few-shot run cannot leak.
    assign_folds(items, k=args.folds, seed=23, group_by="doc_id")

    n = save_items(ROOT / args.out, items)
    counts = Counter(it.sense_l1 for it in items)
    print(f"wrote {n} items to {args.out}")
    print("classes:", counts.most_common())
    print("explicit:", Counter(it.explicit for it in items))
    print("docs:", len({it.doc_id for it in items}))
    top = max(counts.values()) / sum(counts.values())
    print(f"majority-class baseline: {top:.4f}")
    if min(counts.values()) < 20:
        print("WARNING: at least one class has <20 items — a balanced 4-class "
              "evaluation is not possible at this level; report descriptively.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
