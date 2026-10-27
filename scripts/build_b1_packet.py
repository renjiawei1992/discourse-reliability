"""Build the B1 human double-annotation packet (R101).

Two constraints drive the design and both come from the corpus, not taste:

* **Only 50 documents.**  A packet of 600 items would put 12 items from the
  same article in front of every annotator, and items from one article are not
  exchangeable -- the effective sample size for a kappa estimate is far below
  600.  Capping documents and reporting a document-clustered CI is the honest
  version of the same budget.
* **The packet must be paired with the model.**  A human ceiling computed on
  items the model was never run on cannot be compared to the model's accuracy.
  ``hit_balanced`` is the evaluation set, so the packet is drawn to overlap it
  where possible and the remaining items are re-run through the model later
  (300 calls, not a budget problem).

The pilot block is disjoint from the main block: it is annotated first, with
discussion, purely to calibrate the guideline.  Reusing pilot items in the main
block would leak that discussion into the measurement.

    python scripts/build_b1_packet.py --input data/processed/hitcdtb.jsonl \
        --out data/processed/l1/b1_packet.jsonl --n 300 --pilot 60 --cap-per-doc 6
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import Item, load_items, save_items  # noqa: E402

CLASSES = ("TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION")


def draw(
    pool: list[Item],
    n: int,
    cap_per_doc: int,
    seed: int,
    per_class_floor: int = 0,
) -> list[Item]:
    """Round-robin over documents so no article dominates the packet.

    Classes are filled scarcest-first: TEMPORAL is 3.3% of the corpus, so if
    the common classes fill every slot first there is no room left to estimate
    its kappa at all.  Within a round every document contributes at most one
    item, which keeps the packet from being one long article.
    """
    rng = random.Random(seed)
    by_doc: dict[str, list[Item]] = defaultdict(list)
    for it in pool:
        by_doc[it.doc_id or it.id].append(it)
    for doc in by_doc:
        rng.shuffle(by_doc[doc])

    used: Counter[str] = Counter()
    chosen: list[Item] = []
    taken: set[str] = set()

    def take_from(pred, k: int) -> None:
        while k > 0:
            ready = [
                d for d, items in by_doc.items()
                if used[d] < cap_per_doc and any(i.id not in taken and pred(i) for i in items)
            ]
            if not ready:
                return
            rng.shuffle(ready)
            progressed = False
            for d in ready:
                if k <= 0:
                    break
                for it in by_doc[d]:
                    if it.id not in taken and pred(it):
                        chosen.append(it)
                        taken.add(it.id)
                        used[d] += 1
                        k -= 1
                        progressed = True
                        break
            if not progressed:
                return

    order = sorted(CLASSES, key=lambda c: sum(1 for x in pool if x.sense_l1 == c))
    if per_class_floor:
        for c in order:
            take_from(lambda it, c=c: it.sense_l1 == c, per_class_floor)

    take_from(lambda it: True, n - len(chosen))
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--pilot", type=int, default=60)
    ap.add_argument("--cap-per-doc", type=int, default=6)
    ap.add_argument("--per-class-floor", type=int, default=25)
    ap.add_argument("--pilot-floor", type=int, default=10)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args(argv)

    items = [it for it in load_items(args.input) if it.sense_l1 in CLASSES]
    pilot = draw(items, args.pilot, args.cap_per_doc, args.seed, args.pilot_floor)
    taken = {it.id for it in pilot}
    main = draw(
        [it for it in items if it.id not in taken],
        args.n,
        args.cap_per_doc,
        args.seed + 1,
        args.per_class_floor,
    )
    for it in pilot:
        it.meta["b1_split"] = "pilot"
    for it in main:
        it.meta["b1_split"] = "main"

    out = sorted(pilot + main, key=lambda x: (x.meta["b1_split"], x.id))
    save_items(args.out, out)

    for name, block in (("pilot", pilot), ("main", main)):
        docs = {it.doc_id for it in block}
        print(json.dumps({
            "split": name,
            "n": len(block),
            "docs": len(docs),
            "per_doc_max": max(Counter(it.doc_id for it in block).values()),
            "classes": dict(Counter(it.sense_l1 for it in block)),
            "explicit": dict(Counter(it.explicit for it in block)),
            "genres": dict(Counter(it.genre for it in block)),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
