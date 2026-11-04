"""Build the R152 timed-annotation packet (``annotation/timing/items.js``).

R152 asks what a human annotator actually saves when an LLM answer is put in
front of them, so the page must never let the gold label leak: we ship only
the arg spans, the connective, the sentence, the model's suggested L1 label and
whether the triage flagged the item.

With a single annotator the design is item-level counterbalancing instead of a
between-annotator Latin square:

* the 100 items are split into halves X / Y, stratified by gold L1 and
  explicitness, so the halves are comparable;
* session 1 shows X under condition ``a`` (from scratch) and Y under ``b``
  (proofread the suggestion); session 2 swaps them.  Practice and fatigue
  therefore cancel in the a-vs-b contrast instead of biasing it;
* session 3 is condition ``c``: only the triage-flagged items are shown, and
  the unflagged ones are auto-accepted at the model's label.

The flag set is the lowest ``--review-budget`` share of self-reported
confidence, a fixed count rather than a threshold, so the review cost is
controlled by design and not by where the confidence distribution happens to
fall.

    python scripts/build_timing_site.py \
        --items data/processed/l1/r152_timing_items.jsonl \
        --preds outputs/l1_r152_deepseek-chat_zs.jsonl \
        --self-assess outputs/l1_r152_selfassess.jsonl \
        --review-budget 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items  # noqa: E402


def load_preds(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error") is None and r.get("label_l1"):
                out[r["item_id"]] = r
    return out


def split_halves(items, seed: int) -> set[str]:
    """Split roughly in half, balancing gold class and explicitness.

    A plain shuffle is fine for class balance at n=100, but implicit items are
    harder and slower; if one half held most of them the two sessions would
    differ in workload for reasons unrelated to condition.
    """
    rng = random.Random(seed)
    cells: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for it in items:
        cells[(it.sense_l1, it.explicit)].append(it.id)
    x: list[str] = []
    for key in sorted(cells):
        ids = sorted(cells[key])
        rng.shuffle(ids)
        take = len(ids) // 2
        # Odd cell: alternate which half absorbs the extra item.
        if len(ids) % 2 and rng.random() < 0.5:
            take += 1
        x.extend(ids[:take])
    return set(x)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/processed/l1/r152_timing_items.jsonl")
    ap.add_argument("--preds", default="outputs/l1_r152_deepseek-chat_zs.jsonl")
    ap.add_argument("--self-assess", default="outputs/l1_r152_selfassess.jsonl")
    ap.add_argument("--out", default="annotation/timing/items.js")
    ap.add_argument("--review-budget", type=int, default=50,
                    help="number of items the annotator reviews in condition c")
    ap.add_argument("--annotators", nargs="+", default=["A", "B"],
                    help="annotator ids; each covers half the items, so at most two")
    ap.add_argument("--with-c", action="store_true",
                    help="add a third session that measures condition c directly "
                         "(costs each annotator 25 more items)")
    ap.add_argument("--warmup", type=int, default=4,
                    help="practice items prepended to session 1 and dropped from analysis")
    ap.add_argument("--warmup-from", default="data/processed/l1/hit_balanced.jsonl")
    ap.add_argument("--seed", type=int, default=152)
    args = ap.parse_args(argv)

    items = list(load_items(ROOT / args.items))
    preds = load_preds(ROOT / args.preds)
    assess = load_preds(ROOT / args.self_assess)
    missing = [it.id for it in items if it.id not in preds or it.id not in assess]
    if missing:
        raise SystemExit(f"{len(missing)} items lack a prediction, e.g. {missing[:3]}")

    # Flag the lowest-confidence items.  Ties are broken by item id so the
    # packet is reproducible without recording the browser's sort behaviour.
    ranked = sorted(
        items,
        key=lambda it: (assess[it.id]["meta"].get("self_confidence", 0.5), it.id),
    )
    flagged = {it.id for it in ranked[: args.review_budget]}

    half_x = split_halves(items, args.seed)

    def order(ids: list[str], salt: int) -> list[str]:
        rng = random.Random(args.seed + salt)
        ids = sorted(ids)
        rng.shuffle(ids)
        return ids

    by_id = {it.id: it for it in items}
    all_ids = [it.id for it in items]
    half_ids = {"1": [i for i in all_ids if i in half_x],
                "2": [i for i in all_ids if i not in half_x]}
    # More than two annotators would need the halves to repeat, which would
    # reintroduce shared items; refuse rather than silently duplicate.
    if len(args.annotators) > len(half_ids):
        raise SystemExit("at most two annotators in the split design")

    # Warm-up items are drawn from the same corpus but never enter the timing
    # comparison: their only job is to absorb the first-exposure cost.
    in_packet = {it.id for it in items}
    warm: list = []
    if args.warmup:
        pool = [it for it in load_items(ROOT / args.warmup_from) if it.id not in in_packet]
        rng = random.Random(args.seed + 9)
        by_exp: dict[bool, list] = defaultdict(list)
        for it in pool:
            by_exp[it.explicit].append(it)
        want_exp = round(args.warmup * sum(it.explicit for it in pool) / len(pool))
        for explicit, take in ((True, want_exp), (False, args.warmup - want_exp)):
            rng.shuffle(by_exp[explicit])
            warm.extend(by_exp[explicit][:take])
        rng.shuffle(warm)

    # One slot per annotator: each covers half the items under both a and b.
    # Two sessions per slot swap the two quarters, so within a person half the
    # items are a-first and half b-first.
    by_slot: dict[str, dict] = {}
    for slot, name in enumerate(args.annotators, start=1):
        mine = half_ids[str(slot)]
        sub_x = split_halves([by_id[i] for i in mine], args.seed + slot)
        sess = {
            "1": {
                "order": order([it.id for it in warm], 7)
                         + order([i for i in mine if i in sub_x], 11 + slot)
                         + order([i for i in mine if i not in sub_x], 21 + slot),
                "cond": {**{it.id: "w" for it in warm},
                         **{i: ("a" if i in sub_x else "b") for i in mine}},
            },
            "2": {
                "order": order([i for i in mine if i not in sub_x], 31 + slot)
                         + order([i for i in mine if i in sub_x], 41 + slot),
                "cond": {i: ("b" if i in sub_x else "a") for i in mine},
            },
        }
        if args.with_c:
            # Optional third session: only flagged items from this person's
            # half are reviewed; the rest of the half is auto-accepted.
            mine_flagged = sorted(set(mine) & flagged)
            sess["3"] = {
                "order": order(mine_flagged, 51 + slot),
                "cond": {i: "c" for i in mine_flagged},
                "auto": {i: preds[i]["label_l1"] for i in mine if i not in flagged},
            }
        by_slot[str(slot)] = sess

    payload = {
        "packet": "r152-timing-split",
        "packet_hash": hashlib.sha1(
            json.dumps([[by_slot[k][s]["order"] for s in sorted(by_slot[k])]
                        for k in sorted(by_slot)], ensure_ascii=False).encode()
        ).hexdigest()[:12],
        "n_items": len(items),
        "review_budget": args.review_budget,
        "slots": {name: str(slot) for slot, name in enumerate(args.annotators, start=1)},
        "by_slot": by_slot,
        # Kept for a quick look at the whole design without resolving a slot.
        "sessions": by_slot["1"],
        "items": {
            **{
                it.id: {
                    "doc": it.doc_id,
                    "task": it.task,
                    "arg1": it.arg1 or it.text,
                    "arg2": it.arg2,
                    "conn": it.connective,
                    "text": it.text,
                    "explicit": bool(it.explicit),
                    "genre": it.genre,
                    "suggestion": preds[it.id]["label_l1"],
                    "conf": round(
                        float(assess[it.id]["meta"].get("self_confidence", 0.5)), 3
                    ),
                    "flagged": it.id in flagged,
                }
                for it in items
            },
            **{
                it.id: {
                    "doc": it.doc_id,
                    "task": it.task,
                    "arg1": it.arg1 or it.text,
                    "arg2": it.arg2,
                    "conn": it.connective,
                    "text": it.text,
                    "explicit": bool(it.explicit),
                    "genre": it.genre,
                    "suggestion": "",
                    "conf": None,
                    "flagged": False,
                }
                for it in warm
            },
        },
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "window.R152 = " + json.dumps(payload, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8",
    )

    print(f"wrote {args.out}  hash={payload['packet_hash']}")
    for slot in sorted(by_slot):
        mine = half_ids[slot]
        warm_n = sum(1 for c in by_slot[slot]["1"]["cond"].values() if c == "w")
        shown = sum(1 for c in by_slot[slot]["1"]["cond"].values() if c != "w")
        shown += len(by_slot[slot]["2"]["order"])
        if args.with_c:
            shown += len(by_slot[slot]["3"]["order"])
        print(f"  slot {slot}: {len(mine)} items "
              f"(explicit {sum(1 for i in mine if by_id[i].explicit)})"
              f" -> {shown} shown{' + ' + str(warm_n) + ' warm' if warm_n else ''}")
    print(f"  flagged (condition c) {len(flagged)} / {len(items)}")
    if not args.with_c:
        print("  condition c will be derived from the flagged subset of b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
