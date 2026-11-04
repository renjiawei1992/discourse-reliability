"""Where do two annotators actually disagree?

Counts the label pairs, splits them by explicitness and by whether the corpus
gold sides with one annotator or with neither, and flags corpus defects.
The point is to see the shape of human disagreement before spending a third
annotator's time on adjudication.

    python scripts/annotator_disagreements.py \
        --items data/processed/l1/b1_packet.jsonl \
        --a data/interim/b1_round1/submissions/C_latest.csv \
        --b data/interim/b1_round1/submissions/D_latest.csv \
        --out outputs/l1_annotator_disagreements.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items  # noqa: E402


def read_sheet(path: Path, column: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            val = (row.get(column) or "").strip().upper()
            if val:
                out[row["item_id"]] = val
    return out


def defect(it) -> str:
    if not (it.arg1 or "").strip():
        return "前件缺失"
    if not (it.arg2 or "").strip():
        return "后件缺失"
    norm = lambda x: "".join((x or "").split())
    a1, a2 = norm(it.arg1), norm(it.arg2)
    if a1 and a2 and (a2 in a1 or a1 in a2):
        return "论元重叠"
    # Chinese clauses are short: "３人受伤" (4) and "错了。" (3) are real
    # arguments.  Only a bare noun fragment (<= 2 characters) is a defect.
    if a2 and len(a2) <= 2:
        return "后件过短"
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--column", default="annotator_sense_l1")
    ap.add_argument("--split", choices=["all", "pilot", "main"], default="all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    items = {it.id: it for it in load_items(args.items)}
    a = read_sheet(Path(args.a), args.column)
    b = read_sheet(Path(args.b), args.column)

    ids = [i for i in a if i in b and i in items]
    if args.split != "all":
        ids = [i for i in ids if items[i].meta.get("b1_split") == args.split]

    undecided = [i for i in ids if a[i] == "UNKNOWN" or b[i] == "UNKNOWN"]
    decided = [i for i in ids if i not in set(undecided)]
    disagree = [i for i in decided if a[i] != b[i]]

    pairs = Counter(tuple(sorted((a[i], b[i]))) for i in disagree)
    pairs = Counter({f"{x} | {y}": n for (x, y), n in pairs.items()})
    by_kind = Counter("显式" if items[i].explicit else "隐式" for i in disagree)

    # Who does the corpus gold side with on the disagreeing items?
    gold_side = Counter()
    for i in disagree:
        g = items[i].sense_l1
        if g == a[i] and g == b[i]:
            gold_side["与双方一致"] += 1
        elif g == a[i]:
            gold_side["与 A 一致"] += 1
        elif g == b[i]:
            gold_side["与 B 一致"] += 1
        else:
            gold_side["与双方都不同"] += 1

    conn = Counter(items[i].connective for i in disagree
                   if items[i].connective and items[i].explicit)
    defects = [i for i in disagree if defect(items[i])]

    rows = []
    for i in disagree:
        it = items[i]
        rows.append({
            "item_id": i,
            "split": it.meta.get("b1_split", ""),
            "task": "显式" if it.explicit else "隐式",
            "connective": it.connective,
            "a": a[i], "b": b[i], "gold": it.sense_l1,
            "defect": defect(it),
            "arg1": it.arg1 or it.text,
            "arg2": it.arg2,
        })
    rows.sort(key=lambda r: (r["defect"] != "", r["task"], r["item_id"]))

    result = {
        "split": args.split,
        "n_items": len(ids),
        "n_both_decided": len(decided),
        "n_disagree": len(disagree),
        "disagree_rate_of_decided": round(len(disagree) / max(1, len(decided)), 4),
        "n_undecided": len(undecided),
        "pairs": dict(pairs.most_common()),
        "by_task": dict(by_kind),
        "gold_side": dict(gold_side),
        "connective_top": dict(conn.most_common(12)),
        "n_defective": len(defects),
        "defective_ids": defects,
        "rows": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
