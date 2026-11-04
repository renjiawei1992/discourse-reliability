"""R161: build a blinded packet for a second (human) coder.

The first pass over the 62 errors was coded by Claude; ``typology_sheet.csv``
has empty ``coder2_*`` columns waiting to be filled.  This script exports the
same 62 items for a human coder **without** the first coder's category or
rationale, in a shuffled order, so the second pass is independent.

Two things it deliberately does not do: it does not show ``coder1_reason``
(anchoring), and it does not group items by category (order effects).

    python scripts/build_r161_coder2_packet.py \
        --sheet data/interim/r161/typology_sheet.csv \
        --gold data/processed/l1/hit_balanced.jsonl \
        --preds outputs/l1_hit_balanced_qwen3.5-plus_zs_filled.jsonl \
        --out data/interim/r161/coder2_packet.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items  # noqa: E402

HEADER = [
    "序号", "item_id", "连接词", "论元1", "论元2", "全文",
    "金标准", "模型判断", "coder2_category", "coder2_note",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="data/interim/r161/typology_sheet.csv")
    ap.add_argument("--gold", default="data/processed/l1/hit_balanced.jsonl")
    ap.add_argument("--preds",
                    default="outputs/l1_hit_balanced_qwen3.5-plus_zs_filled.jsonl")
    ap.add_argument("--out", default="data/interim/r161/coder2_packet.csv")
    ap.add_argument("--seed", type=int, default=161)
    args = ap.parse_args(argv)

    with (ROOT / args.sheet).open(encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    wanted = [r["item_id"] for r in sheet]

    gold = {it.id: it for it in load_items(ROOT / args.gold)}
    preds = {}
    with (ROOT / args.preds).open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            preds[o["item_id"]] = o["label_l1"]

    missing = [i for i in wanted if i not in gold or i not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} item ids not found, e.g. {missing[:3]}")

    rows = []
    for i in wanted:
        it = gold[i]
        rows.append([
            "", i, (it.connective or "（无，隐式项）"), it.arg1 or "", it.arg2 or "",
            (it.text or "").replace("\n", " "),
            it.sense_l1, preds[i], "", "",
        ])

    # Shuffle so the coder cannot infer blocks from order; ids stay stable.
    random.Random(args.seed).shuffle(rows)
    for n, row in enumerate(rows, start=1):
        row[0] = n

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)

    print(f"wrote {out} ({len(rows)} items, order shuffled with seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
