"""R140b: rebuild the paired-frame variants without deleting their content.

R140 replaced ``conn_char_start:conn_char_end`` with ``____``.  For a paired
frame ("当初…如今", "目前…到二十一世纪") that span runs from the first token to
the last, so the mutation deleted the whole clause between them.  The M4 check
showed the consequence: humans could judge 99% of single-token deletions but
only 21% of these.

This script builds the fixed variant — each frame token becomes ``____``, the
text between them stays — for the model re-run, and exports a short human
packet covering the same 14 items the annotator already saw, so the two
constructions can be compared item by item.

    python scripts/build_r140b_paired.py
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
sys.path.insert(0, str(ROOT / "scripts"))

from discourse_reliability.counterfactual import (  # noqa: E402
    _frame_token_spans,
    no_connective_tokens,
)
from discourse_reliability.schema import load_items  # noqa: E402
from r140_recount import classify  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-items", default="data/processed/l1/r140b_paired.jsonl")
    ap.add_argument("--m4-dir", default="data/interim/m4")
    ap.add_argument("--m4b-dir", default="data/interim/m4b")
    ap.add_argument("--seed", type=int, default=161)
    args = ap.parse_args(argv)

    gold = {}
    for p in ["data/processed/l1/hit_balanced.jsonl", "data/processed/l1/ted_balanced.jsonl"]:
        for it in load_items(ROOT / p):
            gold[it.id] = it

    base_ids = []
    for p in ["outputs/l1_probe_noconn.jsonl", "outputs/l1_probe_ted_noconn.jsonl"]:
        with (ROOT / p).open(encoding="utf-8") as f:
            for line in f:
                base_ids.append(json.loads(line)["item_id"].replace("::no_conn", ""))

    paired = [gold[i] for i in base_ids if classify(gold[i]) == "paired_frame"]

    variants, failed = [], []
    for it in paired:
        v = no_connective_tokens(it)
        if v is None:
            failed.append(it.id)
            continue
        variants.append(v)
    if failed:
        raise SystemExit(f"{len(failed)} paired frames could not be rebuilt: {failed[:3]}")

    # The between-token text must survive verbatim, and the connective tokens
    # must be gone.  Check both rather than trusting the splice.
    problems = []
    for v in variants:
        base = gold[v.meta["base_id"]]
        spans = _frame_token_spans(base)
        n_marks = v.text.count("____")
        if n_marks != len(spans):
            problems.append((v.id, f"{n_marks} placeholders for {len(spans)} tokens"))
            continue
        # Deleting the placeholders must leave exactly the original minus the
        # connective tokens: no content lost, no token left behind.
        kept, prev = [], 0
        for a, b in spans:
            kept.append(base.text[prev:a])
            prev = b
        kept.append(base.text[prev:])
        if v.text.replace("____", "") != "".join(kept):
            problems.append((v.id, "reconstructed text does not match"))
    if problems:
        raise SystemExit(f"{len(problems)} variants failed verification, e.g. {problems[:3]}")

    out = ROOT / args.out_items
    with out.open("w", encoding="utf-8") as f:
        for v in variants:
            v.meta = {**v.meta, "fold": v.meta.get("fold", 0)}
            f.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")

    # Human follow-up: the 14 items already seen in the old construction.
    key_path = ROOT / args.m4_dir / "m4_key.csv"
    with key_path.open(encoding="utf-8-sig", newline="") as f:
        sampled = [r for r in csv.DictReader(f) if r.get("paired_frame") == "1"]
    by_id = {v.meta["base_id"]: v for v in variants}
    rows = []
    for r in sampled:
        v = by_id.get(r["item_id"])
        if v is None:
            continue
        rows.append([r["pid"], r["item_id"], v.text])
    random.Random(args.seed + 11).shuffle(rows)

    m4b = ROOT / args.m4b_dir
    m4b.mkdir(parents=True, exist_ok=True)
    with (m4b / "m4b_block.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["序号", "pid", "句子", "你的判断", "备注"])
        for n, (pid, _iid, text) in enumerate(rows, start=1):
            w.writerow([n, pid, f"句子（原连接词已被删除，用 ____ 标出）：\n{text}", "", ""])

    print(f"wrote {len(variants)} variants to {args.out_items}")
    print(f"wrote {len(rows)}-item human packet to {args.m4b_dir}/m4b_block.csv")
    for v in variants[:3]:
        print("  " + v.meta["target_connective"] + " -> " + v.text[:70])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
