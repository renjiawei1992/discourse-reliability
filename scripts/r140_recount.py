"""R140 recount: separate cue deletion from content deletion.

The delete-connective condition replaces the span ``conn_char_start:conn_char_end``
with ``____``.  For a single-token connective that removes the cue.  For a
paired frame ("当初…如今") the span covers everything in between, so the mutation
deletes *content*, and the resulting accuracy drop cannot be read as cue
reliance.  The M4 readability check showed this directly: single-token items
stay readable after deletion, paired frames do not.

This script re-splits the whole R140 population and recomputes the headline
numbers on the single-token subset.  It also reports the three probe items that
were never mutated (their connective is absent from the text, so the variant is
byte-identical to the original).

    python scripts/r140_recount.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items  # noqa: E402


def frame_parts(connective: str) -> list[str]:
    c = connective.replace("...", "…").replace("……", "…")
    return [p for p in c.split("…") if p]


def classify(it) -> str:
    s, e = it.conn_char_start, it.conn_char_end
    span = it.text[s:e] if s >= 0 and e > s else ""
    if span and span == it.connective:
        return "single"
    parts = frame_parts(it.connective)
    if span and len(parts) >= 2 and span.startswith(parts[0]) and any(
            span.endswith(p) for p in parts[1:]):
        return "paired_frame"
    if it.connective and it.connective in it.text:
        return "fallback"
    return "unmutated"


def load_preds(path: Path, strip_suffix: str = "") -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            iid = o["item_id"]
            if strip_suffix:
                iid = iid.replace(strip_suffix, "")
            out[iid] = o.get("label_l1") or ""
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/l1_r140_recount.json")
    ap.add_argument("--out-md", default="../refine-logs/R140_RECOUNT.md")
    args = ap.parse_args(argv)

    gold = {}
    for p in ["data/processed/l1/hit_balanced.jsonl", "data/processed/l1/ted_balanced.jsonl"]:
        for it in load_items(ROOT / p):
            gold[it.id] = it

    base = {}
    for p in ["outputs/l1_ds_zeroshot.jsonl", "outputs/l1_ds_ted_zeroshot.jsonl"]:
        base.update(load_preds(ROOT / p))
    noconn = {}
    for p in ["outputs/l1_probe_noconn.jsonl", "outputs/l1_probe_ted_noconn.jsonl"]:
        noconn.update(load_preds(ROOT / p, strip_suffix="::no_conn"))

    kinds: dict[str, list[str]] = {}
    for iid in noconn:
        it = gold.get(iid)
        if it is None:
            continue
        kinds.setdefault(classify(it), []).append(iid)

    def stats(ids: list[str]) -> dict:
        ok_b = sum(1 for i in ids if base.get(i) == gold[i].sense_l1)
        ok_n = sum(1 for i in ids if noconn.get(i) == gold[i].sense_l1)
        flip = sum(1 for i in ids if base.get(i) != noconn.get(i))
        return {"n": len(ids),
                "acc_base": round(ok_b / len(ids), 4),
                "acc_noconn": round(ok_n / len(ids), 4),
                "delta": round((ok_n - ok_b) / len(ids), 4),
                "flip_rate": round(flip / len(ids), 4)}

    out = {"by_kind": {k: stats(v) for k, v in sorted(kinds.items())},
           "single": stats(kinds.get("single", [])),
           "all_mutated": stats([i for k, v in kinds.items() if k != "unmutated" for i in v]),
           "kind_counts": {k: len(v) for k, v in sorted(kinds.items())}}

    p = ROOT / args.out
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# R140 重算：删掉的是线索，还是内容", "",
         "模型 deepseek-chat zero-shot，与 R140 同一批预测。", "",
         "| 删除方式 | n | 原句 acc | 删后 acc | Δ | 翻标签率 |",
         "|---|---|---|---|---|---|"]
    label = {"single": "单词连接词（删的就是那个词）",
             "paired_frame": "成对框架（整个跨论元片段被删）",
             "fallback": "无偏移回退（删首次出现）",
             "unmutated": "**从未被删**（连接词不在正文里）"}
    for k, v in sorted(kinds.items()):
        s = out["by_kind"][k]
        L.append(f"| {label.get(k, k)} | {s['n']} | {s['acc_base']} | {s['acc_noconn']} | "
                 f"{s['delta']:+.4f} | {s['flip_rate']} |")
    s = out["single"]
    L.append(f"| **单词连接词合计** | {s['n']} | {s['acc_base']} | {s['acc_noconn']} | "
             f"{s['delta']:+.4f} | {s['flip_rate']} |")
    L.append("")
    L.append("> `paired_frame` 与 `fallback` 的掉点里混着内容删除，不能算作线索依赖；"
             "`unmutated` 的变异与原句完全相同，本不该进这个条件。"
             "正文应报单词连接词那一行。")
    L.append("")

    mp = ROOT / args.out_md
    mp.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out_md}")
    print(json.dumps(out["kind_counts"], ensure_ascii=False))
    print(json.dumps(out["single"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
