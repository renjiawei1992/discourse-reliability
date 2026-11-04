"""R140 final: single-token deletions plus the rebuilt paired frames.

R140 mixed two mutations in one condition: single-token cue deletion (valid)
and whole-span deletion on paired frames (deletes content as well).  The
paired frames were rebuilt so only their own tokens are blanked
(``no_connective_tokens``) and re-run; this script puts the two halves back
together and reports the number the paper should quote.

    python scripts/r140_final.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from discourse_reliability.schema import load_items  # noqa: E402
from r140_recount import classify, load_preds  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-probe", default="outputs/l1_probe_noconn_tokens.jsonl")
    ap.add_argument("--out", default="outputs/l1_r140_final.json")
    ap.add_argument("--out-md", default="../refine-logs/R140_FINAL.md")
    args = ap.parse_args(argv)

    new_path = ROOT / args.new_probe
    if not new_path.exists():
        raise SystemExit(f"{args.new_probe} not found — run the re-run first")

    gold = {}
    for p in ["data/processed/l1/hit_balanced.jsonl", "data/processed/l1/ted_balanced.jsonl"]:
        for it in load_items(ROOT / p):
            gold[it.id] = it
    base = {}
    for p in ["outputs/l1_ds_zeroshot.jsonl", "outputs/l1_ds_ted_zeroshot.jsonl"]:
        base.update(load_preds(ROOT / p))
    old = {}
    for p in ["outputs/l1_probe_noconn.jsonl", "outputs/l1_probe_ted_noconn.jsonl"]:
        old.update(load_preds(ROOT / p, strip_suffix="::no_conn"))
    new = load_preds(new_path, strip_suffix="::no_conn_tokens")

    kinds: dict[str, list[str]] = {}
    for iid in old:
        kinds.setdefault(classify(gold[iid]), []).append(iid)

    def stats(ids: list[str], preds: dict[str, str]) -> dict:
        ok_b = sum(1 for i in ids if base.get(i) == gold[i].sense_l1)
        ok_n = sum(1 for i in ids if preds.get(i) == gold[i].sense_l1)
        flip = sum(1 for i in ids if base.get(i) != preds.get(i))
        return {"n": len(ids), "acc_base": round(ok_b / len(ids), 4),
                "acc_noconn": round(ok_n / len(ids), 4),
                "delta": round((ok_n - ok_b) / len(ids), 4),
                "flip_rate": round(flip / len(ids), 4)}

    single = kinds.get("single", [])
    paired = kinds.get("paired_frame", [])
    missing = [i for i in paired if i not in new]

    core = single + paired
    ok_b = sum(1 for i in core if base.get(i) == gold[i].sense_l1)
    ok_n = (sum(1 for i in single if old.get(i) == gold[i].sense_l1)
            + sum(1 for i in paired if new.get(i) == gold[i].sense_l1))
    flip = (sum(1 for i in single if base.get(i) != old.get(i))
            + sum(1 for i in paired if base.get(i) != new.get(i)))

    out = {
        "single": stats(single, old),
        "paired_old_span_delete": stats(paired, old),
        "paired_new_token_delete": stats([i for i in paired if i in new], new),
        "corrected_total": {"n": len(core), "acc_base": round(ok_b / len(core), 4),
                            "acc_noconn": round(ok_n / len(core), 4),
                            "delta": round((ok_n - ok_b) / len(core), 4),
                            "flip_rate": round(flip / len(core), 4)},
        "excluded_never_mutated": len(kinds.get("unmutated", [])),
        "missing_new_predictions": missing,
    }
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# R140 最终数字：只删线索，不删内容", "",
         "模型 deepseek-chat zero-shot，与原 R140 同一设置。", "",
         "| 集合 | n | 原句 acc | 删后 acc | Δ | 翻标签率 |",
         "|---|---|---|---|---|---|"]
    rows = [("单词连接词（原构造）", out["single"]),
            ("成对框架 · 旧构造（整段删除）", out["paired_old_span_delete"]),
            ("成对框架 · 新构造（只删框架词）", out["paired_new_token_delete"]),
            ("**校正后合计**", out["corrected_total"])]
    for name, s in rows:
        L.append(f"| {name} | {s['n']} | {s['acc_base']} | {s['acc_noconn']} | "
                 f"{s['delta']:+.4f} | {s['flip_rate']} |")
    L.append("")
    if missing:
        L.append(f"> 注意：{len(missing)} 条新构造没有预测，未计入合计。")
        L.append("")
    L.append(f"> 另有 {out['excluded_never_mutated']} 条的连接词不在正文里，"
             "旧构造从未真正删过，已从条件中剔除。")
    L.append("")
    L.append("正文报「校正后合计」那一行。旧构造的成对框架数字保留在这里，"
             "用来说明为什么不能把两类条目混在一起算。")
    L.append("")
    (ROOT / args.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out_md}")
    print(json.dumps(out["corrected_total"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
