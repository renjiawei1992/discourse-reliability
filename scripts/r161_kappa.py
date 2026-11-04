"""R161: agreement between the two coders of the error typology.

Coder 1 is the first pass (Claude, recorded in ``typology_sheet.csv``); coder 2
is the human who fills ``coder2_packet.csv``.  The two files are merged on
``item_id``, because the packet is shuffled.

Reports Cohen's kappa over the five fixed categories, the confusion matrix, and
the items the two coders disagree on.  If ``--adjudication`` points at an
existing CSV of third-party calls for those disagreements, a third tally is
reported alongside the two coder tallies (the paper reports all three; neither
coder's set is treated as the truth).

    python scripts/r161_kappa.py \
        --sheet data/interim/r161/typology_sheet.csv \
        --coder2 data/interim/r161/coder2_packet.csv \
        --adjudication data/interim/r161/adjudication.csv \
        --out outputs/l1_r161_kappa.json \
        --out-md ../refine-logs/R161_KAPPA.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATS = {
    "C1": "多义连接词义项误读",
    "C2": "论元内部假连接词",
    "C3": "时间框架的归属歧义（时间 vs 对比）",
    "C4": "隐式项的类别边界",
    "C5": "语料切分或标注存疑",
}


def read_csv(path: Path, key: str = "item_id") -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r[key].strip(): r for r in csv.DictReader(f) if r.get(key)}


def cohen_kappa(a: list[str], b: list[str]) -> dict:
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else None
    return {"n": n, "po": round(po, 4),
            "pe": round(pe, 4),
            "kappa": round(k, 4) if k is not None else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="data/interim/r161/typology_sheet.csv")
    ap.add_argument("--coder2", default="data/interim/r161/coder2_packet.csv")
    ap.add_argument("--adjudication", default="data/interim/r161/adjudication.csv",
                    help="optional third-party calls for the disagreement items")
    ap.add_argument("--out", default="outputs/l1_r161_kappa.json")
    ap.add_argument("--out-md", default="../refine-logs/R161_KAPPA.md")
    args = ap.parse_args(argv)

    sheet = read_csv(ROOT / args.sheet)
    packet = read_csv(ROOT / args.coder2)
    adj_path = ROOT / args.adjudication if args.adjudication else None
    adj = read_csv(adj_path) if adj_path and adj_path.exists() else {}

    rows, blank = [], []
    for iid, s in sheet.items():
        p = packet.get(iid)
        c1 = (s.get("coder1_category") or "").strip().upper()
        c2 = ((p or {}).get("coder2_category") or "").strip().upper()
        if not c1:
            continue
        if c2 not in CATS:
            blank.append(iid)
            continue
        rows.append({"item_id": iid, "coder1": c1, "coder2": c2,
                     "gold": s.get("gold", ""), "pred": s.get("pred", ""),
                     "coder2_note": (p or {}).get("coder2_note", "")})

    if blank:
        print(f"WARNING: {len(blank)} items not coded by coder 2 yet "
              f"(e.g. {blank[:3]}) — kappa below is over the {len(rows)} coded ones")

    a = [r["coder1"] for r in rows]
    b = [r["coder2"] for r in rows]
    kappa = cohen_kappa(a, b) if rows else {}
    confusion = Counter((x, y) for x, y in zip(a, b))
    disagreements = [r for r in rows if r["coder1"] != r["coder2"]]
    counts = {c: {"coder1": a.count(c), "coder2": b.count(c)} for c in CATS}

    # Third tally: agreements keep the coder-1 call (they are identical anyway),
    # disagreements take the adjudicated call when one exists.
    third, adj_rows, unadjudicated = [], [], []
    for r in rows:
        if r["coder1"] == r["coder2"]:
            third.append(r["coder1"])
            continue
        cell = (adj.get(r["item_id"], {}).get("adjudicated") or "").strip().upper()
        if cell not in CATS:
            unadjudicated.append(r["item_id"])
            third.append(r["coder1"])
            continue
        third.append(cell)
        adj_rows.append({**r, "adjudicated": cell,
                         "basis": adj[r["item_id"]].get("basis", "")})
    adj_complete = bool(adj_rows) and not unadjudicated
    for c in CATS:
        counts[c]["adjudicated"] = third.count(c)

    out = {"n_coded": len(rows), "kappa": kappa, "counts": counts,
           "confusion": {f"{x}->{y}": n for (x, y), n in sorted(confusion.items())},
           "disagreements": disagreements,
           "adjudication": {"source": str(args.adjudication) if adj else None,
                            "complete": adj_complete,
                            "adjudicated": len(adj_rows),
                            "missing": unadjudicated,
                            "rows": adj_rows}}
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# R161 错误类型学：双编码一致性\n",
         f"编码条目 {len(rows)} 条（qwen3.5-plus 在 hit_balanced 上的错误）。\n"]
    if kappa:
        L.append(f"**Cohen κ = {kappa['kappa']}**，原始一致率 {kappa['po']:.4f}，"
                 f"期望一致率 {kappa['pe']:.4f}。\n")
    has_adj = bool(adj_rows)
    head = "| 类别 | 编码者 1 | 编码者 2 |"
    rule = "|---|---|---|"
    if has_adj:
        head += " 裁决 |"
        rule += "---|"
    L.append(head)
    L.append(rule)
    for c in CATS:
        line = f"| {c} {CATS[c]} | {counts[c]['coder1']} | {counts[c]['coder2']} |"
        if has_adj:
            line += f" {counts[c]['adjudicated']} |"
        L.append(line)
    L.append("")
    L.append("## 混淆矩阵（行 = 编码者 1，列 = 编码者 2）\n")
    L.append("| | " + " | ".join(CATS) + " |")
    L.append("|---|" + "---|" * len(CATS))
    for x in CATS:
        L.append(f"| **{x}** | " + " | ".join(str(confusion.get((x, y), 0)) for y in CATS) + " |")
    L.append("")
    if disagreements:
        L.append(f"## 分歧 {len(disagreements)} 条\n")
        L.append("| item_id | 金标准 | 模型 | 编码者 1 | 编码者 2 | 备注 |")
        L.append("|---|---|---|---|---|---|")
        for r in disagreements:
            L.append(f"| {r['item_id']} | {r['gold']} | {r['pred']} | "
                     f"{r['coder1']} | {r['coder2']} | {r['coder2_note']} |")
        L.append("")

    if has_adj:
        n1 = sum(1 for r in adj_rows if r["adjudicated"] == r["coder1"])
        n2 = sum(1 for r in adj_rows if r["adjudicated"] == r["coder2"])
        n0 = len(adj_rows) - n1 - n2
        L.append("## 裁决\n")
        L.append(f"分歧 {len(disagreements)} 条中，裁决支持编码者 1 {n1} 条、"
                 f"编码者 2 {n2} 条、两者都不支持 {n0} 条。\n")
        if unadjudicated:
            L.append(f"> 未裁决 {len(unadjudicated)} 条，第三套计数在这些条目上暂用编码者 1 的值。\n")
        L.append("| item_id | 金标准 | 模型 | 编码者 1 | 编码者 2 | 裁决 | 依据 |")
        L.append("|---|---|---|---|---|---|---|")
        for r in adj_rows:
            L.append(f"| {r['item_id']} | {r['gold']} | {r['pred']} | {r['coder1']} | "
                     f"{r['coder2']} | **{r['adjudicated']}** | {r['basis']} |")
        L.append("")
        L.append("> 裁决者与编码者 1 是同一人，不是独立第三方，第三套计数可能偏向编码者 1；"
                 "它只给分歧一个可复现的落点，不是「正确答案」。"
                 "正文同时报编码者 1、编码者 2 与裁决三套计数。\n")

    L.append("> 双编码的一致性只说明两套判断可复现，**不说明哪一套对**。"
             "分歧条目裁决后才能报各类占比；正文里应同时给出两套计数。\n")

    mp = ROOT / args.out_md
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out_md}")
    summary = {"n_coded": len(rows), **kappa}
    if has_adj:
        summary["adjudicated_counts"] = {c: counts[c]["adjudicated"] for c in CATS}
        summary["adjudication_complete"] = adj_complete
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
