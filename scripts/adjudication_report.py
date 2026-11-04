"""Summarise a completed adjudication round.

Three questions, in the order that matters for the paper:

    1. coverage -- how many of the worksheets got a real label, and what
       happened to the rest (unresolved / excluded / still pending)
    2. agreement with each source -- only the ten triple disagreements let the
       adjudicator override gold; everywhere else gold stays final, so the
       adjudicator's cell is reported but not applied.  That distinction
       matters because on priority 2 the gold column always equals one of the
       two annotators: "followed gold" and "followed the annotator who matched
       gold" are the same act, and the worksheet shows the gold column.
    3. what the reference does to model scores -- on the disputed items, are
       the models closer to the final reference than to the corpus gold?

    python scripts/adjudication_report.py \
        --items data/processed/l1/b1_packet.jsonl \
        --worksheet data/interim/b1_round1/adjudication_worksheet.csv \
        --a data/interim/b1_round1/submissions/C_latest.csv \
        --b data/interim/b1_round1/submissions/D_latest.csv \
        --preds 'outputs/l1_b1packet_*_zs*.jsonl' \
        --out-json outputs/l1_adjudication.json \
        --out-md refine-logs/L1_ADJUDICATION.md
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
# Sibling script: the override rule has to be the same one merge_worksheet.py
# applies, or the report will describe a reference file it did not build.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from discourse_reliability.labels import L1_CLASSES, normalize_sense  # noqa: E402
from discourse_reliability.schema import load_items, load_predictions  # noqa: E402
from merge_worksheet import is_triple_disagreement  # noqa: E402

DECISION = "最终裁决"
GOLD_SIDE = ("gold=both", "gold=C", "gold=D", "C=D!=gold", "gold=none")


def read_sheet(path: Path, column: str) -> dict[str, str]:
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get(column) or "").strip().upper()
            if v:
                out[row["item_id"]] = v
    return out


def gold_side(g: str, c: str, b: str) -> str:
    if g == c and g == b:
        return "gold=both"
    if g == c:
        return "gold=C"
    if g == b:
        return "gold=D"
    if c == b:
        return "C=D!=gold"
    return "gold=none"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--worksheet", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args(argv)

    items = {it.id: it for it in load_items(args.items)}
    annot_a = read_sheet(Path(args.a), "annotator_sense_l1")
    annot_b = read_sheet(Path(args.b), "annotator_sense_l1")

    rows = []
    status = Counter()
    with Path(args.worksheet).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get(DECISION) or "").strip()
            label = normalize_sense(raw)
            if label in L1_CLASSES:
                kind = "labelled"
            elif not raw:
                kind = "blank"
            elif raw.upper() == "UNKNOWN":
                kind = "unresolved"
            elif "删除" in raw or "剔除" in raw:
                kind = "excluded"
            else:
                kind = "pending"
            status[kind] += 1
            if kind != "labelled":
                continue
            g = normalize_sense(row["语料金标准"])
            c = normalize_sense(row["C 的标注"])
            b = normalize_sense(row["D 的标注"])
            triple = is_triple_disagreement(row)
            # The protocol: the adjudicator's cell becomes the reference only
            # on triple disagreements; everywhere else gold is final.
            final = label if triple else g
            rows.append({
                "item_id": row["item_id"],
                "priority": row["优先级"],
                "task": row["任务"],
                "gold": g, "c": c, "d": b,
                "adjudicator": label, "final": final,
                "triple": triple,
                "side": gold_side(g, c, b),
                "adjudicator_agrees_gold": label == g,
                "changed": final != g,
                "adjudicator_cell_dropped": (not triple) and label != g,
            })

    def tally(sub):
        s = Counter(r["side"] for r in sub)
        agree = sum(r["adjudicator_agrees_gold"] for r in sub)
        return {
            "n": len(sub),
            "by_gold_side": {k: s.get(k, 0) for k in GOLD_SIDE},
            # How the adjudicator's own cell compared to gold -- informative
            # about their reading, but not what the reference file contains.
            "agree_gold": agree,
            "agree_gold_pct": round(agree / max(1, len(sub)), 4),
            "adjudicator_disagreed": len(sub) - agree,
            "changed": sum(r["changed"] for r in sub),
            "triple": sum(r["triple"] for r in sub),
            "cell_dropped": sum(r["adjudicator_cell_dropped"] for r in sub),
        }

    by_priority = {p: tally([r for r in rows if r["priority"] == p]) for p in ("1", "2", "3")}
    by_task = {t: tally([r for r in rows if r["task"] == t]) for t in ("显式", "隐式")}

    # On the priority-2 block gold always equals one of the two annotators, so
    # "followed the gold" and "followed the annotator who agrees with gold" are
    # the same act.  Count that explicitly instead of letting the reader assume
    # the adjudicator verified the label.
    p2 = [r for r in rows if r["priority"] == "2"]
    confounded = {
        "n": len(p2),
        "gold_matches_one_annotator": sum(r["side"] in ("gold=C", "gold=D") for r in p2),
        "decision_equals_gold": sum(r["adjudicator_agrees_gold"] for r in p2),
        "adjudicator_cell_dropped": sum(r["adjudicator_cell_dropped"] for r in p2),
    }

    triple_rows = [r for r in rows if r["triple"]]
    rule = {
        "triple_disagreements": len(triple_rows),
        "reference_adopts_adjudicator": len(triple_rows),
        "reference_keeps_corpus_gold": len(rows) - len(triple_rows),
        "adjudicator_cells_dropped": sum(r["adjudicator_cell_dropped"] for r in rows),
        "reference_changed_vs_corpus_gold": sum(r["changed"] for r in rows),
    }

    models: dict[str, dict[str, str]] = {}
    for spec in args.preds:
        for path in sorted(ROOT.glob(spec)):
            for p in load_predictions(path):
                if p.error is None:
                    models.setdefault(p.model, {})[p.item_id] = p.label_l1

    final_ref = {r["item_id"]: r["final"] for r in rows}
    changed_ids = [r["item_id"] for r in rows if r["changed"]]
    model_scores = []
    for name, pred in models.items():
        seen = [i for i in final_ref if i in pred]
        if not seen:
            continue
        vs_ref = sum(pred[i] == final_ref[i] for i in seen) / len(seen)
        vs_gold = sum(pred[i] == items[i].sense_l1 for i in seen) / len(seen)
        on_changed = sum(pred[i] == final_ref[i] for i in changed_ids if i in pred)
        model_scores.append({
            "model": name, "n": len(seen),
            "acc_vs_final_reference": round(vs_ref, 4),
            "acc_vs_corpus_gold": round(vs_gold, 4),
            "delta": round(vs_ref - vs_gold, 4),
            "right_on_changed": on_changed,
            "n_changed": len(changed_ids),
        })

    report = {
        "worksheet_rows": sum(status.values()),
        "status": dict(status),
        "labelled": len(rows),
        "rule": rule,
        "by_priority": by_priority,
        "by_task": by_task,
        "priority2_confound": confounded,
        "decisions_by_class": dict(Counter(r["final"] for r in rows).most_common()),
        "changed_items": [
            {"item_id": r["item_id"], "gold": r["gold"], "final": r["final"],
             "c": r["c"], "d": r["d"], "priority": r["priority"]}
            for r in rows if r["changed"]
        ],
        "model_scores": model_scores,
    }
    out = ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# 裁决轮次报告", "",
          f"工作表 **{report['worksheet_rows']}** 行，其中 **{len(rows)}** 条得到 L1 标签。", "",
          "| 状态 | 条数 |", "|---|---|"]
    for k, label in (("labelled", "裁决格已填标签"), ("blank", "空白"),
                     ("pending", "待定"), ("unresolved", "裁决者也无法判定"),
                     ("excluded", "已从语料剔除（不在包内）")):
        md.append(f"| {label} | {status.get(k, 0)} |")

    md += ["", "## 采用规则：只有三重分歧才以裁决为准", "",
           f"工作表 91 条里，**{rule['triple_disagreements']} 条是三重分歧**"
           "（语料金标准、标注者 C、标注者 D 三者两两不同），这 10 条以裁决者的判断为最终答案；"
           f"其余 **{rule['reference_keeps_corpus_gold']} 条一律以语料金标准为最终正确答案**，"
           "裁决列的填写只作记录，不改写标签。", "",
           "这条规则排除了一类本来无法成立的证据：优先级 2 的 57 条里，金标准总是等于两位标注者之一，"
           f"所以裁决有 {confounded['decision_equals_gold']} 条与金标准一致，"
           "既可以说是「裁决独立支持了金标准」，也可以说是「裁决跟着那位与金标准一致的标注者走」——"
           "裁决表里金标准列可见，两种解释都成立。把这些单元格直接写成新标签，"
           f"等于凭空制造 {confounded['adjudicator_cell_dropped']} 条「金标准被独立验证」的记录。"
           "按现在的规则，它们不进入参考文件。", "",
           "## 与语料金标准的关系", "",
           "| 优先级 | n | 裁决格与金标准一致 | 采用为参考 | 金标准=C | 金标准=D | 不等于任何一方 |",
           "|---|---|---|---|---|---|---|"]
    for p, t in by_priority.items():
        s = t["by_gold_side"]
        md.append(f"| {p} | {t['n']} | {t['agree_gold']}（{t['agree_gold_pct']:.1%}） | "
                  f"{t['triple']} | {s['gold=C']} | {s['gold=D']} | "
                  f"{s['C=D!=gold'] + s['gold=none']} |")

    md += ["", "## 三个优先级其实在裁三件不同的事", "",
           "| 优先级 | 裁决对象 | n | 裁决格与金标准一致 | 是否采用 |", "|---|---|---|---|---|",
           f"| 1 | 金标准 vs 两位标注者（三重分歧） | {by_priority['1']['n']} | "
           f"{by_priority['1']['agree_gold']} | **全部采用** |",
           f"| 2 | 标注者 C vs 标注者 D | {by_priority['2']['n']} | "
           f"{by_priority['2']['agree_gold']} | 不采用，维持金标准 |",
           f"| 3 | 该条目能不能判 | {by_priority['3']['n']} | "
           f"{by_priority['3']['agree_gold']} | 不采用，维持金标准 |", "",
           f"读者容易把三行加起来说成「金标准被裁决支持 {sum(t['agree_gold'] for t in by_priority.values())} 次」，"
           "但只有优先级 1 是真正的三方对照，而且只有这一块改写了参考标签。", "",
           "### 优先级 2 的混淆", "",
           f"优先级 2 共 {confounded['n']} 条，其中 **{confounded['gold_matches_one_annotator']} 条**"
           "的语料金标准恰好等于两位标注者之一。"
           f"裁决有 {confounded['decision_equals_gold']} 条与金标准一致。", "",
           "也就是说，在这一块里「遵循金标准」和「遵循那位与金标准一致的标注者」"
           "**是同一个动作**，无法区分。裁决表里金标准列是可见的，所以这两个解释都成立。"
           f"其中 {confounded['adjudicator_cell_dropped']} 条裁决格与金标准不同——"
           "按规则不改写标签，但这些分歧本身仍是可用的线索（模型是否也读到同样的区别）。", "",
           "## 改写条目（相对语料金标准）", "",
           f"共 **{rule['reference_changed_vs_corpus_gold']} 条**，全部来自优先级 1：", "",
           "| item_id | 金标准 | 最终参考（裁决） | 优先级 |", "|---|---|---|---|"]
    for r in rows:
        if r["changed"]:
            md.append(f"| `{r['item_id']}` | {r['gold']} | **{r['final']}** | {r['priority']} |")

    if model_scores:
        md += ["", "## 模型在争议条目上更接近哪一边", "",
               f"| 模型 | n | vs 最终参考 | vs 金标准 | Δ | {len(changed_ids)} 条改写中猜对 |",
               "|---|---|---|---|---|---|"]
        for s in model_scores:
            md.append(f"| {s['model']} | {s['n']} | {s['acc_vs_final_reference']:.4f} | "
                      f"{s['acc_vs_corpus_gold']:.4f} | {s['delta']:+.4f} | "
                      f"{s['right_on_changed']}/{s['n_changed']} |")
        md += ["", f"含义：如果模型「只是复现旧标注」，Δ 应当 ≤ 0。"
                   f"改写只有 {len(changed_ids)} 条，Δ 的正负方向可以看，幅度不可读。"]

    # refine-logs/ lives one level above the package root, so a bare relative
    # path has to be probed in both places instead of assumed.
    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = (ROOT / out_md) if (ROOT / out_md).parent.exists() else (ROOT.parent / out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("changed_items",)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
