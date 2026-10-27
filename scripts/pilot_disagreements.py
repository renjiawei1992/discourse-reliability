"""Turn an annotator's disagreements with the corpus gold into a discussion sheet.

Used for the pilot review: the point is not to tell the annotator they were
wrong, but to find the guideline rule that is missing.  Each row therefore
carries a targeted question rather than a correction.

Model predictions are exported to a **separate** section and marked
"do not show the annotator" -- seeing them first would anchor the discussion on
agreement with the model instead of on the guideline.

    python scripts/pilot_disagreements.py \
        --items data/processed/l1/b1_packet.jsonl \
        --annotator data/interim/b1_round1/submissions/D_latest.csv \
        --preds outputs/l1_b1packet_reasoner_zs.jsonl outputs/l1_b1packet_qwen_zs_filled.jsonl \
        --out-csv data/interim/b1_round1/pilot_discussion.csv \
        --out-md data/interim/b1_round1/pilot_discussion.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items, load_predictions  # noqa: E402

# What to ask, keyed by (gold, annotator).  The question targets the distinction
# the pair actually turns on, not the label itself.
QUESTIONS = {
    ("CONTINGENCY", "EXPANSION"):
        "后件是在讲前件的**后果**吗？中间补「所以」通不通？补「而且」通不通？哪个更贴近原意？",
    ("EXPANSION", "CONTINGENCY"):
        "后件是在**补充**前件，还是在说前件造成的**结果**？补「因此」会不会改变意思？",
    ("EXPANSION", "COMPARISON"):
        "两句是在讲同一件事的两个方面，还是彼此**相反/预期冲突**？补「但是」自然吗？",
    ("COMPARISON", "EXPANSION"):
        "后件是在**补充**前件，还是在**否定**前件引出的预期？补「但是」自然吗？",
    ("TEMPORAL", "CONTINGENCY"):
        "两件事是「**然后**」的关系，还是「**所以**」的关系？前件是不是后件的条件？",
    ("TEMPORAL", "EXPANSION"):
        "能换成「**然后**」吗？还是更像「**另外还有**」？",
    ("TEMPORAL", "COMPARISON"):
        "这是在讲**时间先后/同期**，还是在做**对照**？「以前…现在」这类框架你按哪一类判？",
    ("COMPARISON", "TEMPORAL"):
        "这是**对照**还是**时间框架**？「过去…今后」在你看来属于哪一类？",
    ("COMPARISON", "CONTINGENCY"):
        "后件是前件的**结果**，还是与它**相反/让步**？有没有「本该…却」的意思？",
    ("CONTINGENCY", "COMPARISON"):
        "后件是在**解释原因**，还是在**对照/让步**？",
    ("EXPANSION", "TEMPORAL"):
        "两句是**先后发生**，还是**并列列举**？能换成「然后」吗？",
}
DEFAULT_Q = "你补的是哪个连接词？如果两类都能补，哪一类更贴近原意？"


def defect(it) -> str:
    """Flag corpus segmentation problems that make the item unanswerable.

    These are not annotator errors.  Asking someone to judge a relation whose
    Arg2 is literally inside Arg1 wastes the pilot discussion and contaminates
    the kappa.
    """
    # Collapse whitespace first: HIT-CDTB is space-tokenised, so the same span
    # can differ only by an internal space and still be the same text.
    if not (it.arg1 or "").strip():
        return "前件缺失（导出时只能显示全文）"
    if not (it.arg2 or "").strip():
        return "后件缺失"
    norm = lambda x: "".join((x or "").split())
    a1, a2 = norm(it.arg1), norm(it.arg2)
    if a1 and a2 and (a2 in a1 or a1 in a2):
        return "论元重叠：后件包含在前件之中"
    if a2 and len(a2) < 8:
        return "后件过短，可能切分不全"
    return ""


def read_sheet(path: Path, column: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            val = (row.get(column) or "").strip().upper()
            if val:
                out[row["item_id"]] = val
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--column", default="annotator_sense_l1")
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args(argv)

    items = {it.id: it for it in load_items(args.items)}
    ans = read_sheet(Path(args.annotator), args.column)
    rows = []
    for item_id, label in ans.items():
        it = items.get(item_id)
        if not it or not it.sense_l1 or label == it.sense_l1:
            continue
        rows.append({
            "item_id": item_id,
            "split": it.meta.get("b1_split", ""),
            "task": "显式" if it.explicit else "隐性",
            "connective": it.connective,
            "arg1": it.arg1 or it.text,
            "arg2": it.arg2,
            "gold": it.sense_l1,
            "annotator": label,
            "defect": defect(it),
            "question": (defect(it) or
                         QUESTIONS.get((it.sense_l1, label), DEFAULT_Q)),
        })
    rows.sort(key=lambda r: (r["gold"], r["annotator"], r["item_id"]))

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["序号", "item_id", "分段", "任务", "连接词", "论元1", "论元2",
                    "语料金标准", "标注者", "语料问题", "建议问法", "讨论结论", "规则是否要改"])
        for k, r in enumerate(rows, 1):
            w.writerow([k, r["item_id"], r["split"], r["task"], r["connective"],
                        r["arg1"], r["arg2"], r["gold"], r["annotator"], r["defect"],
                        r["question"], "", ""])

    models: dict[str, dict[str, str]] = {}
    for spec in args.preds:
        for path in sorted(ROOT.glob(spec)):
            for p in load_predictions(path):
                if p.error is None:
                    models.setdefault(p.model, {})[p.item_id] = p.label_l1

    counts = Counter((r["gold"], r["annotator"]) for r in rows if not r["defect"])
    n_defect = sum(1 for r in rows if r["defect"])
    who = Path(args.annotator).stem.split("_")[0]
    md = ["# 试标分歧讨论清单", "",
          f"共 **{len(rows)}** 条分歧（标注者：{who}，对照：语料金标准）。",
          "", "## 按类型汇总", "", "| 金标准 → 标注者 | 条数 |", "|---|---|"]
    for (g, a), n in counts.most_common():
        md.append(f"| {g} → {a} | {n} |")
    if n_defect:
        md += ["", f"其中 **{n_defect} 条是语料切分问题**（论元重叠或过短），"
                   "不是标注分歧，单独列出、不计入指南讨论。"]
    md += ["", "> 讨论的重点不是「谁对」，而是**指南里缺了哪条规则**。",
           "> 若双方都认为两类可通，就记下判据（例如「能补『所以』就判因果」），写回 guideline。", ""]

    for k, r in enumerate(rows, 1):
        head = f"## {k}. {r['gold']} → {r['annotator']}"
        if r["defect"]:
            head += "　⚠️ 语料问题"
        md += [head, "",
               f"- 任务：{r['task']}" + (f"｜连接词：**{r['connective']}**" if r["connective"] else ""),
               f"- 论元 1：{r['arg1']}",
               f"- 论元 2：{r['arg2']}",
               f"- 金标准：**{r['gold']}**　标注者：**{r['annotator']}**",
               f"- 建议问法：{r['question']}",
               f"- 讨论结论：　（待填）", ""]

    if models:
        md += ["---", "", "## 附：模型判断（**先不要给标注者看**，会锚定讨论）", "",
               "| # | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for k, r in enumerate(rows, 1):
            md.append("| " + str(k) + " | " +
                      " | ".join(models[m].get(r["item_id"], "—") for m in models) + " |")
        md.append("")

    (ROOT / args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {args.out_csv}, {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
