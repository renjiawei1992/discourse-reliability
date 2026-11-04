"""Build the adjudication worksheet, ordered by what actually needs deciding.

Ordering is deliberate.  A third annotator's time is the scarcest input here, so:

    1. items where the corpus gold agrees with *neither* annotator
       (a real three-way disagreement -- these are where the guideline is thin)
    2. ordinary two-way disagreements, grouped by label pair so one decision
       covers a whole block
    3. items where one annotator abstained (cheapest to settle)

Corpus defects (missing or truncated arguments) are removed before numbering and
listed separately: adjudicating them would be adjudicating a broken item.

    python scripts/build_adjudication_table.py \
        --items data/processed/l1/b1_packet.jsonl \
        --a data/interim/b1_round1/submissions/C_latest.csv \
        --b data/interim/b1_round1/submissions/D_latest.csv \
        --out-csv data/interim/b1_round1/adjudication_worksheet.csv \
        --out-md data/interim/b1_round1/adjudication_worksheet.md
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

PAIR_QUESTION = {
    frozenset(("COMPARISON", "EXPANSION")):
        "后件是在补充前件，还是否定前件引出的预期？补「但是」自然吗？",
    frozenset(("CONTINGENCY", "EXPANSION")):
        "后件是前件造成的后果，还是只是补充说明？补「所以」会改变意思吗？",
    frozenset(("EXPANSION", "TEMPORAL")):
        "是「然后」还是「另外还有」？两件事有没有明确先后？",
    frozenset(("CONTINGENCY", "TEMPORAL")):
        "是「所以」还是「然后」？前件是不是后件发生的条件？",
    frozenset(("COMPARISON", "TEMPORAL")):
        "是对照，还是时间框架（以前…现在）？",
    frozenset(("COMPARISON", "CONTINGENCY")):
        "是让步/对照，还是因果？有没有「本该…却」的意思？",
}
UNKNOWN_QUESTION = "一方无法判定。是条目本身无法判，还是指南没有覆盖这种情况？"


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
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args(argv)

    items = {it.id: it for it in load_items(args.items)}
    a = read_sheet(Path(args.a), args.column)
    b = read_sheet(Path(args.b), args.column)

    defects, three_way, two_way, abstain = [], [], [], []
    for item_id in sorted(set(a) & set(b)):
        it = items.get(item_id)
        if it is None or a[item_id] == b[item_id]:
            continue
        if defect(it):
            defects.append((item_id, defect(it)))
            continue
        la, lb = a[item_id], b[item_id]
        row = {
            "item_id": item_id,
            "split": it.meta.get("b1_split", ""),
            "task": "显式" if it.explicit else "隐式",
            "connective": it.connective,
            "arg1": it.arg1 or it.text,
            "arg2": it.arg2,
            "a": la,
            "b": lb,
            "gold": it.sense_l1,
        }
        if "UNKNOWN" in (la, lb):
            row["kind"] = "单方无法判定"
            row["question"] = UNKNOWN_QUESTION
            abstain.append(row)
        elif it.sense_l1 not in (la, lb):
            row["kind"] = "三方分歧（金标准与两人都不同）"
            row["question"] = PAIR_QUESTION.get(frozenset((la, lb)), "")
            three_way.append(row)
        else:
            row["kind"] = "双方分歧"
            row["question"] = PAIR_QUESTION.get(frozenset((la, lb)), "")
            two_way.append(row)

    for block in (three_way, two_way):
        block.sort(key=lambda r: (sorted((r["a"], r["b"])), r["task"], r["item_id"]))
    abstain.sort(key=lambda r: (r["a"] != "UNKNOWN", r["item_id"]))

    rows = three_way + two_way + abstain
    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["序号", "优先级", "类型", "item_id", "分段", "任务", "连接词",
              "论元1", "论元2", "C 的标注", "D 的标注", "语料金标准",
              "裁决要点", "最终裁决", "理由", "是否要改指南"]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k, r in enumerate(rows, 1):
            prio = 1 if r["kind"].startswith("三方") else (2 if r["kind"] == "双方分歧" else 3)
            w.writerow([k, prio, r["kind"], r["item_id"], r["split"], r["task"],
                        r["connective"], r["arg1"], r["arg2"], r["a"], r["b"],
                        r["gold"], r["question"], "", "", ""])

    counts = Counter(r["kind"] for r in rows)
    pairs = Counter(" ↔ ".join(sorted((r["a"], r["b"]))) for r in two_way + three_way)
    md = ["# 裁决工作表", "",
          f"共 **{len(rows)}** 条待裁决；已剔除 **{len(defects)}** 条语料缺陷（见文末）。", "",
          "## 优先级", "",
          "| 优先级 | 类型 | 条数 | 为什么排在这里 |",
          "|---|---|---|---|"]
    md.append(f"| 1 | 三方分歧（金标准与两人都不同） | {counts.get('三方分歧（金标准与两人都不同）', 0)} | "
              "最能暴露指南缺口，先裁 |")
    md.append(f"| 2 | 双方分歧 | {counts.get('双方分歧', 0)} | 按类别对成块处理 |")
    md.append(f"| 3 | 单方无法判定 | {counts.get('单方无法判定', 0)} | 通常一句话即可定 |")
    md += ["", f"## 全部 {sum(pairs.values())} 条双方分歧的类别对分布（含优先级 1）",
           "", "| 类别对 | 条数 |", "|---|---|"]
    for k, v in pairs.most_common():
        md.append(f"| {k} | {v} |")
    md += ["", "## 使用方式", "",
           "1. 先做优先级 1 的几条，它们决定了指南要不要再改。",
           "2. 优先级 2 按类别对成块裁决：定下第一条的判据后，同块的其余条目直接套用。",
           "3. 优先级 3 只需回答「这条目本身能不能判」。",
           "4. 任何一条若触发了新判据，填在「是否要改指南」列里，统一回写 guideline。", ""]
    if defects:
        md += ["## 已剔除的语料缺陷", "", "| item_id | 问题 |", "|---|---|"]
        for item_id, why in defects:
            md.append(f"| `{item_id}` | {why} |")
        md.append("")

    (ROOT / args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"written": len(rows), "defects": len(defects),
                      "three_way": len(three_way), "two_way": len(two_way),
                      "abstain": len(abstain), "pairs": dict(pairs.most_common())},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
