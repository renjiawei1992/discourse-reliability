"""M4 with two independent annotators.

The protocol assumed one annotator and said so as a limitation.  A second
annotator has since completed all four blocks (``*_1.csv``), which turns the
main weakness into a measurement: how much of "the item is not judgeable"
is the construction, and how much is the person.

Reports, per annotator:
    readability overall / single-token / paired-frame
    gold recall on the original sentence, after deletion, and the drop
    the rebuilt-construction re-check
and between annotators:
    agreement on "judged vs abstained", agreement on the label, and the
    overlap of the abstention sets.

    python scripts/m4_two_annotators.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from discourse_reliability.metrics.agreement import cohen_kappa  # noqa: E402
from m4_readability import (  # noqa: E402
    CLASSES,
    GATE_HUMAN_DROP,
    GATE_READ,
    boot_ci,
    load_key,
    rate,
    read_block,
)

SUFFIX = {"A1": "", "A2": "_1"}


def score(d: Path, suffix: str, key: dict, m4b_dir: Path) -> dict:
    A, _ = read_block(d / f"m4_block_a{suffix}.csv")
    B, _ = read_block(d / f"m4_block_b{suffix}.csv")
    C, _ = read_block(d / f"m4_block_c{suffix}.csv")
    M, _ = read_block(m4b_dir / f"m4b_block{suffix}.csv")

    pids = [p for p in A if p in key]
    gold = {p: key[p]["gold_l1"] for p in pids}
    single = [p for p in pids if key[p].get("paired_frame") != "1"]
    paired = [p for p in pids if key[p].get("paired_frame") == "1"]

    def judged(block, ps):
        return rate(sum(1 for p in ps if block.get(p) in CLASSES), len(ps))

    gb = rate(sum(1 for p in pids if B.get(p) == gold[p]), len(pids))
    ga = rate(sum(1 for p in pids if A.get(p) == gold[p]), len(pids))

    c_pids = [p for p in C if p in key]
    c_twice = [p for p in c_pids if C[p] in CLASSES and A.get(p) in CLASSES]

    m_pids = [p for p in M if p in key]
    out = {
        "n": len(pids),
        "read": judged(A, pids),
        "read_ci95": boot_ci([1 if A[p] in CLASSES else 0 for p in pids]),
        "read_single": judged(A, single),
        "read_paired": judged(A, paired),
        "n_single": len(single), "n_paired": len(paired),
        "abstained": sorted(p for p in pids if A[p] not in CLASSES),
        "gold_b_all": gb, "gold_a_all": ga,
        "human_drop": round(gb - ga, 4) if None not in (gb, ga) else None,
        "gold_b_single": rate(sum(1 for p in single if B.get(p) == gold[p]), len(single)),
        "gold_a_single": rate(sum(1 for p in single if A[p] == gold[p]), len(single)),
        "drop_single": (round(sum(1 for p in single if B.get(p) == gold[p]) / len(single)
                              - sum(1 for p in single if A[p] == gold[p]) / len(single), 4)
                        if single else None),
        "c_read": judged(C, c_pids),
        "c_vs_a": rate(sum(1 for p in c_twice if C[p] == A[p]), len(c_twice)),
        "c_pairs": len(c_twice),
        "m4b": {
            "n": len(m_pids),
            "read_old": judged(A, m_pids),
            "read_new": judged(M, m_pids),
            "gold_old": rate(sum(1 for p in m_pids if A.get(p) == gold[p]), len(m_pids)),
            "gold_new": rate(sum(1 for p in m_pids if M[p] == gold[p]), len(m_pids)),
            "gold_b": rate(sum(1 for p in m_pids if B.get(p) == gold[p]), len(m_pids)),
        },
    }
    out["verdict"] = ("PASS" if out["read"] >= GATE_READ
                      else "SUBSET" if out["read"] >= 0.75 else "VOID")
    if out["human_drop"] is not None and out["human_drop"] > GATE_HUMAN_DROP:
        out["verdict"] += "+RELATION_CHANGED"

    # Keep the raw labels so the between-annotator comparison can be done here.
    out["_A"], out["_B"], out["_C"], out["_M"] = A, B, C, M
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/interim/m4")
    ap.add_argument("--m4b-dir", default="data/interim/m4b")
    ap.add_argument("--out", default="outputs/l1_m4_two_annotators.json")
    ap.add_argument("--out-md", default="../refine-logs/M4_TWO_ANNOTATORS.md")
    args = ap.parse_args(argv)

    d, md = ROOT / args.dir, ROOT / args.m4b_dir
    key = load_key(d / "m4_key.csv")
    s1 = score(d, SUFFIX["A1"], key, md)
    s2 = score(d, SUFFIX["A2"], key, md)

    pids = [p for p in s1["_A"] if p in key]
    gold = {p: key[p]["gold_l1"] for p in pids}

    # --- between annotators ------------------------------------------------
    def kappa(xs, ys, ps):
        ps = [p for p in ps if xs.get(p) in CLASSES and ys.get(p) in CLASSES]
        if not ps:
            return None, 0
        k = cohen_kappa([xs[p] for p in ps], [ys[p] for p in ps])
        return (round(k, 4) if k is not None else None), len(ps)

    k_a4, n_a4 = kappa(s1["_A"], s2["_A"], pids)
    k_b4, n_b4 = kappa(s1["_B"], s2["_B"], pids)
    k_c4, n_c4 = kappa(s1["_C"], s2["_C"], [p for p in s1["_C"] if p in key])
    k_m4, n_m4 = kappa(s1["_M"], s2["_M"], [p for p in s1["_M"] if p in key])

    # Binary "could you commit to a class" judgement — the gate's own quantity.
    def binary_kappa(block1, block2):
        b1 = [1 if block1[p] in CLASSES else 0 for p in pids]
        b2 = [1 if block2[p] in CLASSES else 0 for p in pids]
        k = cohen_kappa([str(x) for x in b1], [str(x) for x in b2])
        return (round(k, 4) if k is not None else None)

    k_bin_a = binary_kappa(s1["_A"], s2["_A"])

    ab1, ab2 = set(s1["abstained"]), set(s2["abstained"])
    both_abstain = len(ab1 & ab2)
    union = len(ab1 | ab2)

    # Consensus readability: an item counts as unjudgeable only if BOTH say so.
    union_abstain = [p for p in pids if p in ab1 or p in ab2]
    intersection_read = rate(len(pids) - union, len(pids))

    single = [p for p in pids if key[p].get("paired_frame") != "1"]
    paired = [p for p in pids if key[p].get("paired_frame") == "1"]
    single_union = [p for p in single if p in ab1 or p in ab2]
    paired_union = [p for p in paired if p in ab1 or p in ab2]
    abstain_kind = {
        "single": {"a1": sum(1 for p in single if p in ab1),
                   "a2": sum(1 for p in single if p in ab2),
                   "both": sum(1 for p in single if p in ab1 and p in ab2), "n": len(single)},
        "paired": {"a1": sum(1 for p in paired if p in ab1),
                   "a2": sum(1 for p in paired if p in ab2),
                   "both": sum(1 for p in paired if p in ab1 and p in ab2), "n": len(paired)},
    }

    stats = {
        "A1": {k: v for k, v in s1.items() if not k.startswith("_")},
        "A2": {k: v for k, v in s2.items() if not k.startswith("_")},
        "between": {
            "kappa_label_a": {"kappa": k_a4, "n": n_a4},
            "kappa_label_b": {"kappa": k_b4, "n": n_b4},
            "kappa_label_c": {"kappa": k_c4, "n": n_c4},
            "kappa_label_m4b": {"kappa": k_m4, "n": n_m4},
            "kappa_judged_a": k_bin_a,
            "abstain_overlap": {"a1": len(ab1), "a2": len(ab2),
                                "both": both_abstain, "union": union},
            "union_readability": intersection_read,
            "union_readability_single": rate(len(single) - len(single_union), len(single)),
            "union_readability_paired": rate(len(paired) - len(paired_union), len(paired)),
            "abstain_by_kind": abstain_kind,
            "n_single": len(single), "n_paired": len(paired),
        },
    }
    (ROOT / args.out).write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    f4 = lambda x: f"{x:.4f}" if isinstance(x, float) else "—"  # noqa: E731
    b = stats["between"]
    L = ["# M4 删连接词条件：两名独立标注者", "",
         "同一套盲表（100 条删连接词 / 100 条原句 / 30 条无占位符 / 14 条重构造），"
         "两个人独立完成。协议原本按单标注者设计，这一节把那条局限变成测量。", "",
         "## 主指标", "",
         "| 指标 | 标注者 1 | 标注者 2 |", "|---|---|---|"]
    rows = [
        ("可判读率（100 条）", s1["read"], s2["read"]),
        ("可判读率 · 单词连接词", s1["read_single"], s2["read_single"]),
        ("可判读率 · 成对框架", s1["read_paired"], s2["read_paired"]),
        ("原句金标准复现", s1["gold_b_all"], s2["gold_b_all"]),
        ("删后金标准复现（弃权记错）", s1["gold_a_all"], s2["gold_a_all"]),
        ("人类掉点（全样本）", s1["human_drop"], s2["human_drop"]),
        ("人类掉点（单词连接词子集）", s1["drop_single"], s2["drop_single"]),
        ("无占位符区块可判读率", s1["c_read"], s2["c_read"]),
        ("重构造 · 旧构造可判读率", s1["m4b"]["read_old"], s2["m4b"]["read_old"]),
        ("重构造 · 新构造可判读率", s1["m4b"]["read_new"], s2["m4b"]["read_new"]),
        ("重构造 · 新构造金标准复现", s1["m4b"]["gold_new"], s2["m4b"]["gold_new"]),
    ]
    for name, x, y in rows:
        L.append(f"| {name} | {f4(x)} | {f4(y)} |")
    L += ["", f"判定：标注者 1 **{s1['verdict']}**，标注者 2 **{s2['verdict']}**。", "",
          "## 两人之间", "",
          "| 量 | 值 | n |", "|---|---|---|",
          f"| 标签一致性 κ（删连接词，四类） | {f4(b['kappa_label_a']['kappa'])} | {b['kappa_label_a']['n']} |",
          f"| 标签一致性 κ（原句，四类） | {f4(b['kappa_label_b']['kappa'])} | {b['kappa_label_b']['n']} |",
          f"| 标签一致性 κ（无占位符） | {f4(b['kappa_label_c']['kappa'])} | {b['kappa_label_c']['n']} |",
          f"| 标签一致性 κ（重构造） | {f4(b['kappa_label_m4b']['kappa'])} | {b['kappa_label_m4b']['n']} |",
          f"| **「判得了 / 判不了」一致 κ** | {f4(b['kappa_judged_a'])} | {len(pids)} |",
          "", f"弃权条目：标注者 1 {b['abstain_overlap']['a1']} 条、标注者 2 "
          f"{b['abstain_overlap']['a2']} 条，**交集只有 {b['abstain_overlap']['both']} 条**"
          f"（并集 {b['abstain_overlap']['union']} 条）。", "",
          "| 弃权 | 单词连接词（n=%d） | 成对框架（n=%d） |" % (b["n_single"], b["n_paired"]),
          "|---|---|---|",
          f"| 标注者 1 | {b['abstain_by_kind']['single']['a1']} | {b['abstain_by_kind']['paired']['a1']} |",
          f"| 标注者 2 | {b['abstain_by_kind']['single']['a2']} | {b['abstain_by_kind']['paired']['a2']} |",
          f"| 两人都弃权 | {b['abstain_by_kind']['single']['both']} | {b['abstain_by_kind']['paired']['both']} |",
          "",
          f"并集口径的可判读率（两人都说判不了才算判不了）：{f4(b['union_readability'])}；"
          f"单词连接词子集 {f4(b['union_readability_single'])}；"
          f"成对框架 {f4(b['union_readability_paired'])}。", "",
          "> 「这条判不了」本身是一次判断，不是客观事实：两人都弃权的条目只有 "
          f"{b['abstain_overlap']['both']} 条，而并集有 {b['abstain_overlap']['union']} 条。"
          "所以主指标要报两个人的区间，并给出并集口径，而不是只报一个人的比例。", ""]
    (ROOT / args.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out_md}")
    print(json.dumps({k: v for k, v in stats["between"].items()
                      if k.startswith("kappa") or k == "abstain_overlap"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
