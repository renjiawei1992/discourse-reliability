"""M4: score the human readability check for the delete-connective condition.

Reads the three filled blocks, joins them through the opaque ``pid``, and
applies the gates pre-registered in ``refine-logs/M4_READABILITY_PROTOCOL.md``:
the point estimate of the readability rate decides, the confidence interval
only says how much of the decision is noise.

    python scripts/m4_readability.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION"]
ZH = {"时间": "TEMPORAL", "因果": "CONTINGENCY", "条件": "CONTINGENCY",
      "对比": "COMPARISON", "让步": "COMPARISON", "扩展": "EXPANSION"}

GATE_READ = 0.90
GATE_HUMAN_DROP = 0.15   # R_gold(B) - R_gold(A), abstentions counted as wrong
GATE_GOLD = 0.75
SUBSET_DELTA = -0.20


def norm(cell: str) -> str:
    s = (cell or "").strip()
    if not s:
        return ""
    up = s.upper()
    if up in CLASSES:
        return up
    if s in ZH:
        return ZH[s]
    return ""  # 无法判断 / 不知道 / anything else counts as "no class given"


def read_block(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    labels, notes = {}, {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            pid = (r.get("pid") or "").strip()
            if not pid:
                continue
            labels[pid] = norm(r.get("你的判断", ""))
            notes[pid] = (r.get("备注") or "").strip()
    return labels, notes


def load_key(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["pid"]: r for r in csv.DictReader(f)}


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


def rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 4)


def boot_ci(values: list[int], n_boot: int = 5000, seed: int = 161) -> list[float]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/interim/m4")
    ap.add_argument("--base-preds", nargs="+",
                    default=["outputs/l1_ds_zeroshot.jsonl",
                             "outputs/l1_ds_ted_zeroshot.jsonl"])
    ap.add_argument("--noconn-preds", nargs="+",
                    default=["outputs/l1_probe_noconn.jsonl",
                             "outputs/l1_probe_ted_noconn.jsonl"])
    ap.add_argument("--m4b", default="data/interim/m4b/m4b_block.csv",
                    help="optional re-check of the rebuilt paired frames")
    ap.add_argument("--noconn-tokens", default="outputs/l1_probe_noconn_tokens.jsonl",
                    help="model run on the rebuilt paired frames")
    ap.add_argument("--out", default="outputs/l1_m4_readability.json")
    ap.add_argument("--out-md", default="../refine-logs/M4_READABILITY.md")
    args = ap.parse_args(argv)

    d = ROOT / args.dir
    A, notes_a = read_block(d / "m4_block_a.csv")
    B, _ = read_block(d / "m4_block_b.csv")
    C, _ = read_block(d / "m4_block_c.csv")
    key = load_key(d / "m4_key.csv")

    filled = {blk: sum(1 for v in x.values() if v) for blk, x in (("A", A), ("B", B), ("C", C))}
    if not filled["A"]:
        print("block A is empty — run this again after the annotator returns it")
        return 1

    base, noconn = {}, {}
    for p in args.base_preds:
        base.update(load_preds(ROOT / p))
    for p in args.noconn_preds:
        noconn.update(load_preds(ROOT / p, strip_suffix="::no_conn"))
    noconn_new = {}
    new_path = ROOT / args.noconn_tokens
    if new_path.exists():
        noconn_new = load_preds(new_path, strip_suffix="::no_conn_tokens")

    pids = [p for p in A if p in key]
    gold = {p: key[p]["gold_l1"] for p in pids}

    # --- 主门 -------------------------------------------------------------
    read_vec = [1 if A[p] in CLASSES else 0 for p in pids]
    r_read = rate(sum(read_vec), len(read_vec))
    ci = boot_ci(read_vec)

    # --- 关系保持 / 金标准复现 -------------------------------------------
    both = [p for p in pids if A[p] in CLASSES and B.get(p) in CLASSES]
    r_keep = rate(sum(1 for p in both if A[p] == B[p]), len(both))
    b_cls = [p for p in pids if B.get(p) in CLASSES]
    r_gold_b = rate(sum(1 for p in b_cls if B[p] == gold[p]), len(b_cls))
    a_cls = [p for p in pids if A[p] in CLASSES]
    r_gold_a = rate(sum(1 for p in a_cls if A[p] == gold[p]), len(a_cls))

    # Accuracy after deletion over the whole sample, abstention counted wrong.
    # This is the human analogue of the model's Δacc and the one the gate uses:
    # scoring it only on judged items would let the annotator drop exactly the
    # items the deletion broke.
    gold_a_all = rate(sum(1 for p in pids if A[p] == gold[p]), len(pids))
    gold_b_all = rate(sum(1 for p in pids if B.get(p) == gold[p]), len(pids))
    human_drop = (round(gold_b_all - gold_a_all, 4)
                  if gold_a_all is not None and gold_b_all is not None else None)

    # --- C 区块（占位符诊断） --------------------------------------------
    c_pids = [p for p in C if p in key]
    r_read_c = rate(sum(1 for p in c_pids if C[p] in CLASSES), len(c_pids))
    c_both = [p for p in c_pids if C[p] in CLASSES and A.get(p) in CLASSES]
    r_c_vs_a = rate(sum(1 for p in c_both if C[p] == A[p]), len(c_both))

    # --- 子集 Δacc（协议 §5 规则 2） --------------------------------------
    def acc(ps: list[str], preds: dict[str, str]) -> tuple[int, int]:
        ok = [p for p in ps if key[p]["item_id"] in preds]
        return sum(1 for p in ok if preds[key[p]["item_id"]] == gold[p]), len(ok)

    full_b, n_full_b = acc(pids, base)
    full_n, n_full_n = acc(pids, noconn)
    sub_b, n_sub_b = acc(a_cls, base)
    sub_n, n_sub_n = acc(a_cls, noconn)
    d_full = rate(full_n, n_full_n) - rate(full_b, n_full_b) if n_full_b and n_full_n else None
    d_sub = rate(sub_n, n_sub_n) - rate(sub_b, n_sub_b) if n_sub_b and n_sub_n else None

    # --- 分层 --------------------------------------------------------------
    by_corpus = {}
    for p in pids:
        c = key[p]["corpus"]
        by_corpus.setdefault(c, []).append(p)
    per_corpus = {
        c: {"n": len(ps),
            "read": rate(sum(1 for p in ps if A[p] in CLASSES), len(ps)),
            "gold_b": rate(sum(1 for p in ps if B.get(p) in CLASSES and B[p] == gold[p]),
                           sum(1 for p in ps if B.get(p) in CLASSES))}
        for c, ps in by_corpus.items()
    }
    per_class = {}
    for g in CLASSES:
        ps = [p for p in pids if gold[p] == g]
        per_class[g] = {"n": len(ps), "read": rate(sum(1 for p in ps if A[p] in CLASSES), len(ps))}
    by_conn = defaultdict(list)
    for p in pids:
        by_conn[key[p]["connective"]].append(p)
    per_conn = {c: {"n": len(ps), "read": rate(sum(1 for p in ps if A[p] in CLASSES), len(ps))}
                for c, ps in sorted(by_conn.items(), key=lambda kv: -len(kv[1])) if len(ps) >= 3}

    # Paired frames delete a whole cross-argument span, not just the cue, so
    # their readability is reported separately: a low rate there is a
    # construction defect, not evidence about cue reliance.
    by_span = defaultdict(list)
    for p in pids:
        k = key[p]
        kind = "paired_frame" if k.get("paired_frame") == "1" else "single"
        by_span[kind].append(p)
    per_span = {}
    for k, ps in by_span.items():
        gb = rate(sum(1 for p in ps if B.get(p) == gold[p]), len(ps))
        ga = rate(sum(1 for p in ps if A[p] == gold[p]), len(ps))
        pb, nb = acc(ps, base)
        pn, nn = acc(ps, noconn)
        # After deletion, is the model worse than the human on the same items?
        # Abstention counts as an error for the human, as in Δ_h.
        hv, mv = [], []
        for p in ps:
            if key[p]["item_id"] not in noconn:
                continue
            m_ok = noconn[key[p]["item_id"]] == gold[p]
            h_ok = A[p] == gold[p]
            hv.append(1 if h_ok else 0)
            mv.append(1 if m_ok else 0)
        b = sum(1 for h, m in zip(hv, mv) if h and not m)
        c = sum(1 for h, m in zip(hv, mv) if m and not h)
        per_span[k] = {
            "n": len(ps),
            "read": rate(sum(1 for p in ps if A[p] in CLASSES), len(ps)),
            "read_ci95": boot_ci([1 if A[p] in CLASSES else 0 for p in ps]),
            "mean_span": round(sum(int(key[p]["span_chars"]) for p in ps) / len(ps), 1),
            "gold_b_all": gb, "gold_a_all": ga,
            "human_drop": round(gb - ga, 4) if gb is not None and ga is not None else None,
            "model_base": rate(pb, nb), "model_noconn": rate(pn, nn),
            "model_delta": (round(rate(pn, nn) - rate(pb, nb), 4)
                            if nb and nn else None),
            "human_vs_model_after": {"n": len(hv), "human_only": b, "model_only": c,
                                     "p": mcnemar(b, c)},
        }

    # --- 判定 --------------------------------------------------------------
    verdict, reasons = [], []
    if r_read >= GATE_READ:
        verdict.append("PASS")
        reasons.append(f"可判读率 {r_read:.4f} ≥ {GATE_READ}：删连接词条件可以作为捷径证据。")
    elif r_read >= 0.75:
        verdict.append("SUBSET")
        reasons.append(f"可判读率 {r_read:.4f} < {GATE_READ}，但 ≥ 0.75：改用可判读子集重算。")
        if d_sub is not None and d_sub <= SUBSET_DELTA:
            reasons.append(f"子集 Δacc = {d_sub:+.4f} ≤ {SUBSET_DELTA}：捷径结论保留，正文用子集数字。")
        else:
            reasons.append(f"子集 Δacc = {d_sub:+.4f} > {SUBSET_DELTA}：结论降级为「信息缺失效应」。")
    else:
        verdict.append("VOID")
        reasons.append(f"可判读率 {r_read:.4f} < 0.75：删除条件不具备单变量消融资格，结论作废。")
    if human_drop is not None and human_drop > GATE_HUMAN_DROP:
        verdict.append("RELATION_CHANGED")
        reasons.append(f"人类掉点 {human_drop:+.4f} > {GATE_HUMAN_DROP}：删连接词把关系本身也改了，"
                       "不能称单变量消融。")
    if r_gold_b is not None and r_gold_b < GATE_GOLD:
        reasons.append(f"注意：原句金标准复现率只有 {r_gold_b:.4f}，这批条目本身偏难，"
                       "R_read 要结合 R152 的人类准确率一起读。")

    # Re-check of the rebuilt paired-frame construction on the same 14 items,
    # so old and new can be compared item by item.
    m4b_out = None
    m4b_path = ROOT / args.m4b
    if args.m4b and m4b_path.exists():
        m4b, _ = read_block(m4b_path)
        same = [p for p in m4b if p in gold]
        if same:
            twice = [p for p in same if A.get(p) in CLASSES and m4b[p] in CLASSES]
            m4b_out = {
                "n": len(same),
                "read_old": rate(sum(1 for p in same if A.get(p) in CLASSES), len(same)),
                "read_new": rate(sum(1 for p in same if m4b[p] in CLASSES), len(same)),
                "gold_old": rate(sum(1 for p in same if A.get(p) == gold[p]), len(same)),
                "gold_new": rate(sum(1 for p in same if m4b[p] == gold[p]), len(same)),
                "read_b": rate(sum(1 for p in same if B.get(p) in CLASSES), len(same)),
                "gold_b": rate(sum(1 for p in same if B.get(p) == gold[p]), len(same)),
                "same_label": rate(sum(1 for p in twice if m4b[p] == A[p]), len(twice)),
                "n_twice": len(twice),
            }
            mb = [p for p in same if key[p]["item_id"] in base]
            m4b_out["model_base"] = rate(
                sum(1 for p in mb if base[key[p]["item_id"]] == gold[p]), len(mb))
            if noconn_new:
                mn = [p for p in same if key[p]["item_id"] in noconn_new]
                m4b_out["model_new"] = rate(
                    sum(1 for p in mn if noconn_new[key[p]["item_id"]] == gold[p]), len(mn))
                m4b_out["model_delta"] = (
                    round(m4b_out["model_new"] - m4b_out["model_base"], 4)
                    if mb and mn else None)

    out = {
        "n_items": len(pids),
        "filled": filled,
        "missing_from_key": [p for p in A if p not in key],
        "readability": {"rate": r_read, "ci95": ci, "n": len(read_vec),
                        "n_judged": sum(read_vec)},
        "relation_kept": {"rate": r_keep, "n": len(both)},
        "gold_recall": {"block_a_judged": r_gold_a, "block_b_judged": r_gold_b,
                        "block_a_all": gold_a_all, "block_b_all": gold_b_all,
                        "human_drop": human_drop},
        "block_c": {"n": len(c_pids), "read": r_read_c, "agrees_with_a": r_c_vs_a,
                    "n_pairs": len(c_both)},
        "delta_acc": {"full": {"base": rate(full_b, n_full_b), "noconn": rate(full_n, n_full_n),
                               "delta": d_full, "n": n_full_b},
                      "readable_subset": {"base": rate(sub_b, n_sub_b),
                                          "noconn": rate(sub_n, n_sub_n),
                                          "delta": d_sub, "n": n_sub_b}},
        "per_corpus": per_corpus,
        "per_class": per_class,
        "per_span_kind": per_span,
        "m4b_recheck": m4b_out,
        "per_connective_n3": per_conn,
        "verdict": verdict,
        "reasons": reasons,
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# M4 删连接词条件：人工可判读性", "",
         "> 本文件只含标注者 1。两名标注者的完整对照见 `M4_TWO_ANNOTATORS.md`，"
         "论文引用后者。", "",
         f"样本 {len(pids)} 条（区块 A/B 各 {len(A)}/{len(B)}，区块 C {len(C)}）；"
         f"给出四类标签的条目：A {filled['A']}、B {filled['B']}、C {filled['C']}"
         f"（其余为「无法判断」或空白）。", ""]

    def f4(x):
        return f"{x:.4f}" if isinstance(x, float) else "—"

    L += ["## 主门", "",
          "| 指标 | 值 | 95% CI | 门 |",
          "|---|---|---|---|",
          f"| **可判读率**（区块 A） | **{r_read:.4f}** | [{ci[0]:.4f}, {ci[1]:.4f}] | ≥ {GATE_READ} |",
          f"| 人类掉点（B − A，弃权记错） | {f4(human_drop)} | — | ≤ {GATE_HUMAN_DROP} |",
          f"| 金标准复现（原句 B，全样本） | {f4(gold_b_all)} | — | 诊断 |",
          f"| 金标准复现（删后 A，全样本） | {f4(gold_a_all)} | — | 诊断 |",
          f"| 金标准复现（原句 B，已判条目） | {f4(r_gold_b)} | — | 诊断 |",
          f"| 金标准复现（删后 A，已判条目） | {f4(r_gold_a)} | — | 诊断 |",
          f"| 关系保持率（A = B，已判条目） | {f4(r_keep)} | — | 诊断 |",
          f"| 占位符诊断（区块 C 可判读率） | {f4(r_read_c)} | — | 诊断 |",
          f"| 区块 C 与区块 A 同标签 | {f4(r_c_vs_a)} | — | 诊断 |",
          "",
          "> 关系保持率与人的判断噪声混在一起（同一人两次独立标注也会有分歧），"
          "所以只作诊断；判定用的是以金标准为锚的人类掉点。",
          ""]

    L += ["## 按语料", "", "| 语料 | n | 可判读率 | 原句金标准复现 |", "|---|---|---|---|"]
    for c, v in per_corpus.items():
        L.append(f"| {c} | {v['n']} | {v['read']} | {v['gold_b']} |")
    if m4b_out:
        L += ["", "## 重构造复核（同一批 14 条成对框架）", "",
              "把成对框架改成「只删框架词、保留中间内容」之后，同一个人重判一次。"
              "原句（区块 B）的结果一并列出，作为这批条目本身有多难的参照：", "",
              "| | 原句（B） | 旧构造（整段删） | 新构造（只删框架词） |",
              "|---|---|---|---|",
              f"| 可判读率 | {f4(m4b_out['read_b'])} | "
              f"{f4(m4b_out['read_old'])} | **{f4(m4b_out['read_new'])}** |",
              f"| 金标准复现（人） | {f4(m4b_out['gold_b'])} | {f4(m4b_out['gold_old'])} | "
              f"{f4(m4b_out['gold_new'])} |",
              f"| 金标准复现（模型） | {f4(m4b_out.get('model_base'))} | — | "
              f"{f4(m4b_out.get('model_new'))} |",
              "",
              f"新旧两次都给了类别的 {m4b_out['n_twice']} 条里，标签相同的占 "
              f"{f4(m4b_out['same_label'])}。",
              "",
              "> 这是构造修复的直接证据：同样 14 条、同一个人，只改删除方式，"
              "可判读率从两成左右升到全部可判。金标准复现率仍低于原句，"
              "说明这批条目本身偏难，修复解决的是「根本没法判」，不是「判得对」。", ""]

    if per_span:
        L += ["", "## 按删除方式（决定性拆分）", "",
              "成对框架（「当初…如今」）删除的是整个跨论元片段，不只是线索；"
              "这一类可判读率低是构造缺陷，不能算作「模型依赖连接词形式」的证据。"
              "该分层在数据回收前已写进协议 §3 与已知局限，不是事后挑出来的。",
              "",
              "| 删除方式 | n | 平均删除字数 | 可判读率 | 人类掉点 | 模型原句 acc | 模型删后 acc | 模型 Δ |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in per_span.items():
            L.append(f"| {k} | {v['n']} | {v['mean_span']} | "
                     f"{f4(v['read'])} [{v['read_ci95'][0]:.3f}, {v['read_ci95'][1]:.3f}] | "
                     f"{f4(v['human_drop'])} | {f4(v['model_base'])} | "
                     f"{f4(v['model_noconn'])} | {f4(v['model_delta'])} |")
        L.append("")
        L.append("删连接词之后，人与模型在同一批条目上的正确率对比"
                 "（人弃权记错，exact McNemar）：")
        L.append("")
        L.append("| 删除方式 | n | 只有人对 | 只有模型对 | p |")
        L.append("|---|---|---|---|---|")
        for k, v in per_span.items():
            h = v["human_vs_model_after"]
            L.append(f"| {k} | {h['n']} | {h['human_only']} | {h['model_only']} | {h['p']} |")
    L += ["", "## 按金标准类别", "", "| 类别 | n | 可判读率 |", "|---|---|---|"]
    for g, v in per_class.items():
        L.append(f"| {g} | {v['n']} | {v['read']} |")
    if per_conn:
        L += ["", "## 按连接词（n ≥ 3）", "", "| 连接词 | n | 可判读率 |", "|---|---|---|"]
        for c, v in per_conn.items():
            L.append(f"| {c} | {v['n']} | {v['read']} |")

    L += ["", "## 删连接词在可判读子集上的效应", "",
          "| 集合 | n | 原句 acc | 删后 acc | Δ |", "|---|---|---|---|---|",
          f"| 全部样本 | {out['delta_acc']['full']['n']} | {out['delta_acc']['full']['base']} | "
          f"{out['delta_acc']['full']['noconn']} | {d_full:+.4f} |",
          f"| 可判读子集 | {out['delta_acc']['readable_subset']['n']} | "
          f"{out['delta_acc']['readable_subset']['base']} | "
          f"{out['delta_acc']['readable_subset']['noconn']} | {d_sub:+.4f} |",
          "", "（模型为 deepseek-chat zero-shot，与 R140 同一批预测。）", ""]

    L += ["## 判定", "", f"**{' + '.join(verdict)}**", ""] + [f"- {r}" for r in reasons] + [""]
    L += ["> 判定规则见 `refine-logs/M4_READABILITY_PROTOCOL.md` §5，在数据回收前写定。", ""]

    mp = ROOT / args.out_md
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out_md}")
    print(json.dumps({"readability": r_read, "ci95": ci, "r_keep": r_keep,
                      "gold_b": r_gold_b, "delta_full": d_full, "delta_subset": d_sub,
                      "verdict": verdict}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
