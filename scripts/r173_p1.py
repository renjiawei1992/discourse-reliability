"""R173: score the ``.p1`` sentence-group level (appendix, descriptive).

``.p1`` is 90% EXPANSION, so accuracy and kappa mostly measure the majority
class.  This script reports them anyway, but alongside the majority baseline,
the predicted distribution, and per-class recall, so the table cannot be read
as a four-class result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.metrics.agreement import (  # noqa: E402
    accuracy,
    cohen_kappa,
    confusion,
    macro_f1,
)
from discourse_reliability.schema import load_items  # noqa: E402

CLASSES = ["TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION"]


def load_preds(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("error") is None and o.get("label_l1"):
                out[o["item_id"]] = o["label_l1"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/processed/l1/hit_p1.jsonl")
    ap.add_argument("--runs", nargs="+", default=[
        "outputs/l1_p1_chat_zs.jsonl", "outputs/l1_p1_reasoner_zs.jsonl"])
    ap.add_argument("--out", default="outputs/l1_r173_p1.json")
    ap.add_argument("--out-md", default="../refine-logs/R173_P1.md")
    args = ap.parse_args(argv)

    items = {it.id: it for it in load_items(ROOT / args.gold)}
    gold_dist = Counter(it.sense_l1 for it in items.values())
    n_gold = len(items)
    majority = max(gold_dist.values()) / n_gold

    results = {}
    for run in args.runs:
        preds = load_preds(ROOT / run)
        ids = [i for i in items if i in preds]
        g = [items[i].sense_l1 for i in ids]
        p = [preds[i] for i in ids]
        per_class = {}
        for c in CLASSES:
            n_c = sum(1 for x in g if x == c)
            per_class[c] = {"n": n_c, "recall": (
                round(sum(1 for x, y in zip(g, p) if x == c and y == c) / n_c, 4)
                if n_c else None)}
        strat = {}
        for flag, name in ((True, "explicit"), (False, "implicit")):
            sub = [i for i in ids if items[i].explicit is flag]
            if sub:
                gg = [items[i].sense_l1 for i in sub]
                pp = [preds[i] for i in sub]
                strat[name] = {"n": len(sub), "accuracy": round(accuracy(gg, pp), 4)}
        results[Path(run).stem] = {
            "n": len(ids),
            "accuracy": round(accuracy(g, p), 4),
            "kappa": (lambda k: round(k, 4) if k is not None else None)(cohen_kappa(g, p)),
            "macro_f1": round(macro_f1(g, p), 4),
            "pred_dist": dict(Counter(p).most_common()),
            "per_class": per_class,
            "by_explicitness": strat,
        }

    out = {"n_gold": n_gold, "gold_dist": dict(gold_dist.most_common()),
           "majority_baseline": round(majority, 4), "runs": results}
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# R173：`.p1` 句群层（附录，描述性）", "",
         f"条目 {n_gold} 条，50 篇文档。类别分布："
         + "、".join(f"{k} {v}" for k, v in gold_dist.most_common())
         + f"。**多数类基线就是 {majority:.4f}**——p1 上 EXPANSION 占九成，"
         "所以 acc 与 κ 主要反映的是多数类，不是一个四类结果。", "",
         "| 模型 | n | acc | κ | macro-F1 |",
         "|---|---|---|---|---|"]
    for name, r in results.items():
        L.append(f"| {name} | {r['n']} | {r['accuracy']} | {r['kappa']} | {r['macro_f1']} |")
    L.append(f"| 多数类基线 | {n_gold} | {majority:.4f} | 0.0 | — |")
    L += ["", "## 每类召回", "",
          "| 模型 | " + " | ".join(f"{c}(n)" for c in CLASSES) + " |",
          "|---|" + "---|" * len(CLASSES)]
    for name, r in results.items():
        cells = []
        for c in CLASSES:
            v = r["per_class"][c]
            cells.append(f"{v['recall']} ({v['n']})" if v["recall"] is not None else f"— ({v['n']})")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## 预测分布", "", "| 模型 | " + " | ".join(CLASSES) + " |",
          "|---|" + "---|" * len(CLASSES)]
    for name, r in results.items():
        L.append(f"| {name} | " + " | ".join(str(r["pred_dist"].get(c, 0)) for c in CLASSES) + " |")
    L += ["", "## 显式 / 隐式", "", "| 模型 | 显式 n / acc | 隐式 n / acc |", "|---|---|---|"]
    for name, r in results.items():
        e = r["by_explicitness"].get("explicit", {})
        i = r["by_explicitness"].get("implicit", {})
        L.append(f"| {name} | {e.get('n','—')} / {e.get('accuracy','—')} | "
                 f"{i.get('n','—')} / {i.get('accuracy','—')} |")
    L += ["", "> 结论只能这样写：p1 上模型与多数类基线基本持平（或高 X 点），"
          "κ 因类别极度不平衡而不可解释；这一层不作四类结论，"
          "也不与 p2/p3 的平衡集数字并列比较。", ""]

    (ROOT / args.out_md).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out} and {args.out_md}")
    for name, r in results.items():
        print(name, {k: r[k] for k in ("n", "accuracy", "kappa", "macro_f1")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
