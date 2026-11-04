"""Per-connective breakdown of explicit-sense accuracy (R172).

Claim C2 says explicit performance is largely reproducible from the connective
itself.  A corpus-level accuracy hides whether that is true: a model can score
0.90 while failing every polysemous connective and getting the rest "right" by
reading the lexeme.  This script opens the black box by grouping explicit items
by connective and reporting, for each connective:

    n                  how many items carry it
    majority_acc       accuracy of always predicting the connective's most
                       frequent gold label on these very items -- the in-sample
                       ceiling of any purely lexical rule
    prior_acc          the five-fold lexeme prior (R110), i.e. the honest
                       lexical baseline
    <model>_acc        each LLM / encoder on the same items
    entropy            Shannon entropy (bits, max 2) of the gold labels here

Then it aggregates: accuracy by ambiguity stratum, Spearman correlation between
gold entropy and accuracy across connectives, and how concentrated the errors
are (a handful of connectives accounting for most mistakes is the finding the
error typology expects, and it is testable without reading any examples).

    python scripts/per_connective.py \
        --gold data/processed/l1/hit_balanced.jsonl \
        --preds 'outputs/l1_hit_balanced_*_zs.jsonl' \
        --prior outputs/l1_prior_hit_balanced.jsonl \
        --out-json outputs/l1_per_connective.json \
        --out-md refine-logs/L1_PER_CONNECTIVE.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items, load_predictions  # noqa: E402

Z = 1.96


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def entropy(counter: Counter) -> float:
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


def spearman(xs: list[float], ys: list[float]) -> tuple[float, int]:
    """Rank correlation with average ranks for ties; no scipy dependency."""
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    n = len(xs)
    if n < 3:
        return (float("nan"), n)
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return (float("nan"), n)
    return (num / (dx * dy), n)


def load_model(path: Path) -> tuple[str, dict[str, str]]:
    name, out = None, {}
    for p in load_predictions(path):
        if p.error is not None:
            continue
        name = name or p.model
        out[p.item_id] = p.label_l1
    return (name or path.stem, out)


def pick_files(specs: list[str], item_ids: set[str]) -> dict[str, tuple[Path, dict[str, str]]]:
    """One file per model: prefer the variant that covers the most items.

    Several models have a ``_zs`` file and a ``_zs_filled`` retry file.  Taking
    whichever glob sorts last would silently decide coverage, and coverage is
    not random -- the retried items are the ones the first pass failed on, i.e.
    the hard ones.  Keep the widest file and record which one it was.
    """
    best: dict[str, tuple[Path, dict[str, str]]] = {}
    for spec in specs:
        for path in sorted(ROOT.glob(spec)):
            name, pred = load_model(path)
            if not pred:
                continue
            cover = len(item_ids & set(pred))
            prev = best.get(name)
            if prev is None or cover > len(item_ids & set(prev[1])):
                best[name] = (path, pred)
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--prior", default="")
    ap.add_argument("--min-n", type=int, default=3, help="min items for the printed table")
    ap.add_argument("--human-a", default="", help="annotator A submission CSV (optional)")
    ap.add_argument("--human-b", default="", help="annotator B submission CSV (optional)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args(argv)

    items = [it for it in load_items(ROOT / args.gold) if it.explicit and it.connective]
    item_ids = {it.id for it in items}
    picked = pick_files(args.preds, item_ids)
    models: dict[str, dict[str, str]] = {name: pred for name, (_, pred) in picked.items()}
    system_files = {name: str(path.relative_to(ROOT)) for name, (path, _) in picked.items()}
    if args.prior:
        prior = load_model(ROOT / args.prior)[1]
        models = {"lexeme_prior": prior, **models}
        system_files = {"lexeme_prior": args.prior, **system_files}

    # Restrict every system to the same item set so the columns are comparable.
    shared = [it for it in items if all(it.id in m for m in models.values())]
    dropped = len(items) - len(shared)
    if not shared:
        raise SystemExit("no item is covered by every system")

    by_conn: dict[str, list] = defaultdict(list)
    for it in shared:
        by_conn[it.connective].append(it)

    # Fair lexical baseline: predict each item from the majority label of the
    # *other* items sharing its connective (falling back to the global majority
    # for singleton connectives).  The in-sample majority in the table below is
    # an upper bound, not an achievable rule; this one is what a real lookup
    # table scores on unseen items.
    global_majority = Counter(it.sense_l1 for it in shared).most_common(1)[0][0]
    loo_hits = 0
    for group in by_conn.values():
        for it in group:
            others = [o.sense_l1 for o in group if o is not it]
            pred = Counter(others).most_common(1)[0][0] if others else global_majority
            loo_hits += pred == it.sense_l1
    loo_majority_acc = round(loo_hits / len(shared), 4)
    n_singleton = sum(1 for g in by_conn.values() if len(g) == 1)

    rows = []
    for conn, group in by_conn.items():
        golds = [it.sense_l1 for it in group]
        dist = Counter(golds)
        maj = dist.most_common(1)[0][0]
        row = {
            "connective": conn,
            "n": len(group),
            "gold_distribution": dict(dist.most_common()),
            "entropy_bits": round(max(0.0, entropy(dist)), 3),
            "majority_label": maj,
            "majority_acc": round(sum(g == maj for g in golds) / len(golds), 4),
            "n_lexicon_senses": group[0].n_lexicon_senses,
            "lexicon_match": group[0].meta.get("lexicon_match"),
            "stratum": group[0].stratum()["ambiguity"],
        }
        for name, pred in models.items():
            hits = sum(pred[it.id] == it.sense_l1 for it in group)
            lo, hi = wilson(hits, len(group))
            row[f"{name}_acc"] = round(hits / len(group), 4)
            row[f"{name}_ci95"] = [round(lo, 3), round(hi, 3)]
            row[f"{name}_hits"] = hits
        rows.append(row)
    rows.sort(key=lambda r: (-r["n"], r["connective"]))

    # --- ambiguity strata ---------------------------------------------------
    strata: dict[str, dict] = {}
    for label in ("low", "high", "unknown"):
        group = [r for r in rows if r["stratum"] == label]
        n_items = sum(r["n"] for r in group)
        if not n_items:
            continue
        entry = {"n_connectives": len(group), "n_items": n_items}
        for name in models:
            hits = sum(r[f"{name}_hits"] for r in group)
            entry[f"{name}_acc"] = round(hits / n_items, 4)
        entry["majority_acc"] = round(
            sum(r["majority_acc"] * r["n"] for r in group) / n_items, 4
        )
        strata[label] = entry

    # --- does accuracy track how ambiguous the connective is? ----------------
    # Only connectives with enough items to estimate an accuracy at all.
    usable = [r for r in rows if r["n"] >= args.min_n]
    correlations = {}
    for name in models:
        rho, n = spearman([r["entropy_bits"] for r in usable],
                          [r[f"{name}_acc"] for r in usable])
        correlations[name] = {"spearman_entropy_vs_acc": round(rho, 3) if rho == rho else None,
                              "n_connectives": n}
    # One connective can carry a whole correlation: a single high-entropy
    # connective that every model fails will produce a strong negative rho on
    # its own.  Report how far rho moves when any one connective is removed.
    for name in models:
        loo = []
        for i in range(len(usable)):
            sub = usable[:i] + usable[i + 1:]
            rho, _ = spearman([r["entropy_bits"] for r in sub],
                              [r[f"{name}_acc"] for r in sub])
            if rho == rho:
                loo.append((rho, usable[i]["connective"]))
        if loo:
            correlations[name]["rho_leave_one_out"] = [round(min(r for r, _ in loo), 3),
                                                      round(max(r for r, _ in loo), 3)]
            base = correlations[name]["spearman_entropy_vs_acc"] or 0.0
            worst = max(loo, key=lambda t: abs(t[0] - base))
            correlations[name]["most_influential_connective"] = worst[1]
            correlations[name]["rho_without_it"] = round(worst[0], 3)

    matched = [r for r in usable if r["lexicon_match"] is True]
    for name in models:
        rho, n = spearman([r["n_lexicon_senses"] for r in matched],
                          [r[f"{name}_acc"] for r in matched])
        correlations[name]["spearman_nsenses_vs_acc"] = round(rho, 3) if rho == rho else None
        correlations[name]["n_connectives_matched"] = n

    # --- error concentration -------------------------------------------------
    concentration = {}
    for name in models:
        errs = sorted(((r["n"] - r[f"{name}_hits"], r["connective"], r["n"]) for r in rows),
                      reverse=True)
        total = sum(e for e, _, _ in errs)
        if total == 0:
            concentration[name] = {"total_errors": 0}
            continue
        cum, need = 0, None
        for k, (e, _, _) in enumerate(errs, 1):
            cum += e
            if cum >= 0.5 * total and need is None:
                need = k
        top5 = errs[:5]
        concentration[name] = {
            "total_errors": total,
            "n_connectives_with_error": sum(e > 0 for e, _, _ in errs),
            "connectives_for_half_the_errors": need,
            "top5": [{"connective": c, "errors": e, "n": n} for e, c, n in top5],
            "top5_error_share": round(sum(e for e, _, _ in top5) / total, 3),
        }

    # --- optional human cross-check ------------------------------------------
    # The per-connective picture can look like a model failure when the corpus
    # gold follows a lexical convention that the discourse does not support.
    # The B1 packet gives a small human sample on the same connectives, so the
    # table says who actually reads it the gold way.
    human = {}
    if args.human_a and args.human_b:
        import csv

        def read_sub(path: str) -> dict[str, str]:
            with open(ROOT / path, encoding="utf-8-sig", newline="") as f:
                return {r["item_id"]: (r.get("annotator_sense_l1") or "").strip().upper()
                        for r in csv.DictReader(f)}

        subs = {"C": read_sub(args.human_a), "D": read_sub(args.human_b)}
        for r in rows:
            if r["n"] < args.min_n:
                continue
            group = by_conn[r["connective"]]
            hit = {k: sum(sub.get(it.id) == it.sense_l1 for it in group) for k, sub in subs.items()}
            cover = {k: sum(1 for it in group if sub.get(it.id)) for k, sub in subs.items()}
            if not any(cover.values()):
                continue
            human[r["connective"]] = {
                "n_covered": cover,
                "acc_vs_gold": {k: round(hit[k] / cover[k], 3) if cover[k] else None
                                for k in hit},
            }

    report = {
        "human_crosscheck": human,
        "gold": args.gold,
        "n_explicit": len(items),
        "n_shared": len(shared),
        "n_dropped_for_missing_pred": dropped,
        "systems": list(models),
        "system_files": system_files,
        "coverage": {name: len(item_ids & set(pred)) for name, pred in models.items()},
        "loo_majority_acc": loo_majority_acc,
        "n_singleton_connectives": n_singleton,
        "n_connectives": len(rows),
        "min_n_for_table": args.min_n,
        "strata": strata,
        "correlations": correlations,
        "concentration": concentration,
        "rows": rows,
    }
    out = ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- markdown ------------------------------------------------------------
    systems = list(models)
    md = ["# 显式关系的逐连接词分解（R172）", "",
          f"数据：`{args.gold}`，显式条目 {len(items)} 条，"
          f"被所有系统共同覆盖 {len(shared)} 条（{len(models)} 个系统），"
          f"连接词 {len(rows)} 个。", "",
          "`majority` 是「对这个连接词永远预测它最常见的金标签」在这批条目上的准确率，"
          "即纯词汇规则的**样本内上限**；`lexeme_prior` 是 R110 的五折先验（样本外）。"
          "两者之间的差是词汇规则本身的不稳定，两者与 LLM 的差才是模型读到的语用信息。", "",
          "各系统取覆盖最全的那个文件（`_zs` 与 `_zs_filled` 并存时取覆盖多者），"
          "因此同一模型在不同段落里的口径一致：", "",
          "| 系统 | 覆盖显式条目 | 文件 |", "|---|---|---|"]
    for name in systems:
        md.append(f"| {name} | {report['coverage'][name]}/{len(items)} | "
                  f"`{system_files[name]}` |")
    in_sample = sum(r["majority_acc"] * r["n"] for r in rows) / len(shared)
    prior_acc = (sum(r["lexeme_prior_acc"] * r["n"] for r in rows) / len(shared)
                 if "lexeme_prior" in models else None)
    md += ["", "### 三个词汇基线口径", "",
           "| 基线 | 显式 acc | 说明 |", "|---|---|---|",
           f"| 样本内众数 | {in_sample:.4f} | 每个连接词永远预测它在这批条目上最常见的标签；**上限，不是可用规则** |",
           f"| 留一众数 | {loo_majority_acc:.4f} | 用该连接词的其余条目定标签，单例回退到全局众数；跨条目可用的词汇查表 |"]
    if prior_acc is not None:
        md.append(f"| 五折词形先验 | {prior_acc:.4f} | R110，基于中文歧义词典的词表先验（样本外） |")
    md += ["", f"其中 {n_singleton} 个连接词在这批数据里只出现过一次；"
           "留一众数在这些条目上只能回退到全局众数，所以留一基线整体偏低。", "",
           "## 1. 按连接词（n ≥ {}）".format(args.min_n), ""]
    header = "| 连接词 | n | 金标签分布 | H(bits) | majority | " + " | ".join(systems) + " |"
    md += [header, "|" + "---|" * (5 + len(systems))]
    for r in rows:
        if r["n"] < args.min_n:
            continue
        dist = " / ".join(f"{k}:{v}" for k, v in r["gold_distribution"].items())
        cells = [f"{r[f'{s}_acc']:.3f}" for s in systems]
        md.append(f"| `{r['connective']}` | {r['n']} | {dist} | {r['entropy_bits']:.2f} | "
                  f"{r['majority_acc']:.3f} | " + " | ".join(cells) + " |")
    small = [r for r in rows if r["n"] < args.min_n]
    if small:
        md += ["", f"另有 {len(small)} 个连接词只出现 1–{args.min_n - 1} 次，"
                   "单条准确率没有信息量，保留在 JSON 里。"]

    if human:
        md += ["", "## 1.5 人工对照（B1 包）", "",
               "同一批条目里，C/D 两位标注者相对**金标准**的准确率。"
               "模型与人工差别很大的连接词，先要问是模型错了，还是金标准用了没有文本证据的约定。", "",
               "| 连接词 | n | C 覆盖 | C vs 金标准 | D 覆盖 | D vs 金标准 |",
               "|---|---|---|---|---|---|"]
        for r in rows:
            h = human.get(r["connective"])
            if not h:
                continue
            c, d = h["acc_vs_gold"]["C"], h["acc_vs_gold"]["D"]
            md.append(f"| `{r['connective']}` | {r['n']} | {h['n_covered']['C']} | "
                      f"{c if c is not None else '—'} | {h['n_covered']['D']} | "
                      f"{d if d is not None else '—'} |")

    md += ["", "## 2. 歧义分层", "",
           "`low` = 词表义项 ≤ 2，`high` = ≥ 3，`unknown` = 连接词不在中文歧义词典里"
           "（HIT-CDTB 上占多数，不能当作「低歧义」）。", "",
           "| 分层 | 连接词数 | 条目数 | majority | " + " | ".join(systems) + " |",
           "|" + "---|" * (4 + len(systems))]
    for label in ("low", "high", "unknown"):
        s = strata.get(label)
        if not s:
            continue
        cells = [f"{s[f'{name}_acc']:.3f}" for name in systems]
        md.append(f"| {label} | {s['n_connectives']} | {s['n_items']} | "
                  f"{s['majority_acc']:.3f} | " + " | ".join(cells) + " |")

    md += ["", "## 3. 准确率跟连接词的歧义程度走吗", "",
           f"只在 n ≥ {args.min_n} 的连接词上计算 Spearman ρ。", "",
           "| 系统 | ρ(H, acc) | 去掉一个连接词后的 ρ 范围 | 最影响 ρ 的连接词 | ρ(义项数, acc) |",
           "|---|---|---|---|---|"]
    for name in systems:
        c = correlations[name]
        loo = c.get("rho_leave_one_out")
        loo_txt = f"[{loo[0]}, {loo[1]}]" if loo else "—"
        infl = c.get("most_influential_connective")
        infl_txt = f"`{infl}` ({c['rho_without_it']})" if infl else "—"
        md.append(f"| {name} | {c['spearman_entropy_vs_acc']} | {loo_txt} | {infl_txt} | "
                  f"{c['spearman_nsenses_vs_acc']} |")
    md += ["", "ρ 明显为负说明「连接词越歧义、准确率越低」，即准确率确实由词汇层面驱动；"
               "ρ 接近 0 说明模型在歧义连接词上并没有系统性掉分。"
               "留一列是稳健性检查：如果去掉一个连接词 ρ 就翻号，这条相关只能算一个连接词的现象，"
               "不能当成普遍规律。"]

    md += ["", "## 4. 错误集中度", "",
           "| 系统 | 错误数 | 有错误的连接词 | 占一半错误所需的连接词数 | top-5 错误占比 |",
           "|---|---|---|---|---|"]
    for name in systems:
        c = concentration[name]
        if c["total_errors"] == 0:
            md.append(f"| {name} | 0 | 0 | — | — |")
            continue
        md.append(f"| {name} | {c['total_errors']} | {c['n_connectives_with_error']} | "
                  f"{c['connectives_for_half_the_errors']} | {c['top5_error_share']:.1%} |")

    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = (ROOT / out_md) if (ROOT / out_md).parent.exists() else (ROOT.parent / out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
