"""Analyse the R152 timed-annotation submissions.

Conditions, as implemented in ``annotation/timing/``:

    a  annotate from scratch (no model output)
    b  proofread a pre-filled model suggestion
    c  review only the triage-flagged items; the rest accept the model label

Every one of the 100 items is observed under ``a`` exactly once and under ``b``
exactly once (the halves swap between sessions 1 and 2), so a-vs-b is paired at
the item level.  Condition ``c`` covers only the ~50 flagged items; the
unflagged ones are counted at 0 ms and at the model's label, because in that
workflow they cost the annotator nothing.

The unit of the bootstrap is the item, not the annotator: with one annotator
there is no between-annotator variance to estimate, and pretending otherwise
would understate uncertainty.  Item-level resampling still captures the
uncertainty in *which* 100 items were drawn.

    python scripts/r152_analyze.py \
        --gold data/processed/l1/r152_timing_items.jsonl \
        --submissions data/interim/r152/submissions \
        --out-json outputs/l1_r152_timing.json \
        --out-md ../refine-logs/R152_RESULTS.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items  # noqa: E402

CONDS = ("a", "b", "c")


# ---------------------------------------------------------------- loading

def load_submissions(paths: list[Path], alias: dict[str, str] | None = None) -> list[dict]:
    alias = alias or {}
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("item_id"):
                    continue
                if (r.get("condition") or "").strip() == "w":
                    continue  # warm-up, never analysed
                r["annotator"] = alias.get(r["annotator"], r["annotator"])
                r["_file"] = path.name
                rows.append(r)
    return rows


def dedupe(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep the latest submission per (annotator, session).

    Re-submitting after a break is normal; merging two versions of the same
    session would double-count items and corrupt both the timing sums and the
    accuracy.
    """
    latest: dict[tuple[str, str], dict] = {}
    notes: list[str] = []
    for r in rows:
        key = (r["annotator"], r["session"])
        prev = latest.get(key)
        if prev is None or r["_file"] > prev["_file"]:
            if prev is not None:
                notes.append(
                    f"{key[0]}/s{key[1]}: {prev['_file']} superseded by {r['_file']}"
                )
            latest[key] = r
    # One row per submission is enough to order them; the payload rows were
    # read per file, so group by file instead.
    keep_files = {r["_file"] for r in latest.values()}
    kept = [r for r in rows if r["_file"] in keep_files]
    dropped = sorted({r["_file"] for r in rows} - keep_files)
    if dropped:
        notes.append(f"superseded files: {', '.join(dropped)}")
    return kept, notes


# ---------------------------------------------------------------- stats

def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def boot_ratio(num: list[float], den: list[float], n_boot: int, seed: int):
    """Bootstrap a ratio of paired sums, resampling items.

    ``saving = (sum(den) - sum(num)) / sum(den)`` with ``den`` = condition-a
    time.  Resampling items keeps each item's a/b/c times together, which is
    what "paired" is supposed to buy.
    """
    rng = random.Random(seed)
    n = len(num)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = [den[i] for i in idx]
        m = [num[i] for i in idx]
        sd = sum(d)
        if sd <= 0:
            continue
        vals.append((sd - sum(m)) / sd)
    vals.sort()
    if not vals:
        return (None, None, None)
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return (vals[len(vals) // 2], lo, hi)


def robust_pair(num: dict[str, float], den: dict[str, float],
                keys: list[str], cap_s: float = 60.0) -> dict:
    """Sensitivity checks for a paired saving.

    A ratio of sums is dominated by the few longest items, and one break-time
    that was not left out of the clock can flip its sign.  These three
    statistics answer "would the conclusion survive that one item?":

    * ``median_rel``  median over items of ``(t_cond - t_a) / t_a`` (negative =
      the condition was faster on the typical item);
    * ``frac_faster`` share of items on which the condition was faster;
    * ``saving_winsor`` re-summed after capping each item at ``cap_s``;
    * ``saving_drop_max`` re-summed after removing the condition's longest item.
    """
    cap_ms = cap_s * 1000.0
    usable = [i for i in keys if den[i] > 0]
    if not usable:
        return {}
    rel = sorted((num[i] - den[i]) / den[i] for i in usable)
    tw_n = sum(min(cap_ms, num[i]) for i in usable)
    tw_d = sum(min(cap_ms, den[i]) for i in usable)
    worst = max(usable, key=lambda i: num[i])
    td_n = sum(num[i] for i in usable if i != worst)
    td_d = sum(den[i] for i in usable if i != worst)
    return {
        "median_rel": round(statistics.median(rel), 4),
        "frac_faster": round(sum(1 for x in rel if x < 0) / len(rel), 4),
        f"saving_winsor{int(cap_s)}s": round((tw_d - tw_n) / tw_d, 4) if tw_d else None,
        "saving_drop_max": round((td_d - td_n) / td_d, 4) if td_d else None,
        "max_item_s": round(num[worst] / 1000, 1),
    }


def describe(ms: list[float]) -> dict:
    ms = sorted(ms)
    n = len(ms)
    if not n:
        return {"n": 0}
    def pct(p: float) -> float:
        return ms[min(n - 1, int(p * n))]
    return {
        "n": n,
        "total_s": round(sum(ms) / 1000, 1),
        "mean_s": round(statistics.mean(ms) / 1000, 2),
        "median_s": round(statistics.median(ms) / 1000, 2),
        "p90_s": round(pct(0.9) / 1000, 2),
        "max_s": round(ms[-1] / 1000, 1),
    }


# ---------------------------------------------------------------- analysis

def analyse_one(ann: str, rows: list[dict], gold: dict[str, dict], n_boot: int,
                seed: int) -> dict:
    # One record per (item, condition); later rows win so a corrected re-answer
    # is respected.
    picked: dict[tuple[str, str], dict] = {}
    for r in rows:
        cond = (r.get("condition") or "").strip()
        if cond not in CONDS:
            continue
        picked[(r["item_id"], cond)] = r

    ids = sorted(gold)
    per_cond: dict[str, list[dict]] = defaultdict(list)
    for i in ids:
        for cond in CONDS:
            r = picked.get((i, cond))
            if r is not None:
                per_cond[cond].append(r)

    # ---- condition c without a third session
    # Reviewing a flagged item is the same operation in b and c, and unflagged
    # items are never shown, so c's cost is the flagged subset of b plus zero.
    # This is a reconstruction, not an independent measurement: it assumes the
    # difficulty premium of a flagged item does not change when *every* item in
    # the session is flagged.  Flagged below and in the report.
    c_derived = not per_cond.get("c")
    if c_derived and per_cond.get("b"):
        synth = []
        for r in per_cond["b"]:
            row = dict(r)
            if (r.get("flagged") or "0") != "1":
                row.update(label=r.get("suggestion", ""), ms=0, auto="1",
                           changed="", visits="0", answered_at="")
            else:
                row["auto"] = "0"
            synth.append(row)
        per_cond["c"] = synth

    # The split design gives each annotator only half the items, so "missing"
    # must be judged against the items this annotator actually covered, not
    # against the gold file.
    by_cond = {c: {r["item_id"]: r for r in per_cond.get(c, [])} for c in CONDS}
    mine = set().union(*[set(d) for d in by_cond.values()]) if by_cond else set()
    missing = {c: sorted(mine - set(by_cond[c])) for c in CONDS}
    missing = {k: v for k, v in missing.items() if v}

    # ---- accuracy
    acc = {}
    for cond in CONDS:
        rs = per_cond[cond]
        if not rs:
            continue
        hits = sum(1 for r in rs if (r["label"] or "").strip() == gold[r["item_id"]]["sense_l1"])
        unknown = sum(1 for r in rs if (r["label"] or "").strip() == "UNKNOWN")
        acc[cond] = {
            "n": len(rs),
            "correct": hits,
            "acc": round(hits / len(rs), 4),
            "unknown": unknown,
            "n_auto": sum(1 for r in rs if r.get("auto") == "1"),
        }

    # ---- time
    times: dict[str, dict[str, float]] = {}
    for cond in CONDS:
        times[cond] = {
            r["item_id"]: float(r.get("ms") or 0) for r in per_cond[cond]
        }

    hits = {
        cond: {r["item_id"]: (r["label"] or "").strip() == gold[r["item_id"]]["sense_l1"]
               for r in per_cond.get(cond, [])}
        for cond in CONDS
    }

    # Human labels only: condition c's auto-accepted rows are the *same* model
    # label for every annotator, so including them would manufacture agreement.
    labels = {
        cond: {r["item_id"]: (r["label"] or "").strip()
               for r in per_cond.get(cond, []) if r.get("auto") != "1"}
        for cond in CONDS
    }

    common = [i for i in ids if all(i in times[c] for c in CONDS)]
    summary = {}
    for cond in CONDS:
        if times[cond]:
            summary[cond] = describe(list(times[cond].values()))

    savings = {}
    for cond in ("b", "c"):
        keys = [i for i in ids if i in times["a"] and i in times[cond]]
        if not keys:
            continue
        num = [times[cond][i] for i in keys]
        den = [times["a"][i] for i in keys]
        med, lo, hi = boot_ratio(num, den, n_boot, seed)
        ta, tc = sum(den), sum(num)
        savings[f"a_vs_{cond}"] = {
            "n_pairs": len(keys),
            "total_a_s": round(ta / 1000, 1),
            "total_cond_s": round(tc / 1000, 1),
            "net_saving": round((ta - tc) / ta, 4) if ta else None,
            "boot_median": round(med, 4) if med is not None else None,
            "boot_ci": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "robust": robust_pair(times[cond], times["a"], keys),
        }

    # ---- paired accuracy contrasts (McNemar exact on shared items)
    contrasts = {}
    for c1, c2 in (("a", "b"), ("a", "c"), ("b", "c")):
        keys = sorted(set(by_cond[c1]) & set(by_cond[c2]))
        if not keys:
            continue
        right1 = [(by_cond[c1][i]["label"] or "").strip() == gold[i]["sense_l1"]
                  for i in keys]
        right2 = [(by_cond[c2][i]["label"] or "").strip() == gold[i]["sense_l1"]
                  for i in keys]
        d12 = sum(1 for x, y in zip(right1, right2) if x and not y)
        d21 = sum(1 for x, y in zip(right1, right2) if y and not x)
        contrasts[f"{c1}_vs_{c2}"] = {
            "n": len(keys), f"{c1}_only": d12, f"{c2}_only": d21,
            "p_exact": round(exact_mcnemar(d12, d21), 4),
        }

    # ---- condition c: what the review actually bought
    c_rows = per_cond.get("c", [])
    flagged_times = [float(r.get("ms") or 0) for r in c_rows if r.get("auto") != "1"]
    auto_rows = [r for r in c_rows if r.get("auto") == "1"]
    c_detail = {
        "reviewed": len(flagged_times),
        "auto_accepted": len(auto_rows),
        "review_time_s": round(sum(flagged_times) / 1000, 1),
        "per_reviewed_item_s": round(statistics.mean(flagged_times) / 1000, 2)
        if flagged_times else None,
        "auto_acc": round(
            sum(1 for r in auto_rows
                if (r["label"] or "").strip() == gold[r["item_id"]]["sense_l1"])
            / len(auto_rows), 4) if auto_rows else None,
    }

    # ---- same-item comparison on the triage-flagged subset
    # Reviewing a flagged item must be compared with annotating *that* item
    # from scratch, not with the average item: flagged items are the hard ones
    # and are slower for reasons that have nothing to do with the suggestion.
    b_flag = {r["item_id"]: (r.get("flagged") or "0") == "1"
              for r in per_cond.get("b", [])}
    fk = [i for i in ids if i in times["a"] and i in times["b"] and b_flag.get(i)]
    review_vs_scratch = {}
    if fk:
        ta_f = sum(times["a"][i] for i in fk)
        tb_f = sum(times["b"][i] for i in fk)
        review_vs_scratch = {
            "n": len(fk),
            "a_mean_s": round(ta_f / len(fk) / 1000, 2),
            "b_mean_s": round(tb_f / len(fk) / 1000, 2),
            "delta_pct": round((tb_f - ta_f) / ta_f, 4) if ta_f else None,
        }

    # ---- where the time goes
    by_group = {}
    for cond in ("a", "b"):
        for label, pred in (("explicit", lambda i: gold[i]["explicit"]),
                            ("implicit", lambda i: not gold[i]["explicit"])):
            vals = [times[cond][i] for i in ids if i in times[cond] and pred(i)]
            if vals:
                by_group[f"{cond}:{label}"] = describe(vals)
    for label, pred in (("reviewed", lambda i: i in {r["item_id"] for r in c_rows if r.get("auto") != "1"}),):
        vals = [times["c"][i] for i in ids if i in times["c"] and pred(i)]
        if vals:
            by_group[f"c:{label}"] = describe(vals)

    # ---- changed-answer rate in b/c (was the suggestion accepted?)
    changed = {}
    for cond in ("b", "c"):
        rs = [r for r in per_cond.get(cond, []) if r.get("auto") != "1" and r.get("changed") != ""]
        if rs:
            changed[cond] = {
                "n": len(rs),
                "changed": sum(1 for r in rs if r.get("changed") == "1"),
                "changed_rate": round(
                    sum(1 for r in rs if r.get("changed") == "1") / len(rs), 4
                ),
            }

    # ---- anchoring: what happens when the suggestion is wrong
    # In a the suggestion is not displayed, so "accepting" it there is just
    # agreement.  In b it is displayed and pre-filled, so a row whose label
    # equals a *wrong* suggestion is a swallowed model error.
    follow = {}
    b_rows = per_cond.get("b", [])
    a_rows = per_cond.get("a", [])
    if b_rows:
        m_right = [r for r in b_rows if r.get("suggestion") == gold[r["item_id"]]["sense_l1"]]
        m_wrong = [r for r in b_rows if r.get("suggestion") != gold[r["item_id"]]["sense_l1"]]
        acc_ = lambda rs: (sum(1 for r in rs
                               if (r["label"] or "").strip() == gold[r["item_id"]]["sense_l1"])
                           / len(rs)) if rs else None
        follow = {
            "model_acc_on_items": round(
                sum(1 for r in b_rows
                    if r.get("suggestion") == gold[r["item_id"]]["sense_l1"]) / len(b_rows), 4),
            "n_model_right": len(m_right),
            "n_model_wrong": len(m_wrong),
            "acc_when_suggestion_right": round(acc_(m_right), 4) if m_right else None,
            "acc_when_suggestion_wrong": round(acc_(m_wrong), 4) if m_wrong else None,
            "swallowed_wrong": sum(1 for r in m_wrong
                                   if (r["label"] or "").strip() == r.get("suggestion")),
            "broke_right": sum(1 for r in m_right
                               if (r["label"] or "").strip() != r.get("suggestion")),
        }
        if a_rows:
            follow["a_agrees_with_model"] = round(
                sum(1 for r in a_rows if (r["label"] or "").strip() == r.get("suggestion"))
                / len(a_rows), 4)

    # ---- fatigue: mean per-item time in each fifth of a session
    fatigue = {}
    for cond in ("a", "b"):
        rs = per_cond.get(cond, [])
        rs = [r for r in rs if r.get("answered_at")]
        if len(rs) < 10:
            continue
        rs.sort(key=lambda r: r["answered_at"])
        step = max(1, len(rs) // 5)
        fatigue[cond] = [
            round(statistics.mean([float(x.get("ms") or 0) for x in rs[i:i + step]]) / 1000, 2)
            for i in range(0, len(rs), step)
        ]

    return {
        "annotator": ann,
        "n_items_gold": len(ids),
        "missing": {k: len(v) for k, v in missing.items() if v},
        "accuracy": acc,
        "time": summary,
        "time_paired_n": len(common),
        "savings": savings,
        "accuracy_contrasts": contrasts,
        "condition_c": c_detail,
        "c_derived": c_derived,
        "by_group": by_group,
        "changed": changed,
        "follow": follow,
        "review_vs_scratch": review_vs_scratch,
        "flagged_ids": fk,
        "fatigue_mean_s_by_fifth": fatigue,
        # Kept for the cross-annotator pass; also written to JSON so the
        # agreement numbers can be recomputed without re-parsing the CSVs.
        "labels": labels,
        "hits": {c: hits[c] for c in CONDS if hits[c]},
        "times": {c: times[c] for c in CONDS if times[c]},
    }


def cohen_kappa(a: dict[str, str], b: dict[str, str], keys: list[str]) -> dict:
    """Unweighted 4-class Cohen's kappa between two annotators."""
    n = len(keys)
    pa, pb = Counter(a[k] for k in keys), Counter(b[k] for k in keys)
    po = sum(1 for k in keys if a[k] == b[k]) / n
    pe = sum(pa[c] * pb[c] for c in set(pa) | set(pb)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else None
    return {"n": n, "po": round(po, 4),
            "kappa": round(k, 4) if k is not None else None}


def cross_annotator(results: list[dict], n_boot: int, seed: int) -> dict | None:
    """Agreement and pooled timing across annotators.

    Agreement under b/c is *not* a human ceiling: both annotators see the same
    model suggestion, so a shared anchor inflates it by construction.  It is
    reported to make that inflation visible, and the comparable number to the
    B1 four-annotator kappa is condition a.
    """
    live = [r for r in results if r.get("labels")]
    if len(live) < 2:
        return None

    agreement = {}
    for cond in CONDS:
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                A, B = live[i], live[j]
                la, lb = A["labels"].get(cond) or {}, B["labels"].get(cond) or {}
                keys = sorted(set(la) & set(lb))
                if not keys:
                    continue
                agreement.setdefault(cond, []).append(
                    {"pair": f"{A['annotator']}-{B['annotator']}",
                     **cohen_kappa(la, lb, keys)}
                )

    # Pooled time: average each item's time over the annotators who ran it,
    # then bootstrap items.  With two annotators this is "the average
    # annotator", not a population estimate.
    # Union, not intersection: in the split design each item is annotated by
    # exactly one person, so intersecting would leave nothing to pool.
    ids = sorted(set().union(*[set(r["times"]["a"]) for r in live]))
    pooled = {}
    per_item = {"a": [], "b": [], "c": []}
    for i in ids:
        for cond in CONDS:
            vals = [r["times"][cond][i] for r in live
                    if i in r["times"].get(cond, {})]
            if vals:
                per_item[cond].append(statistics.mean(vals))
    for cond in ("b", "c"):
        if not per_item[cond] or not per_item["a"]:
            continue
        med, lo, hi = boot_ratio(per_item[cond], per_item["a"], n_boot, seed)
        ta, tc = sum(per_item["a"]), sum(per_item[cond])
        pooled[cond] = {
            "n_items": len(per_item[cond]),
            "total_a_s": round(ta / 1000, 1),
            "total_cond_s": round(tc / 1000, 1),
            "net_saving": round((ta - tc) / ta, 4) if ta else None,
            "boot_ci": [round(lo, 4), round(hi, 4)]
            if lo is not None and hi is not None else None,
        }
    # Pooled accuracy: in the split design every item is answered under each
    # condition by exactly one person, so the union is a clean 100-item score.
    merged = {c: {} for c in CONDS}
    for r in live:
        for cond in CONDS:
            merged[cond].update(r.get("hits", {}).get(cond, {}))
    acc_pool = {}
    for cond in CONDS:
        if merged[cond]:
            k = sum(merged[cond].values())
            acc_pool[cond] = {"n": len(merged[cond]), "correct": k,
                              "acc": round(k / len(merged[cond]), 4)}
    contrasts_pool = {}
    for c1, c2 in (("a", "b"), ("a", "c"), ("b", "c")):
        keys = sorted(set(merged[c1]) & set(merged[c2]))
        if not keys:
            continue
        d12 = sum(1 for i in keys if merged[c1][i] and not merged[c2][i])
        d21 = sum(1 for i in keys if merged[c2][i] and not merged[c1][i])
        contrasts_pool[f"{c1}_vs_{c2}"] = {
            "n": len(keys), f"{c1}_only": d12, f"{c2}_only": d21,
            "p_exact": round(exact_mcnemar(d12, d21), 4),
        }

    # Paired a-vs-b on the flagged subset only, pooled over annotators.
    rv = {}
    for r in live:
        for i in r.get("flagged_ids", []):
            if i in rv:
                continue
            rv[i] = (r["times"]["a"].get(i), r["times"]["b"].get(i))
    rv = {i: v for i, v in rv.items() if v[0] is not None and v[1] is not None}
    review_pool = {}
    if rv:
        ta_ = sum(v[0] for v in rv.values())
        tb_ = sum(v[1] for v in rv.values())
        review_pool = {
            "n": len(rv),
            "a_mean_s": round(ta_ / len(rv) / 1000, 2),
            "b_mean_s": round(tb_ / len(rv) / 1000, 2),
            "delta_pct": round((tb_ - ta_) / ta_, 4) if ta_ else None,
        }

    return {"n_annotators": len(live), "agreement": agreement, "pooled": pooled,
            "accuracy_pooled": acc_pool, "contrasts_pooled": contrasts_pool,
            "review_vs_scratch_pooled": review_pool}


def render_cross_md(results: list[dict], cross: dict | None) -> list[str]:
    if not cross:
        return []
    L = ["## 跨标注者\n"]
    L.append(f"{cross['n_annotators']} 名标注者，同一盲标包；split 设计下各覆盖一半条目、"
             "互不重叠。每条节省区间仍是**条目级** bootstrap。\n")

    L.append("### 逐标注者\n")
    L.append("| 标注者 | b 净节省 [95% CI] | c 净节省 [95% CI] | acc a | acc b | acc c |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        cells = []
        for key in ("a_vs_b", "a_vs_c"):
            s = r["savings"].get(key)
            if not s or s["net_saving"] is None:
                cells.append("n/a")
            else:
                ci = s["boot_ci"]
                ci_txt = (f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"
                          if ci and ci[0] is not None else "")
                cells.append(f"{s['net_saving']*100:.1f}% {ci_txt}")
        accs = [f"{r['accuracy'][c]['acc']:.3f}" if c in r["accuracy"] else "n/a"
                for c in CONDS]
        L.append(f"| {r['annotator']} | {cells[0]} | {cells[1]} | "
                 + " | ".join(accs) + " |")
    L.append("")

    if cross["pooled"]:
        L.append("### 两标注者合并（每条由一人标注）\n")
        L.append("| 对比 | 条目数 | a 总时 | 该条件总时 | 净节省 | 95% CI |")
        L.append("|---|---|---|---|---|---|")
        for cond, label in (("b", "b vs a"), ("c", "c vs a")):
            p = cross["pooled"].get(cond)
            if not p:
                continue
            ci = p["boot_ci"]
            ci_txt = (f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"
                      if ci and ci[0] is not None else "n/a")
            L.append(f"| {label} | {p['n_items']} | {p['total_a_s']} "
                     f"| {p['total_cond_s']} | "
                     + (f"{p['net_saving']*100:.1f}%" if p["net_saving"] is not None
                        else "n/a")
                     + f" | {ci_txt} |")
        L.append("")

    rp = cross.get("review_vs_scratch_pooled") or {}
    if rp:
        L.append("### 被标记条目的单条复核成本（同一批题，a vs b）\n")
        L.append(f"合并 n={rp['n']}：从零标 {rp['a_mean_s']} s → 看建议 {rp['b_mean_s']} s "
                 f"（{rp['delta_pct']*100:+.1f}%）。这条必须和**同一批条目**比："
                 "被标记的都是难题，和全体均值比会虚报。\n")

    if cross.get("accuracy_pooled"):
        L.append("### 两标注者合并准确率\n")
        L.append("split 设计下每条在每个条件下只由一人作答，合并后是干净的 100 条目分数。\n")
        L.append("| 条件 | n | 正确 | acc |")
        L.append("|---|---|---|---|")
        for cond in CONDS:
            a = cross["accuracy_pooled"].get(cond)
            if a:
                L.append(f"| {cond} | {a['n']} | {a['correct']} | {a['acc']:.4f} |")
        L.append("")
        if cross.get("contrasts_pooled"):
            L.append("配对 McNemar（exact，合并 n=100）：\n")
            L.append("| 对比 | n | 仅前者对 | 仅后者对 | p |")
            L.append("|---|---|---|---|---|")
            for k, v in cross["contrasts_pooled"].items():
                c1, c2 = k.split("_vs_")
                L.append(f"| {k} | {v['n']} | {v[f'{c1}_only']} | {v[f'{c2}_only']} "
                         f"| {v['p_exact']:.4f} |")
            L.append("")

    if cross["agreement"]:
        L.append("### 条件内人-人一致性\n")
        L.append("| 条件 | 标注者对 | n | 原始一致率 | κ |")
        L.append("|---|---|---|---|---|")
        for cond in CONDS:
            for row in cross["agreement"].get(cond, []):
                k = "n/a" if row["kappa"] is None else f"{row['kappa']:.4f}"
                L.append(f"| {cond} | {row['pair']} | {row['n']} "
                         f"| {row['po']:.4f} | {k} |")
        L.append("")
        L.append("条件 a 的一致性才是这批条目上的人-人上限；b/c 里两人看的是同一条模型建议，"
                 "一致性被共同锚定抬高，不能当效度证据。\n")
    return L


def render_md(results: list[dict], notes: list[str], n_boot: int,
              cross: dict | None = None) -> str:
    L: list[str] = []
    L.append("# R152 真实计时：结果\n")
    L.append("由 `scripts/r152_analyze.py` 生成；`s` = 秒。条件 a = 从零标，"
             "b = 校对模型建议，c = 只查被标记项（未标记项自动采用模型答案、计 0 秒）。"
             "若没有单独跑 c（split 设计），c 由 b 的被标记子集重建，报告里会注明。\n")
    L.append(f"配对 bootstrap 重采样 {n_boot} 次，重采样单位是**条目**。\n")
    if notes:
        L.append("数据说明：\n")
        for n in notes:
            L.append(f"- {n}")
        L.append("")

    for r in results:
        L.append(f"## 标注者 {r['annotator']}\n")
        if r["missing"]:
            L.append(f"⚠ 缺失条目：{r['missing']}\n")

        L.append("### 每条件用时\n")
        L.append("| 条件 | n | 总计 | 均值 | 中位数 | P90 | 最大 |")
        L.append("|---|---|---|---|---|---|---|")
        for cond, label in (("a", "a 从零标"), ("b", "b 校对建议"), ("c", "c 只查标记")):
            t = r["time"].get(cond)
            if t:
                L.append(f"| {label} | {t['n']} | {t['total_s']} | {t['mean_s']} "
                         f"| {t['median_s']} | {t['p90_s']} | {t['max_s']} |")
        L.append("")

        L.append("### 净节省（配对，百分位 bootstrap CI）\n")
        L.append("| 对比 | 配对数 | a 总时 | 该条件总时 | 净节省 | 95% CI |")
        L.append("|---|---|---|---|---|---|")
        for key, label in (("a_vs_b", "b vs a"), ("a_vs_c", "c vs a")):
            s = r["savings"].get(key)
            if not s:
                continue
            if s["net_saving"] is None:
                L.append(f"| {label} | {s['n_pairs']} | {s['total_a_s']} "
                         f"| {s['total_cond_s']} | n/a | n/a |")
                continue
            ci = s["boot_ci"]
            ci_txt = (f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"
                      if ci and ci[0] is not None else "n/a")
            L.append(f"| {label} | {s['n_pairs']} | {s['total_a_s']} | {s['total_cond_s']} "
                     f"| {s['net_saving']*100:.1f}% | {ci_txt} |")
        L.append("")
        L.append("**稳健性**：净节省是总时之比，对最长的几条极敏感。另外三种口径如下，"
                 "方向不一致时只能报告「无可测节省」。\n")
        L.append("| 对比 | 条目中位相对变化 | 该条件更快的条目 | winsor@60s | 去掉最长条目 | 最长条目 |")
        L.append("|---|---|---|---|---|---|")
        for key, label in (("a_vs_b", "b vs a"), ("a_vs_c", "c vs a")):
            rb = (r["savings"].get(key) or {}).get("robust") or {}
            if not rb:
                continue
            wins = rb.get("saving_winsor60s")
            drop = rb.get("saving_drop_max")
            wins_txt = f"{wins*100:+.1f}%" if wins is not None else "n/a"
            drop_txt = f"{drop*100:+.1f}%" if drop is not None else "n/a"
            L.append(
                f"| {label} | {rb['median_rel']*100:+.1f}% | {rb['frac_faster']*100:.0f}% "
                f"| {wins_txt} | {drop_txt} | {rb['max_item_s']} s |"
            )
        L.append("\n中位相对变化为负表示该条件下的题目更快（省时）。c 的未标记条目计 0 秒，"
                 "因此 c 的「中位相对变化」几乎总是 −100%，只有 winsor 与去最长条目两列有信息量。\n")

        L.append("### 准确率（对金标准 L1）\n")
        L.append("| 条件 | n | 正确 | acc | 无法判定 | 自动采用 |")
        L.append("|---|---|---|---|---|---|")
        for cond, label in (("a", "a"), ("b", "b"), ("c", "c")):
            a = r["accuracy"].get(cond)
            if a:
                L.append(f"| {label} | {a['n']} | {a['correct']} | {a['acc']:.4f} "
                         f"| {a['unknown']} | {a['n_auto']} |")
        L.append("")
        if r["accuracy_contrasts"]:
            L.append("配对 McNemar（exact）：\n")
            L.append("| 对比 | n | 仅前者对 | 仅后者对 | p |")
            L.append("|---|---|---|---|---|")
            for k, v in r["accuracy_contrasts"].items():
                c1, c2 = k.split("_vs_")
                L.append(f"| {k} | {v['n']} | {v[f'{c1}_only']} | {v[f'{c2}_only']} "
                         f"| {v['p_exact']:.4f} |")
            L.append("")

        c = r["condition_c"]
        if c["reviewed"]:
            L.append("### 条件 c 的构成\n")
            L.append(f"- 人工复核 {c['reviewed']} 条，自动采用 {c['auto_accepted']} 条")
            L.append(f"- 复核总用时 {c['review_time_s']} s，单条均值 {c['per_reviewed_item_s']} s")
            L.append(f"- 自动采用部分的准确率 {c['auto_acc']}")
            rv = r.get("review_vs_scratch") or {}
            if rv:
                L.append(f"- 同一批被标记条目（n={rv['n']}）：从零标 {rv['a_mean_s']} s → "
                         f"看建议复核 {rv['b_mean_s']} s（{rv['delta_pct']*100:+.1f}%）")
            if r.get("c_derived"):
                L.append("- **c 由 b 重建**：被标记条目沿用 b 的用时与标签，未标记条目计 0 秒并采用模型答案。"
                         "这是估算，前提是「整场都是难题」不改变单条复核速度。")
            L.append("")

        f = r.get("follow") or {}
        if f:
            L.append("### 建议的锚定效应\n")
            L.append(f"模型在这 50 条上的准确率 {f['model_acc_on_items']:.4f}；"
                     f"其中答错 {f['n_model_wrong']} 条。\n")
            L.append("| 情形 | 人在这批上的 acc |")
            L.append("|---|---|")
            if f["acc_when_suggestion_right"] is not None:
                L.append(f"| 建议正确 | {f['acc_when_suggestion_right']:.4f} |")
            if f["acc_when_suggestion_wrong"] is not None:
                L.append(f"| 建议错误 | {f['acc_when_suggestion_wrong']:.4f} |")
            L.append("")
            L.append(f"- 建议错误时仍被采纳（吞下模型错误）：{f['swallowed_wrong']}/{f['n_model_wrong']}")
            L.append(f"- 建议正确时仍被改掉（改坏建议）：{f['broke_right']}/{f['n_model_right']}")
            if f.get("a_agrees_with_model") is not None:
                L.append(f"- 无建议时（a）人的标签恰好等于模型标签：{f['a_agrees_with_model']*100:.1f}%")
            L.append("")

        for cond in ("b", "c"):
            ch = r["changed"].get(cond)
            if ch:
                L.append(f"- 条件 {cond}：改动建议 {ch['changed']}/{ch['n']} "
                         f"（{ch['changed_rate']*100:.1f}%）")
        if any(r["changed"] for r in results):
            L.append("")

        if r["by_group"]:
            L.append("### 时间分解（秒／条）\n")
            L.append("| 组 | n | 均值 | 中位数 |")
            L.append("|---|---|---|---|")
            for k, v in r["by_group"].items():
                L.append(f"| {k} | {v['n']} | {v['mean_s']} | {v['median_s']} |")
            L.append("")

        if r["fatigue_mean_s_by_fifth"]:
            L.append("### 疲劳检查：按作答顺序分五段的单条均值（秒）\n")
            for cond, vals in r["fatigue_mean_s_by_fifth"].items():
                L.append(f"- {cond}: {vals}")
            L.append("")

    L.extend(render_cross_md(results, cross))
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/processed/l1/r152_timing_items.jsonl")
    ap.add_argument("--submissions", default="data/interim/r152/submissions")
    ap.add_argument("--files", nargs="*", help="explicit CSV paths (overrides --submissions)")
    ap.add_argument("--out-json", default="outputs/l1_r152_timing.json")
    # The write-up logs live one level above the code repo.
    ap.add_argument("--out-md", default="../refine-logs/R152_RESULTS.md")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=152)
    ap.add_argument(
        "--note", action="append", default=[], metavar="TEXT",
        help="free-text provenance note to print at the top of the report "
             "(use for things the CSVs cannot show, e.g. an aborted run)",
    )
    ap.add_argument(
        "--alias", action="append", default=[], metavar="OLD=NEW",
        help="merge an annotator name into another, e.g. --alias 'Annotator 1=A'. "
             "Applied before de-duplication.",
    )
    args = ap.parse_args(argv)

    alias: dict[str, str] = {}
    for spec in args.alias:
        if "=" not in spec:
            raise SystemExit(f"--alias expects OLD=NEW, got {spec!r}")
        old, new = spec.split("=", 1)
        alias[old.strip()] = new.strip()

    gold = {it.id: {"sense_l1": it.sense_l1, "explicit": it.explicit}
            for it in load_items(ROOT / args.gold)}

    if args.files:
        paths = [Path(p) for p in args.files]
    else:
        d = ROOT / args.submissions
        paths = sorted(p for p in d.glob("*.csv") if not p.name.endswith("_latest.csv")) if d.exists() else []
    if not paths:
        raise SystemExit(
            f"no submissions found under {args.submissions}; "
            "run the annotation sessions first (or pass --files)"
        )

    rows, notes = dedupe(load_submissions(paths, alias))
    notes = list(args.note) + notes
    repro, skip = [], False
    for a in sys.argv:
        if skip:
            skip = False
            continue
        if a in ("--note",):
            skip = True
            continue
        if a.startswith("--note="):
            continue
        repro.append(a)
    notes.append("复现命令：`python " + " ".join(repro) + "`")
    if alias:
        notes.append("annotator alias: "
                     + ", ".join(f"{k} -> {v}" for k, v in alias.items()))
    # A split-design annotator needs both sessions; say so instead of silently
    # reporting a half-empty table.
    sess_seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        sess_seen[r["annotator"]].add(r["session"])
    for ann, sess in sorted(sess_seen.items()):
        if len(sess) < 2:
            notes.append(f"{ann}: only session(s) {sorted(sess)} found — "
                         "a/b are not paired for this annotator")
    # Session spacing: the protocol asks for sessions to be separated so that
    # fatigue and learning do not ride on the session boundary.  Report the
    # actual gap rather than assuming it was followed.
    span: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        if r.get("answered_at"):
            span[(r["annotator"], r["session"])].append(r["answered_at"])
    for ann in sorted(sess_seen):
        ordered = sorted(s for s in sess_seen[ann])
        for prev, nxt in zip(ordered, ordered[1:]):
            a = span.get((ann, prev))
            b = span.get((ann, nxt))
            if not a or not b:
                continue
            try:
                import datetime as _dt
                t1 = _dt.datetime.fromisoformat(max(a).replace("Z", "+00:00"))
                t2 = _dt.datetime.fromisoformat(min(b).replace("Z", "+00:00"))
                gap = (t2 - t1).total_seconds() / 3600
            except ValueError:
                continue
            dur = (_dt.datetime.fromisoformat(max(a).replace("Z", "+00:00"))
                   - _dt.datetime.fromisoformat(min(a).replace("Z", "+00:00"))).total_seconds() / 60
            notes.append(
                f"{ann}: session {prev} lasted {dur:.0f} min, then session {nxt} "
                f"started {gap:.2f} h after it ended"
                + ("  ← less than the 2 h the protocol asks for" if gap < 2 else "")
            )
    by_ann: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ann[r["annotator"]].append(r)

    results = [analyse_one(ann, rs, gold, args.n_boot, args.seed + k)
               for k, (ann, rs) in enumerate(sorted(by_ann.items()))]
    cross = cross_annotator(results, args.n_boot, args.seed)

    out = {"n_files": len(paths), "gold_n": len(gold), "results": results,
           "cross": cross, "notes": notes, "n_boot": args.n_boot}
    p = ROOT / args.out_json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = render_md(results, notes, args.n_boot, cross)
    mp = ROOT / args.out_md
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(md, encoding="utf-8")

    print(f"wrote {args.out_json} and {args.out_md}")
    for r in results:
        s = r["savings"]
        got = {k: v["net_saving"] for k, v in s.items()}
        print(f"  {r['annotator']}: saving {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
