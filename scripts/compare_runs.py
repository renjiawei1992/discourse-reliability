"""Paired comparison of two prediction files on the same gold set.

Eyeballing two accuracies that differ by 1--2 points is not evidence: the
evaluation sets are 400 items, so a 2.5-point gap is ten items and well inside
sampling noise.  This script reports the paired difference with a bootstrap CI
and a McNemar test over the items where the two systems disagree.

    python scripts/compare_runs.py --gold data/processed/l1/hit_balanced.jsonl \
        --base outputs/l1_ds_zeroshot.jsonl \
        --alt outputs/l1_ds_fewshot.jsonl outputs/l1_transfer_ted2hit_fs4.jsonl \
        --out outputs/l1_compare.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.metrics.agreement import accuracy, cohen_kappa  # noqa: E402
from discourse_reliability.schema import load_items, load_predictions  # noqa: E402


def labels(path: str, level: str) -> dict[str, str]:
    return {
        p.item_id: getattr(p, f"label_{level}") or ""
        for p in load_predictions(path)
        if p.error is None
    }


def macro_f1(gold: list[str], pred: list[str]) -> float:
    classes = sorted(set(gold) | set(pred))
    scores = []
    for c in classes:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores) if scores else float("nan")


def mcnemar(a_right: list[bool], b_right: list[bool]) -> dict:
    """Exact-ish binomial McNemar on the discordant pairs."""
    b = sum(1 for x, y in zip(a_right, b_right) if x and not y)
    c = sum(1 for x, y in zip(a_right, b_right) if not x and y)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    # two-sided exact binomial at p=0.5, computed in log space so n stays stable
    log_half = n * math.log(0.5)
    p = 0.0
    for k in range(0, min(b, c) + 1):
        p += math.exp(log_half + math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    return {"b": b, "c": c, "p": min(1.0, 2 * p)}


def compare(
    ids: list[str],
    gold: list[str],
    base: list[str],
    alt: list[str],
    n_boot: int = 2000,
    seed: int = 13,
) -> dict:
    rng = random.Random(seed)
    n = len(ids)
    base_right = [g == p for g, p in zip(gold, base)]
    alt_right = [g == p for g, p in zip(gold, alt)]
    diffs = [int(a) - int(b) for a, b in zip(alt_right, base_right)]
    point = sum(diffs) / n
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    return {
        "n": n,
        "base": {
            "accuracy": accuracy(gold, base),
            "kappa": cohen_kappa(gold, base),
            "macro_f1": macro_f1(gold, base),
        },
        "alt": {
            "accuracy": accuracy(gold, alt),
            "kappa": cohen_kappa(gold, alt),
            "macro_f1": macro_f1(gold, alt),
        },
        "delta_accuracy": point,
        "delta_ci95": [lo, hi],
        "mcnemar": mcnemar(base_right, alt_right),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", nargs="+", required=True)
    ap.add_argument("--level", default="l1", choices=["l1", "l2", "l3"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    items = {it.id: it for it in load_items(args.gold)}
    base_all = labels(args.base, args.level)
    report: dict[str, dict] = {}
    for alt_path in args.alt:
        alt_all = labels(alt_path, args.level)
        ids = [i for i in items if base_all.get(i) and alt_all.get(i)]
        gold = [getattr(items[i], f"sense_{args.level}") or items[i].gold for i in ids]
        rep = compare(
            ids,
            gold,
            [base_all[i] for i in ids],
            [alt_all[i] for i in ids],
            n_boot=args.n_boot,
        )
        report[alt_path] = rep
        d = rep["delta_accuracy"]
        lo, hi = rep["delta_ci95"]
        mc = rep["mcnemar"]
        print(
            f"{Path(alt_path).name}: acc {rep['base']['accuracy']:.4f} -> "
            f"{rep['alt']['accuracy']:.4f} (d={d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], "
            f"McNemar b={mc['b']} c={mc['c']} p={mc['p']:.4f})"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
