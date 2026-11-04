"""R160: are the model's errors the same errors humans make?

Two very different pictures fit the same accuracy number:

* the model fails where the task is genuinely hard -- humans disagree on those
  same items, so the model's error is *task difficulty*;
* the model fails where humans agree -- the label is clear to people and the
  model is simply wrong, which is a model problem.

This compares model error against human disagreement item by item:

    P(model wrong | humans disagree)  vs  P(model wrong | humans agree)
    AUC of model error predicting human disagreement
    Jaccard overlap of the two error sets
    JS divergence between the confusion patterns
      (which class pairs humans argue about vs which the model mixes up)

It also reports the strictest case: items where both humans agree and the model
still differs.

    python scripts/validity_analysis.py \
        --items data/processed/l1/b1_packet.jsonl \
        --a data/interim/b1_round1/submissions/A_latest.csv \
        --b data/interim/b1_round1/submissions/B_latest.csv \
        --preds outputs/l1_b1packet_qwen_zs.jsonl outputs/l1_b1packet_reasoner_zs.jsonl \
        --level l1 --out outputs/l1_validity.json

With ``--simulate`` it generates two synthetic annotators so the pipeline can be
exercised before the human data exists.  Nothing simulated may be reported as a
result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items, load_predictions  # noqa: E402

CLASSES = ("TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION")


def read_sheet(path: Path, column: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            val = (row.get(column) or "").strip().upper()
            if val:
                out[row["item_id"]] = val
    return out


def js_divergence(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    if not sp or not sq:
        return float("nan")
    total = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        m = (a + b) / 2
        if a:
            total += 0.5 * a * math.log2(a / m)
        if b:
            total += 0.5 * b * math.log2(b / m)
    return total


def auc_binary(score: list[int], target: list[int]) -> float:
    """AUC of a 0/1 score predicting a 0/1 target (ties count half)."""
    pos = [s for s, t in zip(score, target) if t == 1]
    neg = [s for s, t in zip(score, target) if t == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def analyse(items, a: dict, b: dict, preds_by_model: dict[str, dict], level: str) -> dict:
    report: dict = {"models": {}, "n_human_overlap": {}}
    for model, pred in preds_by_model.items():
        ids = [i for i in items if i in a and i in b and i in pred]
        if len(ids) < 20:
            report["models"][model] = {"error": f"only {len(ids)} overlapping items"}
            report["n_human_overlap"][model] = len(ids)
            continue

        human_dis = [1 if a[i] != b[i] else 0 for i in ids]
        model_wrong = [1 if pred[i] != getattr(items[i], f"sense_{level}") else 0 for i in ids]
        gold = {i: getattr(items[i], f"sense_{level}") for i in ids}
        both_agree_human = [i for i in ids if a[i] == b[i]]

        # Three different quantities were previously conflated under one key.
        # ``pred != a`` on agreed items is not "the model is at fault": when the
        # annotators agree with each other but disagree with the corpus, the
        # model may be the one that is right.  Keep them separate.
        err_on_agree = [i for i in both_agree_human if pred[i] != gold[i]]
        disagrees_agree = [i for i in both_agree_human if pred[i] != a[i]]
        # The strict case: the annotators agree with each other, they agree with
        # the corpus, and the model is still wrong.
        strict_wrong = [i for i in both_agree_human
                        if a[i] == gold[i] and pred[i] != gold[i]]
        # Where the annotators agree with each other but not with the corpus.
        humans_off_gold = [i for i in both_agree_human if a[i] != gold[i]]

        def rate(flag_items):
            if not flag_items:
                return None
            return round(sum(1 for i in flag_items if pred[i] != getattr(items[i], f"sense_{level}"))
                         / len(flag_items), 4)

        dis_items = [i for i in ids if a[i] != b[i]]
        dis_rate = rate(dis_items)
        agr_rate = rate(both_agree_human)
        table = {
            "human_disagree": {"n": len(dis_items), "model_error_rate": dis_rate},
            "human_agree": {"n": len(both_agree_human), "model_error_rate": agr_rate},
        }
        rr = (round(dis_rate / agr_rate, 3)
              if dis_rate is not None and agr_rate not in (None, 0.0) else None)

        # Where each side fails: which label pairs they confuse.
        pairs_h = Counter((a[i], b[i]) for i in dis_items)
        pairs_m = Counter((getattr(items[i], f"sense_{level}"), pred[i])
                          for i in ids if pred[i] != getattr(items[i], f"sense_{level}"))
        human_err_set = set(dis_items)
        model_err_set = {i for i in ids if pred[i] != getattr(items[i], f"sense_{level}")}
        union = human_err_set | model_err_set

        by_stratum = {}
        for name, keep in (("explicit", True), ("implicit", False)):
            sub = [i for i in ids if items[i].explicit is keep]
            if not sub:
                continue
            sub_dis = [i for i in sub if a[i] != b[i]]
            sub_agr = [i for i in sub if a[i] == b[i]]
            by_stratum[name] = {
                "n": len(sub),
                "n_human_disagree": len(sub_dis),
                "model_error_rate_on_disagree": rate(sub_dis),
                "model_error_rate_on_agree": rate(sub_agr),
            }

        report["n_human_overlap"][model] = len(ids)
        report["models"][model] = {
            "n": len(ids),
            "human_disagreement_rate": round(sum(human_dis) / len(ids), 4),
            "model_error_rate": round(sum(model_wrong) / len(ids), 4),
            "contingency": table,
            "risk_ratio": rr,
            "auc_model_error_vs_human_disagreement": round(
                auc_binary(model_wrong, human_dis), 4),
            "jaccard_error_sets": round(len(human_err_set & model_err_set) / len(union), 4)
            if union else None,
            "js_divergence_confusions": round(js_divergence(pairs_h, pairs_m), 4),
            "human_confusion_top": {f"{x}->{y}": n for (x, y), n in pairs_h.most_common(5)},
            "model_confusion_top": {f"{x}->{y}": n for (x, y), n in pairs_m.most_common(5)},
            "model_error_on_agreed": {
                "n": len(err_on_agree),
                "share_of_model_errors": round(len(err_on_agree) / max(1, sum(model_wrong)), 4),
                "n_agreed": len(both_agree_human),
            },
            "disagrees_with_agreed_humans": {
                "n": len(disagrees_agree),
                "share_of_model_errors": round(len(disagrees_agree) / max(1, sum(model_wrong)), 4),
            },
            "strictly_model_fault": {
                "n": len(strict_wrong),
                "share_of_model_errors": round(len(strict_wrong) / max(1, sum(model_wrong)), 4),
                "items": strict_wrong[:20],
            },
            "humans_agree_off_gold": {
                "n": len(humans_off_gold),
                "model_sides_with_gold": sum(1 for i in humans_off_gold if pred[i] == gold[i]),
            },
            "by_stratum": by_stratum,
        }
    return report


def simulate(items, seed: int = 7, noise: float = 0.2):
    """Two synthetic annotators for pipeline testing only."""
    rng = np.random.default_rng(seed)
    a, b = {}, {}
    for it in items:
        gold = getattr(it, "sense_l1") or ""
        if not gold:
            continue
        a[it.id] = gold
        b[it.id] = gold if rng.random() > noise else str(rng.choice(CLASSES))
    return a, b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="gold items, e.g. the annotation packet")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--column", default="annotator_sense_l1")
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--level", default="l1", choices=["l1", "l2", "l3"])
    ap.add_argument("--simulate", action="store_true", help="use synthetic annotators (test only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    items = load_items(args.items)
    by_id = {it.id: it for it in items}
    if args.simulate:
        a, b = simulate(items)
        provenance = "SYNTHETIC -- pipeline test only, never report these numbers"
    else:
        if not (args.a and args.b):
            raise SystemExit("need --a and --b, or --simulate")
        a = read_sheet(Path(args.a), args.column)
        b = read_sheet(Path(args.b), args.column)
        provenance = "human annotators"

    preds_by_model: dict[str, dict[str, str]] = {}
    for spec in args.preds:
        for path in sorted(ROOT.glob(spec)):
            for p in load_predictions(path):
                if p.error is None:
                    preds_by_model.setdefault(p.model, {})[p.item_id] = getattr(
                        p, f"label_{args.level}") or ""

    report = {
        "provenance": provenance,
        "items_file": args.items,
        "n_items": len(items),
        # Restrict to items still in the packet: a submission can carry ids that
        # were later excluded from the corpus, and counting them here would
        # report more annotated items than the packet contains.
        "n_annotated": len(set(a) & set(b) & set(by_id)),
        "annotator_agreement": round(
            sum(1 for i in set(a) & set(b) & set(by_id) if a[i] == b[i])
            / max(1, len(set(a) & set(b) & set(by_id))), 4),
        **analyse(by_id, a, b, preds_by_model, args.level),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "provenance": provenance,
        "n_annotated": report["n_annotated"],
        "models": {m: {k: v.get(k) for k in ("n", "auc_model_error_vs_human_disagreement",
                                             "risk_ratio", "js_divergence_confusions")}
                   for m, v in report["models"].items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
