"""How much do the models overlap?

Two questions the per-model table cannot answer:

* **Redundancy.** If every model gets the same items right, a second model adds
  nothing to a human-review triage.  Pairwise kappa and the Jaccard overlap of
  error sets say how much independent signal there actually is.
* **Where they differ.** Explicit vs implicit accuracy per model shows whether
  the ranking is driven by connceptive reading or by real inference.

    python scripts/model_agreement.py --gold data/processed/l1/hit_balanced.jsonl \
        --preds "outputs/l1_hit_balanced_*_zs.jsonl" --level l1 \
        --out outputs/l1_model_agreement.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.metrics.agreement import cohen_kappa  # noqa: E402
from discourse_reliability.schema import load_items, load_predictions  # noqa: E402


def load_run(path: Path, level: str) -> tuple[str, dict[str, str]]:
    preds = load_predictions(path)
    name = preds[0].model if preds else path.stem
    return name, {
        p.item_id: getattr(p, f"label_{level}") or ""
        for p in preds
        if p.error is None
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--level", default="l1", choices=["l1", "l2", "l3"])
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    items = {it.id: it for it in load_items(args.gold)}
    runs: dict[str, dict[str, str]] = {}
    for spec in args.preds:
        paths = sorted(ROOT.glob(spec)) if any(c in spec for c in "*?[") else [Path(spec)]
        for path in paths:
            name, labels = load_run(path, args.level)
            if name in runs:  # keep the first file per model, e.g. first pass
                continue
            runs[name] = labels

    # Pairwise kappa over items both models answered.
    pairwise = {}
    for a, b in combinations(sorted(runs), 2):
        ids = [i for i in items if runs[a].get(i) and runs[b].get(i)]
        if len(ids) < 10:
            continue
        pairwise[f"{a} | {b}"] = {
            "n": len(ids),
            "kappa": round(cohen_kappa([runs[a][i] for i in ids],
                                       [runs[b][i] for i in ids]), 4),
        }

    # Error-set overlap: how often do two models fail on the same items?
    errors = {
        name: {i for i in items if labels.get(i) and
               labels[i] != (getattr(items[i], f"sense_{args.level}") or items[i].gold)}
        for name, labels in runs.items()
    }
    overlap = {}
    for a, b in combinations(sorted(errors), 2):
        inter = errors[a] & errors[b]
        union = errors[a] | errors[b]
        if union:
            overlap[f"{a} | {b}"] = {
                "jaccard": round(len(inter) / len(union), 4),
                "both_wrong": len(inter),
            }

    # Accuracy split by explicitness.
    strata = {}
    for name, labels in runs.items():
        for label, keep in (("explicit", True), ("implicit", False)):
            ids = [i for i in items if items[i].explicit is keep and labels.get(i)]
            if not ids:
                continue
            gold = [getattr(items[i], f"sense_{args.level}") or items[i].gold for i in ids]
            strata.setdefault(name, {})[label] = round(
                sum(1 for i, g in zip(ids, gold) if labels[i] == g) / len(ids), 4
            )

    result = {
        "models": sorted(runs),
        "answered": {k: len(v) for k, v in runs.items()},
        "pairwise_kappa": dict(sorted(pairwise.items(), key=lambda kv: -kv[1]["kappa"])),
        "error_overlap": overlap,
        "accuracy_by_explicitness": strata,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
