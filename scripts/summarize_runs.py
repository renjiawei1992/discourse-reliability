"""One table for many prediction files.

    python scripts/summarize_runs.py --gold data/processed/l1/hit_balanced.jsonl \
        --preds "outputs/l1_hit_balanced_*_zs.jsonl" --level l1 \
        --out outputs/l1_model_table.json

Prints a markdown table sorted by kappa and writes the same numbers as JSON, so
the paper table and the stored record cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.metrics.agreement import (  # noqa: E402
    accuracy,
    cohen_kappa,
    macro_f1,
)
from discourse_reliability.schema import load_items, load_predictions  # noqa: E402


def score(gold: dict, preds: list, level: str) -> dict:
    ok = [p for p in preds if p.error is None and p.item_id in gold]
    out = {
        "n": len(preds),
        "errors": sum(1 for p in preds if p.error),
        "scored": len(ok),
    }
    if not ok:
        return out
    g = [getattr(gold[p.item_id], f"sense_{level}") or gold[p.item_id].gold for p in ok]
    q = [getattr(p, f"label_{level}") or "" for p in ok]
    correct = sum(1 for a, b in zip(g, q) if a == b)
    out.update(
        accuracy=round(accuracy(g, q), 4),
        kappa=round(cohen_kappa(g, q) or 0.0, 4),
        macro_f1=round(macro_f1(g, q), 4),
        # An unparseable answer is a wrong answer in a real deployment.  Report
        # it separately from the parseable-only score, or a model that fails on
        # 13% of items looks better than one that answers everything.
        accuracy_all=round(correct / len(preds), 4),
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="paths or globs")
    ap.add_argument("--level", default="l1", choices=["l1", "l2", "l3"])
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    gold = {it.id: it for it in load_items(args.gold)}
    rows = []
    for spec in args.preds:
        paths = sorted(ROOT.glob(spec)) if any(c in spec for c in "*?[") else [Path(spec)]
        for path in paths:
            preds = load_predictions(path)
            model = preds[0].model if preds else "?"
            regime = preds[0].regime if preds else "?"
            meta = preds[0].meta if preds else {}
            rows.append({
                "file": str(path).replace("\\", "/"),
                "model": model,
                "served": meta.get("served_model"),
                **score(gold, preds, args.level),
            })

    rows.sort(key=lambda r: (-(r.get("kappa") or -1), r["file"]))
    header = ["model", "n", "errors", "acc", "kappa", "macro_f1", "acc_all"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        print("| " + " | ".join([
            r["model"],
            str(r["scored"]),
            str(r["errors"]),
            f"{r.get('accuracy', float('nan')):.4f}" if "accuracy" in r else "-",
            f"{r.get('kappa', float('nan')):.4f}" if "kappa" in r else "-",
            f"{r.get('macro_f1', float('nan')):.4f}" if "macro_f1" in r else "-",
            f"{r.get('accuracy_all', float('nan')):.4f}" if "accuracy_all" in r else "-",
        ]) + " |")

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
