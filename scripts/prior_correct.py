"""Post-hoc class-prior correction for encoder predictions.

The encoder is trained on a corpus that is ~72% EXPANSION, so its argmax is
pulled toward that class.  Inverse-frequency loss weights and balanced sampling
change what the model learns; this changes only the decision rule, using output
probabilities that were already saved -- ``log p(y|x) - tau * log pi_train(y)``,
renormalised.  ``tau=1`` is the exact correction (the resulting scores are the
class-balanced posterior up to a constant); ``tau=0`` reproduces the raw argmax.

The prior comes from the *training* fold of each item, never from the test fold,
so this introduces no leakage.  It is deliberately untuned: tuning ``tau`` on
the evaluation set would make the comparison meaningless.

    python scripts/prior_correct.py \
        --preds outputs/l1_encoder_hit_oof.jsonl \
        --train-template data/processed/l1/hit_fold{k}_train.jsonl \
        --tau 1.0 --tag chinese-roberta-wwm-ext-pc \
        --out outputs/l1_encoder_hit_oof_pc.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.labels import L1_CLASSES  # noqa: E402
from discourse_reliability.schema import (  # noqa: E402
    load_items,
    load_predictions,
    save_predictions,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--train-template", default="data/processed/l1/hit_fold{k}_train.jsonl")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-meta-key", default="probs",
                    help="where the class probabilities live in Prediction.meta")
    args = ap.parse_args(argv)

    preds = load_predictions(ROOT / args.preds)
    folds = sorted({p.meta.get("fold") for p in preds if p.meta.get("fold") is not None})
    if not folds:
        raise SystemExit("predictions carry no meta['fold']; cannot pick the right prior")

    priors: dict[int, dict[str, float]] = {}
    for fold in folds:
        items = load_items(ROOT / args.train_template.format(k=fold))
        counts = Counter(it.sense_l1 for it in items if it.sense_l1 in L1_CLASSES)
        total = sum(counts.values())
        if not total:
            raise SystemExit(f"fold {fold}: no labelled training items")
        priors[fold] = {c: counts.get(c, 0) / total for c in L1_CLASSES}
        print(f"fold {fold} prior " + " ".join(f"{c}={priors[fold][c]:.3f}" for c in L1_CLASSES))

    out = []
    changed = 0
    for p in preds:
        fold = p.meta.get("fold")
        probs = p.meta.get(args.from_meta_key)
        if probs is None or fold not in priors:
            out.append(p)
            continue
        scores = {}
        for c in L1_CLASSES:
            pc = max(float(probs.get(c, 0.0)), 1e-12)
            scores[c] = math.log(pc) - args.tau * math.log(priors[fold][c])
        top = max(scores.values())
        exps = {c: math.exp(v - top) for c, v in scores.items()}
        z = sum(exps.values())
        adj = {c: exps[c] / z for c in L1_CLASSES}
        label = max(adj, key=adj.get)
        if label != p.label_l1:
            changed += 1
        out.append(type(p)(
            item_id=p.item_id, model=f"encoder:{args.tag}", regime=p.regime,
            label_l1=label, label_l2=p.label_l2, label_l3=p.label_l3,
            raw_output=p.raw_output,
            confidence=round(adj[label], 4), samples=p.samples,
            latency_ms=p.latency_ms, error=p.error,
            meta={"fold": fold, "probs": {c: round(adj[c], 4) for c in L1_CLASSES},
                  "probs_raw": probs, "tau": args.tau},
        ))

    n = save_predictions(ROOT / args.out, out)
    print(f"tau={args.tau}: {changed}/{n} labels moved -> {args.out}")
    print(json.dumps({"n": n, "changed": changed, "delta_rate": round(changed / max(1, n), 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
