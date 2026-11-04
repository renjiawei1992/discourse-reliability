"""Run an L1 evaluation set fold by fold, with fold-correct few-shot exemplars.

The balanced evaluation set spans all cross-validation folds, so a single shot
pool would leak: some exemplars would come from the same document as the item
being scored.  This driver groups the evaluation items by ``meta["fold"]``,
draws exemplars only from ``<train-template with k = fold>`` and concatenates
the predictions.

    python scripts/run_l1_cv.py --eval data/processed/l1/hit_balanced.jsonl \
        --train-template data/processed/l1/hit_fold{k}_train.jsonl \
        --out outputs/l1_fs.jsonl --model deepseek:deepseek-chat \
        --regime few_shot --k-shots 4 --level l1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.models import get_adapter  # noqa: E402
from discourse_reliability.runner import few_shot_pool, run  # noqa: E402
from discourse_reliability.schema import load_items, save_predictions  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--train-template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--regime", default="few_shot", choices=["zero_shot", "few_shot", "cot"])
    ap.add_argument("--k-shots", type=int, default=4)
    ap.add_argument("--level", default="l1")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-variant", default="v0", choices=["v0", "v1", "v2"],
                    help="R171 wording variants; v0 is the template used everywhere else")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    items = load_items(args.eval)
    if args.limit:
        items = items[: args.limit]
    if not all("fold" in it.meta for it in items):
        raise SystemExit(
            "evaluation items lack meta['fold']; run scripts/build_l1_blocks.py first"
        )

    adapter = get_adapter(args.model)
    preds = []
    folds = sorted({it.meta["fold"] for it in items})
    for fold in folds:
        part = [it for it in items if it.meta["fold"] == fold]
        shots = []
        if args.regime in {"few_shot", "cot"}:
            train_path = Path(args.train_template.format(k=fold))
            if not train_path.exists():
                raise SystemExit(f"missing shots file: {train_path}")
            shots = few_shot_pool(load_items(train_path), args.k_shots, level=args.level)
            if len(shots) < 4:
                print(f"WARNING: fold {fold} only yielded {len(shots)} exemplars")
        out = run(
            part,
            adapter,
            regime=args.regime,
            shots=shots,
            n_samples=args.samples,
            temperature=args.temperature,
            workers=args.workers,
            level=args.level,
            max_tokens=args.max_tokens,
            variant=args.prompt_variant,
        )
        errs = sum(1 for p in out if p.error)
        print(f"fold {fold}: n={len(part)} errors={errs} shots={[s.sense_l1 for s in shots]}")
        preds.extend(out)

    n = save_predictions(args.out, preds)
    print(json.dumps({"written": n, "out": args.out, "model": adapter.name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
