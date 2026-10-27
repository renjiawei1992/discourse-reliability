"""Re-run only the items a model failed to answer, until none are left.

Why this exists: on a gateway, a failure is usually not the model's answer.  A
ReadTimeout, a 5xx, or the gateway's own canned fallback text says something
about the middleman, not about whether the model can classify the item.  Folding
those into accuracy would report the gateway's reliability as the model's error
rate.  So the main table scores answers that were actually obtained, and the
failure rate is reported separately.

Retrying is still a harness decision and must leave a trace: every recovered
prediction is marked ``meta["retried"] = True`` with the round it took, and the
original file is never modified.

    python scripts/retry_unparseable.py \
        --preds outputs/l1_hit_balanced_qwen3.5-plus_zs.jsonl \
        --items data/processed/l1/hit_balanced.jsonl \
        --out outputs/l1_hit_balanced_qwen3.5-plus_zs_filled.jsonl \
        --rounds 5 --workers 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.models import get_adapter  # noqa: E402
from discourse_reliability.runner import run  # noqa: E402
from discourse_reliability.schema import (  # noqa: E402
    load_items,
    load_predictions,
    save_predictions,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="l1", choices=["l1", "l2", "l3"])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--max-retries", type=int, default=1,
                    help="in-attempt retries; the --rounds loop is the outer retry")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between rounds")
    args = ap.parse_args(argv)

    preds = load_predictions(args.preds)
    pending = {p.item_id: p for p in preds if p.error}
    if not pending:
        print(json.dumps({"failed": 0, "note": "nothing to retry"}))
        return 0

    model = preds[0].model
    regime = preds[0].regime
    by_id = {it.id: it for it in load_items(args.items)}
    adapter = get_adapter(model, timeout=args.timeout, max_retries=args.max_retries)

    first_errors = Counter((p.error or "")[:40] for p in pending.values())
    log = []
    for round_no in range(1, args.rounds + 1):
        items = [by_id[i] for i in sorted(pending) if i in by_id]
        if not items:
            break
        fresh = run(
            items, adapter, regime=regime, level=args.level,
            workers=args.workers, max_tokens=args.max_tokens,
        )
        recovered = 0
        for p in fresh:
            if p.error:
                pending[p.item_id] = p
                continue
            old = pending.pop(p.item_id)
            p.meta = {**p.meta, "retried": True, "retry_round": round_no,
                      "first_error": (old.error or "")[:80]}
            preds = [x for x in preds if x.item_id != p.item_id] + [p]
            recovered += 1
        log.append({"round": round_no, "attempted": len(items),
                    "recovered": recovered, "remaining": len(pending)})
        print(json.dumps(log[-1], ensure_ascii=False))
        if not pending:
            break
        if recovered == 0 and round_no >= 2:
            # Two dry rounds in a row: hammering the gateway further will not
            # change the answer, only the bill.
            break
        time.sleep(args.sleep)

    preds.sort(key=lambda p: p.item_id)
    save_predictions(args.out, preds)
    print(json.dumps({
        "failed_first_pass": sum(first_errors.values()),
        "first_pass_errors": dict(first_errors),
        "still_failing": len(pending),
        "still_failing_errors": dict(Counter((p.error or "")[:40] for p in pending.values())),
        "out": args.out,
        "model": model,
        "rounds": log,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
