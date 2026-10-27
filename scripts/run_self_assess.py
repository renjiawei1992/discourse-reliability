"""Run the L1 task while asking the model to rate its own certainty (R151).

The label comes from the same definitions and the same item as the plain run;
only the output contract adds ``confidence``.  That makes the accuracy directly
comparable, and gives the triage experiment a "does the model know when it is
wrong" signal to test against consensus entropy.

    python scripts/run_self_assess.py --input data/processed/l1/hit_balanced.jsonl \
        --out outputs/l1_selfassess_chat.jsonl --model deepseek:deepseek-chat \
        --workers 6 --max-tokens 256
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.labels import sense_levels  # noqa: E402
from discourse_reliability.models import get_adapter  # noqa: E402
from discourse_reliability.prompts import (  # noqa: E402
    PROMPT_VERSION,
    build_self_assess_messages,
    parse_self_assess,
)
from discourse_reliability.schema import (  # noqa: E402
    Item,
    Prediction,
    load_items,
    save_predictions,
)

ALLOWED = ("TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION")


def one(item: Item, adapter, max_tokens: int) -> Prediction:
    msgs = build_self_assess_messages(item)
    gens = adapter.generate(msgs, n=1, temperature=0.0, max_tokens=max_tokens)
    g = gens[0]
    if not g.text:
        return Prediction(item_id=item.id, model=adapter.name, regime="self_assess",
                          error=g.error or "empty response")
    sense, conf = parse_self_assess(g.text, ALLOWED)
    if not sense:
        return Prediction(item_id=item.id, model=adapter.name, regime="self_assess",
                          error="unparseable", raw_output=g.text)
    l1, l2, l3 = sense_levels(sense)
    served = g.usage.get("served_model")
    return Prediction(
        item_id=item.id,
        model=adapter.name,
        regime="self_assess",
        label_l1=l1,
        label_l2=l2,
        label_l3=l3,
        raw_output=g.text,
        confidence=conf,
        latency_ms=g.latency_ms,
        meta={"prompt_version": PROMPT_VERSION + "+selfassess", "max_tokens": max_tokens,
              "self_confidence": conf, "temperature": 0.0,
              **({"served_model": served} if served else {})},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek:deepseek-chat")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    items = load_items(args.input)
    if args.limit:
        items = items[: args.limit]
    adapter = get_adapter(args.model, timeout=args.timeout, max_retries=args.max_retries)

    out: list[Prediction] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(one, it, adapter, args.max_tokens) for it in items]
        for fut in as_completed(futures):
            out.append(fut.result())
    out.sort(key=lambda p: p.item_id)
    n = save_predictions(args.out, out)
    print(json.dumps({
        "written": n,
        "errors": sum(1 for p in out if p.error),
        "missing_confidence": sum(1 for p in out if not p.error and p.confidence is None),
        "model": adapter.name,
        "out": args.out,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
