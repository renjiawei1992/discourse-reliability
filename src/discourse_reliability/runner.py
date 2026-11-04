"""Run a model over a set of items."""

from __future__ import annotations

import random
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Sequence

from .labels import sense_levels
from .models.base import Adapter
from .prompts import PROMPT_VERSION, build_messages, parse_sense, sense_inventory
from .schema import Item, Prediction


def few_shot_pool(
    items: Sequence[Item], k: int, seed: int = 11, level: str = "l3"
) -> list[Item]:
    """Pick k exemplars, one per distinct sense where possible.

    ``level="l1"`` draws one exemplar per top-level class, which is the only
    useful coverage for a 4-way task (L3 would spend all 8 shots on EXPANSION).
    """
    rng = random.Random(seed)
    by_sense: dict[str, list[Item]] = {}
    for it in items:
        if it.gold:
            key = it.sense_l1 if level == "l1" else (it.sense_l2 or it.sense_l1)
            by_sense.setdefault(key, []).append(it)
    senses = sorted(by_sense)
    rng.shuffle(senses)
    shots: list[Item] = []
    for s in senses:
        pool = sorted(by_sense[s], key=lambda x: x.id)
        rng.shuffle(pool)
        shots.append(pool[0])
        if len(shots) >= k:
            break
    return shots


def predict_one(
    item: Item,
    adapter: Adapter,
    regime: str,
    shots: Sequence[Item] = (),
    allowed: Sequence[str] | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    level: str = "l3",
    max_tokens: int = 256,
    variant: str = "v0",
) -> Prediction:
    messages = build_messages(
        item, regime=regime, shots=shots, extra_senses=allowed or (), level=level,
        variant=variant,
    )
    gens = adapter.generate(
        messages, n=n_samples, temperature=temperature, max_tokens=max_tokens
    )
    texts = [g.text for g in gens]
    errors = [g.error for g in gens if g.error]
    if not any(texts):
        return Prediction(item_id=item.id, model=adapter.name, regime=regime, error="; ".join(errors) or "empty response")

    senses = [parse_sense(t, allowed) for t in texts]
    senses = [s for s in senses if s]
    if not senses:
        truncated = any((g.error or "").startswith("truncated") for g in gens)
        return Prediction(
            item_id=item.id, model=adapter.name, regime=regime,
            error="truncated" if truncated else "unparseable",
            raw_output=texts[0],
        )

    counts = Counter(senses)
    top, top_n = counts.most_common(1)[0]
    l1, l2, l3 = sense_levels(top)
    lat = [g.latency_ms for g in gens if g.latency_ms]
    served = next((g.usage.get("served_model") for g in gens if g.usage.get("served_model")), None)
    return Prediction(
        item_id=item.id,
        model=adapter.name,
        regime=regime,
        label_l1=l1,
        label_l2=l2,
        label_l3=l3,
        raw_output=texts[0],
        samples=senses,
        confidence=top_n / len(senses),
        latency_ms=statistics.mean(lat) if lat else None,
        meta={"n_samples": len(senses), "temperature": temperature,
              "max_tokens": max_tokens,
              "prompt_version": PROMPT_VERSION if variant == "v0" else f"{PROMPT_VERSION}-{variant}",
              "prompt_variant": variant,
              **({"served_model": served} if served else {})},
    )


def run(
    items: Iterable[Item],
    adapter: Adapter,
    regime: str = "zero_shot",
    shots: Sequence[Item] = (),
    allowed: Sequence[str] | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    workers: int = 8,
    level: str = "l3",
    max_tokens: int = 256,
    variant: str = "v0",
) -> list[Prediction]:
    items = list(items)
    if allowed is None:
        allowed = sense_inventory(items, level=level)
    out: list[Prediction] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                predict_one,
                it,
                adapter,
                regime,
                shots,
                allowed,
                n_samples,
                temperature,
                level,
                max_tokens,
                variant,
            ): it.id
            for it in items
        }
        for fut in as_completed(futures):
            out.append(fut.result())
    out.sort(key=lambda p: p.item_id)
    return out
