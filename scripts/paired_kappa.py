"""Paired document-clustered bootstrap for the difference in Cohen's kappa.

Two systems scored on the same items give two correlated kappas.  Comparing
their marginal intervals is not a test: overlapping CIs can coexist with a
reliable difference, and disjoint CIs are not required for one.  This resamples
documents (not items) with replacement, recomputes both kappas on each replicate
and reports the distribution of ``kappa_base - kappa_alt``.

Resampling documents rather than items keeps items from the same document
together, which is what the evaluation sets need: HIT-CDTB has ~50 documents
and items inside one document are not independent.

    python scripts/paired_kappa.py \
        --gold data/processed/l1/hit_balanced.jsonl \
        --base outputs/l1_hit_balanced_qwen3.5-plus_zs.jsonl \
        --alt outputs/l1_encoder_hit_oof_mac_cw_pc.jsonl \
        --stratum implicit --out outputs/l1_paired_kappa_qwen_encoder.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.metrics.agreement import cohen_kappa  # noqa: E402
from discourse_reliability.schema import load_items, load_predictions  # noqa: E402


def labels(path: Path, level: str) -> dict[str, str]:
    return {
        p.item_id: getattr(p, f"label_{level}") or ""
        for p in load_predictions(path)
        if p.error is None
    }


def kappa_on(idx: list[int], gold: list[str], a: list[str], b: list[str]):
    g = [gold[i] for i in idx]
    return cohen_kappa(g, [a[i] for i in idx]), cohen_kappa(g, [b[i] for i in idx])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--base", required=True, help="system A (the one whose lead is claimed)")
    ap.add_argument("--alt", required=True, help="system B")
    ap.add_argument("--level", default="l1", choices=["l1", "l2", "l3"])
    ap.add_argument("--stratum", default="all", choices=["all", "explicit", "implicit"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    base = labels(ROOT / args.base, args.level)
    alt = labels(ROOT / args.alt, args.level)

    items = []
    for it in load_items(ROOT / args.gold):
        if it.id not in base or it.id not in alt:
            continue
        if args.stratum != "all":
            want_explicit = args.stratum == "explicit"
            if it.explicit is not want_explicit:
                continue
        items.append(it)
    if not items:
        raise SystemExit("no items shared by both prediction files")

    gold = [getattr(it, f"sense_{args.level}") or it.gold for it in items]
    a = [base[it.id] for it in items]
    b = [alt[it.id] for it in items]
    docs = [it.doc_id or it.id for it in items]

    by_doc: dict[str, list[int]] = {}
    for i, d in enumerate(docs):
        by_doc.setdefault(d, []).append(i)
    doc_ids = sorted(by_doc)

    ka, kb = kappa_on(list(range(len(items))), gold, a, b)
    rng = random.Random(args.seed)
    deltas = []
    for _ in range(args.n_boot):
        picked: list[int] = []
        for _ in range(len(doc_ids)):
            picked.extend(by_doc[rng.choice(doc_ids)])
        va, vb = kappa_on(picked, gold, a, b)
        if va is None or vb is None:
            continue
        deltas.append(va - vb)

    deltas.sort()
    n = len(deltas)
    report = {
        "n_items": len(items),
        "n_docs": len(doc_ids),
        "stratum": args.stratum,
        "base": {"file": args.base, "kappa": None if ka is None else round(ka, 4)},
        "alt": {"file": args.alt, "kappa": None if kb is None else round(kb, 4)},
        "delta": None if ka is None or kb is None else round(ka - kb, 4),
        "delta_ci95": [round(deltas[int(0.025 * n)], 4), round(deltas[int(0.975 * n) - 1], 4)]
        if n else None,
        "p_delta_le_0": round(sum(d <= 0 for d in deltas) / n, 4) if n else None,
        "n_boot_kept": n,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
