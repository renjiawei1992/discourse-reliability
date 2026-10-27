"""Materialise the L1 (4-class) experiment blocks.

Reads the processed corpora and writes every file an L1 block consumes, plus a
manifest with the counts that decide whether a block is runnable at all.

    python scripts/build_l1_blocks.py \
        --hit data/processed/hitcdtb.jsonl \
        --ted data/processed/tedcdb.jsonl \
        --out-dir data/processed/l1

Design decisions baked in here (see refine-logs/L1_EXPERIMENT_PLAN.md):

* **Document-level folds.**  Relations from one document are never split across
  folds.  HIT annotates the same document at p2 and p3, so an item-level split
  would put near-duplicate text on both sides.
* **Balanced sets are evaluation-only.**  EXPANSION is 72% of HIT-CDTB; a plain
  random test set makes accuracy a measure of the majority class.  The headline
  per-class table uses a class-balanced set, and the corpus-distributed set is
  reported alongside it.
* **Cross-corpus targets are balanced too**, so the HIT->TED and TED->HIT
  numbers are comparable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.data.sampling import assign_folds, assign_splits  # noqa: E402
from discourse_reliability.schema import Item, load_items, save_items  # noqa: E402


def balanced(items: list[Item], per_class: int, seed: int = 13) -> list[Item]:
    """Up to ``per_class`` items per L1 class, deterministic."""
    rng = random.Random(seed)
    groups: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        if it.sense_l1:
            groups[it.sense_l1].append(it)
    out: list[Item] = []
    for label in sorted(groups):
        pool = sorted(groups[label], key=lambda x: x.id)
        rng.shuffle(pool)
        out.extend(pool[:per_class])
    out.sort(key=lambda x: x.id)
    return out


def human_sample(items: list[Item], n: int, seed: int = 31) -> list[Item]:
    """B1 annotation packet: class quotas first, then fill in corpus proportions.

    The human ceiling is only meaningful if every class has enough items to
    estimate a per-class kappa, but a fully balanced 600 would over-represent
    TEMPORAL (104 items in the whole sample) at the cost of the classes that
    dominate real annotation work.  Quota first, proportional fill second.
    """
    rng = random.Random(seed)
    groups: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        if it.sense_l1:
            groups[it.sense_l1].append(it)
    for pool in groups.values():
        rng.shuffle(pool)
    quota = max(1, n // (2 * max(1, len(groups))))  # e.g. 75 per class at n=600
    chosen: list[Item] = []
    for label in sorted(groups):
        chosen.extend(groups[label][:quota])
    picked = {it.id for it in chosen}
    rest = [it for it in items if it.id not in picked and it.sense_l1]
    rng.shuffle(rest)
    chosen.extend(rest[: max(0, n - len(chosen))])
    chosen.sort(key=lambda x: x.id)
    return chosen


def class_counts(items: list[Item]) -> dict[str, int]:
    return dict(Counter(it.sense_l1 for it in items if it.sense_l1).most_common())


def fold_table(items: list[Item], k: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in range(k):
        part = [it for it in items if it.meta.get("fold") == f]
        out[str(f)] = {
            "n": len(part),
            "docs": len({it.doc_id for it in part}),
            "classes": class_counts(part),
        }
    return out


def write(path: Path, items: list[Item]) -> int:
    n = save_items(path, items)
    print(f"  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {n}")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit", required=True, help="processed HIT-CDTB jsonl")
    ap.add_argument("--ted", default=None, help="processed TED-CDB jsonl (optional)")
    ap.add_argument("--out-dir", default="data/processed/l1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--per-class", type=int, default=100,
                    help="items per L1 class in each balanced evaluation set")
    ap.add_argument("--human-n", type=int, default=600)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "protocol": {
            "level": "l1",
            "folds": args.folds,
            "split_unit": "document",
            "balanced_per_class": args.per_class,
        },
        "corpora": {},
        "blocks": {},
    }

    hit = load_items(args.hit)
    assign_folds(hit, k=args.folds, seed=23, group_by="doc_id")
    assign_splits(hit, seed=7, group_by="doc_id")
    print(f"HIT-CDTB: {len(hit)} items, {len({it.doc_id for it in hit})} docs")

    write(out_dir / "hit_cv.jsonl", hit)
    for f in range(args.folds):
        test = [it for it in hit if it.meta.get("fold") == f]
        train = [it for it in hit if it.meta.get("fold") != f]
        write(out_dir / f"hit_fold{f}_test.jsonl", test)
        write(out_dir / f"hit_fold{f}_train.jsonl", train)

    write(out_dir / "hit_balanced.jsonl", balanced(hit, args.per_class))
    write(out_dir / "hit_distributed.jsonl", [it for it in hit if it.split == "test"])
    write(out_dir / "b1_human.jsonl", human_sample(hit, args.human_n))

    manifest["corpora"]["hit-cdtb"] = {
        "n": len(hit),
        "docs": len({it.doc_id for it in hit}),
        "classes": class_counts(hit),
        "split_sizes": dict(Counter(it.split for it in hit)),
        "folds": fold_table(hit, args.folds),
    }

    if args.ted:
        ted = load_items(args.ted)
        assign_folds(ted, k=args.folds, seed=23, group_by="doc_id")
        assign_splits(ted, seed=7, group_by="doc_id")
        print(f"TED-CDB: {len(ted)} items, {len({it.doc_id for it in ted})} docs")
        write(out_dir / "ted_cv.jsonl", ted)
        write(out_dir / "ted_balanced.jsonl", balanced(ted, args.per_class))
        write(out_dir / "ted_distributed.jsonl", [it for it in ted if it.split == "test"])
        for f in range(args.folds):
            write(out_dir / f"ted_fold{f}_test.jsonl",
                  [it for it in ted if it.meta.get("fold") == f])
            write(out_dir / f"ted_fold{f}_train.jsonl",
                  [it for it in ted if it.meta.get("fold") != f])
        manifest["corpora"]["ted-cdb"] = {
            "n": len(ted),
            "docs": len({it.doc_id for it in ted}),
            "classes": class_counts(ted),
            "split_sizes": dict(Counter(it.split for it in ted)),
            "folds": fold_table(ted, args.folds),
        }

    # Block-level pointers, so a run command never has to guess a filename.
    manifest["blocks"] = {
        "B1_human_ceiling": {"items": "b1_human.jsonl", "n": len(human_sample(hit, args.human_n))},
        "B2_main_matrix": {
            "eval_balanced": "hit_balanced.jsonl",
            "eval_distributed": "hit_distributed.jsonl",
            "folds": [f"hit_fold{f}_test.jsonl for f in range({args.folds})"],
            "shots_from": "hit_fold{k}_train.jsonl (k = the evaluated fold)",
            "primary_level": "l1",
        },
        "B3_cross_corpus": {
            "hit_to_ted": {"eval": "ted_balanced.jsonl", "shots": "hit_fold{k}_train.jsonl"},
            "ted_to_hit": {"eval": "hit_balanced.jsonl", "shots": "ted_fold{k}_train.jsonl"},
        },
        "B4_shortcuts": {"source": "hit_fold{k}_test.jsonl", "explicit_only": True},
        "B5_triage": {"source": "hit_fold{k}_test.jsonl", "level": "l1"},
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out_dir / 'manifest.json'}")
    for name, info in manifest["corpora"].items():
        print(f"{name}: {info['classes']}")
        for f, fold in info["folds"].items():
            print(f"  fold {f}: n={fold['n']:5d} docs={fold['docs']:3d} {fold['classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
