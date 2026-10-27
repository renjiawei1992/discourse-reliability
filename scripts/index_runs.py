"""Index every prediction file under outputs/ into one table.

File names accumulated ad hoc across sessions (``l1_ds_zeroshot.jsonl`` next to
``l1_hit_balanced_gpt-6-luna_zs.jsonl``), so "which file is the result" stopped
being answerable from the directory listing alone.  This walks the files, reads
the metadata the runner already wrote, infers the evaluation set by item-id
overlap, and emits both a markdown table and a JSON manifest.

    python scripts/index_runs.py --out-md outputs/RUNS_INDEX.md \
        --out-json outputs/RUNS_INDEX.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items, load_predictions  # noqa: E402

GOLD = {
    "hit_balanced": "data/processed/l1/hit_balanced.jsonl",
    "hit_distributed": "data/processed/l1/hit_distributed.jsonl",
    "ted_balanced": "data/processed/l1/ted_balanced.jsonl",
    "hit_probe_explicit": "data/processed/l1/probe_hit_balanced_explicit.jsonl",
    "ted_probe_explicit": "data/processed/l1/probe_ted_balanced_explicit.jsonl",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def infer_set(ids: set[str], gold: dict[str, set[str]]) -> str:
    best, score = "?", 0.0
    for name, gids in gold.items():
        if not ids:
            continue
        overlap = len(ids & gids) / len(ids)
        if overlap > score:
            best, score = name, overlap
    return best if score >= 0.9 else "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default="outputs/RUNS_INDEX.md")
    ap.add_argument("--out-json", default="outputs/RUNS_INDEX.json")
    args = ap.parse_args(argv)

    gold = {k: {it.id for it in load_items(v)} for k, v in GOLD.items()}
    rows = []
    for path in sorted((ROOT / "outputs").glob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        try:
            preds = load_predictions(path)
        except Exception:
            continue
        if not preds or not hasattr(preds[0], "label_l1"):
            continue
        first = preds[0]
        ids = {p.item_id for p in preds}
        # Filled files repeat the same run with failures repaired; both are kept
        # so the first-pass failure rate can always be recomputed.
        filled = path.stem.endswith("_filled")
        rows.append({
            "file": path.relative_to(ROOT).as_posix(),
            "sha256_16": sha256(path),
            "model": first.model,
            "regime": first.regime,
            "served_model": first.meta.get("served_model"),
            "prompt_version": first.meta.get("prompt_version"),
            "max_tokens": first.meta.get("max_tokens"),
            "evaluation_set": infer_set(ids, gold),
            "items": len(preds),
            "errors": sum(1 for p in preds if p.error),
            "is_filled": filled,
            "base_run": path.stem[:-7] if filled else path.stem,
            "bytes": path.stat().st_size,
        })

    canonical = {}
    for r in rows:
        if r["evaluation_set"] == "?":
            continue
        key = (r["evaluation_set"], r["model"], r["regime"], r["base_run"])
        current = canonical.get(key)
        if current is None or (r["is_filled"] and not current["is_filled"]):
            canonical[key] = r
    for r in rows:
        r["canonical"] = any(
            r is c for c in canonical.values()
        )

    md = ["# 预测文件索引", "",
          "由 `scripts/index_runs.py` 生成。`canonical=true` 的是该 run 应当引用的文件"
          "（有补跑版本时用补跑版）；其余是原始快照或探针，保留以便复算第一遍失败率。", "",
          "| 集合 | 模型 | 文件 | 条目 | 错误 | 补跑 | canonical |",
          "|---|---|---|---|---|---|---|"]
    order = {"hit_balanced": 0, "hit_distributed": 1, "ted_balanced": 2}
    for r in sorted(rows, key=lambda x: (order.get(x["evaluation_set"], 9),
                                         x["evaluation_set"], x["model"], x["file"])):
        md.append(
            f"| {r['evaluation_set']} | {r['model']} | `{r['file']}` | {r['items']} | "
            f"{r['errors']} | {'是' if r['is_filled'] else ''} | "
            f"{'**是**' if r['canonical'] else ''} |"
        )
    (ROOT / args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / args.out_json).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "files": len(rows),
        "canonical": sum(1 for r in rows if r["canonical"]),
        "out_md": args.out_md, "out_json": args.out_json,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
