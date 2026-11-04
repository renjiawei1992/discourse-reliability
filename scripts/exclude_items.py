"""Remove specific items from every derived artefact, with an audit trail.

Used when an item has to come out of the dataset for editorial reasons (rather
than because it is malformed).  The rule is that the removal is *recorded*, not
silent: every purge appends to ``exclusions.jsonl`` so the denominator change
(N=360 -> N=359) can be explained in the paper, and so the same id can be
filtered out of files regenerated later.

``--ids`` takes ``item_id:reason`` pairs.  JSONL files are filtered on ``id``,
CSV files on ``item_id``; both are rewritten in place after a ``.bak`` copy.

    python scripts/exclude_items.py \
        --ids 'hitcdtb:bn/cbs_0013#:p3:6:政治敏感内容' \
        --manifest data/processed/l1/exclusions.jsonl \
        --files data/processed/hitcdtb.jsonl data/processed/l1/b1_packet.jsonl ...
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def purge_jsonl(path: Path, drop: set[str]) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    keep, removed = [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            keep.append(line)
            continue
        if obj.get("id") in drop:
            removed += 1
        else:
            keep.append(line)
    if removed:
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return len(keep), removed


def purge_csv(path: Path, drop: set[str]) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0].keys()) if rows else []
    keep = [r for r in rows if r.get("item_id") not in drop]
    removed = len(rows) - len(keep)
    if removed:
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(keep)
    return len(keep), removed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True, help="item_id:reason")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    reasons = {}
    for spec in args.ids:
        # Item ids contain colons ("hitcdtb:bn/cbs_0013#:p3:6"), so the reason is
        # whatever follows the *last* one.  Reasons must therefore not contain
        # a colon themselves.
        iid, _, why = spec.rpartition(":")
        if not iid:
            raise SystemExit(f"--ids entry {spec!r} has no item_id:reason split")
        reasons[iid] = why or "unspecified"
    drop = set(reasons)

    manifest = ROOT / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    report = []
    for spec in args.files:
        p = ROOT / spec
        if not p.exists():
            report.append({"file": spec, "status": "missing"})
            continue
        if p.suffix == ".csv":
            n, removed = (0, 0) if args.dry_run else purge_csv(p, drop)
        else:
            n, removed = (0, 0) if args.dry_run else purge_jsonl(p, drop)
        report.append({"file": spec, "kept": n, "removed": removed})

    if not args.dry_run:
        with manifest.open("a", encoding="utf-8") as f:
            for iid, why in reasons.items():
                f.write(json.dumps({"item_id": iid, "reason": why,
                                    "date": date.today().isoformat()},
                                   ensure_ascii=False) + "\n")
    print(json.dumps({"dry_run": args.dry_run, "excluded": reasons, "files": report},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
