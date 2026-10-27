"""Freeze the v0 baseline: hashes of every input and every prediction file.

The point is attribution.  Without a frozen snapshot, a later "improvement"
cannot be separated from a silent upstream change -- a gateway swapping the
model behind an alias, a prompt edit, or a regenerated data block.

    python scripts/freeze_baseline.py --out outputs/v0/manifest.json

Re-running later and diffing the manifest tells you what actually moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.prompts import PROMPT_VERSION  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(patterns: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                rel = path.relative_to(ROOT).as_posix()
                out[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return out


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/v0/manifest.json")
    ap.add_argument("--predictions", nargs="*", default=["outputs/l1_*.jsonl"])
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    manifest = {
        "frozen_on": date.today().isoformat(),
        "prompt_version": PROMPT_VERSION,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "note": args.note,
        "data": collect([
            "data/processed/hitcdtb.jsonl",
            "data/processed/tedcdb.jsonl",
            "data/processed/l1/*.jsonl",
            "data/processed/l1/manifest.json",
        ]),
        "predictions": collect(args.predictions),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "out": args.out,
        "prompt_version": PROMPT_VERSION,
        "data_files": len(manifest["data"]),
        "prediction_files": len(manifest["predictions"]),
        "git_dirty": manifest["git_dirty"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
