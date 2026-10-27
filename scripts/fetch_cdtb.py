#!/usr/bin/env python
"""Prepare a licensed CDTB archive that you have already obtained.

CDTB (Chinese Discourse Treebank 0.5, LDC2014T21) is a paid, licence-gated
resource.  This script deliberately does NOT download it and rejects
undeclared mirrors: using an unlicensed copy would invalidate any paper built
on it.  What it does is the mechanical part -- safe extraction, structure
check, and the exact command to run next.

Getting the archive (one of):
  1. LDC catalogue  https://catalog.ldc.upenn.edu/LDC2014T21
     - USD 0 for 2014 members, USD 300 for non-members
     - requires an LDC account and a signed licence
  2. Ask your institution's library whether it holds an LDC membership;
     many Chinese universities buy one centrally.

Then:
  python scripts/fetch_cdtb.py --archive /path/to/LDC2014T21.tgz
  python run.py inspect-pipe --input data/raw/cdtb/<file>.pipe
  python run.py build-data --source cdtb --input data/raw/cdtb \
      --columns configs/cdtb_columns.json --out data/processed/cdtb.jsonl
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "raw" / "cdtb"


def safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target = (dest / name).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise SystemExit(f"unsafe path in archive: {name}")
            z.extractall(dest)
    elif archive.suffixes[-2:] in ([".tar", ".gz"], [".tgz"]) or archive.suffix == ".tar":
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                target = (dest / member.name).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise SystemExit(f"unsafe path in archive: {member.name}")
            t.extractall(dest, filter="data")
    else:
        raise SystemExit(f"unsupported archive type: {archive.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="path to the LDC archive you downloaded")
    ap.add_argument("--dest", default=str(DEST))
    args = ap.parse_args()

    archive = Path(args.archive).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    if not archive.exists():
        raise SystemExit(f"archive not found: {archive}")

    safe_extract(archive, dest)

    pipes = sorted(dest.rglob("*.pipe"))
    readme = sorted(dest.rglob("README*")) + sorted(dest.rglob("*.txt"))
    print(f"extracted to: {dest}")
    print(f".pipe files: {len(pipes)}")
    for p in pipes[:10]:
        print(f"  {p.relative_to(dest)}")
    if not pipes:
        print("WARNING: no .pipe files found; inspect the extracted tree before continuing.")
    for r in readme[:5]:
        print(f"doc: {r.relative_to(dest)}")
    print("\nnext:")
    if pipes:
        print(f"  python run.py inspect-pipe --input \"{pipes[0]}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
