"""Loader for PDTB-style Chinese treebanks (CDTB 0.5, LDC2014T21).

CDTB is distributed by the LDC under a paid licence.  This module does not
download it and must not be pointed at a copy obtained elsewhere; see
``scripts/fetch_cdtb.py`` for the licensed acquisition path.

The on-disk format is pipe-delimited with a variable number of columns
(CDTB 0.5 is described as 27 fields, PDTB 2.0 uses 24).  Because the column
order cannot be verified without the licensed archive, the loader supports:

* ``--columns path.json`` -- an explicit ``field -> 0-based index`` map, and
* ``autodetect_columns()`` -- content-based inference, which is only a
  starting point and should be checked with ``inspect_pipe()``.

Always run ``python -m discourse_reliability inspect-pipe <file>`` on the real
data first and pin the map in ``configs/cdtb_columns.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ..labels import is_known_level1, normalize_sense, sense_levels
from ..schema import Item

RELATION_TYPES = {"explicit", "implicit", "altlex", "entrel", "norel"}

DEFAULT_COLUMNS: dict[str, int] = {
    "relation": 0,
    "conn_head": 1,
    "conn1": 2,
    "conn2": 3,
    "conn_start_tok": 4,
    "conn_end_tok": 5,
    "arg1_start_sent": 6,
    "arg1_start_tok": 7,
    "arg1_end_sent": 8,
    "arg1_end_tok": 9,
    "arg2_start_sent": 10,
    "arg2_start_tok": 11,
    "arg2_end_sent": 12,
    "arg2_end_tok": 13,
    "conn_raw": 14,
    "arg1_raw": 15,
    "arg2_raw": 16,
    "sense": 24,
}

_CJK = re.compile(r"[一-鿿]")


def _split(line: str, delimiter: str) -> list[str]:
    return [c.strip() for c in line.rstrip("\n").split(delimiter)]


def _looks_like_chinese(text: str) -> bool:
    return bool(_CJK.search(text))


def _score_relation(col: list[str]) -> int:
    vals = [v.lower() for v in col if v]
    if not vals:
        return 0
    return sum(v in RELATION_TYPES for v in vals)


def _score_sense(col: list[str]) -> int:
    hits = 0
    for v in col:
        if not v:
            continue
        first = v.split(":")[0].strip().upper()
        if first in {"TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION"}:
            hits += 1
        elif ":" in v and re.match(r"^[A-Za-z ]+:", v):
            hits += 1
    return hits


def autodetect_columns(rows: list[list[str]]) -> dict[str, int]:
    """Best-effort column inference.  Verify against ``inspect_pipe`` output."""
    ncols = max((len(r) for r in rows), default=0)
    cols = [[r[i] if i < len(r) else "" for r in rows] for i in range(ncols)]

    rel = max(range(ncols), key=lambda i: _score_relation(cols[i])) if ncols else 0
    sense = max(range(ncols), key=lambda i: _score_sense(cols[i])) if ncols else 0

    # Raw argument columns: Chinese text, long.
    candidates = [
        i
        for i in range(ncols)
        if i not in (rel, sense)
        and sum(_looks_like_chinese(v) for v in cols[i]) > max(1, len(rows) // 2)
    ]
    # Among the two largest Chinese-text columns, the left one is Arg1:
    # PDTB-style layouts always place Arg1 before Arg2.
    candidates.sort(key=lambda i: sum(len(v) for v in cols[i]), reverse=True)
    top2 = sorted(candidates[:2])
    arg1 = top2[0] if top2 else 15
    arg2 = top2[1] if len(top2) > 1 else 16

    # Connective raw: short Chinese string, not equal to the argument columns.
    short = [
        i
        for i in range(ncols)
        if i not in (rel, sense, arg1, arg2)
        and cols[i]
        and sum(_looks_like_chinese(v) for v in cols[i]) > len(rows) // 2
        and sum(len(v) for v in cols[i]) / max(1, len(rows)) < 12
    ]
    conn = max(short, key=lambda i: sum(_looks_like_chinese(v) for v in cols[i])) if short else 14

    return {
        "relation": rel,
        "sense": sense,
        "arg1_raw": arg1,
        "arg2_raw": arg2,
        "conn_raw": conn,
    }


def inspect_pipe(path: str | Path, n: int = 3, delimiter: str = "|") -> dict[str, Any]:
    """Return a preview of the first rows with column indices and widths."""
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(_split(line, delimiter))
            if len(rows) >= n:
                break
    in_hand = rows[0] if rows else []
    out_rows = [
        [{"index": i, "value": (r[i] if i < len(r) else "")[:120]} for i in range(max(len(r) for r in rows))]
        for r in rows
    ]
    return {
        "path": str(path),
        "n_columns_in_first_row": len(in_hand),
        "preview": out_rows,
        "autodetected_first": autodetect_columns(rows) if rows else {},
    }


def load_cdtb(
    root: str | Path,
    columns: dict[str, int] | None = None,
    delimiter: str = "|",
    include_implicit: bool = True,
    max_rows: int | None = None,
) -> tuple[list[Item], dict[str, int]]:
    """Load every ``*.pipe`` under ``root``.

    Returns ``(items, used_columns)`` so the caller can record which map was
    actually applied -- important for reproducibility.
    """
    root = Path(root)
    files = sorted(root.rglob("*.pipe"))
    if not files:
        raise FileNotFoundError(f"no .pipe files under {root}")

    first_rows: list[list[str]] = []
    with files[0].open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                first_rows.append(_split(line, delimiter))
            if len(first_rows) >= 50:
                break
    used = dict(columns) if columns else autodetect_columns(first_rows)

    def get(row: list[str], field: str) -> str:
        i = used.get(field, -1)
        return row[i] if 0 <= i < len(row) else ""

    items: list[Item] = []
    for path in files:
        split = next((s for s in ("train", "dev", "test") if s in path.parts), "train")
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = _split(line, delimiter)
                rel = get(row, "relation").lower()
                if rel in {"entrel", "norel"} or not rel:
                    continue
                explicit = rel == "explicit"
                if not explicit and not include_implicit:
                    continue

                sense = normalize_sense(get(row, "sense").split(",")[0])
                if not is_known_level1(sense):
                    continue
                l1, l2, l3 = sense_levels(sense)

                arg1 = get(row, "arg1_raw")
                arg2 = get(row, "arg2_raw")
                conn = get(row, "conn_raw") or get(row, "conn1")
                if not arg1 or not arg2:
                    continue

                if explicit and conn:
                    text = f"{arg1} {conn} {arg2}"
                    start = len(arg1) + 1
                    end = start + len(conn)
                else:
                    text = f"{arg1} {arg2}"
                    start = end = -1
                    conn = ""

                try:
                    a1s = int(get(row, "arg1_end_sent") or 0)
                    a2s = int(get(row, "arg2_start_sent") or 0)
                    cross = bool(a1s and a2s and a1s != a2s)
                except ValueError:
                    cross = False

                items.append(
                    Item(
                        id=f"{path.stem}:{lineno}",
                        corpus="cdtb",
                        task="explicit_sense" if explicit else "implicit_relation",
                        text=text,
                        connective=conn,
                        arg1=arg1,
                        arg2=arg2,
                        conn_char_start=start,
                        conn_char_end=end,
                        explicit=explicit,
                        paired=bool(get(row, "conn2")) or ("..." in conn) or ("…" in conn),
                        cross_sentence=cross,
                        sense_l1=l1,
                        sense_l2=l2,
                        sense_l3=l3,
                        genre="news",
                        split=split if split in {"train", "dev", "test"} else "train",
                        source=str(path),
                        doc_id=path.stem,
                        sent_id=get(row, "arg1_start_sent"),
                    )
                )
                if max_rows and len(items) >= max_rows:
                    return items, used
    return items, used


def load_columns_config(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
