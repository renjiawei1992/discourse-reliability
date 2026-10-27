"""Loader for TED-CDB (Long, Webber & Xiong, EMNLP 2020).

TED-CDB is the closest free substitute for CDTB: Chinese TED talks annotated
under PDTB 3.0 principles, 15k relations over 72 talks, with full three-level
sense strings and a second annotator recorded in ``adjudication_groups.xml``.

Two differences from CDTB matter scientifically and must be reported:

* **Genre.** TED-CDB is planned *spoken monologue*; CDTB is Xinhua newswire.
  Published 4-way F1 drops from ~95 on CDTB to ~60 on TED-CDB, so numbers are
  not comparable across the two -- never mix them in one table.
* **Licence.** The repo ships no licence file.  The paper states the data
  "will be made freely available", and the 16-talk predecessor (LREC 2020,
  github.com/tjunlp-lab/...) is explicitly CC BY 4.0.  If you need an
  unambiguous licence for redistribution, use the 16-talk subset.

On-disk layout (``TED_CDB_Data``)::

    Ann/tokens/<talk>.txt   34 pipe-separated fields per relation
    raw/tokens/<talk>.txt   UTF-8, space-separated tokens (the offsets' basis)
    Ann/adjudication_groups.xml

Field map was recovered by inspection, not documented upstream; it is
re-checked against spans in ``tests/test_core.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..labels import is_known_level1, normalize_sense, sense_levels
from ..schema import Item

#: ``None`` entries are carried into ``Item.meta`` but not otherwise used.
FIELDS = {
    0: "relation",
    1: "conn_span",
    2: "speaker",       # Wr (writer) / Ot (other)
    3: "comm",
    4: "attrib1",
    5: "attrib2",
    6: "conn_span2",
    7: "connective",    # explicit surface form, "NONE" when implicit
    8: "sense",
    9: "sense2",        # second admissible sense, when the annotators kept one
    14: "arg1_span",
    15: "arg1_flag",
    20: "arg2_span",
    21: "arg2_flag",
    31: "conn_span3",
}

SKIP_RELATIONS = {"entrel", "norel"}

#: CJK ideographs, CJK punctuation/fullwidth forms, and ``○`` (U+25CB), which
#: is used as a digit in Chinese dates ("八○年代") and behaves like one here.
_CJK = r"㐀-鿿　-〿＀-￯○"
_CJK_RE = re.compile(rf"[{_CJK}]")


def detok_map(tokens: str) -> tuple[str, list[int]]:
    """Detokenize and return, for each output char, its index in ``tokens``.

    Chinese is written without spaces, so a space between two CJK characters
    is a tokenisation artefact; spaces next to Latin/digits are real and are
    preserved.  Line wraps are always artefacts.
    """
    chars: list[str] = []
    pos: list[int] = []
    n = len(tokens)
    for i, ch in enumerate(tokens):
        if ch in "\r\n":
            continue
        if ch == " ":
            j = i + 1
            while j < n and tokens[j] in " \r\n":
                j += 1
            prev = chars[-1] if chars else ""
            nxt = tokens[j] if j < n else ""
            if prev and nxt and _CJK_RE.match(prev) and _CJK_RE.match(nxt):
                continue
            if prev == " ":
                continue
        chars.append(ch)
        pos.append(i)
    s = "".join(chars)
    lead = len(s) - len(s.lstrip())
    end = len(s.rstrip())
    return s[lead:end], pos[lead:end]


def detokenize(tokens: str) -> str:
    return detok_map(tokens)[0]


def _span(raw: str) -> tuple[int, int] | None:
    """Parse ``"37..40"`` (inclusive) into ``(start, end_exclusive)``."""
    if not raw or ".." not in raw:
        return None
    a, b = raw.split("..", 1)
    try:
        start, end = int(a), int(b)
    except ValueError:
        return None
    return start, end + 1


def _segments(raw: str) -> list[tuple[int, int]]:
    """Parse ``"587..588;601..603"`` into ``[(587, 589), (601, 604)]``.

    Discontinuous spans encode paired connectives ("当 ... 之后") and arguments
    interrupted by a parenthetical; they are common enough in TED-CDB that a
    single-span parser silently drops a few hundred relations.
    """
    segs = []
    for part in (raw or "").split(";"):
        sp = _span(part)
        if sp:
            segs.append(sp)
    return segs


def _slice(tokens: str, raw: str, sep: str = " ") -> str:
    segs = _segments(raw)
    return sep.join(detokenize(tokens[a:b]) for a, b in segs)


def _row_fields(line: str) -> dict[str, str]:
    cells = line.split("|")
    return {name: (cells[i] if i < len(cells) else "") for i, name in FIELDS.items()}


def load_tedcdb(
    root: str | Path,
    include_implicit: bool = True,
    max_rows: int | None = None,
) -> list[Item]:
    """Load every ``Ann/tokens/*.txt`` under ``TED_CDB_Data``.

    ``root`` may point at ``TED_CDB_Data`` or at its parent.
    """
    root = Path(root)
    ann_dir = next((p for p in root.rglob("Ann") if (p / "tokens").is_dir()), None)
    if ann_dir is None:
        raise FileNotFoundError(f"no Ann/tokens directory under {root}")
    raw_dir = ann_dir.parent / "raw" / "tokens"
    split = next((s for s in ("train", "dev", "test") if s in root.parts), "external")

    items: list[Item] = []
    for ann_path in sorted((ann_dir / "tokens").glob("*.txt")):
        raw_path = raw_dir / ann_path.name
        if not raw_path.exists():
            continue
        tokens = raw_path.read_text(encoding="utf-8")
        for lineno, line in enumerate(ann_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = _row_fields(line)
            rel = row["relation"].strip().lower()
            if rel in SKIP_RELATIONS:
                continue
            if rel != "explicit" and not include_implicit:
                continue

            sense = normalize_sense(row["sense"].split(",")[0])
            if not sense or not is_known_level1(sense):
                continue

            a1segs = _segments(row["arg1_span"])
            a2segs = _segments(row["arg2_span"])
            if not a1segs or not a2segs:
                continue
            arg1 = _slice(tokens, row["arg1_span"])
            arg2 = _slice(tokens, row["arg2_span"])
            if not arg1 or not arg2:
                continue

            start = min(a1segs[0][0], a2segs[0][0])
            end = max(max(e for _, e in a1segs), max(e for _, e in a2segs))
            text, tpos = detok_map(tokens[start:end])
            # Map token-stream offsets onto the detokenized string.
            idx = {p + start: k for k, p in enumerate(tpos)}

            first, second = (a1segs, a2segs) if a1segs[0][0] <= a2segs[0][0] else (a2segs, a1segs)
            gap_start, gap_end = min(e for _, e in first), second[0][0]
            gap = tokens[gap_start:gap_end] if gap_start < gap_end else ""
            # Cross-sentence only if a sentence boundary falls *between* the args.
            cross = any(p in gap for p in "。！？")

            # Field 7 holds the surface connective for Explicit rows but the
            # *annotation-only* implied connective for Implicit rows.  Using
            # the latter as ``Item.connective`` would leak a label the model
            # never sees, so it is kept in ``meta`` instead.
            conn_field = row["connective"].strip()
            if conn_field.upper() == "NONE":
                conn_field = ""
            csegs = _segments(row["conn_span"])
            if rel == "explicit":
                conn = conn_field or _slice(tokens, row["conn_span"], sep="...")
                implied = ""
            else:
                conn = ""
                implied = conn_field

            l1, l2, l3 = sense_levels(sense)
            alt = normalize_sense(row["sense2"])
            inside = [idx[p] for a, b in csegs for p in range(a, b) if p in idx]
            if inside:
                c_start, c_end = min(inside), max(inside) + 1
            else:
                c_start = c_end = -1
            items.append(
                Item(
                    id=f"tedcdb:{ann_path.stem}:{lineno}",
                    corpus="ted-cdb",
                    task="explicit_sense" if rel == "explicit" else (
                        "altlex_sense" if rel == "altlex" else "implicit_relation"
                    ),
                    text=text,
                    connective=conn,
                    arg1=arg1,
                    arg2=arg2,
                    conn_char_start=c_start,
                    conn_char_end=c_end,
                    explicit=rel == "explicit",
                    paired=len(csegs) > 1 or ("..." in conn),
                    cross_sentence=cross,
                    sense_l1=l1,
                    sense_l2=l2,
                    sense_l3=l3,
                    genre="ted_talk_spoken",
                    split=split,
                    source=str(ann_path),
                    doc_id=ann_path.stem,
                    sent_id=str(lineno),
                    meta={
                        "relation_type": row["relation"],
                        "speaker": row["speaker"],
                        "arg1_flag": row["arg1_flag"],
                        "arg2_flag": row["arg2_flag"],
                        "alt_sense": alt,
                        "implicit_connective": implied,
                        "altlex_trigger": _slice(tokens, row["conn_span"]) if rel == "altlex" else "",
                        "arg_order": "arg1_first" if a1segs[0][0] <= a2segs[0][0] else "arg2_first",
                        "tokenized": tokens[start:end],
                    },
                )
            )
            if max_rows and len(items) >= max_rows:
                return items
    return items
