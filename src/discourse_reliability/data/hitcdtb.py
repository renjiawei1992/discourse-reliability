"""Loader for HIT-CDTB (Harbin Institute of Technology Chinese Discourse Treebank).

525 OntoNotes documents annotated at three granularities::

    .p1  sentence-group relations   (句群)
    .p2  sentence relations         (复句)
    .p3  clause relations           (分句)

Each ``.pN`` file holds one block per relation::

    --------------------------------------------------
    ____Explicit____
    Source: <sentence>
    ____Connective____
    Span:244...245
    Content:也
    ____Arg1____
    Span:233...240
    Content: ...
    ____Arg2____
    Span:241...278
    Content: ...
    ____Sense____
    RelNO:4-2
    Content:递进
    ____Annotation____
    All:233 240 241 278 1 0 244 245 1 4-2递进

Spans are half-open character offsets into the matching ``.txt``, which is
GB18030-encoded and space-tokenised.  For implicit relations the connective
span is a bare insertion point and its ``Content`` is the connective the
annotators supplied -- keep it in ``meta``, never in the prompt.

The relation set is HIT's own four-level hierarchy, not PDTB.  ``HIT_TO_PDTB``
below is a *lossy* mapping written for this project; see ``MAPPING_NOTE``.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..labels import sense_levels
from ..schema import Item
from .tedcdb import detok_map

#: HIT first digit -> HIT top-level name.
HIT_L1 = {
    "1": "时序",
    "2": "因果",
    "3": "比较",
    "4": "扩展",
}

#: HIT ``RelNO`` (or its longest matching prefix) -> PDTB 3.0 sense string.
#: ``True`` marks a mapping where HIT draws a distinction PDTB 3.0 has no
#: counterpart for; the distinction survives only in ``meta["hit_label"]``.
HIT_TO_PDTB: dict[str, tuple[str, bool]] = {
    # 1 时序
    "1-1": ("TEMPORAL:Synchronous", False),
    "1-2-1": ("TEMPORAL:Asynchronous:Precedence", False),
    "1-2-2": ("TEMPORAL:Asynchronous:Succession", False),
    # 2 因果 (含条件)
    "2-1-1": ("CONTINGENCY:Cause:Reason", False),
    "2-1-2": ("CONTINGENCY:Cause:Result", False),
    "2-2-1": ("CONTINGENCY:Cause+belief:Evidence", True),
    "2-2-2": ("CONTINGENCY:Cause+belief:Conclusion", True),
    "2-3-1": ("CONTINGENCY:Purpose:Arg1-as-goal", False),
    "2-3-2": ("CONTINGENCY:Purpose:Arg2-as-goal", False),
    "2-4-1": ("CONTINGENCY:Condition:Arg1-as-cond", True),
    "2-4-2": ("CONTINGENCY:Condition:Arg2-as-cond", True),
    "2-5": ("CONTINGENCY:Condition", True),
    "2-6": ("CONTINGENCY:Condition", True),
    # 3 比较
    "3-1": ("COMPARISON:Contrast", False),
    "3-2": ("COMPARISON:Contrast", True),
    "3-3": ("COMPARISON:Concession", False),
    "3-4": ("COMPARISON:Concession", True),
    # 4 扩展 (含并列)
    "4-1": ("EXPANSION:Conjunction", True),
    "4-2": ("EXPANSION:Progression", False),
    "4-3-1": ("EXPANSION:Level-of-detail", False),
    "4-3-2": ("EXPANSION:Instantiation", False),
    "4-3-3": ("EXPANSION:Exception", False),
    "4-4": ("EXPANSION:Level-of-detail", True),
    "4-5": ("EXPANSION:Conjunction", True),
    "4-6": ("EXPANSION:Disjunction", False),
    "4-7": ("EXPANSION:Conjunction", True),
}

#: Fallback used when a ``RelNO`` is finer than anything in ``HIT_TO_PDTB``
#: (or newer than this table).  Keeping the relation with an L1-only gold
#: label is better than dropping it: L1 is the level we can defend, and a
#: silent drop would bias every class count.
HIT_L1_TO_PDTB = {
    "1": "TEMPORAL",
    "2": "CONTINGENCY",
    "3": "COMPARISON",
    "4": "EXPANSION",
}

MAPPING_NOTE = (
    "HIT_TO_PDTB is a project-level, lossy mapping. Top-level classes align "
    "one-to-one (1->TEMPORAL, 2->CONTINGENCY, 3->COMPARISON, 4->EXPANSION), so "
    "L1 results are directly comparable with TED-CDB; L2/L3 are not. Rows "
    "flagged True collapse a HIT distinction with no PDTB 3.0 counterpart. "
    "Sub-codes missing from the table fall back to their top-level class."
)

_SECTION = re.compile(r"^_{4}([A-Za-z0-9]+)_{4}\s*$", re.M)
_DASHES = re.compile(r"^-{10,}\s*$", re.M)
_SEG = re.compile(r"(\d+)\.\.\.(\d+)")


def map_sense(relno: str) -> tuple[str, str, bool]:
    """Return ``(pdtb_sense, matched_prefix, lossy)`` for a HIT ``RelNO``.

    Longest prefix wins, so ``4-3-2-2`` resolves through ``4-3-2``.  A code with
    no matching prefix at all still resolves to its top-level class (``"4"`` ->
    ``EXPANSION``) so that no relation is silently dropped; such rows carry an
    L1-only gold label, which is exactly the level the mapping can defend.
    """
    key = (relno or "").strip()
    while key:
        if key in HIT_TO_PDTB:
            sense, lossy = HIT_TO_PDTB[key]
            return sense, key, lossy
        if "-" not in key:
            break
        key = key.rsplit("-", 1)[0]
    top = (relno or "").strip().split("-")[0]
    if top in HIT_L1_TO_PDTB:
        return HIT_L1_TO_PDTB[top], top, True
    return "", "", True


def _read(path: Path) -> str:
    """Read a ``.txt``/``.pN`` file with newlines normalised to ``\\n``.

    The annotation offsets were computed under Python's universal-newline
    mode, i.e. against a text where every ``\\r\\n`` is a single ``\\n``.
    ``read_bytes().decode()`` keeps the ``\\r``, which shifts every offset by
    the number of line breaks in the document -- silently, and by an amount
    that differs per file, so it is not a constant that could be corrected
    downstream.
    """
    raw = path.read_bytes()
    for enc in ("utf-8", "gb18030"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("gb18030", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _blocks(text: str) -> list[dict[str, dict[str, str]]]:
    """Split a ``.pN`` file into ``{section: {key: value}}`` records."""
    out = []
    for chunk in _DASHES.split(text):
        if "____" not in chunk:
            continue
        marks = list(_SECTION.finditer(chunk))
        if not marks:
            continue
        header = chunk[: marks[0].start()].strip().strip("_").strip()
        # The relation type is itself written as a section ("____Explicit____"),
        # so it is usually marks[0] rather than a bare header line.
        first = marks[0].group(1)
        if not header and first.lower() in {"explicit", "implicit"}:
            header = first
            marks = marks[1:]
            if not marks:
                continue
        sections: dict[str, dict[str, str]] = {}
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(chunk)
            body = chunk[m.end() : end]
            fields = {}
            for line in body.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    fields[k.strip()] = v.strip()
                elif line.startswith("Span"):
                    fields["Span"] = line[4:].strip().lstrip(":").strip()
            sections[m.group(1)] = fields
        out.append({"type": header, **sections})
    return out


def _segments(raw: str) -> list[tuple[int, int]]:
    """Parse a ``Span`` value into half-open ``(start, end)`` segments.

    HIT uses ``a...b`` with an *exclusive* end (unlike TED-CDB's inclusive
    ``a..b``).  Paired connectives and parenthetical-interrupted arguments are
    written as ``"504...507;525...527"``; 162 of the 1,050 explicit relation
    blocks in the 50-doc sample take this form, so a single-span parser would
    mis-handle them rather than merely drop them.
    """
    segs = []
    for a, b in _SEG.findall(raw or ""):
        i, j = int(a), int(b)
        if j > i:
            segs.append((i, j))
    return segs


def _detok_slice(tokens: str, segs: list[tuple[int, int]], sep: str = "") -> str:
    return sep.join(detok_map(tokens[a:b])[0] for a, b in segs)


def _cross_sentence(
    tokens: str, a1s: list[tuple[int, int]], a2s: list[tuple[int, int]]
) -> bool:
    """True when a sentence boundary falls *between* the two arguments.

    Only the gap counts: a full stop inside an argument is not a cross-sentence
    relation.
    """
    first, second = (a1s, a2s) if a1s[0][0] <= a2s[0][0] else (a2s, a1s)
    gap_start = max(e for _, e in first)
    gap_end = second[0][0]
    return gap_start < gap_end and any(p in tokens[gap_start:gap_end] for p in "。！？")


def load_hitcdtb(
    root: str | Path,
    levels: tuple[str, ...] = ("p2", "p3"),
    include_implicit: bool = True,
    max_rows: int | None = None,
) -> list[Item]:
    """Load ``.p1``/``.p2``/``.p3`` files under ``root``.

    ``levels`` defaults to sentence- and clause-level relations; ``p1``
    (sentence groups) is excluded because PDTB has no counterpart granularity
    and including it would triple-count the same text.
    """
    root = Path(root)
    items: list[Item] = []
    for lvl in levels:
        for rel_path in sorted(root.rglob(f"*.{lvl}")):
            txt_path = rel_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            tokens = _read(txt_path)
            genre = rel_path.parent.name
            for idx, blk in enumerate(_blocks(_read(rel_path)), 1):
                rtype = (blk.get("type") or "").strip().lower()
                if rtype not in {"explicit", "implicit"}:
                    continue
                if rtype == "implicit" and not include_implicit:
                    continue

                sense_sec = blk.get("Sense", {})
                relno = sense_sec.get("RelNO", "").strip()
                label = sense_sec.get("Content", "").strip()
                sense, matched, lossy = map_sense(relno)
                if not sense:
                    continue

                a1s = _segments(blk.get("Arg1", {}).get("Span", ""))
                a2s = _segments(blk.get("Arg2", {}).get("Span", ""))
                if not a1s or not a2s:
                    continue
                start = min(a1s[0][0], a2s[0][0])
                end = max(e for _, e in a1s + a2s)
                if end <= start or end > len(tokens):
                    continue
                text, tpos = detok_map(tokens[start:end])
                if not text:
                    continue
                idx_map = {p + start: k for k, p in enumerate(tpos)}

                arg1 = _detok_slice(tokens, a1s)
                arg2 = _detok_slice(tokens, a2s)

                csegs = _segments(blk.get("Connective", {}).get("Span", ""))
                conn_raw = blk.get("Connective", {}).get("Content", "").strip()
                if conn_raw.upper() == "NULL":
                    conn_raw = ""
                if rtype == "explicit":
                    # ``Content`` carries the annotators' own surface form; it
                    # separates paired connectives with ";" ("不仅;同时").
                    conn = conn_raw or _detok_slice(tokens, csegs, sep="…")
                    conn = "…".join(p.strip() for p in re.split(r"[;；]", conn) if p.strip())
                    implied = ""
                else:
                    conn = ""
                    implied = conn_raw

                # Each connective segment mapped onto ``text``; for a paired
                # connective ``conn_char_start/end`` is the enclosing envelope,
                # so the exact pieces are kept here as well.
                conn_segments: list[list[int]] = []
                for a, b in csegs:
                    hit = [idx_map[p] for p in range(a, b) if p in idx_map]
                    if not hit:
                        continue
                    s, e = min(hit), max(hit) + 1
                    # A span that ends against Latin text keeps the separating
                    # space; it belongs to neither the connective nor the arg.
                    while s < e and text[s].isspace():
                        s += 1
                    while e > s and text[e - 1].isspace():
                        e -= 1
                    if e > s:
                        conn_segments.append([s, e])
                if conn_segments and rtype == "explicit":
                    c_start = min(s for s, _ in conn_segments)
                    c_end = max(e for _, e in conn_segments)
                else:
                    c_start = c_end = -1

                l1, l2, l3 = sense_levels(sense)
                items.append(
                    Item(
                        id=f"hitcdtb:{rel_path.parent.name}/{rel_path.stem}:{lvl}:{idx}",
                        corpus="hit-cdtb",
                        task="explicit_sense" if rtype == "explicit" else "implicit_relation",
                        text=text,
                        connective=conn,
                        arg1=arg1,
                        arg2=arg2,
                        conn_char_start=c_start,
                        conn_char_end=c_end,
                        explicit=rtype == "explicit",
                        paired=len(csegs) > 1,
                        cross_sentence=_cross_sentence(tokens, a1s, a2s),
                        sense_l1=l1,
                        sense_l2=l2,
                        sense_l3=l3,
                        genre=f"ontonotes_{genre}",
                        split="external",
                        source=str(rel_path),
                        doc_id=rel_path.stem,
                        sent_id=f"{lvl}:{idx}",
                        meta={
                            "hit_level": lvl,
                            "hit_relno": relno,
                            "hit_label": label,
                            "hit_l1": HIT_L1.get(relno.split("-")[0], ""),
                            "mapping_key": matched,
                            "mapping_lossy": lossy,
                            "relation_type": blk.get("type", ""),
                            "implicit_connective": implied,
                            "conn_char_segments": conn_segments,
                            # Literal text under the span, kept so that the two
                            # places where the corpus's Content disagrees with
                            # its own offsets stay visible instead of silent.
                            "conn_span_text": _detok_slice(tokens, csegs, sep="…")
                            if rtype == "explicit"
                            else "",
                        },
                    )
                )
                if max_rows and len(items) >= max_rows:
                    return items
    return items
