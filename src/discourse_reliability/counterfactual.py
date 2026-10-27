"""Counterfactual variants for the shortcut-control block (Block 3).

Three mutations, each targeting a different possible shortcut:

    no_connective     remove the connective  -> does the model need its form?
    synonym_swap      replace with a near-synonym of the same sense
                                             -> is the label tied to the lexeme?
    reorder_context   move the first sentence of the context to the end,
                      falling back to swapping arguments
                                             -> does the model use global order?

Every variant keeps the gold label: a correct mutation must not change the
relation.  ``validate_csv`` exports them for the human check required before
the main run.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import re
from pathlib import Path
from typing import Iterable

from .schema import Item

#: Each group must be interchangeable WITHOUT changing the PDTB sense.
#: Swaps are validated by a human before the main run (see validate_csv).
SYNONYM_GROUPS = [
    ["但是", "然而", "不过", "可是", "但"],
    ["因为", "由于", "鉴于"],
    ["所以", "因此", "因而", "故而", "于是"],
    ["如果", "假如", "要是", "倘若", "一旦"],
    ["虽然", "尽管", "虽"],
    ["即使", "即便", "哪怕"],
    ["只要", "只需"],
    ["只有", "除非"],
    ["无论", "不论", "不管"],
    ["而且", "并且", "且"],
    ["同时", "与此同时"],
    ["例如", "比如", "譬如", "诸如"],
    ["也就是说", "换言之", "即"],
    ["或者", "或", "抑或"],
    ["然后", "接着", "随后", "此后"],
    ["此外", "另外"],
    ["总之", "总而言之"],
    ["最后", "最终"],
]

GROUP_OF = {w: i for i, g in enumerate(SYNONYM_GROUPS) for w in g}

_SENT_END = re.compile(r"(?<=[。！？])")


def _pick_synonym(word: str, seed: str) -> str | None:
    gi = GROUP_OF.get(word)
    if gi is None:
        return None
    group = SYNONYM_GROUPS[gi]
    h = int(hashlib.sha256(f"{seed}|{word}".encode()).hexdigest(), 16)
    options = [w for w in group if w != word]
    if not options:
        return None
    return options[h % len(options)]


def no_connective(item: Item) -> Item:
    v = copy.deepcopy(item)
    if item.conn_char_start >= 0 and item.conn_char_end > item.conn_char_start:
        s, e = item.conn_char_start, item.conn_char_end
        v.text = item.text[:s] + "____" + item.text[e:]
        v.conn_char_start = s
        v.conn_char_end = s + 4
    else:
        v.text = item.text.replace(item.connective, "____", 1) if item.connective else item.text
        v.conn_char_start = v.conn_char_end = -1
    v.connective = ""
    v.meta = {
        **item.meta,
        "variant": "no_connective",
        "base_id": item.id,
        "target_connective": item.connective,
        "original_text": item.text,
    }
    v.id = f"{item.id}::no_conn"
    return v


def synonym_swap(item: Item) -> Item | None:
    syn = _pick_synonym(item.connective, item.id)
    if not syn:
        return None
    v = copy.deepcopy(item)
    if item.conn_char_start >= 0 and item.conn_char_end > item.conn_char_start:
        s, e = item.conn_char_start, item.conn_char_end
        if item.text[s:e] != item.connective:
            return None
        v.text = item.text[:s] + syn + item.text[e:]
        v.conn_char_end = s + len(syn)
    else:
        if item.connective not in item.text:
            return None
        v.text = item.text.replace(item.connective, syn, 1)
    v.connective = syn
    v.meta = {
        **item.meta,
        "variant": "synonym_swap",
        "base_id": item.id,
        "original_connective": item.connective,
        "original_text": item.text,
    }
    v.id = f"{item.id}::syn"
    return v


def reorder_context(item: Item) -> Item | None:
    """Move the first sentence to the end, or swap the two arguments.

    Returns ``None`` when neither transformation is possible (a single-sentence
    example with no stored arguments).  Emitting an empty or ungrammatical
    variant would silently corrupt the control condition.
    """
    v = copy.deepcopy(item)
    sentences = [s for s in _SENT_END.split(item.text) if s.strip()]
    if len(sentences) >= 2:
        v.text = "".join(sentences[1:] + sentences[:1])
        variant = "reorder_context"
    elif item.arg1 and item.arg2:
        v.text = f"{item.arg2} {item.arg1}"
        variant = "swap_args"
    else:
        return None
    v.conn_char_start = v.text.find(item.connective) if item.connective else -1
    v.conn_char_end = (
        v.conn_char_start + len(item.connective) if v.conn_char_start >= 0 else -1
    )
    v.meta = {**item.meta, "variant": variant, "base_id": item.id, "original_text": item.text}
    v.id = f"{item.id}::reorder"
    return v


def generate(items: Iterable[Item], only_explicit: bool = True) -> list[Item]:
    out: list[Item] = []
    for it in items:
        if only_explicit and not it.explicit:
            continue
        out.append(no_connective(it))
        syn = synonym_swap(it)
        if syn:
            out.append(syn)
        reordered = reorder_context(it)
        if reordered is not None:
            out.append(reordered)
    return out


def validate_csv(items: Iterable[Item], path: str | Path) -> int:
    """Export original/variant pairs for human plausibility checking."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant_id", "base_id", "variant", "connective", "sense", "original", "variant_text", "ok_1/0", "note"])
        for it in items:
            w.writerow(
                [
                    it.id,
                    it.meta.get("base_id", ""),
                    it.meta.get("variant", ""),
                    it.connective,
                    it.gold,
                    it.meta.get("original_text", ""),
                    it.text,
                    "",
                    "",
                ]
            )
            n += 1
    return n
