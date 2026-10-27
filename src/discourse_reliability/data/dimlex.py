"""Loader for Chinese-DiMLex (Wan et al., LRE 2025).

Chinese-DiMLex is a lexicon of 226 Chinese discourse connectives with PDTB 3.0
senses and one illustrative example per sense.  It is NOT a corpus: its
examples were selected by lexicographers to make a sense clear, so they are
easier than naturally occurring instances.  Treat them as (a) the sense
inventory, (b) a smoke-test set, and (c) a source of few-shot exemplars --
never as an unbiased estimate of model reliability.

License: CC BY-NC-SA 4.0 (inherited from German DiMLex).
Repo: https://github.com/discourse-lab/chinese-dimlex
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..labels import normalize_sense, sense_levels
from ..schema import Item


def _parts(word: str) -> list[str]:
    return [p for p in word.split("...") if p]


def load_dimlex(xml_path: str | Path) -> list[Item]:
    root = ET.parse(str(xml_path)).getroot()
    entries = root.findall("entry")
    items: list[Item] = []

    for entry in entries:
        entry_id = entry.get("id", "")
        word = (entry.get("word") or "").strip()
        if not word:
            continue
        syns = entry.findall("syn")
        # A connective can have several <syn> blocks; senses live under each.
        for syn in syns:
            sems = syn.findall("sem")
            n_senses = len(sems)
            for j, sem in enumerate(sems):
                rel = sem.find("pdtb3_relation")
                sense = normalize_sense(rel.get("sense") if rel is not None else "")
                example_el = sem.find("example")
                example = (example_el.text or "").strip() if example_el is not None else ""
                if not example or not sense:
                    continue
                l1, l2, l3 = sense_levels(sense)
                parts = _parts(word)
                paired = len(parts) > 1
                missing = [p for p in parts if p not in example]
                items.append(
                    Item(
                        id=f"dimlex:{entry_id}:{j}",
                        corpus="chinese-dimlex",
                        task="explicit_sense",
                        text=example,
                        connective=word,
                        paired=paired,
                        explicit=True,
                        cross_sentence=("。" in example) or ("！" in example) or ("？" in example),
                        n_lexicon_senses=n_senses,
                        sense_l1=l1,
                        sense_l2=l2,
                        sense_l3=l3,
                        genre="lexicon_example",
                        split="external",
                        source="dimlex_example",
                        meta={
                            "entry_id": entry_id,
                            "syn_cat": (syn.findtext("cat") or "").strip(),
                            "english_equivalent": (sem.findtext("english_equivalent") or "").strip(),
                            "parts_missing_from_example": missing,
                            "example_bias": True,
                        },
                    )
                )
    return items


def enrich_from_inventory(items: list[Item], inventory: dict[str, int]) -> dict[str, int]:
    """Attach lexicon ambiguity (number of senses) to corpus items.

    Items whose connective is absent from Chinese-DiMLex keep ``n_lexicon_senses=1``
    and are flagged in ``meta["lexicon_match"]`` so the ambiguity stratum can be
    reported with an explicit coverage figure.
    """
    stats = {"matched": 0, "unmatched": 0}
    for it in items:
        n = inventory.get(it.connective)
        if n is None and "..." in it.connective:
            # "因为...所以..." may be stored as one entry; try the head part alone.
            n = inventory.get(it.connective.split("...")[0])
        if n is None:
            it.meta["lexicon_match"] = False
            stats["unmatched"] += 1
        else:
            it.n_lexicon_senses = n
            it.meta["lexicon_match"] = True
            stats["matched"] += 1
    return stats


def dimlex_inventory(xml_path: str | Path) -> dict[str, int]:
    """Map normalised connective -> number of distinct lexicon senses."""
    root = ET.parse(str(xml_path)).getroot()
    inv: dict[str, int] = {}
    for entry in root.findall("entry"):
        word = (entry.get("word") or "").strip()
        if not word:
            continue
        senses = {
            normalize_sense(r.get("sense"))
            for r in entry.iter("pdtb3_relation")
            if r.get("sense")
        }
        inv[word] = len(senses)
    return inv
