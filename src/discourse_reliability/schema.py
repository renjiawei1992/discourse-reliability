"""Core data types and JSONL I/O.

Every loader (CDTB, Chinese-DiMLex, ad-hoc JSONL) normalises into :class:`Item`.
Every model adapter emits :class:`Prediction`.  Keeping both flat and
JSON-serialisable means intermediate files are inspectable and diffable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Item:
    id: str
    corpus: str
    task: str  # "explicit_sense" | "implicit_relation"
    text: str
    connective: str = ""
    arg1: str = ""
    arg2: str = ""
    conn_char_start: int = -1
    conn_char_end: int = -1
    sense_l1: str = ""
    sense_l2: str = ""
    sense_l3: str = ""
    explicit: bool = True
    paired: bool = False
    cross_sentence: bool = False
    n_lexicon_senses: int = 1
    genre: str = "unknown"
    split: str = "train"
    source: str = ""
    doc_id: str = ""
    sent_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def gold(self) -> str:
        """Best available gold label (L3 preferred, then L2, then L1)."""
        return self.sense_l3 or self.sense_l2 or self.sense_l1

    def stratum(self) -> dict[str, str]:
        """The four stratification axes from the experiment plan.

        ``ambiguity`` is ``unknown`` when the connective is absent from
        Chinese-DiMLex.  The enrichment leaves ``n_lexicon_senses`` at its
        default of 1 in that case, so without this check every unmatched
        connective would be silently reported as "low ambiguity" -- and on
        HIT-CDTB that is 84% of the corpus.
        """
        matched = self.meta.get("lexicon_match")
        ambiguity = (
            "unknown"
            if matched is False
            else ("low" if self.n_lexicon_senses <= 2 else "high")
        )
        return {
            "explicitness": "explicit" if self.explicit else "implicit",
            "ambiguity": ambiguity,
            "connective_type": "paired" if self.paired else "single",
            "span": "cross_sentence" if self.cross_sentence else "intra_sentence",
        }

    def stratum_key(self) -> str:
        return "|".join(f"{k}={v}" for k, v in self.stratum().items())


@dataclass
class Prediction:
    item_id: str
    model: str
    regime: str
    label_l1: str = ""
    label_l2: str = ""
    label_l3: str = ""
    raw_output: str = ""
    confidence: float | None = None
    samples: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prediction":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_items(path: str | Path) -> list[Item]:
    return [Item.from_dict(d) for d in read_jsonl(path)]


def load_predictions(path: str | Path) -> list[Prediction]:
    return [Prediction.from_dict(d) for d in read_jsonl(path)]


def save_items(path: str | Path, items: Iterable[Item]) -> int:
    return write_jsonl(path, (it.to_dict() for it in items))


def save_predictions(path: str | Path, preds: Iterable[Prediction]) -> int:
    return write_jsonl(path, (p.to_dict() for p in preds))
