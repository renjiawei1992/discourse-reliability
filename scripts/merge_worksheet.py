"""Fold the filled adjudication worksheet back into a reference standard.

The worksheet is the working document (it carries the targeted question, both
annotators' labels and the corpus gold side by side); this script turns the
`最终裁决` column into an actual label and writes a reference file usable by
the scoring pipeline.

Who may change the label.  The adjudicator only overrides the corpus gold when
the item is a *triple* disagreement -- gold, annotator C and annotator D name
three different things.  In every other case (gold sides with C or D, or one
annotator returned UNKNOWN) the corpus gold is the final answer and the
adjudicator's cell is recorded, not applied.  This matters for priority 2 of the
worksheet: there gold always equals one of the two annotators, so "the
adjudicator agreed with gold" cannot be told apart from "the adjudicator agreed
with the annotator who happened to match gold" -- applying those cells would
manufacture agreement that was never demonstrated independently.

Provenance is kept per item rather than implied, because the sources are not
interchangeable in a paper:

    adjudicated  the item was a triple disagreement and the third person's
                 label is now the reference  -> highest confidence
    corpus       the corpus gold stands (possibly after adjudicator review)

Items where both annotators already agree are *not* rewritten here.  Their
consensus is a different reference standard (human, post-hoc) from the corpus
gold every table in L1_RESULTS.md was computed against; silently swapping it
would make the new numbers incomparable to the old ones.  `--consensus-file`
does exactly that swap when you deliberately want the human reference.

    python scripts/merge_worksheet.py \
        --items data/processed/l1/b1_packet.jsonl \
        --worksheet data/interim/b1_round1/adjudication_worksheet.csv \
        --out data/interim/b1_round1/b1_gold.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.labels import L1_CLASSES, normalize_sense, sense_levels  # noqa: E402

DECISION_COLUMN = "最终裁决"
GOLD_COLUMN = "语料金标准"
ANNOTATOR_COLUMNS = ("C 的标注", "D 的标注")


def is_triple_disagreement(row: dict[str, str]) -> bool:
    """True when gold, C and D name three different L1 classes.

    This is the only case in which the adjudicator's cell overrides gold.  A
    UNKNOWN from either annotator is not a disagreement about which class the
    item belongs to, so it does not qualify.
    """
    gold = normalize_sense(row.get(GOLD_COLUMN, ""))
    labels = [normalize_sense(row.get(col, "")) for col in ANNOTATOR_COLUMNS]
    if any(lab not in L1_CLASSES for lab in labels):
        return False
    c, d = labels
    return c != d and gold != c and gold != d


def read_worksheet(
    path: Path,
) -> tuple[dict[str, str], dict[str, tuple[str, str]], set[str]]:
    """Return (label decisions, other decisions, triple-disagreement ids).

    Not every filled cell is a label.  Three things show up in practice:

    * ``UNKNOWN`` -- the adjudicator could not decide either.  The item stays
      unresolved rather than being forced into a class.
    * a free-text note asking for the item to be dropped -- treated as an
      *exclusion*, not a label.
    * anything else -- still pending; a question is not a decision.
    """
    labels: dict[str, str] = {}
    other: dict[str, tuple[str, str]] = {}
    triple: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            iid = row["item_id"]
            if is_triple_disagreement(row):
                triple.add(iid)
            raw = (row.get(DECISION_COLUMN) or "").strip()
            if not raw:
                continue
            label = normalize_sense(raw)
            if label in L1_CLASSES:
                labels[iid] = label
            elif raw.upper() == "UNKNOWN":
                other[iid] = ("unresolved", raw)
            elif "删除" in raw or "剔除" in raw:
                other[iid] = ("excluded", raw)
            else:
                other[iid] = ("pending", raw)
    return labels, other, triple


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--worksheet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--consensus-file", default=None,
                    help="CSV where both annotators agreed; used only when you "
                         "deliberately want the human reference instead of corpus gold")
    ap.add_argument("--column", default="annotator_sense_l1")
    args = ap.parse_args(argv)

    decisions, flagged, triple = read_worksheet(Path(args.worksheet))
    consensus: dict[str, str] = {}
    if args.consensus_file:
        with Path(args.consensus_file).open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                v = normalize_sense(row.get(args.column, ""))
                if v in L1_CLASSES:
                    consensus[row["item_id"]] = v

    stats = Counter()
    changed = []
    reviewed_not_applied = []
    out = []
    seen: set[str] = set()
    with Path(args.items).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            seen.add(d["id"])
            original = normalize_sense(d.get("sense_l1"))
            meta = d.setdefault("meta", {})
            meta["ref_corpus_gold"] = original
            if d["id"] in decisions and d["id"] in triple:
                label, source = decisions[d["id"]], "adjudicated"
                meta["ref_adjudicated"] = decisions[d["id"]]
            elif d["id"] in decisions:
                # The adjudicator read the item but the protocol only lets them
                # override gold on triple disagreements.  Record the cell so the
                # disagreement is visible, keep gold as the reference.
                label, source = original, "corpus"
                meta["ref_adjudicated"] = decisions[d["id"]]
                if decisions[d["id"]] != original:
                    reviewed_not_applied.append(
                        (d["id"], original, decisions[d["id"]]))
            elif d["id"] in flagged:
                source, note = flagged[d["id"]]
                label = original
                meta["ref_note"] = note
            elif d["id"] in consensus:
                label, source = consensus[d["id"]], "human_consensus"
            else:
                label, source = original, "corpus"
            meta["ref_source"] = source
            if label and label != original:
                changed.append((d["id"], original, label, source))
                l1, l2, l3 = sense_levels(label)
                d.update(sense_l1=l1, sense_l2=l2, sense_l3=l3)
            stats[source] += 1
            out.append(d)

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    report = {
        "written": str(path).replace("\\", "/"),
        "n_items": len(out),
        "by_source": dict(stats),
        "changed_vs_corpus_gold": len(changed),
        "triple_disagreements": {"n": len(triple),
                                 "with_label": len(triple & set(decisions))},
        "reviewed_but_not_applied": [
            {"item_id": i, "corpus": a, "adjudicated": b}
            for i, a, b in reviewed_not_applied],
        "corpus_gold_kept_by_rule": {
            "n": len(set(decisions) - triple),
            "adjudicator_disagreed": len(reviewed_not_applied),
        },
        # Flagged ids that were since purged from the packet (see
        # scripts/exclude_items.py) are dropped from this list: reporting an
        # excluded item that is no longer in the file would suggest it is still
        # there under a flag.
        "not_labelled": {k: [{"item_id": i, "note": n}
                             for i, (s, n) in flagged.items() if s == k and i in seen]
                         for k in ("pending", "unresolved", "excluded")},
        "changes": [{"item_id": i, "corpus": a, "ref": b, "source": s}
                    for i, a, b, s in changed],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
