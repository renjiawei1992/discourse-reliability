"""Two-annotator agreement and adjudication sheets.

Workflow (experiment plan, Block 1):
    1. python run.py prep-annotation --input ... --out-dir data/interim/r1
    2. give each annotator a copy of annotation_sheet.csv
    3. python run.py agreement --a a.csv --b b.csv --out-dir data/interim/r1
       -> exports agreement.json and adjudication.csv
    4. a third person fills the adjudicated column in adjudication.csv
    5. python run.py merge-adjudication --items items.jsonl --adjudication ... \
       --out gold.jsonl
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .labels import normalize_sense, sense_levels
from .metrics.agreement import cohen_kappa

UNDECIDED = {"UNKNOWN", "UNSURE", "?"}


def _read(path: str | Path, column: str = "annotator_sense_l3") -> dict[str, str]:
    out: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            val = normalize_sense(row.get(column, ""))
            if val:
                out[row["item_id"]] = val
    return out


def _item_meta(path: str | Path | None) -> dict[str, dict]:
    if not path:
        return {}
    meta: dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                meta[d["id"]] = d
    return meta


def bootstrap_kappa(
    x: list[str],
    y: list[str],
    groups: list[str],
    n_boot: int = 1000,
    seed: int = 13,
) -> tuple[float, float]:
    """Kappa CI by resampling *documents*, not items.

    Items from one document share context and topic, so their labels are not
    exchangeable.  Resampling items would understate the interval -- by how
    much depends on how clustered the packet is, which is exactly the thing
    that must not be assumed away.
    """
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    keys = sorted(by_group)
    if len(keys) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n_boot):
        idx: list[int] = []
        for _ in keys:
            idx.extend(by_group[keys[rng.randrange(len(keys))]])
        k = cohen_kappa([x[i] for i in idx], [y[i] for i in idx])
        # kappa is undefined when a resample happens to contain one class;
        # dropping those draws is the only honest option (imputing 1.0 or 0.0
        # would manufacture certainty in exactly the degenerate strata).
        if k is not None:
            out.append(k)
    if len(out) < max(30, n_boot // 10):
        return float("nan"), float("nan")
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def agreement(
    path_a: str | Path,
    path_b: str | Path,
    column: str = "annotator_sense_l3",
    items_path: str | Path | None = None,
    n_boot: int = 1000,
) -> dict:
    raw_a = _read(path_a, column)
    raw_b = _read(path_b, column)
    ids = sorted(set(raw_a) & set(raw_b))

    # An item dropped from the packet (e.g. an editorial exclusion) is still in
    # the annotators' submissions.  Scoring it would keep a purged item in the
    # kappa and in any example drawn from this report, so when an items file is
    # given it defines the universe.
    meta = _item_meta(items_path)
    if meta:
        ids = [i for i in ids if i in meta]

    # UNKNOWN is not a sense.  Counting two matching UNKNOWNs as agreement
    # inflates kappa, so undecided items are reported separately and kappa is
    # computed on the items both annotators actually decided.
    undecided = [i for i in ids if raw_a[i].upper() in UNDECIDED or raw_b[i].upper() in UNDECIDED]
    ids = [i for i in ids if i not in set(undecided)]
    levels = {"l1": [], "l2": [], "l3": []}
    ag, bg, groups = [], [], []
    for i in ids:
        la, lb = sense_levels(raw_a[i]), sense_levels(raw_b[i])
        for k in range(3):
            levels[["l1", "l2", "l3"][k]].append((la[k], lb[k]))
        ag.append(raw_a[i])
        bg.append(raw_b[i])
        groups.append(str(meta.get(i, {}).get("doc_id") or i))

    report: dict = {
        "n": len(ids),
        "undecided": len(undecided),
        "undecided_ids": undecided,
        "n_docs": len(set(groups)),
        "by_level": {},
    }
    for lvl, pairs in levels.items():
        if not pairs:
            continue
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        entry = {
            "agreement": sum(g == h for g, h in pairs) / len(pairs),
            "kappa": cohen_kappa(x, y),
        }
        if lvl == "l1" and len(set(groups)) >= 2:
            lo, hi = bootstrap_kappa(x, y, groups, n_boot=n_boot)
            entry["kappa_ci95_doc_cluster"] = [lo, hi]
        report["by_level"][lvl] = entry
    if ag:
        report["kappa_l3"] = cohen_kappa(ag, bg)

    # Per-stratum kappa at L1: the gate's headline number can hide a class in
    # which the guideline simply does not work.
    if meta:
        strata: dict[str, list[int]] = defaultdict(list)
        for k, i in enumerate(ids):
            m = meta.get(i, {})
            strata["explicit" if m.get("explicit") else "implicit"].append(k)
        report["by_stratum"] = {}
        for name, idx in strata.items():
            x = [levels["l1"][k][0] for k in idx]
            y = [levels["l1"][k][1] for k in idx]
            g = [groups[k] for k in idx]
            lo, hi = bootstrap_kappa(x, y, g, n_boot=n_boot) if len(set(g)) >= 2 else (float("nan"),) * 2
            report["by_stratum"][name] = {
                "n": len(idx),
                "agreement": sum(a == b for a, b in zip(x, y)) / len(idx),
                "kappa": cohen_kappa(x, y),
                "kappa_ci95_doc_cluster": [lo, hi],
            }

    # Which pairs of classes the annotators actually confuse.
    conf = Counter()
    for k in range(len(ids)):
        conf[(levels["l1"][k][0], levels["l1"][k][1])] += 1
    report["confusion_l1"] = [
        {"a": p, "b": q, "n": n} for (p, q), n in sorted(conf.items()) if p != q
    ]
    report["disagreements"] = [
        {"item_id": i, "a": raw_a[i], "b": raw_b[i]} for i in ids if raw_a[i] != raw_b[i]
    ]
    return report


def write_adjudication(
    path_a: str | Path,
    path_b: str | Path,
    items_path: str | Path,
    out_csv: str | Path,
    column: str = "annotator_sense_l3",
) -> int:
    a = _read(path_a, column)
    b = _read(path_b, column)
    meta: dict[str, dict] = {}
    with Path(items_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                meta[d["id"]] = d
    n = 0
    with Path(out_csv).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "text", "connective", "annotator_a", "annotator_b", "adjudicated", "reason"])
        for i in sorted(set(a) | set(b)):
            if a.get(i) == b.get(i):
                continue
            it = meta.get(i, {})
            w.writerow([i, it.get("text", ""), it.get("connective", ""), a.get(i, ""), b.get(i, ""), "", ""])
            n += 1
    return n


def merge_adjudication(
    items_path: str | Path,
    adjudication_csv: str | Path,
    out_path: str | Path,
) -> dict:
    """Apply adjudicated labels; unresolved disagreements keep annotator A's label."""
    adj: dict[str, str] = {}
    with Path(adjudication_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = normalize_sense(row.get("adjudicated", ""))
            if v:
                adj[row["item_id"]] = v

    stats = Counter()
    out = []
    with Path(items_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["id"] in adj:
                l1, l2, l3 = sense_levels(adj[d["id"]])
                d.update(sense_l1=l1, sense_l2=l2, sense_l3=l3, source=d.get("source", "") + "+adjudicated")
                stats["adjudicated"] += 1
            else:
                stats["unchanged"] += 1
            out.append(d)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return {"written": len(out), **stats}
