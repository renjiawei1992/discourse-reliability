"""M4: build the human readability packets for the delete-connective condition.

Three blocks, joined by an opaque ``pid`` the annotator never resolves:

    block_a  the item exactly as the model saw it (connective -> ____)
    block_b  the original item, connective marked with brackets
    block_c  the connective simply removed, no placeholder (diagnostic only)

The base items are recovered from the probe prediction files, so the sample is
exactly the population R140 ran on rather than a different draw.  See
``refine-logs/M4_READABILITY_PROTOCOL.md`` for the sampling rule and the
pre-registered decision gates.

    python scripts/build_m4_readability_packet.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.counterfactual import no_connective  # noqa: E402
from discourse_reliability.schema import load_items  # noqa: E402

CLASSES = ["TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION"]
HEADER = ["序号", "pid", "句子", "你的判断", "备注"]


def frame_parts(connective: str) -> list[str]:
    """Split a paired connective ("当初…如今", "不是...是") into its parts."""
    c = connective.replace("...", "…").replace("……", "…")
    return [p for p in c.split("…") if p]


def paired_frame_ok(connective: str, span: str) -> bool:
    """The span covers the frame; a malformed trailing part is tolerated.

    ``tedcdb:34_w:49`` is labelled 「虽然...但是...你」 but the span ends at
    「但是」 — the deletion itself is right, only the label has a stray tail.
    """
    parts = frame_parts(connective)
    return (bool(span) and len(parts) >= 2 and span.startswith(parts[0])
            and any(span.endswith(p) for p in parts[1:]))


def base_ids_from_probe(path: Path) -> list[str]:
    ids = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ids.append(o["item_id"].replace("::no_conn", ""))
    return ids


def allocate(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Largest-remainder allocation, never exceeding the group size."""
    n = sum(weights.values())
    exact = {k: total * v / n for k, v in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k, v in out.items():
        out[k] = min(out[k], weights[k])
    while sum(out.values()) < total:
        room = [k for k in weights if out[k] < weights[k]]
        if not room:
            break
        k = max(room, key=lambda x: exact[x] - out[x])
        out[k] += 1
    return out


def spread(items: list, k: int) -> list:
    """Take k of len(items) evenly spaced, preserving the given sort order."""
    if k >= len(items):
        return list(items)
    if k <= 0:
        return []
    m = len(items)
    idx = [round(i * (m - 1) / (k - 1)) if k > 1 else 0 for i in range(k)]
    seen, picked = set(), []
    for i in idx:
        while i in seen:
            i += 1
            if i >= m:
                i = 0
                while i in seen:
                    i += 1
        seen.add(i)
        picked.append(items[i])
    return picked


def write_block(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit-probe", default="outputs/l1_probe_noconn.jsonl")
    ap.add_argument("--ted-probe", default="outputs/l1_probe_ted_noconn.jsonl")
    ap.add_argument("--hit", default="data/processed/l1/hit_balanced.jsonl")
    ap.add_argument("--ted", default="data/processed/l1/ted_balanced.jsonl")
    ap.add_argument("--out-dir", default="data/interim/m4")
    ap.add_argument("--n-hit", type=int, default=60)
    ap.add_argument("--n-ted", type=int, default=40)
    ap.add_argument("--n-c", type=int, default=30)
    ap.add_argument("--seed", type=int, default=161)
    args = ap.parse_args(argv)

    corpus_paths = {"hit": ROOT / args.hit, "ted": ROOT / args.ted}
    probe_paths = {"hit": ROOT / args.hit_probe, "ted": ROOT / args.ted_probe}
    targets = {"hit": args.n_hit, "ted": args.n_ted}

    items: dict[str, object] = {}
    for name, path in corpus_paths.items():
        for it in load_items(path):
            items[it.id] = it

    # Verify the recovered ids can actually be mutated the way the model saw
    # them.  Three shapes are legitimate: a single token, a paired frame
    # ("当初…如今", deleted as one span), and the no-offset fallback (first
    # occurrence of the connective).  Items where none applies were NOT
    # mutated at all — the probe ran the original sentence — so they are
    # excluded from the population and reported.
    kinds = Counter()
    unusable = []
    for name, path in probe_paths.items():
        for iid in base_ids_from_probe(path):
            it = items.get(iid)
            if it is None:
                unusable.append((iid, "missing from corpus file"))
                continue
            if not it.explicit:
                unusable.append((iid, "not explicit"))
                continue
            s, e = it.conn_char_start, it.conn_char_end
            span = it.text[s:e] if s >= 0 and e > s else ""
            if span == it.connective:
                kinds["single"] += 1
            elif paired_frame_ok(it.connective, span):
                kinds["paired"] += 1
            elif it.connective and it.connective in it.text:
                kinds["fallback"] += 1
            else:
                unusable.append((iid, f"connective {it.connective!r} absent from text, span {span!r}"))
    if unusable:
        print(f"WARNING: {len(unusable)} probe item(s) were never mutated; excluded")
        for iid, why in unusable[:10]:
            print(f"  {iid}: {why}")
        print(f"usable population: {dict(kinds)}")

    rng = random.Random(args.seed)
    sample: list[str] = []
    excluded = {iid for iid, _ in unusable}
    for name, path in probe_paths.items():
        ids = [i for i in base_ids_from_probe(path) if i in items and i not in excluded]
        by_class = defaultdict(list)
        for iid in ids:
            by_class[items[iid].sense_l1].append(iid)
        alloc = allocate(min(targets[name], len(ids)), {c: len(v) for c, v in by_class.items()})
        # Within a class, sort by connective frequency so the evenly-spaced pick
        # does not collapse onto the single most frequent connective.
        for c, group_ids in by_class.items():
            freq = Counter(items[i].connective for i in group_ids)
            group_ids.sort(key=lambda i: (-freq[items[i].connective], i))
            sample.extend(spread(group_ids, alloc[c]))

    sample = sorted(set(sample))
    rng.shuffle(sample)

    pids = [f"m4-{rng.randrange(16**6):06x}" for _ in sample]
    if len(set(pids)) != len(pids):
        raise SystemExit("pid collision, rerun with another seed")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def render(it, variant: str) -> str:
        s, e = it.conn_char_start, it.conn_char_end
        if variant == "delete":
            # Same code path as R140, so block A is literally the model's input.
            return f"句子（原连接词已被删除，用 ____ 标出）：\n{no_connective(it).text}"
        if variant == "plain":
            if s >= 0 and e > s:
                return f"句子（原连接词已被删除，没有占位符）：\n{it.text[:s]}{it.text[e:]}"
            return ("句子（原连接词已被删除，没有占位符）：\n"
                    + it.text.replace(it.connective, "", 1))
        # Same rendering as prompts._render_explicit: bracket the whole span.
        return f"句子（连接词已用【】标出）：\n{it.text[:s]}【{it.text[s:e]}】{it.text[e:]}"

    def block(rows_to_write, variant: str, name: str, seed: int) -> None:
        order = list(rows_to_write)
        random.Random(seed).shuffle(order)
        rows = [[n, pid, render(items[iid], variant), "", ""]
                for n, (pid, iid) in enumerate(order, start=1)]
        write_block(out_dir / f"m4_block_{name}.csv", rows)

    pairs = list(zip(pids, sample))
    block(pairs, "delete", "a", args.seed + 1)
    block(pairs, "bracket", "b", args.seed + 2)

    c_pairs = rng.sample(pairs, min(args.n_c, len(pairs)))
    c_order = list(c_pairs)
    random.Random(args.seed + 3).shuffle(c_order)
    block(c_order, "plain", "c", args.seed + 4)

    # ``span_chars`` vs the connective's own length tells us how much *content*
    # the deletion removed: for paired frames ("目前…到二十一世纪") the span is
    # often a whole clause, so the mutation destroys information, not just the
    # cue.  That is a covariate of the readability rate, not a bug to hide.
    with (out_dir / "m4_key.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "item_id", "corpus", "connective", "gold_l1", "in_block_c",
                    "span_chars", "connective_chars", "paired_frame"])
        c_set = {p for p, _ in c_pairs}
        for pid, iid in pairs:
            it = items[iid]
            s, e = it.conn_char_start, it.conn_char_end
            span = it.text[s:e] if s >= 0 and e > s else ""
            conn_len = len("".join(frame_parts(it.connective))) or len(it.connective)
            w.writerow([pid, iid, it.corpus, it.connective, it.sense_l1, int(pid in c_set),
                        len(span), conn_len, int(bool(span) and span != it.connective)])

    print(f"wrote {len(pairs)} items to {out_dir}")
    print("corpus:", Counter(items[i].corpus for _, i in pairs))
    print("gold:", Counter(items[i].sense_l1 for _, i in pairs).most_common())
    print("connective:", Counter(items[i].connective for _, i in pairs).most_common(12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
