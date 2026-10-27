"""Command line interface.

    python run.py build-data      --source dimlex --input <xml> --out data/processed/dimlex.jsonl
    python run.py inspect-pipe    --input <cdtb.pipe>
    python run.py build-data      --source cdtb --input <cdtb_root> --columns configs/cdtb_columns.json --out ...
    python run.py sample          --input ... --n 600 --out ...
    python run.py prep-annotation --input ... --out-dir data/interim/round1
    python run.py run             --input ... --model mock --regime zero_shot --out preds.jsonl
    python run.py evaluate        --gold ... --preds ... --out report.json
    python run.py prior           --input ... --out prior.jsonl
    python run.py counterfactual  --input ... --out variants.jsonl --validate-csv review.csv
    python run.py triage          --gold ... --preds ... --out triage.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from .counterfactual import generate as make_variants
from .counterfactual import validate_csv
from .data.cdtb import inspect_pipe, load_cdtb, load_columns_config
from .data.dimlex import load_dimlex
from .data.hitcdtb import load_hitcdtb
from .data.tedcdb import load_tedcdb
from .data.sampling import (
    assign_folds,
    assign_splits,
    cap_per_connective,
    describe,
    stratified_sample,
)
from .labels import L1_ZH
from .metrics.evaluate import evaluate_by_model
from .schema import Item, Prediction, load_items, load_predictions, save_items, save_predictions, write_jsonl
from .baselines.lexeme_prior import attach_prior_entropy, prior_predictions
from .runner import few_shot_pool, run as run_models
from .triage import compare_policies


def _dump(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_build(args) -> None:
    if args.source == "dimlex":
        items = load_dimlex(args.input)
    elif args.source == "cdtb":
        cols = load_columns_config(args.columns) if args.columns else None
        items, used = load_cdtb(
            args.input,
            columns=cols,
            delimiter=args.delimiter,
            include_implicit=not args.explicit_only,
            max_rows=args.limit,
        )
        print(json.dumps({"used_columns": used}, ensure_ascii=False, indent=2))
    elif args.source == "tedcdb":
        items = load_tedcdb(
            args.input,
            include_implicit=not args.explicit_only,
            max_rows=args.limit,
        )
    elif args.source == "hitcdtb":
        levels = tuple(s.strip() for s in args.hit_levels.split(",") if s.strip())
        items = load_hitcdtb(
            args.input,
            levels=levels,
            include_implicit=not args.explicit_only,
            max_rows=args.limit,
        )
    elif args.source == "jsonl":
        items = load_items(args.input)
    else:
        raise SystemExit(f"unknown source {args.source}")

    if args.dimlex_xml:
        from .data.dimlex import dimlex_inventory, enrich_from_inventory

        stats = enrich_from_inventory(items, dimlex_inventory(args.dimlex_xml))
        print(json.dumps({"lexicon_enrichment": stats}, ensure_ascii=False, indent=2))

    n = save_items(args.out, items)
    print(json.dumps({"written": n, "out": args.out, **describe(items)}, ensure_ascii=False, indent=2))


def cmd_inspect(args) -> None:
    print(json.dumps(inspect_pipe(args.input, n=args.n, delimiter=args.delimiter), ensure_ascii=False, indent=2))


def cmd_sample(args) -> None:
    items = load_items(args.input)
    if args.cap_per_connective:
        items = cap_per_connective(items, args.cap_per_connective, seed=args.seed)
    sampled = stratified_sample(
        items,
        n=args.n,
        seed=args.seed,
        explicit_fraction=args.explicit_fraction,
        min_per_stratum=args.min_per_stratum,
    )
    split_by = None if args.split_by in {"", "none"} else args.split_by
    if args.assign_splits:
        assign_splits(sampled, seed=args.seed, group_by=split_by)
    if args.cv_folds:
        assign_folds(sampled, k=args.cv_folds, seed=args.seed, group_by=split_by or "doc_id")
    n = save_items(args.out, sampled)
    print(json.dumps({"written": n, "out": args.out, **describe(sampled)}, ensure_ascii=False, indent=2))


def cmd_prep_annotation(args) -> None:
    items = load_items(args.input)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_items(out / "items.jsonl", items)
    # Arg1/Arg2 are the two spans that bear the relation.  Without them an
    # implicit item is unnannotatable: the surrounding sentence often contains
    # some unrelated connective, and the reader has no way to know which two
    # clause groups the label applies to.
    header = ["item_id", "doc_id", "split", "task", "arg1", "arg2", "connective",
              "text", "annotator_sense_l1", "annotator_sense_l2",
              "annotator_sense_l3", "note"]
    # Each annotator gets their own row order.  A shared order lets fatigue and
    # topic drift land on the same items twice, which shows up as agreement.
    for k in range(args.annotators):
        name = chr(ord("A") + k)
        order = list(items)
        random.Random(args.seed + k).shuffle(order)
        with (out / f"annotation_sheet_{name}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            w = csv.writer(f)
            w.writerow(header)
            for it in order:
                w.writerow([it.id, it.doc_id, it.meta.get("b1_split", ""), it.task,
                            it.arg1, it.arg2, it.connective, it.text,
                            "", "", "", ""])
    l1 = "\n".join(f"- {k}（{v}）" for k, v in L1_ZH.items())
    (out / "guideline.md").write_text(
        "# 汉语连接词 Sense 标注指南（草案）\n\n"
        "## 标签体系\n\n" + l1 + "\n\n"
        "## 标注步骤\n\n"
        "1. 通读句子与上下文，不要只看连接词。\n"
        "2. 先判大类（L1），再判二级/三级。\n"
        "3. 配对连接词（如「虽然…但是…」）以整体表达的关系为准。\n"
        "4. 遇逗号切分句，先补出隐含的逻辑主语。\n"
        "5. 确实无法判定时填 `UNKNOWN` 并在 note 说明。\n\n"
        "## 分歧处理\n\n两人独立标注，分歧交第三人裁决；裁决理由必须记录。\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {"dir": str(out), "n": len(items), "sheets": args.annotators},
        ensure_ascii=False, indent=2,
    ))


def cmd_run(args) -> None:
    from .models import get_adapter

    items = load_items(args.input)
    if args.limit:
        items = items[: args.limit]
    shots: list[Item] = []
    if args.regime in {"few_shot", "cot"}:
        if args.shots_from:
            pool = load_items(args.shots_from)
        else:
            pool = [it for it in items if it.split == "train"]
            if any("fold" in it.meta for it in items):
                print(
                    "WARNING: exemplars are being drawn from the evaluation file. "
                    "For cross-validation pass --shots-from l1/hit_fold{k}_train.jsonl "
                    "with k the evaluated fold.",
                    file=sys.stderr,
                )
        shots = few_shot_pool(pool or items, args.k_shots, seed=args.seed, level=args.level)
    # Network policy belongs to the harness, not the model: a gateway that
    # times out should fail fast and be filled later, not stall the batch for
    # 4.5 minutes per item across three retries.
    adapter_kwargs = {}
    if not args.model.startswith("mock"):
        adapter_kwargs = {"timeout": args.timeout, "max_retries": args.max_retries}
    adapter = get_adapter(args.model, **adapter_kwargs)
    preds = run_models(
        items,
        adapter,
        regime=args.regime,
        shots=shots,
        n_samples=args.samples,
        temperature=args.temperature,
        workers=args.workers,
        level=args.level,
        max_tokens=args.max_tokens,
    )
    n = save_predictions(args.out, preds)
    errors = sum(1 for p in preds if p.error)
    print(json.dumps({"written": n, "errors": errors, "model": adapter.name, "regime": args.regime}, ensure_ascii=False, indent=2))


def cmd_evaluate(args) -> None:
    gold = load_items(args.gold)
    preds = load_predictions(args.preds)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    report = evaluate_by_model(
        gold, preds, n_boot=args.n_boot, seed=args.seed, levels=levels
    )
    _dump(args.out, report)
    for key, rep in report.items():
        for level in levels:
            lv = rep["levels"].get(level) or {}
            if lv.get("n"):
                print(
                    f"{key} [{level}]: n={lv.get('n')} acc={lv.get('accuracy')} "
                    f"kappa={lv.get('kappa')} macro_f1={lv.get('macro_f1')}"
                )


def cmd_prior(args) -> None:
    items = load_items(args.input)
    preds = prior_predictions(items, k=args.folds, level=args.level)
    n = save_predictions(args.out, preds)
    print(json.dumps({"written": n, "out": args.out}, ensure_ascii=False, indent=2))


def cmd_counterfactual(args) -> None:
    items = load_items(args.input)
    variants = make_variants(items, only_explicit=not args.include_implicit)
    if args.validate_csv:
        validate_csv(variants, args.validate_csv)
    n = save_items(args.out, variants)
    print(json.dumps({"written": n, "out": args.out, "validate_csv": args.validate_csv}, ensure_ascii=False, indent=2))


def cmd_triage(args) -> None:
    gold = load_items(args.gold)
    preds = load_predictions(args.preds)
    # Predictions from every model stay in the pool: models other than the base
    # one supply the cross-model consensus signal.  Filtering here would leave
    # the consensus policy scoring an empty pool.
    if args.regime:
        preds = [p for p in preds if p.regime == args.regime]
    base_model = args.base_model or args.model
    prior_ent = attach_prior_entropy(gold, k=args.folds, level=args.level)
    result = compare_policies(
        gold, preds, prior_ent, n_folds=args.folds, level=args.level,
        base_model=base_model,
    )
    _dump(args.out, result)
    print(json.dumps({"auc": result["auc"]}, ensure_ascii=False, indent=2))


def cmd_agreement(args) -> None:
    from .annotation import agreement, write_adjudication

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = agreement(
        args.a, args.b, column=args.column, items_path=args.items, n_boot=args.n_boot
    )
    if report["n"] == 0 and args.column != "annotator_sense_l1":
        # The web app only asks for the top-level class.  Returning n=0 with an
        # empty report would look like "the annotators agreed on nothing".
        print(
            f"note: no labels in column {args.column!r}; "
            "falling back to annotator_sense_l1",
            file=sys.stderr,
        )
        args.column = "annotator_sense_l1"
        report = agreement(
            args.a, args.b, column=args.column, items_path=args.items, n_boot=args.n_boot
        )
    if report["n"] == 0:
        raise SystemExit(
            "no item has a label in both files -- check the column name and that "
            "both annotators used the same sheet"
        )
    n = write_adjudication(args.a, args.b, args.items, out / "adjudication.csv", column=args.column)
    _dump(out / "agreement.json", report)
    print(json.dumps(
        {
            "n": report["n"],
            "undecided": report["undecided"],
            "by_level": report["by_level"],
            "by_stratum": report.get("by_stratum"),
            "adjudication_rows": n,
        },
        ensure_ascii=False, indent=2,
    ))


def cmd_merge(args) -> None:
    from .annotation import merge_adjudication

    stats = merge_adjudication(args.items, args.adjudication, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="discourse_reliability")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data")
    b.add_argument("--source", required=True, choices=["dimlex", "cdtb", "tedcdb", "hitcdtb", "jsonl"])
    b.add_argument("--input", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--columns")
    b.add_argument("--delimiter", default="|")
    b.add_argument("--explicit-only", action="store_true")
    b.add_argument("--limit", type=int)
    b.add_argument("--dimlex-xml", help="enrich corpus items with lexicon sense counts")
    b.add_argument("--hit-levels", default="p2,p3", help="HIT-CDTB levels: comma list of p1,p2,p3")
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("inspect-pipe")
    i.add_argument("--input", required=True)
    i.add_argument("--n", type=int, default=3)
    i.add_argument("--delimiter", default="|")
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("sample")
    s.add_argument("--input", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=600)
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--explicit-fraction", type=float, default=1 / 3)
    s.add_argument("--min-per-stratum", type=int, default=5)
    s.add_argument("--cap-per-connective", type=int, default=0)
    s.add_argument("--assign-splits", action="store_true")
    s.add_argument("--split-by", default="doc_id",
                   help="group splits by this Item field (default doc_id; use 'none' for item-level)")
    s.add_argument("--cv-folds", type=int, default=0,
                   help="assign document-level CV folds to meta['fold'] (0 = off)")
    s.set_defaults(func=cmd_sample)

    a = sub.add_parser("prep-annotation")
    a.add_argument("--input", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--annotators", type=int, default=2,
                   help="write one independently ordered sheet per annotator")
    a.add_argument("--seed", type=int, default=17)
    a.set_defaults(func=cmd_prep_annotation)

    r = sub.add_parser("run")
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="mock")
    r.add_argument("--regime", default="zero_shot", choices=["zero_shot", "few_shot", "cot"])
    r.add_argument("--shots-from")
    r.add_argument("--k-shots", type=int, default=8)
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--limit", type=int)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--level", default="l3", choices=["l1", "l2", "l3"],
                   help="label granularity the prompt asks for (default l3)")
    r.add_argument("--max-tokens", type=int, default=256,
                   help="completion budget; reasoning models need >1000")
    r.add_argument("--timeout", type=int, default=90, help="per-request timeout (s)")
    r.add_argument("--max-retries", type=int, default=3,
                   help="in-run retries; failed items can be filled later with "
                        "scripts/retry_unparseable.py")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate")
    e.add_argument("--gold", required=True)
    e.add_argument("--preds", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--n-boot", type=int, default=1000)
    e.add_argument("--seed", type=int, default=13)
    e.add_argument("--levels", default="l1,l2,l3",
                   help="comma list of levels to score (default all three)")
    e.set_defaults(func=cmd_evaluate)

    pr = sub.add_parser("prior")
    pr.add_argument("--input", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--folds", type=int, default=5)
    pr.add_argument("--level", default="l3", choices=["l1", "l2", "l3"])
    pr.set_defaults(func=cmd_prior)

    c = sub.add_parser("counterfactual")
    c.add_argument("--input", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--validate-csv")
    c.add_argument("--include-implicit", action="store_true")
    c.set_defaults(func=cmd_counterfactual)

    t = sub.add_parser("triage")
    t.add_argument("--gold", required=True)
    t.add_argument("--preds", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", help="alias for --base-model")
    t.add_argument("--base-model",
                   help="model whose labels are triaged; others feed consensus only")
    t.add_argument("--regime")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--level", default="l3", choices=["l1", "l2", "l3"],
                   help="label level the triage curve is computed on")
    t.set_defaults(func=cmd_triage)

    ag = sub.add_parser("agreement")
    ag.add_argument("--a", required=True)
    ag.add_argument("--b", required=True)
    ag.add_argument("--items", required=True)
    ag.add_argument("--out-dir", required=True)
    ag.add_argument("--column", default="annotator_sense_l3")
    ag.add_argument("--n-boot", type=int, default=1000,
                    help="document-clustered bootstrap draws for the kappa CI")
    ag.set_defaults(func=cmd_agreement)

    mg = sub.add_parser("merge-adjudication")
    mg.add_argument("--items", required=True)
    mg.add_argument("--adjudication", required=True)
    mg.add_argument("--out", required=True)
    mg.set_defaults(func=cmd_merge)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
