"""R111: supervised encoder baseline on HIT-CDTB (answers A5, "do we need an LLM?").

Fine-tunes a Chinese RoBERTa on the four L1 classes, fold by fold, and writes
out-of-fold predictions in the same JSONL format the LLM runs use, so the two
can be scored side by side by the same code.

Input format.  Both arms are ``<arg1> [SEP] <arg2>``; for explicit items the
connective is prepended to arg2, which is roughly where it sits in the source
sentence.  The LLM prompt shows explicit items as the full bracketed sentence
and implicit ones as two arguments, so neither side gets information the other
lacks -- but the encoder also never sees text outside the two arguments.

What this baseline is FOR.  If the encoder matches the LLM, the LLM is not
buying anything the task requires; if it loses badly on implicit items and
wins on explicit ones, that isolates where context matters.  Treat a single
fold's number as noise: HIT-CDTB has only 50 documents across 5 folds, so the
document-clustered interval is wide by construction.

    python scripts/train_l1_encoder.py \
        --train-template data/processed/l1/hit_fold{k}_train.jsonl \
        --test-template  data/processed/l1/hit_fold{k}_test.jsonl \
        --out outputs/l1_encoder_hit_oof.jsonl \
        --report outputs/l1_encoder_hit.json

Weights: huggingface.co is often unreachable from here.  ``HF_ENDPOINT=https://hf-mirror.com``
works, and ``--model /path/to/local/dir`` (config + vocab + pytorch_model.bin) works with
``HF_HUB_OFFLINE=1`` when the download has to be done by hand.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.annotation import bootstrap_kappa  # noqa: E402
from discourse_reliability.labels import L1_CLASSES  # noqa: E402
from discourse_reliability.metrics.agreement import (  # noqa: E402
    accuracy,
    cohen_kappa,
    macro_f1,
)
from discourse_reliability.schema import Prediction, load_items, save_predictions  # noqa: E402

LABEL_TO_ID = {c: i for i, c in enumerate(L1_CLASSES)}


def _texts(it) -> tuple[str, str]:
    a = it.arg1 or it.text
    if it.explicit and it.connective:
        b = f"{it.connective} {it.arg2}" if it.arg2 else it.text
    else:
        b = it.arg2 or ""
    return a, b


def per_class(gold: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for lab in L1_CLASSES:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {
            "n_gold": sum(g == lab for g in gold),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    return out


def summarise(items, preds_by_id: dict[str, str], n_boot: int = 1000) -> dict:
    keep = [it for it in items if it.id in preds_by_id]
    gold = [it.sense_l1 for it in keep]
    pred = [preds_by_id[it.id] for it in keep]
    groups = [it.doc_id or it.id for it in keep]
    k = cohen_kappa(gold, pred)
    lo, hi = bootstrap_kappa(gold, pred, groups, n_boot=n_boot)
    out = {
        "n": len(keep),
        "n_docs": len(set(groups)),
        "accuracy": round(accuracy(gold, pred), 4),
        "kappa": None if k is None else round(k, 4),
        "kappa_ci95_doc_cluster": [round(lo, 4), round(hi, 4)]
        if lo == lo and hi == hi else [None, None],
        "macro_f1": round(macro_f1(gold, pred), 4),
        "per_class": per_class(gold, pred),
        "pred_distribution": dict(Counter(pred).most_common()),
    }
    strata = {}
    for name, keep_flag in (("explicit", True), ("implicit", False)):
        idx = [i for i, it in enumerate(keep) if it.explicit is keep_flag]
        if len(idx) < 20:
            continue
        g = [gold[i] for i in idx]
        p = [pred[i] for i in idx]
        gr = [groups[i] for i in idx]
        kk = cohen_kappa(g, p)
        slo, shi = (bootstrap_kappa(g, p, gr, n_boot=n_boot)
                    if len(set(gr)) >= 2 else (float("nan"),) * 2)
        strata[name] = {
            "n": len(idx),
            "accuracy": round(accuracy(g, p), 4),
            "kappa": None if kk is None else round(kk, 4),
            "kappa_ci95_doc_cluster": [round(slo, 4), round(shi, 4)]
            if slo == slo and shi == shi else [None, None],
            "macro_f1": round(macro_f1(g, p), 4),
        }
    out["by_stratum"] = strata
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-template", default="data/processed/l1/hit_fold{k}_train.jsonl")
    ap.add_argument("--test-template", default="data/processed/l1/hit_fold{k}_test.jsonl")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--model", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--tag", default=None, help="short name used in the prediction model field")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--class-weight", default="none", choices=["none", "inverse"])
    ap.add_argument("--sampler", default="shuffle", choices=["shuffle", "balanced"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--cache-dir", default=None, help="where HF downloads land")
    args = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or args.model.split("/")[-1]
    print(f"device={device} model={args.model} tag={tag}", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

    class Encoded(Dataset):
        def __init__(self, items, labels):
            self.items = items
            self.labels = labels
            self.enc = [
                tok(*_texts(it), truncation=True, max_length=args.max_len,
                    add_special_tokens=True)
                for it in items
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.enc[i], self.labels[i]

    def collate(batch):
        encs = [b[0] for b in batch]
        out = tok.pad(encs, padding=True, return_tensors="pt")
        out["labels"] = torch.tensor([b[1] for b in batch])
        return out

    preds: list[Prediction] = []
    fold_log = []
    for fold in args.folds:
        t0 = time.time()
        train_items = [it for it in load_items(args.train_template.format(k=fold))
                       if it.sense_l1 in LABEL_TO_ID]
        test_items = [it for it in load_items(args.test_template.format(k=fold))
                      if it.sense_l1 in LABEL_TO_ID]
        tr_ds = Encoded(train_items, [LABEL_TO_ID[it.sense_l1] for it in train_items])
        te_ds = Encoded(test_items, [LABEL_TO_ID[it.sense_l1] for it in test_items])

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(L1_CLASSES), cache_dir=args.cache_dir).to(device)
        if args.sampler == "balanced":
            # Resampling instead of reweighting: every batch sees the classes in
            # roughly equal proportion, which is a stronger correction than
            # inverse-frequency loss weights and closer to what a practitioner
            # tuning for the balanced evaluation set would actually do.
            lab_counts = Counter(tr_ds.labels)
            sample_w = [1.0 / lab_counts[y] for y in tr_ds.labels]
            sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w),
                                            replacement=True)
            train_dl = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                                  collate_fn=collate, drop_last=True)
        else:
            train_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate,
                                  drop_last=len(tr_ds) > args.batch_size)
        test_dl = DataLoader(te_ds, batch_size=64, shuffle=False, collate_fn=collate)

        # HIT-CDTB is ~72% EXPANSION, so plain cross-entropy teaches the model
        # that guessing EXPANSION is usually right -- which is true in-distribution
        # and catastrophic on the balanced evaluation set.  The weighted variant
        # exists so a low balanced-set score can't be blamed on an untuned
        # baseline rather than on the architecture.
        class_w = None
        if args.class_weight == "inverse":
            counts = Counter(it.sense_l1 for it in train_items)
            k = len(L1_CLASSES)
            class_w = torch.tensor(
                [len(train_items) / (k * max(1, counts.get(c, 0))) for c in L1_CLASSES],
                dtype=torch.float32, device=device)
            print(f"  fold {fold} class weights {class_w.tolist()}", flush=True)

        steps_per_epoch = max(1, len(train_dl))
        total = int(steps_per_epoch * args.epochs)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        sched = get_linear_schedule_with_warmup(
            opt, int(args.warmup * total), max(1, total))

        model.train()
        step = 0
        epoch = 0
        while step < total:
            epoch += 1
            run_loss = 0.0
            seen = 0
            for batch in train_dl:
                if step >= total:
                    break
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(
                    logits, labels, weight=class_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                run_loss += float(loss) * len(labels)
                seen += len(labels)
                step += 1
            if seen:
                print(f"  fold {fold} epoch {epoch}: loss {run_loss / seen:.4f} "
                      f"({step}/{total} steps)", flush=True)

        model.eval()
        correct = total_n = 0
        with torch.no_grad():
            for batch in test_dl:
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                probs = torch.softmax(logits.float(), dim=-1).cpu()
                pred_ids = probs.argmax(dim=-1)
                correct += int((pred_ids == labels).sum())
                total_n += len(labels)
                # DataLoader preserves order, so the slice lines up with the items.
                start = total_n - len(labels)
                for j in range(len(labels)):
                    it = test_items[start + j]
                    pr = probs[j]
                    preds.append(Prediction(
                        item_id=it.id,
                        model=f"encoder:{tag}",
                        regime="supervised",
                        label_l1=L1_CLASSES[int(pred_ids[j])],
                        raw_output=json.dumps(
                            {c: round(float(pr[i]), 4) for i, c in enumerate(L1_CLASSES)}),
                        confidence=round(float(pr.max()), 4),
                        meta={"fold": fold,
                              "probs": {c: round(float(pr[i]), 4)
                                        for i, c in enumerate(L1_CLASSES)}},
                    ))
        acc = correct / max(1, total_n)
        print(f"fold {fold}: test acc {acc:.4f} n={total_n} "
              f"({time.time() - t0:.0f}s)", flush=True)
        fold_log.append({"fold": fold, "n_train": len(train_items), "n_test": total_n,
                         "test_accuracy": round(acc, 4),
                         "seconds": round(time.time() - t0, 1)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = save_predictions(ROOT / args.out, preds)
    print(f"wrote {n} predictions -> {args.out}", flush=True)

    # Score on the pooled out-of-fold set, then on the two shared evaluation
    # sets so the encoder's numbers sit next to the LLM's on identical items.
    by_id = {p.item_id: p.label_l1 for p in preds}
    pooled: list = []
    for fold in args.folds:
        pooled.extend(load_items(ROOT / args.test_template.format(k=fold)))

    report = {
        "model": args.model,
        "tag": tag,
        "device": device,
        "hyper": {"epochs": args.epochs, "batch_size": args.batch_size,
                  "lr": args.lr, "max_len": args.max_len,
                  "weight_decay": args.weight_decay, "seed": args.seed},
        "folds": fold_log,
        "oof": summarise(pooled, by_id),
        "eval_sets": {},
    }
    for name, path in (("hit_balanced", "data/processed/l1/hit_balanced.jsonl"),
                       ("hit_distributed", "data/processed/l1/hit_distributed.jsonl")):
        p = ROOT / path
        if p.exists():
            report["eval_sets"][name] = summarise(load_items(p), by_id)

    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("oof", "eval_sets")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
