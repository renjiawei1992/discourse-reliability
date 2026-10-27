"""R161: code the model's L1 errors into five categories.

The coding below is a **single coder's judgement** (Claude, reading all 62
errors of qwen3.5-plus on hit_balanced).  It is not a substitute for the
two-annotator protocol; ``--sheet`` exports the same 62 items so a human can
re-code them and a kappa can be computed.

Category definitions, fixed before coding so the counts are not post hoc:

    C1 多义连接词义项误读  A cue is present and recognised, but the wrong sense
                          is assigned (同时 as conjunction; 继/又/并/原来/预计).
    C2 论元内部假连接词    The connective sits *inside* one argument and links
                          its own sub-clauses; the model treats it as the
                          Arg1--Arg2 marker.
    C3 时间框架归属歧义    "以前…现在" / "过去…今后" / "未来…现有" frames: the
                          temporal contrast can be labelled either way, and gold
                          and model pick opposite sides.
    C4 隐式项的类别边界    No cue; the gold falls on one side of a boundary the
                          model does not see (causal vs elaborative vs
                          comparative).
    C5 语料切分或标注存疑  Arguments are not two propositions, or the gold is
                          defensible only under HIT's own conventions.

    python scripts/error_typology.py \
        --preds outputs/l1_hit_balanced_qwen3.5-plus_zs_filled.jsonl \
        --gold data/processed/l1/hit_balanced.jsonl \
        --out outputs/l1_error_typology.json \
        --sheet data/interim/r161/typology_sheet.csv
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

from discourse_reliability.schema import load_items, load_predictions  # noqa: E402

CATS = {
    "C1": "多义连接词义项误读",
    "C2": "论元内部假连接词",
    "C3": "时间框架的归属歧义（时间 vs 对比）",
    "C4": "隐式项的类别边界",
    "C5": "语料切分或标注存疑",
}

# item suffix -> (category, one-line rationale)
CODING = {
    "bn/cbs_0013#:p3:4": ("C5", "Arg1 是名词短语而非命题，因果/时间无从判定"),
    "bn/cbs_0033#:p3:5": ("C1", "「继…之后」表序位对比，模型读成时间先后"),
    "bn/cnr_0133#:p3:7": ("C1", "「并」在否定句中表转折，模型读成并列"),
    "bn/ctv_0119:p2:3": ("C2", "「为了」在 Arg2 内部，与 Arg1 无关"),
    "bn/ctv_0145:p2:7": ("C4", "「据悉」引出后续安排，金标准为因果，线索极弱"),
    "bn/ctv_0145:p3:4": ("C4", "「前所未有的/最积极」是程度比较，无连接词"),
    "bn/vom_0311:p2:10": ("C4", "Arg2 解释高温成因，模型把解释读成因果"),
    "mz/chtb_1002#:p3:11": ("C3", "「现在…在四百年前」是时间框架，模型读成对照"),
    "mz/chtb_1002#:p3:7": ("C1", "「又」表经历接续，模型读成递进"),
    "mz/chtb_1007#:p2:54": ("C2", "「所以」在引语内部，不属于 Arg1--Arg2"),
    "mz/chtb_1007#:p2:63": ("C5", "段落是时间推进，金标准标为对比"),
    "mz/chtb_1007#:p3:30": ("C3", "「以往…如今」时间框架"),
    "mz/chtb_1007#:p3:70": ("C1", "「而是」表替换，模型读成对比"),
    "mz/chtb_1007#:p3:80": ("C3", "「以往…现在」时间框架"),
    "mz/chtb_1010#:p2:45": ("C4", "两句各自陈述，模型读进对比"),
    "mz/chtb_1010#:p3:21": ("C1", "「预计」表将来时间，模型读成推断因果"),
    "mz/chtb_1010#:p3:32": ("C1", "「原来」表先前时间，模型读成解释因果"),
    "mz/chtb_1010#:p3:51": ("C1", "「据说」是传闻标记，模型读成扩展"),
    "mz/chtb_1017#:p3:37": ("C5", "「否则」按 PDTB 属条件，金标准标为对比"),
    "mz/chtb_1017#:p3:9": ("C4", "「一来…其次」是论据列举，金标准为因果"),
    "mz/chtb_1021#:p2:80": ("C4", "「除了…唯一」隐含对比，模型读成扩展"),
    "mz/chtb_1021#:p3:46": ("C3", "「以前…现在」时间框架"),
    "mz/chtb_1028#:p2:80": ("C2", "「为此/所以」在引语内部"),
    "mz/chtb_1028#:p3:19": ("C1", "「同时」表同步，模型读成并列"),
    "mz/chtb_1035#:p2:111": ("C4", "论证文：后句是对前句的论据，金标准为因果"),
    "mz/chtb_1035#:p2:112": ("C4", "同上，论据关系无显式标记"),
    "mz/chtb_1035#:p2:115": ("C4", "同上，劝诫句作为前句的支撑"),
    "mz/chtb_1035#:p2:170": ("C4", "同上，结论句作为前句的推断"),
    "mz/chtb_1035#:p2:190": ("C3", "「过去…今后」时间框架"),
    "mz/chtb_1035#:p2:223": ("C2", "「假如…」在 Arg2 内部，是条件从句而非关系标记"),
    "mz/chtb_1035#:p3:15": ("C4", "论证文，后句展开前句的设定"),
    "mz/chtb_1035#:p3:60": ("C4", "「这代表」引出推论，金标准为因果"),
    "mz/chtb_1048#:p2:17": ("C4", "叙事推进，无显式时间标记"),
    "mz/chtb_1048#:p2:43": ("C4", "两段都在描写画风，实为并列展开"),
    "mz/chtb_1071#:p3:1": ("C1", "「也」表并列，金标准标为时间"),
    "nw/chtb_0049#:p3:1": ("C1", "「同时」表同步，模型读成并列"),
    "nw/chtb_0135#:p3:1": ("C1", "「同时」表同步，模型读成并列"),
    "nw/chtb_0143#:p2:12": ("C4", "段落以「同时」起，但两句是对照"),
    "wb/cmn_0006:p2:99": ("C1", "「同时」表同步，模型读成并列"),
    "wb/cmn_0006:p3:23": ("C4", "「但是…要想」隐含条件，金标准为扩展"),
    "wb/cmn_0006:p3:3": ("C4", "后句是前句的具体化，金标准为因果"),
    "wb/cmn_0009:p2:12": ("C4", "后句是前句的反面推论，模型读成因果"),
    "wb/cmn_0009:p2:15": ("C4", "「而是」引出更深层判断，金标准为因果"),
    "wb/cmn_0009:p2:73": ("C5", "Arg2 是「三、取消个税」这样的列表项"),
    "wb/cmn_0009:p2:83": ("C4", "两句是并列论据，模型读成对比"),
    "wb/cmn_0009:p3:16": ("C1", "「…的同时」表同步，模型读成并列"),
    "wb/cmn_0009:p3:7": ("C4", "后句是前句的推论，金标准为扩展"),
    "wb/cmn_0015:p2:11": ("C1", "「同时」表同步，模型读成并列"),
    "wb/cmn_0029:p2:27": ("C4", "叙事推进，无显式时间标记"),
    "wb/cmn_0029:p2:8": ("C4", "叙事推进，模型读成因果"),
    "wb/dev_09_cmn_0000:p3:3": ("C3", "「未来…现有」时间框架"),
    "wb/dev_09_cmn_0000:p3:7": ("C3", "同上，且与上一条重复"),
    "wb/dev_09_cmn_0039:p2:28": ("C4", "两个假设句并列，金标准为对比"),
    "wb/dev_09_cmn_0039:p2:81": ("C4", "后句是对前句话题的转折评论"),
    "wb/dev_09_cmn_0043:p2:32": ("C4", "后句举例，金标准为因果"),
    "wb/e2c_0001:p2:34": ("C4", "两句并置，模型读进因果"),
    "wb/e2c_0001:p2:53": ("C4", "两句并置，模型读进因果"),
    "wb/e2c_0001:p2:61": ("C4", "两句并置，模型读进因果"),
    "wb/e2c_0001:p2:67": ("C4", "两句并置，模型读进因果"),
    "wb/e2c_0001:p2:7": ("C4", "两类显示器并置，模型读成对比"),
    "wb/e2c_0001:p2:99": ("C4", "观察与主张并置，模型读进因果"),
    "wb/e2c_0006:p2:61": ("C4", "文本残缺（省略号），关系无法判定"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet")
    args = ap.parse_args(argv)

    items = {it.id: it for it in load_items(args.gold)}
    preds = [p for p in load_predictions(args.preds) if p.error is None]
    errors = [p for p in preds if p.label_l1 != items[p.item_id].sense_l1]

    rows, uncoded = [], []
    for p in errors:
        it = items[p.item_id]
        key = it.id.split("hitcdtb:", 1)[-1]
        cat, why = CODING.get(key, ("", ""))
        if not cat:
            uncoded.append(it.id)
        rows.append({
            "id": it.id,
            "gold": it.sense_l1,
            "pred": p.label_l1,
            "explicit": it.explicit,
            "connective": it.connective,
            "category": cat,
            "category_zh": CATS.get(cat, ""),
            "rationale": why,
            "arg1": it.arg1 or it.text,
            "arg2": it.arg2,
            "text": it.text,
        })

    counts = Counter(r["category"] for r in rows)
    by_dir = Counter(f"{r['gold']}->{r['pred']}" for r in rows)

    result = {
        "model": preds[0].model if preds else "",
        "n_items": len(preds),
        "n_errors": len(rows),
        "error_rate": round(len(rows) / len(preds), 4) if preds else None,
        "categories": {
            k: {
                "zh": v,
                "n": counts.get(k, 0),
                "share": round(counts.get(k, 0) / len(rows), 4) if rows else 0,
            }
            for k, v in CATS.items()
        },
        "by_direction": dict(by_dir.most_common()),
        "unmapped": uncoded,
        "coded": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.sheet:
        sheet = ROOT / args.sheet
        sheet.parent.mkdir(parents=True, exist_ok=True)
        with sheet.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "gold", "pred", "arg1", "arg2",
                        "coder1_category", "coder1_reason", "coder2_category", "note"])
            for r in rows:
                w.writerow([r["id"], r["gold"], r["pred"], r["arg1"], r["arg2"],
                            r["category"], r["rationale"], "", ""])

    print(json.dumps({
        "n_errors": result["n_errors"],
        "categories": {k: v["n"] for k, v in result["categories"].items()},
        "unmapped": len(uncoded),
        "sheet": args.sheet,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
