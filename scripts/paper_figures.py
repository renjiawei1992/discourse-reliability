"""Figures for §5 of the manuscript.

Every panel is computed from the same output files the text cites, except the
0.84 s per-item inference cost, which is a constant recorded in
``R152_PROTOCOL.md``.  The script prints a verification table so the numbers can
be diffed against ``L1_RESULTS.md`` each time the figures are rebuilt.

    python -I scripts/paper_figures.py --out ../refine-logs/figures

Writes PDF (for LaTeX) and PNG (for review) at 300 dpi.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discourse_reliability.schema import load_items, load_predictions  # noqa: E402

# ---------------------------------------------------------------- style

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

C_EXP = "#2f6f9f"
C_IMP = "#c4622d"
C_BASE = "#333333"
C_GREY = "#9a9a9a"
C_A1 = "#3b7a57"
C_A2 = "#8a5a9b"
C_OP = "#b03a2e"
C_KAPPA = "#7a3b8f"

MODELS_HIT = [
    ("qwen3.5-plus", "l1_hit_balanced_qwen3.5-plus_zs_filled.jsonl"),
    ("deepseek-reasoner", "l1_hit_balanced_reasoner_zs.jsonl"),
    ("glm-5.3-flash", "l1_hit_balanced_glm-5.3-flash_zs_filled.jsonl"),
    ("gpt-6-luna", "l1_hit_balanced_gpt-6-luna_zs.jsonl"),
    ("deepseek-chat", "l1_ds_zeroshot.jsonl"),
    ("gemini-3.1-flash-lite", "l1_hit_balanced_gemini-3.1-flash-lite-preview_zs.jsonl"),
    ("claude-haiku-5-5", "l1_hit_balanced_claude-haiku-5-5_zs_filled.jsonl"),
]

MODELS_TED = [
    ("qwen3.5-plus", "l1_ted_balanced_qwen3.5-plus_zs_filled.jsonl"),
    ("gpt-6-luna", "l1_ted_balanced_gpt-6-luna_zs_filled.jsonl"),
    ("deepseek-chat", "l1_ds_ted_zeroshot.jsonl"),
    ("deepseek-reasoner", "l1_ted_reasoner_zs.jsonl"),
]

SHORT = {
    "qwen3.5-plus": "qwen3.5-plus",
    "deepseek-reasoner": "deepseek-reasoner",
    "glm-5.3-flash": "glm-5.3-flash",
    "gpt-6-luna": "gpt-6-luna",
    "deepseek-chat": "deepseek-chat",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "claude-haiku-5-5": "claude-haiku-5-5",
}


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------- data

def stratified(items, preds_path: Path) -> dict:
    """Per-stratum accuracy for one model on one evaluation set."""
    items = {it.id: it for it in items}
    ok = 0
    exp = [0, 0]
    imp = [0, 0]
    for p in load_predictions(preds_path):
        it = items.get(p.item_id)
        if it is None or p.error is not None:
            continue
        gold = it.sense_l1
        pred = getattr(p, "label_l1") or ""
        if not gold or not pred:
            continue
        right = pred == gold
        ok += right
        bucket = exp if it.explicit else imp
        bucket[0] += right
        bucket[1] += 1
    return {
        "n": exp[1] + imp[1],
        "exp_acc": exp[0] / max(1, exp[1]),
        "imp_acc": imp[0] / max(1, imp[1]),
        "n_exp": exp[1],
        "n_imp": imp[1],
        "acc": ok / max(1, exp[1] + imp[1]),
    }


# ---------------------------------------------------------------- Fig 1

def fig_tiering(out: Path, hit, ted, checks) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(5.4, 4.5),
        gridspec_kw={"height_ratios": [7, 4], "hspace": 0.42})

    for ax, (title, items, specs) in zip(
        axes,
        [("(a) HIT-CDTB (news, 244 explicit / 156 implicit)",
          hit, MODELS_HIT),
         ("(b) TED-CDB (planned speech, 181 / 219)", ted, MODELS_TED)],
    ):
        rows = []
        for name, f in specs:
            s = stratified(items, ROOT / "outputs" / f)
            rows.append((name, s))
            checks.setdefault(title, {})[name] = s
        rows.sort(key=lambda r: r[1]["acc"])
        for i, (name, s) in enumerate(rows):
            ax.plot([s["imp_acc"], s["exp_acc"]], [i, i],
                    color="#c8c8c8", lw=2.0, zorder=1,
                    solid_capstyle="round")
            ax.scatter(s["exp_acc"], i, s=48, color=C_EXP, zorder=3,
                       edgecolor="white", linewidth=0.8)
            ax.scatter(s["imp_acc"], i, s=48, color=C_IMP, zorder=3,
                       edgecolor="white", linewidth=0.8)
            ax.text(s["exp_acc"] + 0.013, i, f"{s['exp_acc']:.2f}",
                    va="center", ha="left", fontsize=7.2, color=C_EXP)
            ax.text(s["imp_acc"] - 0.013, i, f"{s['imp_acc']:.2f}",
                    va="center", ha="right", fontsize=7.2, color=C_IMP)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([SHORT[n] for n, _ in rows], fontsize=8)
        ax.set_xlim(0.40, 1.00)
        ax.set_ylim(-0.65, len(rows) - 0.35)
        ax.set_xlabel("Accuracy, PDTB Level 1 (4 classes)")
        ax.set_title(title, loc="left", pad=6)
        ax.grid(axis="y", visible=False)

    fig.legend(
        handles=[Patch(color=C_EXP, label="explicit items"),
                 Patch(color=C_IMP, label="implicit items"),
                 plt.Line2D([], [], color="#c8c8c8", lw=2.0,
                            label="drop from explicit to implicit")],
        loc="upper center", ncol=3, fontsize=8,
        bbox_to_anchor=(0.5, 1.015), handlelength=1.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.91, bottom=0.10)
    save(fig, out, "fig1_tiering")


# ---------------------------------------------------------------- Fig 2

SCORER_LABEL = {
    "feature": "static descriptors (6)",
    "confidence": "5-sample agreement",
    "consensus": "cross-model disagreement",
    "self_assess": "self-assessment",
    "lr_base": "LR (8 features)",
    "lr_plus_self": "LR + self-assessment",
    "lr_all": "LR (all features)",
    "simple_mean": "equal-weight mean",
    "gbdt_all": "GBDT (all features)",
}


def fig_frontier(out: Path, tri: dict, metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9),
                             gridspec_kw={"width_ratios": [1.1, 1.0],
                                          "wspace": 0.72})

    ax = axes[0]
    budgets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    q = tri["quality_at_budget"]
    base = metrics["hit_balanced/chat_n5_t07"]["kappa"]
    series = [
        ("lr_base", "LR (8 features), our protocol", "#111111", 2.4, "-", "o"),
        ("self_assess", "self-assessment", "#2e8b57", 1.6, "-", "s"),
        ("consensus", "cross-model disagreement", "#1f6fb2", 1.6, "--", "^"),
        ("confidence", "5-sample agreement", "#e07b39", 1.6, ":", "D"),
        ("oracle", "oracle", "#b3b3b3", 1.2, (0, (4, 1.5)), "x"),
    ]
    for key, label, colour, lw, ls, mk in series:
        ys = [base] + [q[key][f"{b:.2f}"] for b in budgets[1:]]
        ax.plot(budgets, ys, color=colour, lw=lw, ls=ls, label=label,
                marker=mk, ms=3.4, mew=0.9,
                zorder=3 if key == "lr_base" else 2)
    ax.axvline(0.2, color=C_OP, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.scatter([0.2], [q["lr_base"]["0.20"]], s=42, color=C_OP, zorder=5,
               edgecolor="white", linewidth=0.8)
    ax.set_xlim(-0.01, 0.52)
    ax.set_ylim(0.71, 1.02)
    ax.set_xlabel("Fraction of items reviewed")
    ax.set_ylabel("κ after review")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    leg = ax.legend(loc="lower right", fontsize=6.6, handlelength=2.2,
                    borderaxespad=0.2, labelspacing=0.3)
    leg.set_frame_on(True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")

    # (a) title under the panel
    ax.text(0.0, -0.25, "(a) The cost–quality frontier", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2)

    ax = axes[1]
    auc = tri["auc"]
    order = ["feature", "confidence", "consensus", "self_assess",
             "lr_base", "lr_plus_self", "lr_all", "simple_mean", "gbdt_all"]
    # One hue per scorer: the four single signals are cool/warm, the learned
    # combiners run black -> teal -> purple -> brown -> red, so adjacent bars
    # never share a colour.
    BAR_COLOUR = {
        "feature": "#9a9a9a",      # static descriptors
        "confidence": "#e0a020",   # 5-sample agreement
        "consensus": "#1f6fb2",    # cross-model disagreement
        "self_assess": "#2e8b57",  # self-assessment
        "lr_base": "#111111",      # our protocol
        "lr_plus_self": "#2a8f8f",
        "lr_all": "#7a5aa8",
        "simple_mean": "#a05a2c",
        "gbdt_all": "#b03a2e",
    }
    # A distinct hatch on top of the hue, so the bars stay separable in
    # greyscale and for readers who cannot tell the hues apart.
    # No horizontal-line patterns: the bars are short and a horizontal hatch
    # collapses into the edge itself.
    BAR_HATCH = {
        "feature": "//",
        "confidence": "\\\\",
        "consensus": "|",
        "self_assess": "..",
        "lr_base": "xx",
        "lr_plus_self": "oo",
        "lr_all": "*",
        "simple_mean": "//\\\\",
        "gbdt_all": "O",
    }
    with plt.rc_context({"hatch.linewidth": 0.55}):
        for i, key in enumerate(order):
            v = auc[key]
            ax.barh(i, v, height=0.72, color=BAR_COLOUR[key], alpha=0.92,
                    edgecolor="white", linewidth=0.6,
                    hatch=BAR_HATCH[key])
            ax.text(v + 0.007, i, f"{v:.3f}", va="center", fontsize=7.2)
    ax.axvline(0.5, color="#bbbbbb", lw=0.8, ls=":")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SCORER_LABEL[k] for k in order], fontsize=7.4)
    ax.set_xlim(0.45, 0.94)
    ax.set_xlabel("Error-detection AUC")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.text(0.0, -0.25, "(b) What each signal knows", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2)
    save(fig, out, "fig3_frontier")


# ---------------------------------------------------------------- Fig 3

def fig_deletion(out: Path, r140: dict, m4: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.3, 1.0],
                                          "wspace": 0.42})

    new_pt = r140["paired_new_token_delete"]["delta"] * 100
    old_pt = r140["paired_old_span_delete"]["delta"] * 100
    ax = axes[0]
    rows = [
        ("model · paired frame, rebuilt", new_pt, C_EXP),
        ("annotator 2 · single token", -m4["A2"]["drop_single"] * 100, C_A2),
        ("annotator 1 · single token", -m4["A1"]["drop_single"] * 100, C_A1),
        ("model · single token", r140["single"]["delta"] * 100, C_EXP),
        ("model · paired frame, original", old_pt, C_OP),
    ]
    for i, (label, v, colour) in enumerate(rows):
        ax.barh(i, v, height=0.58, color=colour, alpha=0.9,
                edgecolor="white", linewidth=0.6)
        ax.text(v - 1.4, i, f"{v:.1f}", va="center", ha="right",
                fontsize=7.4, color=colour)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    ax.set_xlim(-64, 4)
    ax.set_xlabel("Δ accuracy after deleting the connective (points)")
    ax.set_title("(a) The deletion effect, decomposed", loc="left", pad=6)
    ax.grid(axis="y", visible=False)
    ax.annotate("", xy=(new_pt, 4.46), xytext=(old_pt, 4.46),
                arrowprops=dict(arrowstyle="<->", color=C_OP, lw=0.8,
                                shrinkA=0, shrinkB=0))
    ax.text((new_pt + old_pt) / 2, 4.60,
            f"{abs(old_pt - new_pt):.0f} of these points were deleted content",
            ha="center", fontsize=6.9, color=C_OP)
    ax.set_ylim(-0.6, 5.05)
    ax.legend(handles=[Patch(color=C_EXP, label="model"),
                       Patch(color=C_A1, label="annotator 1"),
                       Patch(color=C_A2, label="annotator 2"),
                       Patch(color=C_OP, label="defective construction")],
              loc="lower left", fontsize=6.6, handlelength=1.0,
              borderaxespad=0.3, labelspacing=0.25)

    ax = axes[1]
    groups = ["single token\n(n = 86)", "paired frame\noriginal (14)",
              "paired frame\nrebuilt (14)"]
    a1 = [m4["A1"]["read_single"], m4["A1"]["read_paired"],
          m4["A1"]["m4b"]["read_new"]]
    a2 = [m4["A2"]["read_single"], m4["A2"]["read_paired"],
          m4["A2"]["m4b"]["read_new"]]
    x = np.arange(len(groups))
    w = 0.34
    ax.bar(x - w / 2, a1, w, color=C_A1, alpha=0.9, label="annotator 1",
           edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, a2, w, color=C_A2, alpha=0.9, label="annotator 2",
           edgecolor="white", linewidth=0.6)
    for xs, vals in ((x - w / 2, a1), (x + w / 2, a2)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.022, f"{v:.2f}", ha="center", fontsize=7.0)
    ax.axhline(0.90, color="#bbbbbb", lw=0.9, ls=":")
    ax.text(0.70, 0.925, "pre-registered\ngate: 0.90", ha="left", va="top",
            fontsize=6.4, color="#777777", linespacing=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=7.2)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.24)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Share of items a human can judge")
    ax.set_title("(b) Readability of the mutated items", loc="left", pad=6)
    ax.legend(loc="upper center", fontsize=7.0, handlelength=1.0,
              ncol=2, bbox_to_anchor=(0.5, 1.02), columnspacing=1.2)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig2_deletion")


# ---------------------------------------------------------------- Fig 4

def fig_time(out: Path, timing: dict, tri: dict, metrics: dict) -> None:
    """Deployment arithmetic for 400 items (the manuscript's Table 6).

    Per-item times come from the R152 timed study; the 0.84 s inference cost is
    the measured mean over 100 items recorded in ``R152_PROTOCOL.md``.  The
    right-hand axis carries the kappa column of the old Table 6, taken from the
    triage curve (Fig. 3), so the figure replaces the table.
    """
    INFER_S = 0.84
    n = 400
    pooled = timing["cross"]["pooled"]
    scratch_s = pooled["b"]["total_a_s"] / pooled["b"]["n_items"]
    b_s = pooled["b"]["total_cond_s"] / pooled["b"]["n_items"]
    review_s = timing["cross"]["review_vs_scratch_pooled"]["b_mean_s"]
    manual_min = n * scratch_s / 60

    # kappa after review: the base rate at 0% and the LR protocol's curve.
    q = tri["quality_at_budget"]["lr_base"]
    base = metrics["hit_balanced/chat_n5_t07"]["kappa"]
    kappa = [base, q["0.10"], q["0.20"], q["0.30"]]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    budgets = ["0%", "10%", "20%", "30%", "100%"]
    review = [0.0,
              n * 0.1 * review_s / 60,
              n * 0.2 * review_s / 60,
              n * 0.3 * review_s / 60,
              n * b_s / 60]
    inference = [n * INFER_S / 60] * 5
    x = np.arange(len(budgets))
    ax.bar(x, inference, 0.55, color=C_GREY, label="model inference",
           edgecolor="white", linewidth=0.6)
    ax.bar(x, review, 0.55, bottom=inference, color=C_EXP,
           label="human review", edgecolor="white", linewidth=0.6)
    totals = [i + r for i, r in zip(inference, review)]
    labels = [f"{(t / manual_min - 1) * 100:+.0f}%".replace("-", "−")
              for t in totals[:4]] + [f"{(totals[4] / manual_min - 1) * 100:+.0f}%"]
    for xi, t, lab in zip(x, totals, labels):
        ax.text(xi, t + 2.2, f"{t:.1f} min", ha="center", fontsize=7.2)
        ax.text(xi, t + 7.0, lab, ha="center", fontsize=7.2,
                color=C_OP if "+" in lab else "#555555",
                fontweight="bold" if "+" in lab else "normal")
    ax.axhline(manual_min, color=C_OP, lw=1.0, ls="--")
    ax.text(1.0, manual_min + 1.8,
            f"full manual pass, {manual_min:.1f} min", ha="center",
            fontsize=6.9, color=C_OP)
    ticks = ["%s\n%d items\nκ %.3f" % (b, n * k, kv)
             for b, k, kv in zip(budgets[:4], (0, .1, .2, .3), kappa)]
    ticks.append("100%%\n%d items\nκ —" % n)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=7.4)
    ax.set_xlabel("Review budget")
    ax.set_ylabel("Minutes for a 400-item set")
    ax.set_ylim(0, 96)
    ax.set_xlim(-0.6, 4.6)
    ax.grid(axis="x", visible=False)

    # kappa after review, right-hand axis (100% is not plotted: every item is
    # human-labelled there, so the curve has no value to report)
    ax2 = ax.twinx()
    ax2.plot(x[:4], kappa, color=C_KAPPA, lw=1.2, marker="s", ms=4.0,
             mew=0.8, mec="white", zorder=6, label="κ after review")
    ax2.set_ylim(0.65, 1.03)
    ax2.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax2.set_ylabel("κ after review", color=C_KAPPA)
    ax2.tick_params(axis="y", colors=C_KAPPA, labelsize=7.4)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_KAPPA)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    handles, labels_ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=C_KAPPA, lw=1.2, marker="s", ms=4.0,
                          mec="white"))
    labels_.append("κ after review")
    ax.legend(handles, labels_, loc="upper left", fontsize=6.7,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.3)
    save(fig, out, "fig4_time")


# ---------------------------------------------------------------- Fig 5

def fig_errors(out: Path, typology: dict, validity: dict,
               r161: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0),
                             gridspec_kw={"width_ratios": [1.35, 1.0],
                                          "wspace": 0.52})

    ax = axes[0]
    c1 = {k: v["n"] for k, v in typology["categories"].items()}
    c2 = {k: v["coder2"] for k, v in r161["counts"].items()}
    adj = {k: v["adjudicated"] for k, v in r161["counts"].items()}
    names = {
        "C4": "C4  implicit class boundary",
        "C1": "C1  polysemous connective",
        "C5": "C5  corpus label in doubt",
        "C3": "C3  paired temporal frame",
        "C2": "C2  connective inside an argument",
    }
    order = ["C4", "C1", "C5", "C3", "C2"]
    # Three separate hues rather than two opacities of one blue: at this size
    # coder 1 and coder 2 read as the same colour otherwise.
    C_C1, C_C2 = "#8fb8d6", "#8fc06a"
    h = 0.26
    for i, k in enumerate(order):
        for off, vals, colour, alpha, lab in (
            (-h, c1, C_C1, 0.95, "coder 1"),
            (0.0, c2, C_C2, 0.95, "coder 2"),
            (+h, adj, C_OP, 0.95, "adjudicated"),
        ):
            v = vals[k]
            ax.barh(i + off, v, height=h * 0.92, color=colour, alpha=alpha,
                    edgecolor="white", linewidth=0.5, label=lab if i == 0 else None)
        # clear of the longest of the three bars, not just the adjudicated one
        ax.text(max(c1[k], c2[k], adj[k]) + 1.0, i,
                f"{adj[k]}  ({adj[k] / 62:.0%})",
                va="center", fontsize=7.2, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in order], fontsize=7.4)
    ax.set_xlim(0, 45)
    ax.set_xlabel("Errors (of 62)")
    ax.text(0.0, -0.25, "(a) What the errors are", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", fontsize=6.8,
              handlelength=1.0, borderaxespad=0.3)
    ax.annotate(f"adjudication moved\nC5 from {c1['C5']} to {adj['C5']}",
                xy=(adj["C5"] + 0.4, 2 + h), xytext=(17.0, 2.45),
                fontsize=6.7,
                color=C_OP, va="center",
                arrowprops=dict(arrowstyle="->", color=C_OP, lw=0.7))

    ax = axes[1]
    models = [("qwen3.5-plus", "chatanywhere:qwen3.5-plus"),
              ("deepseek-\nreasoner", "deepseek:deepseek-reasoner")]
    x = np.arange(len(models))
    w = 0.33
    dis = [validity["models"][m]["contingency"]["human_disagree"]["model_error_rate"]
           for _, m in models]
    agr = [validity["models"][m]["contingency"]["human_agree"]["model_error_rate"]
           for _, m in models]
    rr = [validity["models"][m]["risk_ratio"] for _, m in models]
    ax.bar(x - w / 2, dis, w, color=C_OP, alpha=0.9,
           label="annotators disagree", edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, agr, w, color=C_EXP, alpha=0.9,
           label="annotators agree", edgecolor="white", linewidth=0.6)
    for xi, v in zip(x - w / 2, dis):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.2)
    for xi, v in zip(x + w / 2, agr):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.2)
    for xi, v in zip(x, rr):
        ax.text(xi, 0.385, f"RR = {v:.2f}", ha="center", fontsize=7.6,
                color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in models], fontsize=7.6)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 0.52)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylabel("Model error rate")
    ax.text(0.0, -0.25, "(b) Errors track human disagreement",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.2)
    ax.legend(loc="upper left", fontsize=6.9, handlelength=1.0,
              borderaxespad=0.3)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig5_errors")


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="refine-logs/figures")
    args = ap.parse_args(argv)
    out = (ROOT / args.out).resolve()

    hit = load_items(ROOT / "data/processed/l1/hit_balanced.jsonl")
    ted = load_items(ROOT / "data/processed/l1/ted_balanced.jsonl")
    load = lambda name: json.loads((ROOT / "outputs" / name).read_text("utf-8"))
    tri = load("l1_triage_overbuilt.json")
    typ = load("l1_error_typology.json")
    val = load("l1_validity.json")
    r140 = load("l1_r140_final.json")
    m4 = load("l1_m4_two_annotators.json")
    r152 = load("l1_r152_timing.json")
    r161 = load("l1_r161_kappa.json")
    metrics = load("l1_metrics_summary.json")

    checks: dict = {}
    print("figure 1 -- stratified accuracy recomputed from predictions")
    fig_tiering(out, hit, ted, checks)
    print("figure 3 -- triage frontier")
    fig_frontier(out, tri, metrics)
    print("figure 2 -- deletion decomposition")
    fig_deletion(out, r140, m4)
    print("figure 4 -- time budget")
    fig_time(out, r152, tri, metrics)
    print("figure 5 -- error typology and validity")
    fig_errors(out, typ, val, r161)

    print("\nrecomputed strata (compare with L1_RESULTS section 2):")
    for panel, rows in checks.items():
        print(f"  {panel}")
        for name, s in rows.items():
            print(f"    {name:24s} n={s['n']:4d} "
                  f"exp={s['exp_acc']:.4f} ({s['n_exp']}) "
                  f"imp={s['imp_acc']:.4f} ({s['n_imp']}) "
                  f"all={s['acc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
