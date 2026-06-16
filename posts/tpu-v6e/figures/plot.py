#!/usr/bin/env python
"""Publication-quality figures for TPU v6e-4 benchmark blog.

Run: /tmp/plotenv/bin/python figures/plot.py
Generates fig1..fig7 as both .png (300dpi) and .pdf in figures/.

fig1  – prefill throughput vs context length
fig2  – decode p50 TPOT vs batch size (ctx 1024 solid, ctx 4096 dashed)
fig3  – decode aggregate throughput vs batch size (ctx 1024 solid, ctx 4096 dashed)
fig4  – E2E serving Pareto (output throughput vs p50 TPOT)
fig5  – summary bar chart normalized to dense-32B
fig7  – E2E latency sweep: TTFT and TPOT vs output throughput, two panels
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data.json")

with open(DATA) as f:
    D = json.load(f)
M = D["models"]

# ── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial"],
    "font.size":             10,
    "axes.labelsize":        11,
    "axes.titlesize":        10,
    "legend.fontsize":       9,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "axes.linewidth":        0.8,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.alpha":            0.22,
    "grid.linewidth":        0.5,
    "grid.color":            "0.78",
    "axes.axisbelow":        True,
    "xtick.direction":       "in",
    "ytick.direction":       "in",
    "xtick.major.size":      3.5,
    "ytick.major.size":      3.5,
    "xtick.major.width":     0.7,
    "ytick.major.width":     0.7,
    "figure.dpi":            120,
    "savefig.dpi":           300,
    "mathtext.fontset":      "stixsans",
})

# Per-model identity: Okabe–Ito colorblind-safe palette, consistent across all figs.
STYLE = {
    "Qwen3.5-4B":    {"color": "#0072B2", "marker": "o",
                      "label": "Qwen3.5-4B (dense, 4B)"},
    "Qwen3-30B-A3B": {"color": "#D55E00", "marker": "s",
                      "label": "Qwen3-30B-A3B (MoE, 3B active)"},
    "Qwen3-32B":     {"color": "#009E73", "marker": "^",
                      "label": "Qwen3-32B (dense, 32B)"},
}
ORDER = ["Qwen3.5-4B", "Qwen3-30B-A3B", "Qwen3-32B"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _ctx_style_legend(ax, loc="lower right"):
    """Small secondary legend for solid=ctx1024 / dashed=ctx4096."""
    handles = [
        Line2D([0], [0], color="0.35", linewidth=1.5, linestyle="-",  label="ctx 1024"),
        Line2D([0], [0], color="0.35", linewidth=1.5, linestyle="--", label="ctx 4096"),
    ]
    return ax.legend(handles=handles, frameon=False, loc=loc,
                     handlelength=1.8, fontsize=8.5)


# ── fig1: prefill throughput ──────────────────────────────────────────────────
def fig1():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ctxs = [512, 1024, 2048, 4096, 8192]

    for m in ORDER:
        s = STYLE[m]
        pts = {p["ctx"]: p["prefill_tok_s"] / 1000 for p in M[m]["prefill"]}
        ax.plot(ctxs, [pts[c] for c in ctxs],
                color=s["color"], marker=s["marker"],
                markersize=6, linewidth=1.8, label=s["label"])

    ax.set_xscale("log", base=2)
    ax.set_xticks(ctxs)
    ax.set_xticklabels([str(c) for c in ctxs])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Prefill throughput (k tok/s)  ↑")
    ax.set_ylim(0, 30)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)
    save(fig, "fig1_prefill")


# ── fig2: decode TPOT p50 vs batch size (ctx 1024 + 4096) ────────────────────
def fig2():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bss = [1, 4, 16, 64]
    model_lines = []

    for m in ORDER:
        s = STYLE[m]
        d1 = {p["bs"]: p["p50_tpot_ms"] for p in M[m]["decode"] if p["ctx"] == 1024}
        d4 = {p["bs"]: p["p50_tpot_ms"] for p in M[m]["decode"] if p["ctx"] == 4096}
        ln, = ax.plot(bss, [d1[b] for b in bss],
                      color=s["color"], marker=s["marker"],
                      markersize=6, linewidth=1.8, label=s["label"])
        ax.plot(bss, [d4[b] for b in bss],
                color=s["color"], marker=s["marker"],
                markersize=6, linewidth=1.8, linestyle="--", alpha=0.75)
        model_lines.append(ln)

    # Annotate BS=1 for MoE vs dense-32B contrast; white bbox keeps them readable
    for m in ["Qwen3-30B-A3B", "Qwen3-32B"]:
        val = next(p["p50_tpot_ms"] for p in M[m]["decode"]
                   if p["ctx"] == 1024 and p["bs"] == 1)
        ax.annotate(f"{val:.1f} ms", xy=(1, val),
                    xytext=(7, 0), textcoords="offset points",
                    fontsize=8.5, ha="left", va="center", color=STYLE[m]["color"],
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=1.2, alpha=0.88))

    ax.set_xscale("log", base=2)
    ax.set_xticks(bss)
    ax.set_xticklabels([str(b) for b in bss])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Decode p50 TPOT (ms)  ↓")
    ax.set_ylim(0, 108)

    leg1 = ax.legend(handles=model_lines, frameon=False, loc="upper left",
                     handlelength=1.5, fontsize=8.5)
    ax.add_artist(leg1)
    _ctx_style_legend(ax, loc="lower right")
    save(fig, "fig2_decode_tpot")


# ── fig3: aggregate decode throughput vs batch size (ctx 1024 + 4096) ─────────
def fig3():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bss = [1, 4, 16, 64]
    model_lines = []

    for m in ORDER:
        s = STYLE[m]
        t1 = {p["bs"]: p["bs"] * 1000.0 / p["p50_tpot_ms"]
              for p in M[m]["decode"] if p["ctx"] == 1024}
        t4 = {p["bs"]: p["bs"] * 1000.0 / p["p50_tpot_ms"]
              for p in M[m]["decode"] if p["ctx"] == 4096}
        ln, = ax.plot(bss, [t1[b] for b in bss],
                      color=s["color"], marker=s["marker"],
                      markersize=6, linewidth=1.8, label=s["label"])
        ax.plot(bss, [t4[b] for b in bss],
                color=s["color"], marker=s["marker"],
                markersize=6, linewidth=1.8, linestyle="--", alpha=0.75)
        model_lines.append(ln)

    ax.set_xscale("log", base=2)
    ax.set_xticks(bss)
    ax.set_xticklabels([str(b) for b in bss])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Decode throughput (tok/s)  ↑")
    ax.set_ylim(0, 1480)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(200))

    leg1 = ax.legend(handles=model_lines, frameon=False, loc="upper left",
                     handlelength=1.5, fontsize=8.5)
    ax.add_artist(leg1)
    _ctx_style_legend(ax, loc="lower right")
    save(fig, "fig3_decode_throughput")


# ── fig4: E2E Pareto (output throughput vs p50 TPOT) ─────────────────────────
def fig4():
    fig, ax = plt.subplots(figsize=(5.6, 3.6))

    for m in ORDER:
        s = STYLE[m]
        e = M[m]["e2e"]
        clean = sorted([p for p in e if p["rate"] != "inf"],
                       key=lambda p: float(p["rate"]))
        inf_p = next((p for p in e if p["rate"] == "inf"), None)

        if m == "Qwen3.5-4B":
            ax.plot([p["out_tok_s"] for p in clean],
                    [p["p50_tpot_ms"] for p in clean],
                    color=s["color"], marker=s["marker"],
                    markersize=6, linewidth=1.8, label=s["label"])
            ax.plot(inf_p["out_tok_s"], inf_p["p50_tpot_ms"],
                    marker="X", markersize=9, color=s["color"],
                    markeredgecolor="white", markeredgewidth=0.8,
                    linestyle="none", zorder=5)
            ax.annotate("concurrency\ncollapse",
                        xy=(inf_p["out_tok_s"], inf_p["p50_tpot_ms"]),
                        xytext=(inf_p["out_tok_s"] + 140, inf_p["p50_tpot_ms"] - 22),
                        fontsize=8, color=s["color"], style="italic", ha="left",
                        arrowprops=dict(arrowstyle="->", color=s["color"],
                                        lw=0.9, shrinkA=2, shrinkB=4))
            sat = clean[-1]
        else:
            all_pts = clean + [inf_p]
            ax.plot([p["out_tok_s"] for p in all_pts],
                    [p["p50_tpot_ms"] for p in all_pts],
                    color=s["color"], marker=s["marker"],
                    markersize=6, linewidth=1.8, label=s["label"])
            sat = inf_p

        if m == "Qwen3.5-4B":
            # White bbox lets us place above without clashing with nearby lines
            ax.annotate(f"{sat['out_tok_s']:.0f} tok/s",
                        xy=(sat["out_tok_s"], sat["p50_tpot_ms"]),
                        xytext=(0, 7), textcoords="offset points",
                        fontsize=8.5, color=s["color"], ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  pad=1.2, alpha=0.88))
        else:
            ax.annotate(f"{sat['out_tok_s']:.0f} tok/s",
                        xy=(sat["out_tok_s"], sat["p50_tpot_ms"]),
                        xytext=(0, 7), textcoords="offset points",
                        fontsize=8.5, color=s["color"], ha="center", va="bottom")

    ax.set_xlabel("Output throughput (tok/s)  →  higher is better")
    ax.set_ylabel("p50 TPOT (ms)  ↓  lower is better")
    ax.set_ylim(0, 155)
    ax.set_xlim(0, 1460)
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)
    save(fig, "fig4_e2e_pareto")


# ── fig5: summary bar chart normalized to dense-32B ──────────────────────────
def fig5():
    def tpot_bs1(m):
        return next(p["p50_tpot_ms"] for p in M[m]["decode"]
                    if p["ctx"] == 1024 and p["bs"] == 1)
    def sat_reqs(m):
        return max(p["req_s"] for p in M[m]["e2e"])
    def prefill_peak(m):
        return max(p["prefill_tok_s"] for p in M[m]["prefill"])

    base   = "Qwen3-32B"
    labels = ["Decode latency\n(speedup, bs=1)",
              "Serving capacity\n(req/s @ sat.)",
              "Prefill peak\n(tok/s)"]
    raw    = {m: {"tpot": tpot_bs1(m), "cap": sat_reqs(m), "pre": prefill_peak(m)}
              for m in ORDER}
    norm   = {m: [raw[base]["tpot"] / raw[m]["tpot"],
                  raw[m]["cap"]    / raw[base]["cap"],
                  raw[m]["pre"]    / raw[base]["pre"]]
              for m in ORDER}
    rawlbl = {m: [f"{raw[m]['tpot']:.1f} ms",
                  f"{raw[m]['cap']:.2f} req/s",
                  f"{raw[m]['pre']/1000:.1f}k tok/s"]
              for m in ORDER}

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x  = np.arange(len(labels))
    w  = 0.25
    dx = {"Qwen3.5-4B": -w, "Qwen3-30B-A3B": 0.0, "Qwen3-32B": w}

    for m in ORDER:
        s = STYLE[m]
        ax.bar(x + dx[m], norm[m], width=w, color=s["color"],
               label=s["label"], edgecolor="white", linewidth=0.5)
        for xi, (v, rl) in enumerate(zip(norm[m], rawlbl[m])):
            ax.text(xi + dx[m], v + 0.03, rl,
                    ha="center", va="bottom", fontsize=6.8,
                    rotation=90, color=s["color"])

    ax.axhline(1.0, color="0.3", linewidth=1.0, linestyle="--", zorder=0)
    # Label sits just above the top spine (outside the bar cluster entirely)
    ax.text(0.99, 1.01, "─── Qwen3-32B = 1.0",
            transform=ax.transAxes, fontsize=7.5, color="0.4",
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative to Qwen3-32B (dense)  ↑")
    ax.set_ylim(0, max(max(v) for v in norm.values()) * 1.28)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, fontsize=7.8, columnspacing=0.8,
              handlelength=1.2, handletextpad=0.4)
    save(fig, "fig5_summary")


# ── fig7: E2E serving sweep (TTFT + TPOT vs output throughput), two panels ───
def fig7():
    fig, (ax_ttft, ax_tpot) = plt.subplots(1, 2, figsize=(7.2, 3.5))
    YLIM_TTFT = 1350   # 4B collapse TTFT (6762 ms) is clipped

    for m in ORDER:
        s     = STYLE[m]
        e     = M[m]["e2e"]
        clean = sorted([p for p in e if p["rate"] != "inf"],
                       key=lambda p: float(p["rate"]))
        inf_p = next((p for p in e if p["rate"] == "inf"), None)
        xs_c  = [p["out_tok_s"] for p in clean]

        if m == "Qwen3.5-4B":
            ax_ttft.plot(xs_c, [p["p50_ttft_ms"] for p in clean],
                         color=s["color"], marker=s["marker"],
                         markersize=6, linewidth=1.8, label=s["label"])
            ax_tpot.plot(xs_c, [p["p50_tpot_ms"] for p in clean],
                         color=s["color"], marker=s["marker"],
                         markersize=6, linewidth=1.8, label=s["label"])
            # Collapse: clip TTFT, keep TPOT (133 ms fits in ylim=150)
            ax_ttft.plot(inf_p["out_tok_s"], YLIM_TTFT - 50,
                         marker="X", markersize=9, color=s["color"],
                         markeredgecolor="white", markeredgewidth=0.8,
                         linestyle="none", zorder=5)
            ax_tpot.plot(inf_p["out_tok_s"], inf_p["p50_tpot_ms"],
                         marker="X", markersize=9, color=s["color"],
                         markeredgecolor="white", markeredgewidth=0.8,
                         linestyle="none", zorder=5)
            # TTFT collapse is 6762 ms — well beyond the y-axis (clipped to YLIM_TTFT).
            # Annotate with an "off-scale" label inside the visible area.
            ax_ttft.annotate("TTFT = 6762 ms\n(off-scale, clipped)",
                             xy=(inf_p["out_tok_s"], YLIM_TTFT - 30),
                             xytext=(inf_p["out_tok_s"] + 220, YLIM_TTFT - 420),
                             fontsize=7.5, color=s["color"],
                             style="italic", ha="left",
                             arrowprops=dict(arrowstyle="->", color=s["color"],
                                             lw=0.8, shrinkA=2, shrinkB=4),
                             bbox=dict(facecolor="white", edgecolor="none",
                                       pad=1.5, alpha=0.9))
            ax_tpot.annotate("134 ms TPOT\n(collapse)",
                             xy=(inf_p["out_tok_s"], inf_p["p50_tpot_ms"]),
                             xytext=(inf_p["out_tok_s"] + 130, inf_p["p50_tpot_ms"] - 22),
                             fontsize=7.5, color=s["color"],
                             style="italic", ha="left",
                             arrowprops=dict(arrowstyle="->", color=s["color"],
                                             lw=0.8, shrinkA=2, shrinkB=4))
        else:
            all_pts = clean + [inf_p]
            xs_all  = [p["out_tok_s"] for p in all_pts]
            ax_ttft.plot(xs_all, [p["p50_ttft_ms"] for p in all_pts],
                         color=s["color"], marker=s["marker"],
                         markersize=6, linewidth=1.8, label=s["label"])
            ax_tpot.plot(xs_all, [p["p50_tpot_ms"] for p in all_pts],
                         color=s["color"], marker=s["marker"],
                         markersize=6, linewidth=1.8, label=s["label"])
            # Label saturation throughput
            ax_ttft.annotate(f"{inf_p['out_tok_s']:.0f}",
                             xy=(inf_p["out_tok_s"], inf_p["p50_ttft_ms"]),
                             xytext=(0, 7), textcoords="offset points",
                             fontsize=7.5, color=s["color"], ha="center", va="bottom")

    ax_ttft.set_xlabel("Output throughput (tok/s)")
    ax_ttft.set_ylabel("p50 TTFT (ms)  ↓")
    ax_ttft.set_xlim(0, 1460)
    ax_ttft.set_ylim(0, YLIM_TTFT)
    ax_ttft.yaxis.set_major_locator(mticker.MultipleLocator(200))
    ax_ttft.set_title("(a) Time to first token", fontsize=10, pad=5)

    ax_tpot.set_xlabel("Output throughput (tok/s)")
    ax_tpot.set_ylabel("p50 TPOT (ms)  ↓")
    ax_tpot.set_xlim(0, 1460)
    ax_tpot.set_ylim(0, 155)
    ax_tpot.set_title("(b) Time per output token", fontsize=10, pad=5)

    # Single shared legend below both panels — keeps both axes data-clear.
    handles = [Line2D([0], [0], color=STYLE[m]["color"], marker=STYLE[m]["marker"],
                      markersize=6, linewidth=1.8, label=STYLE[m]["label"])
               for m in ORDER]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False,
               fontsize=8.5, handlelength=1.5, handletextpad=0.4,
               columnspacing=1.0)

    fig.subplots_adjust(wspace=0.32, bottom=0.18)
    save(fig, "fig7_e2e_sweep")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig7()
    print("done")
