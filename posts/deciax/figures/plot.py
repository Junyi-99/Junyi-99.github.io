"""Rebuild the blog figures from the reviewed report extracts.

Run with Python and matplotlib. Values are in ../benchmark-data.json.
The source report hash and measurement conditions are recorded there.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT.parent / "benchmark-data.json").read_text())
COLORS = {"gpu": "#64748b", "jax": "#b7791f", "pallas": "#2563eb"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def plot(questions):
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 8.3), layout="constrained")
    domain = max(r["p95_ms"] for r in DATA["latency"]
                 if r["questions"] == questions
                 and (not r["device"].startswith("TPU") or r["backend"] == "pallas")) * 1.4
    for ax, model in zip(axes, ["english", "multilingual"]):
        rows = [r for r in DATA["latency"]
                if r["model"] == model and r["questions"] == questions
                and (not r["device"].startswith("TPU") or r["backend"] == "pallas")]
        labels = [r["device"] + (" / " + r["backend"]
                  if r["device"].startswith("TPU") else "") for r in rows]
        values = [r["p50_ms"] for r in rows]
        colors = [COLORS[r["backend"]] if r["device"].startswith("TPU")
                  else COLORS["gpu"] for r in rows]
        ax.barh(labels, values, height=.63, color=colors)
        ax.invert_yaxis()
        tails = [r["p95_ms"] for r in rows]
        ax.set_xlim(0, domain)
        ax.set_xlabel("In-process p50 per call (ms) · lower is better")
        ax.set_title(model.capitalize(), loc="left", fontweight="bold")
        ax.xaxis.grid(True, color="#e2e8f0", linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#94a3b8")
        percentiles(ax, values, tails)
    fig.suptitle(f"Optimized inference · {questions} question{'s' if questions != 1 else ''} per call",
                 fontsize=17, fontweight="bold")
    fig.legend(handles=[Patch(color=COLORS[k], label=label)
                        for k, label in [("gpu", "GPU: fastest measured variant"),
                                         ("pallas", "deciax: Pallas")]] +
               [Line2D([], [], color="#0f172a", linestyle="--", label="p95")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    save(fig, f"latency-{questions}q")


def save(fig, name):
    fig.savefig(ROOT / f"{name}.png", dpi=160, facecolor="white")
    svg = ROOT / f"{name}.svg"
    fig.savefig(svg, facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def style(ax):
    ax.xaxis.grid(True, color="#e2e8f0", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")


def percentiles(ax, medians, tails):
    margin = ax.get_xlim()[1] * .013
    for i, (median, tail) in enumerate(zip(medians, tails)):
        ax.hlines(i, median, tail, color="#0f172a", linestyle="--", linewidth=1.2)
        ax.vlines(tail, i - .38, i + .38, color="#0f172a", linestyle="--", linewidth=1.2)
        ax.text(tail + margin, i, f"{median:.2f} / {tail:.2f}", va="center", fontsize=9)


def baseline():
    fig, axes = plt.subplots(4, 1, figsize=(9.6, 13.5), layout="constrained")
    for ax, (model, questions) in zip(axes, [("english", 1), ("multilingual", 1),
                                           ("english", 50), ("multilingual", 50)]):
        rows = [r for r in DATA["baseline_latency"]
                if r["model"] == model and r["questions"] == questions]
        labels = [r["device"] + (" / JAX" if r["backend"] == "jax" else "") for r in rows]
        values = [r["p50_ms"] for r in rows]
        colors = [COLORS["jax"] if r["backend"] == "jax" else COLORS["gpu"] for r in rows]
        ax.barh(labels, values, height=.65, color=colors)
        ax.invert_yaxis()
        limit = 37 if questions == 1 else 300
        ax.set_xlim(0, limit)
        ax.set_xlabel("In-process p50 per call (ms) · lower is better")
        ax.set_title(f"{model.capitalize()} · {questions} question{'s' if questions != 1 else ''}",
                     loc="left", fontweight="bold")
        percentiles(ax, values, [r["p95_ms"] for r in rows])
        style(ax)
    fig.suptitle("Baseline · PyTorch eager and plain JAX", fontsize=17, fontweight="bold")
    fig.legend(handles=[Patch(color=COLORS[k], label=label) for k, label in
                        [("gpu", "GPU: PyTorch eager"), ("jax", "TPU: plain JAX")]] +
               [Line2D([], [], color="#0f172a", linestyle="--", label="p95")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    save(fig, "baseline-latency")


def decomposition():
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 8.3), layout="constrained")
    for ax, questions in zip(axes, [1, 50]):
        rows = [r for r in DATA["baseline_decomposition"]
                if r["model"] == "english" and r["questions"] == questions]
        labels = [r["device"] + (" / JAX" if r["backend"] == "jax" else " / eager")
                  for r in rows]
        busy = [r["profiled_busy_ms"] for r in rows]
        remainder = [r["remainder_estimate_ms"] for r in rows]
        wall = [r["unprofiled_p50_ms"] for r in rows]
        ax.barh(labels, busy, height=.65, color="#2563eb")
        ax.barh(labels, remainder, left=busy, height=.65, color="#e2e8f0",
                edgecolor="#94a3b8", linewidth=.4, hatch="///")
        limit = max(wall) * 1.18
        ax.set_xlim(0, limit)
        ax.invert_yaxis()
        ax.set_xlabel("Time per call (ms) · approximate decomposition")
        ax.set_title(f"English · {questions} question{'s' if questions != 1 else ''}",
                     loc="left", fontweight="bold")
        for i, value in enumerate(wall):
            ax.text(value + limit * .012, i, f"{value:.2f}", va="center", fontsize=10)
        style(ax)
    fig.suptitle("Baseline time breakdown · English", fontsize=17, fontweight="bold")
    fig.legend(handles=[Patch(color="#2563eb", label="Profiled device busy estimate"),
                        Patch(facecolor="#e2e8f0", edgecolor="#94a3b8", hatch="///",
                              label="Remaining wall-time estimate")],
               loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    save(fig, "baseline-breakdown")


def curve():
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 8.3), layout="constrained")
    series = [("H100 PCIe*", None, "H100 PCIe", "gpu", "o", "-"),
              ("H200 NVL*", None, "H200 NVL", "jax", "s", "--"),
              ("TPU v6e", "pallas", "TPU v6e: Pallas", "pallas", "^", "-")]
    for ax, model in zip(axes, ["english", "multilingual"]):
        for device, backend, label, color, marker, line in series:
            rows = sorted([r for r in DATA["latency_curve"]
                           if r["model"] == model and r["device"] == device
                           and (backend is None or r["backend"] == backend)],
                          key=lambda r: r["questions"])
            assert [r["questions"] for r in rows] == [1, 5, 10, 50]
            xs = [r["questions"] for r in rows]
            ys = [r["p50_ms"] for r in rows]
            ax.plot(xs, ys, label=label, color=COLORS[color], marker=marker,
                    linestyle=line, linewidth=1.8, markersize=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 5, 10, 50], labels=["1", "5", "10", "50"])
        ax.set_yticks([1, 2, 5, 10, 20, 50],
                     labels=["1", "2", "5", "10", "20", "50"])
        ax.set_ylim(.9, 55)
        ax.set_xlim(.85, 60)
        ax.set_xlabel("Questions per call (log scale)")
        ax.set_ylabel("In-process p50 (ms, log scale)")
        ax.set_title(model.capitalize(), loc="left", fontweight="bold")
        ax.grid(True, which="major", color="#e2e8f0", linewidth=.7)
        ax.minorticks_off()
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#94a3b8")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False,
               fontsize=9)
    fig.suptitle("Latency vs questions per call", fontsize=17, fontweight="bold")
    save(fig, "latency-curve")


def cost():
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.8), layout="constrained")
    names = {"GeForce RTX 3090": "RTX 3090", "A100-SXM4-80GB": "A100 SXM4*",
             "RTX A6000": "RTX A6000", "H100 PCIe": "H100 PCIe*",
             "H200 NVL": "H200 NVL*", "TPU v6 lite": "TPU v6e / Pallas"}
    for ax, questions in zip(axes, [1, 50]):
        rows = [r for r in DATA["cost"]["records"] if r["questions"] == questions
                and r["host"] in DATA["cost"]["chart_hosts"]]
        domain = max(r[field] for r in rows for field in
                     ["on_demand_million_tokens_per_usd", "preemptible_million_tokens_per_usd"]) * 1.22
        positions = list(range(len(rows)))
        labels = [names[r["device"]] for r in rows]
        for offset, field, color in [(-.18, "on_demand_million_tokens_per_usd", "#2563eb"),
                                     (.18, "preemptible_million_tokens_per_usd", "#b7791f")]:
            values = [r[field] for r in rows]
            bars = ax.barh([p + offset for p in positions], values, height=.31, color=color)
            for bar, value in zip(bars, values):
                ax.text(value + domain * .015,
                        bar.get_y() + bar.get_height()/2, f"{value:,.1f}",
                        va="center", fontsize=9)
        ax.set_yticks(positions, labels=labels)
        ax.invert_yaxis()
        ax.set_xlim(0, domain)
        ax.set_xlabel("Million model input tokens / US dollar · higher is better")
        ax.set_title(f"English · {questions} question{'s' if questions != 1 else ''}",
                     loc="left", fontweight="bold")
        style(ax)
    fig.suptitle("Million model input tokens per US dollar", fontsize=16, fontweight="bold")
    fig.legend(handles=[Patch(color="#2563eb", label="On-demand"),
                        Patch(color="#b7791f", label="Preemptible / spot")],
               loc="outside lower center", ncol=2, frameon=False)
    save(fig, "cost-english")


if __name__ == "__main__":
    baseline()
    decomposition()
    plot(1)
    plot(50)
    curve()
    cost()
