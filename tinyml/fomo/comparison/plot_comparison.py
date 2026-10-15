"""Thesis figure: FOMO v2 (M0) versus candidates on accuracy, object size, and desktop cost."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import common  # noqa: F401,E402
from config import ROOT  # noqa: E402

ART = ROOT / "artifacts"
# Validated categorical slots 1-2 (blue, orange): CVD dE 24.7, normal-vision dE 33.6 on the light surface.
COLORS = {"M0": "#2a78d6", "M4": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def serif():
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in ("Times New Roman", "Liberation Serif") if n in names), "serif")


def load(directory):
    report = json.loads((ART / directory / "report.json").read_text(encoding="utf-8"))
    return report["results_row"], report["selected"]


def bars(ax, groups, values, labels, fmt, ylabel, ylim=None):
    width = 0.36
    for i, (model, series) in enumerate(values.items()):
        xs = [g + (i - 0.5) * (width + 0.02) for g in range(len(groups))]
        rects = ax.bar(xs, series, width, color=COLORS[model], label=labels[model], zorder=3)
        for rect, value in zip(rects, series):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + (ylim or 1) * 0.015, fmt(value),
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(ylabel, color=MUTED)
    if ylim:
        ax.set_ylim(0, ylim)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "figures" / "m0_vs_m4.png"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": serif(), "font.size": 10})
    rows = {"M0": load("m0-fomo-evaluation"), "M4": load("m4-ssdlite-evaluation")}
    labels = {"M0": f"FOMO v2 (threshold {rows['M0'][1]['threshold']:.2f})",
              "M4": f"SSDLite-MobileNetV2 (threshold {rows['M4'][1]['threshold']:.2f})"}
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    accuracy = {m: [rows[m][0][k] for k in ("precision", "recall", "f1")] for m in rows}
    bars(axes[0], ["Precision", "Recall", "F1"], accuracy, labels, pct, "Cityscapes validation", 1.0)
    style(axes[0], "a. Detection accuracy")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    sizes = {m: [rows[m][1]["coarse_size_recall"][k]["recall"] for k in ("lt2", "2to8", "ge8")] for m in rows}
    bars(axes[1], ["< 2 px", "2 to 8 px", ">= 8 px"], sizes, labels, pct, "Recall", 1.0)
    style(axes[1], "b. Recall by object size (160 x 160 input)")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    cost_keys = (("int8_bytes", "INT8 file"), ("params", "Parameters"), ("macs", "MACs"),
                 ("desktop_median_ms", "Desktop latency"))
    ratios = [rows["M4"][0][k] / rows["M0"][0][k] for k, _ in cost_keys]
    ax = axes[2]
    ys = range(len(cost_keys))
    ax.barh(list(ys), ratios, 0.5, color=COLORS["M4"], zorder=3)
    ax.axvline(1, color=COLORS["M0"], linewidth=2, zorder=4)
    ax.text(1.08, -0.62, "FOMO v2 = 1x", color=INK, fontsize=8, va="center")
    for y, value in zip(ys, ratios):
        ax.text(value * 1.08, y, f"{value:.1f}x", va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 80)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}x"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(list(ys), [name for _, name in cost_keys])
    ax.set_ylim(len(cost_keys) - 0.4, -0.95)
    ax.set_xlabel("SSDLite cost relative to FOMO v2 (log scale)", color=MUTED)
    style(ax, "c. Cost (lower is better)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)

    handles = [matplotlib.patches.Patch(color=COLORS[m], label=labels[m]) for m in rows]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, -0.02, "Same rebuilt Cityscapes data, INT8 models, shared centroid scorer and threshold rule. "
             "Validation split also selected checkpoints and thresholds: not an independent test. "
             "Latency is desktop CPU (1 thread), not ESP32.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(output)


if __name__ == "__main__":
    main()
