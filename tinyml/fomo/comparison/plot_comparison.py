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
# Validated categorical slots 1-5 in bar order (blue, orange, aqua, yellow, magenta), adjacent pairs: CVD dE >= 9.1,
# normal-vision dE >= 19.6. Aqua, yellow and magenta are below 3:1 on the light surface, so every mark carries a label.
MODELS = (("M0", "m0-fomo-evaluation", "FOMO v2", "#2a78d6"),
          ("M4", "m4-ssdlite-evaluation", "SSDLite-MobileNetV2", "#eb6834"),
          ("M1", "m1-espdet-evaluation-1200", "ESPDet-Pico", "#1baf7a"),
          ("M3", "m3-yolo11n-evaluation", "YOLO11n", "#eda100"),
          ("M2", "m2-yolofastest-evaluation", "YOLO-FastestV2", "#e87ba4"))
# Scatter label offsets (points) where the default upper-right position collides with a neighbour.
LABEL_OFFSETS = {"M0": (8, -26), "M1": (12, -30)}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def serif():
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in ("Times New Roman", "Liberation Serif") if n in names), "serif")


def load(directory):
    report = json.loads((ART / directory / "report.json").read_text(encoding="utf-8"))
    return report["results_row"], report["selected"]


def bars(ax, groups, values, colors, fmt, ylabel):
    count = len(values)
    width = 0.8 / count
    for i, (model, series) in enumerate(values.items()):
        xs = [g + (i - (count - 1) / 2) * width for g in range(len(groups))]
        rects = ax.bar(xs, series, width - 0.02, color=colors[model], zorder=3)
        for rect, value in zip(rects, series):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015, fmt(value),
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


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
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "figures" / "comparison.png"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": serif(), "font.size": 10})
    rows = {m: load(folder) for m, folder, _, _ in MODELS}
    colors = {m: color for m, _, _, color in MODELS}
    labels = {m: f"{name} (threshold {rows[m][1]['threshold']:.2f})" for m, _, name, _ in MODELS}
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    bars(axes[0], ["Precision", "Recall", "F1"],
         {m: [rows[m][0][k] for k in ("precision", "recall", "f1")] for m in rows}, colors, pct, "Cityscapes validation")
    style(axes[0], "a. Detection accuracy")
    bars(axes[1], ["< 2 px", "2 to 8 px", ">= 8 px"],
         {m: [rows[m][1]["coarse_size_recall"][k]["recall"] for k in ("lt2", "2to8", "ge8")] for m in rows},
         colors, pct, "Recall")
    style(axes[1], "b. Recall by object size (160 x 160 input)")

    cost_keys = (("int8_bytes", "INT8 file"), ("params", "Parameters"), ("macs", "MACs"),
                 ("desktop_median_ms", "Desktop latency"))
    ax = axes[2]
    others = [m for m in rows if m != "M0"]
    for i, model in enumerate(others):
        offset = (i - (len(others) - 1) / 2) * 0.2
        for y, (key, _) in enumerate(cost_keys):
            value, base = rows[model][0][key], rows["M0"][0][key]
            if value is None:
                ax.text(1.08, y + offset, "n/a (simulator)", va="center", fontsize=6, color=MUTED)
                continue
            ratio = value / base
            ax.barh(y + offset, ratio, 0.18, color=colors[model], zorder=3)
            ax.text(ratio * 1.08, y + offset, f"{ratio:.1f}x", va="center", fontsize=7, color=INK)
    ax.axvline(1, color=colors["M0"], linewidth=2, zorder=4)
    ax.text(1.08, -0.72, "FOMO v2 = 1x", color=INK, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 150)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}x"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(range(len(cost_keys)), [name for _, name in cost_keys])
    ax.set_ylim(len(cost_keys) - 0.4, -1.0)
    ax.set_xlabel("Cost relative to FOMO v2 (log scale)", color=MUTED)
    style(ax, "c. Cost (lower is better)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)

    handles = [matplotlib.patches.Patch(color=colors[m], label=labels[m]) for m in rows]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, -0.03, "Same rebuilt Cityscapes data, INT8 models, shared centroid scorer and pre-registered threshold "
             "rule (0.20-0.95, precision >= 70%). Validation split also selected checkpoints and thresholds: not an "
             "independent test.\nLatency is desktop CPU (1 thread), not ESP32. ESPDet-Pico INT8 runs in the ESP-PPQ "
             "simulator, as do YOLO11n and YOLO-FastestV2. Extended sweep to 0.05: ESPDet-Pico F1 59.4% (0.10), "
             "YOLO11n F1 66.9% (0.05); M0, M2 and M4 unchanged.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(output)
    plt.close(fig)
    print(tradeoff(rows, colors, output.parent / "accuracy_vs_cost.png"))


def tradeoff(rows, colors, output):
    """F1 against compute: filled = pre-registered rule, hollow = extended-sweep sensitivity."""
    sensitivity = json.loads((ART / "threshold-sensitivity-final" / "sensitivity.json").read_text(encoding="utf-8"))
    names = {m: name for m, _, name, _ in MODELS}
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    for model, (row, _) in rows.items():
        x = row["macs"] / 1e6
        primary = row["f1"]
        extended = sensitivity["models"][model]["extended_0.05_0.95"]["f1"]
        if abs(extended - primary) > 1e-9:
            ax.plot([x, x], [primary, extended], color=colors[model], linewidth=1.5, linestyle=":", zorder=2)
            ax.scatter([x], [extended], s=70, facecolor=SURFACE, edgecolor=colors[model], linewidth=2, zorder=4)
        ax.scatter([x], [primary], s=90, color=colors[model], edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.annotate(f"{names[model]}\n{row['int8_bytes'] / 1000:,.0f} KB INT8", (x, max(primary, extended)),
                    xytext=LABEL_OFFSETS.get(model, (8, 6)), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(10, 400)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([10, 20, 50, 100, 200]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} M"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0.3, 0.75)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Compute per frame, MACs at 160 x 160 (log scale)", color=MUTED)
    ax.set_ylabel("F1, Cityscapes validation (INT8)", color=MUTED)
    style(ax, "Accuracy versus compute")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.scatter([], [], s=70, color=MUTED, label="Pre-registered rule (0.20-0.95)")
    ax.scatter([], [], s=60, facecolor=SURFACE, edgecolor=MUTED, linewidth=2, label="Sweep extended to 0.05")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.text(0.01, -0.02, "Upper left is better. ESP32-CAM flash, RAM and FPS gates are not yet measured.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return output


if __name__ == "__main__":
    main()
