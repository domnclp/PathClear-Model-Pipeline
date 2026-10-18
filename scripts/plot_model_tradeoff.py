"""Render the thesis tradeoff figure from evaluate_model_tradeoff.py output.

Three small-multiple panels (mIoU, pixel accuracy, road-as-sidewalk error
rate), one row per model, each a point estimate with its 95% bootstrap
interval. Each panel keeps its own x-axis because the metrics have different
ranges; models keep the same row and color in every panel. Writes PNG and PDF.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette (all-pairs safe).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PANELS = (
    ("miou", "mIoU (%)", "higher is better"),
    ("pixel_accuracy", "Pixel accuracy (%)", "higher is better"),
    ("road_as_sidewalk", "Road-as-sidewalk error (%)", "lower is better"),
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--labels", nargs="*", default=[], metavar="NAME=LABEL",
                        help="Display label per model name.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path without extension.")
    return parser.parse_args()


def main():
    args = _arguments()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    labels = dict(spec.split("=", 1) for spec in args.labels)
    names = [model["name"] for model in report["models"]]
    if len(names) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} models fit the validated palette.")
    results = report["cityscapes"]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(10, 0.55 * len(names) + 1.5), sharey=True)
    rows = list(range(len(names)))[::-1]
    for ax, (key, title, direction) in zip(axes, PANELS):
        for row, name, color in zip(rows, names, SERIES):
            value = 100 * results[name][key]
            low, high = (100 * bound for bound in results[name]["ci95"][key])
            ax.plot([low, high], [row, row], color=color, linewidth=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(value, row, "o", markersize=8, color=color,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            ax.annotate(f"{value:.2f}", (high, row), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=INK)
        ax.set_title(f"{title}\n{direction}", fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + 0.18 * (right - left))
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([labels.get(name, name) for name in names])
    axes[0].set_ylim(-0.6, len(names) - 0.4)
    protocol = report["protocol"]
    fig.text(0.01, 0.01,
             f"{protocol['dataset']}, {protocol['frames']} frames; Road / Sidewalk / "
             f"Boundary pixels on a {protocol['evaluation_grid'][0]}x"
             f"{protocol['evaluation_grid'][1]} grid. Lines: 95% bootstrap intervals.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=200, facecolor="white")
    print(f"Saved {args.output.with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main()
