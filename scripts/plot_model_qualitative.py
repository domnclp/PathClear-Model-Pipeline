"""Qualitative thesis figures: MicroSeg vs Fast-SCNN predictions side by side.

Uses the same shared label space and mapping as evaluate_model_tradeoff.py
(Road, Sidewalk, Boundary, Crosswalk, Other) so the pictures match the
reported metrics. Two figure types:

- Cityscapes val frames (--cityscapes N, spaced evenly through the split, not
  hand-picked): input, shared-class ground truth, each model's prediction,
  and each model's error map (red = road predicted as sidewalk, the
  safety-critical error; blue = any other error on a scored pixel).
- Unlabelled photos (--images ...): input and each model's prediction.

Writes PNG and PDF. Read-only with respect to the checkpoints.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Patch

from evaluate_model_tradeoff import (
    GT_LOOKUP, IGNORE, ROAD, SIDEWALK, Model, _cityscapes_pairs, _named, _nearest,
)
from pathclear_pipeline import sha256

EVAL_HEIGHT, EVAL_WIDTH = 512, 1024
CLASS_COLORS = {  # PathClear class colors, so figures match earlier panels
    "Road": (128, 64, 128), "Sidewalk": (244, 35, 232), "Boundary": (130, 110, 80),
    "Crosswalk": (255, 255, 100), "Other": (170, 170, 170),
}
LOOKUP = np.zeros((256, 3), np.uint8)  # 255 (not scored) stays black
for index, color in enumerate(CLASS_COLORS.values()):
    LOOKUP[index] = color
ROAD_AS_SIDEWALK, OTHER_ERROR = (227, 73, 72), (42, 120, 214)
INK, MUTED = "#0b0b0b", "#52514e"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", required=True, metavar="NAME=PATH")
    parser.add_argument("--micro", required=True, metavar="NAME=PATH")
    parser.add_argument("--cityscapes", type=int, default=0, help="Frames to show.")
    parser.add_argument("--images", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _predict(model, rgb):
    return _nearest(model.to_eval[model.predict(rgb)], EVAL_HEIGHT, EVAL_WIDTH)


def _error_map(rgb_small, gt, pred):
    overlay = (rgb_small * 0.35).astype(np.uint8)
    scored = gt != IGNORE
    wrong = scored & (pred != gt)
    overlay[wrong] = OTHER_ERROR
    overlay[scored & (gt == ROAD) & (pred == SIDEWALK)] = ROAD_AS_SIDEWALK
    road = scored & (gt == ROAD)
    ras = float(((pred == SIDEWALK) & road).sum() / max(road.sum(), 1))
    accuracy = float((scored & ~wrong).sum() / max(scored.sum(), 1))
    return overlay, accuracy, ras


def _style(ax, image, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=9, color=INK)
    ax.axis("off")


def _legend(fig, errors):
    names = {"Road": "Road (MicroSeg: Non-walkable, incl. sky)",
             "Sidewalk": "Sidewalk (MicroSeg: Walkable)",
             "Other": "Other (Fast-SCNN classes MicroSeg lacks)"}
    handles = [Patch(color=np.array(c) / 255, label=names.get(n, n))
               for n, c in CLASS_COLORS.items()]
    handles.append(Patch(color="black", label="Not scored"))
    if errors:
        handles += [Patch(color=np.array(ROAD_AS_SIDEWALK) / 255, label="Error: road as sidewalk"),
                    Patch(color=np.array(OTHER_ERROR) / 255, label="Error: other")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5,
               frameon=False)


def _save(fig, path):
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved {path.with_suffix('.png')} and .pdf", flush=True)


def cityscapes_figure(models, count, output):
    pairs = _cityscapes_pairs("val", 0)
    picks = np.linspace(0, len(pairs) - 1, count).round().astype(int)
    fig, axes = plt.subplots(count, 6, figsize=(22, 2.05 * count + 0.8), squeeze=False)
    for row, pick in enumerate(picks):
        image_path, mask_path = pairs[pick]
        rgb = tf.io.decode_png(tf.io.read_file(str(image_path)), channels=3)
        label_ids = tf.io.decode_png(tf.io.read_file(str(mask_path)), channels=1).numpy()[..., 0]
        gt = _nearest(GT_LOOKUP[label_ids], EVAL_HEIGHT, EVAL_WIDTH)
        rgb_small = tf.image.resize(rgb, [EVAL_HEIGHT, EVAL_WIDTH]).numpy().astype(np.uint8)
        name = image_path.name.replace("_leftImg8bit.png", "")
        _style(axes[row, 0], rgb_small, name)
        _style(axes[row, 1], LOOKUP[gt], "Ground truth (shared classes)")
        for offset, model in enumerate(models):
            pred = _predict(model, rgb)
            overlay, accuracy, ras = _error_map(rgb_small, gt, pred)
            _style(axes[row, 2 + offset], LOOKUP[pred], model.label)
            _style(axes[row, 4 + offset], overlay,
                   f"{model.label} errors: acc {100 * accuracy:.1f}%, "
                   f"road-as-sidewalk {100 * ras:.1f}%")
    _legend(fig, errors=True)
    fig.suptitle("Cityscapes val, frames spaced evenly through the split "
                 "(not hand-picked)", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.07,
                        wspace=0.03, hspace=0.22)
    _save(fig, output)


def photos_figure(models, images, output):
    fig, axes = plt.subplots(len(images), 3, figsize=(13, 2.6 * len(images) + 0.8),
                             squeeze=False)
    for row, path in enumerate(images):
        rgb = tf.io.decode_image(tf.io.read_file(str(path)), channels=3,
                                 expand_animations=False)
        shown = tf.image.resize(rgb, [EVAL_HEIGHT, EVAL_WIDTH]).numpy().astype(np.uint8)
        _style(axes[row, 0], shown, path.stem.replace("_", " "))
        for offset, model in enumerate(models):
            pred = _predict(model, rgb)
            crosswalk = 100 * float((pred == 3).mean())
            _style(axes[row, 1 + offset], LOOKUP[pred],
                   f"{model.label} (Crosswalk {crosswalk:.1f}% of frame)")
    _legend(fig, errors=False)
    fig.suptitle("Real pedestrian photos (unlabelled; no ground truth)", fontsize=11,
                 color=INK)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.08,
                        wspace=0.03, hspace=0.2)
    _save(fig, output)


def main():
    args = _arguments()
    micro, fast = _named([args.micro], "micro")[0], _named([args.fast], "fast")[0]
    models = []
    for spec, label in ((micro, "MicroSeg"), (fast, "Fast-SCNN")):
        model = Model(spec)
        model.label = label
        models.append(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cityscapes > 0:
        cityscapes_figure(models, args.cityscapes, args.output_dir / "qualitative_cityscapes")
    if args.images:
        photos_figure(models, args.images, args.output_dir / "qualitative_photos")
    for model in models:
        if sha256(model.path) != model.sha256:
            raise AssertionError(f"Plotting changed checkpoint weights: {model.path}")


if __name__ == "__main__":
    main()
