"""Run PathClear checkpoints on unlabeled images and render one comparison grid.

Each row is one image: input, one prediction per --compare checkpoint
(repeatable; defaults to the pre-KPV start point), the --checkpoint
prediction, and its Crosswalk overlay. There is no ground truth, so nothing here is a metric; the per-image
class shares in the JSON report describe predictions only.

Preprocessing matches training: bilinear resize to 512x1024, divide by 255.
Non-2:1 images are stretched; predictions are resized back to the original
size with nearest neighbor for display. Read-only with respect to both
checkpoints (hashed before and after).
"""
import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Patch
from PIL import Image

import pathclear_stages  # noqa: F401  (registers ActiveClasses for deserialization)
from kpv_loader import CROSSWALK_ID
from pathclear_pipeline import sha256
from pathclear_taxonomy import CLASS_NAMES, COLORS, colorize

DEFAULT_CHECKPOINT = Path("/home/nacsi/pathclear_runs/kpv_full_10ep/best.keras")
DEFAULT_COMPARE = Path(
    "/home/nacsi/pathclear_runs/shared_bnd_epoch1_20261006/best.keras"
)
_GATE_NAME = "pathclear_active_classes"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--compare", type=Path, action="append",
                        help="Comparison checkpoint; repeat for more columns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path):
    model = keras.models.load_model(path, compile=False, safe_mode=False)
    try:
        gate = model.get_layer(_GATE_NAME)
    except ValueError:
        gate = None
    return model, gate


def predict(model, gate, rgb):
    image = tf.image.resize(rgb, [512, 1024], method="bilinear") / 255.0
    logits = model(image[None], training=False)
    if gate is not None:
        logits = gate(logits)
    mask = tf.argmax(logits[0], axis=-1).numpy().astype(np.uint8)
    size = (rgb.shape[1], rgb.shape[0])
    return np.array(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))


def shares(mask):
    counts = np.bincount(mask.ravel(), minlength=len(CLASS_NAMES))
    return {CLASS_NAMES[i]: float(counts[i] / mask.size) for i in range(len(CLASS_NAMES))}


def main():
    args = _arguments()
    compares = args.compare or [DEFAULT_COMPARE]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {p: sha256(p) for p in (args.checkpoint, *compares)}
    main_model = load(args.checkpoint)
    compare_models = [load(path) for path in compares]

    columns = 3 + len(compares)
    fig, axes = plt.subplots(len(args.images), columns,
                             figsize=(5 * columns, 3.3 * len(args.images)), squeeze=False)
    rows = []
    for row, path in enumerate(args.images):
        rgb = np.array(Image.open(path).convert("RGB"))
        befores = [predict(*model, rgb) for model in compare_models]
        after = predict(*main_model, rgb)
        crosswalk = after == CROSSWALK_ID
        overlay = rgb.copy()
        overlay[crosswalk] = (0.45 * overlay[crosswalk] + 0.55 * np.array(COLORS[CROSSWALK_ID])).astype(np.uint8)
        share = float(crosswalk.mean())
        panels = (
            (rgb, f"{path.name}"),
            *((colorize(before), compare.parent.name)
              for before, compare in zip(befores, compares)),
            (colorize(after), args.checkpoint.parent.name),
            (overlay, f"Crosswalk ({args.checkpoint.parent.name}): "
                      f"{100 * share:.1f}% of frame"),
        )
        for ax, (panel, title) in zip(axes[row], panels):
            ax.imshow(panel)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        rows.append({
            "image": str(path),
            "size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "crosswalk_share": share,
            "class_share": shares(after),
            "class_share_compare": {str(compare): shares(before)
                                    for compare, before in zip(compares, befores)},
        })
        print(f"{path.name}: Crosswalk {100 * share:.1f}% of frame", flush=True)

    handles = [Patch(color=np.asarray(COLORS[i]) / 255.0, label=f"{i} {name}")
               for i, name in enumerate(CLASS_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False)
    fig.suptitle("PathClear Fast-SCNN on unlabeled images (no ground truth)",
                 fontsize=12)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.06,
                        wspace=0.03, hspace=0.15)
    grid = args.output_dir / "comparison.jpg"
    fig.savefig(grid, dpi=110)
    plt.close(fig)

    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise AssertionError(f"Inference changed checkpoint weights: {path}")
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashes[args.checkpoint],
        "compare": {str(path): hashes[path] for path in compares},
        "grid": str(grid),
        "images": rows,
        "note": "Unlabeled images; shares describe predictions, not accuracy. "
                "Inputs are stretched to 512x1024 as in training.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"Saved {grid}", flush=True)


if __name__ == "__main__":
    main()
