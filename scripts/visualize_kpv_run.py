"""Render KPV training curves and held-out prediction panels for one run.

Two outputs per run directory:
- training_curves.png: Cityscapes val mIoU, KPV full-frame val IoU, the
  harmonic combined score, and the flood excess per epoch, with the best
  checkpoint epoch marked.
- predictions/NN.jpg: for the first --samples held-out real KPV validation
  frames, a four-panel figure (input image, horizon-split ground truth,
  model prediction, Crosswalk overlay) rendered from the run's best.keras.
- cityscapes/NN.jpg: for --cityscapes-samples Cityscapes val frames spaced
  evenly through the split, a four-panel figure (input image, ten-class
  ground truth, model prediction, error overlay with Crosswalk spill marked).
  Cityscapes has no Crosswalk truth, so every Crosswalk pixel there is spill.

Read-only with respect to the checkpoint: weights are hashed before and
after and must match. CPU-only inference, no training.
"""
import argparse
import csv
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Patch

import pathclear_stages  # noqa: F401  (registers ActiveClasses for deserialization)
from dataset_loader import collect_pairs
from kpv_loader import CROSSWALK_ID, load_kpv_pair, pairs_from_split_manifest
from pathclear_pipeline import load_cityscapes_pair, sha256
from pathclear_taxonomy import CLASS_NAMES, COLORS, IGNORE_LABEL, colorize

DEFAULT_SPLIT_MANIFEST = Path("/home/nacsi/microseg_runs/crosswalk_split.json")


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best.keras")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--samples", type=int, default=6, help="KPV frames to render.")
    parser.add_argument(
        "--cityscapes-samples",
        type=int,
        default=6,
        help="Cityscapes val frames to render, spaced evenly; 0 skips them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/visuals.",
    )
    return parser.parse_args()


def plot_curves(history_path, output_path, checkpoint_epoch):
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty history: {history_path}")
    epochs = [int(row["epoch"]) + 1 for row in rows]
    series = (
        ("val_mean_iou", "Cityscapes val mIoU, logged (%)", 100.0),
        ("val_kpv_iou", "KPV full-frame val IoU (%)", 100.0),
        ("val_combined_score", "Combined score (harmonic)", 1.0),
        ("val_kpv_flood_excess", "Flood excess (over 2x tolerance)", 1.0),
    )
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    best_index = max(
        range(len(rows)), key=lambda i: float(rows[i]["val_combined_score"])
    )
    for ax, (key, label, scale) in zip(axes, series):
        values = [float(row[key]) * scale for row in rows]
        ax.plot(epochs, values, marker="o", color="#1f77b4", linewidth=1.6)
        ax.axvline(
            epochs[best_index],
            color="#d62728",
            linestyle="--",
            linewidth=1,
            label=f"best (epoch {epochs[best_index]})",
        )
        if key == "val_kpv_flood_excess":
            ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"PathClear 10-class + KPV head-only probe ({len(rows)} epochs; "
        f"best checkpoint epoch {checkpoint_epoch})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def render_predictions(model, pairs, output_dir, samples):
    rows = []
    for index, (image_path, mask_path) in enumerate(pairs[:samples]):
        image, targets, weights = load_kpv_pair(image_path, mask_path)
        logits = model(image[None], training=False)
        prediction = tf.argmax(logits[0], axis=-1).numpy().astype(np.uint8)
        truth = targets.numpy().astype(np.uint8)
        # Fade the weak-weight negatives so the marked crossing stands out.
        weak = weights.numpy() < 1.0
        truth_display = truth.copy()
        truth_display[weak & (truth != CROSSWALK_ID)] = IGNORE_LABEL
        pred_display = prediction.copy()
        pred_display[weak & (prediction != CROSSWALK_ID)] = IGNORE_LABEL

        rgb = (image.numpy() * 255.0).astype(np.uint8)
        overlay = rgb.copy()
        true_crosswalk = truth == CROSSWALK_ID
        pred_crosswalk = prediction == CROSSWALK_ID
        overlay[true_crosswalk & ~pred_crosswalk] = (255, 40, 40)   # missed
        overlay[pred_crosswalk & ~true_crosswalk] = (40, 80, 255)  # spurious
        overlay[true_crosswalk & pred_crosswalk] = (40, 220, 60)   # hit

        fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
        panels = (
            (rgb, "Input image"),
            (colorize(truth_display), "Ground truth (faded = weak negative)"),
            (colorize(pred_display), "Prediction"),
            (overlay, "Crosswalk: green hit | red missed | blue spurious"),
        )
        for ax, (panel, title) in zip(axes, panels):
            ax.imshow(panel)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        hits = int((true_crosswalk & pred_crosswalk).sum())
        true_total = int(true_crosswalk.sum())
        pred_total = int(pred_crosswalk.sum())
        recall = hits / max(true_total, 1)
        precision = hits / max(pred_total, 1)
        name = Path(image_path).stem
        fig.suptitle(
            f"{name} | Crosswalk P={precision:.2f} R={recall:.2f} "
            f"(pred {pred_total}px / true {true_total}px)",
            fontsize=11,
        )
        class_legend(fig)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.85, bottom=0.16, wspace=0.03)
        path = output_dir / f"{index:02d}_{name}.jpg"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        rows.append({
            "image": str(image_path),
            "panel": str(path),
            "crosswalk_precision": precision,
            "crosswalk_recall": recall,
            "predicted_crosswalk_px": pred_total,
            "true_crosswalk_px": true_total,
        })
        print(f"Rendered {path.name}: P={precision:.2f} R={recall:.2f}", flush=True)
    return rows


def class_legend(fig):
    handles = [
        Patch(color=np.asarray(COLORS[index]) / 255.0, label=f"{index} {name}")
        for index, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               frameon=False)


def render_cityscapes(model, pairs, output_dir, samples):
    """Render Cityscapes val frames to show what the KPV head costs there."""
    rows = []
    picks = np.linspace(0, len(pairs) - 1, samples).round().astype(int)
    for index, pick in enumerate(picks):
        image_path, mask_path = pairs[pick]
        image, targets, weights = load_cityscapes_pair(image_path, mask_path)
        logits = model(image[None], training=False)
        prediction = tf.argmax(logits[0], axis=-1).numpy().astype(np.uint8)
        valid = weights.numpy() > 0
        truth = np.where(valid, targets.numpy(), IGNORE_LABEL).astype(np.uint8)

        rgb = (image.numpy() * 255.0).astype(np.uint8)
        wrong = valid & (prediction != truth)
        spill = prediction == CROSSWALK_ID
        overlay = (rgb * 0.45).astype(np.uint8)
        overlay[wrong] = (255, 40, 40)   # misclassified
        overlay[spill] = (40, 80, 255)   # Crosswalk, never true on Cityscapes

        fig, axes = plt.subplots(1, 4, figsize=(20, 3.4))
        panels = (
            (rgb, "Input image"),
            (colorize(truth), "Ground truth (black = void)"),
            (colorize(prediction), "Prediction"),
            (overlay, "Errors: red misclassified | blue Crosswalk spill"),
        )
        for ax, (panel, title) in zip(axes, panels):
            ax.imshow(panel)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        accuracy = float((valid & ~wrong).sum() / max(valid.sum(), 1))
        spill_px = int(spill.sum())
        spill_on_road = int((spill & (truth == 0)).sum())
        name = Path(image_path).stem.replace("_leftImg8bit", "")
        fig.suptitle(
            f"{name} | pixel acc {accuracy:.3f} | Crosswalk spill {spill_px}px "
            f"({spill_on_road}px on Road)",
            fontsize=11,
        )
        class_legend(fig)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.84, bottom=0.14, wspace=0.03)
        path = output_dir / f"{index:02d}_{name}.jpg"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        rows.append({
            "image": str(image_path),
            "panel": str(path),
            "pixel_accuracy": accuracy,
            "crosswalk_spill_px": spill_px,
            "crosswalk_spill_on_road_px": spill_on_road,
        })
        print(f"Rendered {path.name}: acc={accuracy:.3f} spill={spill_px}px", flush=True)
    return rows


def main():
    args = _arguments()
    run_dir = args.run_dir.resolve()
    output = (args.output_dir or run_dir / "visuals").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)

    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing history.csv in {run_dir}")
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    best_index = max(
        range(len(rows)), key=lambda i: float(rows[i]["val_combined_score"])
    )
    checkpoint_epoch = 1 + best_index
    plot_curves(history_path, output / "training_curves.png", checkpoint_epoch)
    print(f"Saved {output / 'training_curves.png'}", flush=True)

    checkpoint = run_dir / args.checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    before_hash = sha256(checkpoint)
    model = keras.models.load_model(checkpoint, compile=False)

    pairs = pairs_from_split_manifest(
        args.split_manifest, split="validation", source="real"
    )
    panel_rows = render_predictions(model, pairs, output / "predictions", args.samples)
    cityscapes_rows = []
    if args.cityscapes_samples > 0:
        (output / "cityscapes").mkdir(exist_ok=True)
        city_images, city_masks = collect_pairs("val")
        cityscapes_rows = render_cityscapes(
            model, list(zip(city_images, city_masks)), output / "cityscapes",
            args.cityscapes_samples,
        )

    after_hash = sha256(checkpoint)
    if before_hash != after_hash:
        raise AssertionError("Visualization changed checkpoint weights.")

    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": after_hash,
        "best_epoch": checkpoint_epoch,
        "best_epoch_val_metrics": {
            key: float(value) for key, value in rows[best_index].items()
        },
        "panels": panel_rows,
        "cityscapes_panels": cityscapes_rows,
        "note": (
            "Best-checkpoint validation metrics are the history.csv row of "
            "the best epoch (the state best.keras captures); panel "
            "precision/recall are recomputed on the first held-out real "
            "frames only. The logged val_mean_iou counts Crosswalk as an "
            "eighth class at 0 IoU on Cityscapes; Cityscapes panels report "
            "per-frame pixel accuracy and Crosswalk spill instead."
        ),
    }
    (output / "visualization_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {output / 'visualization_report.json'}", flush=True)


if __name__ == "__main__":
    main()

