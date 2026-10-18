"""Evaluate a Fast-SCNN checkpoint remapped onto MicroSeg classes.

No training happens here. Accepts either a 19-class Cityscapes baseline or a
ten-class PathClear checkpoint at 512x1024; argmax predictions are remapped to
the four MicroSeg IDs (19-class goes through the PathClear taxonomy first)
and scored against the same Cityscapes validation masks MicroSeg uses.
Classes neither model predicts (Person, Vehicle, Obstacle, Vegetation,
Crosswalk) stay ignored, so the comparison is honest about what each model
can actually see.
"""
import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from dataset_config import CITYSCAPES_ROOT
from dataset_loader import collect_pairs, load_pair
from fast_scnn import NUM_CLASSES as BASELINE_NUM_CLASSES
from microseg.taxonomy import (
    CLASS_NAMES as MICROSEG_CLASS_NAMES,
    IGNORE_LABEL as MICROSEG_IGNORE,
    NUM_CLASSES as MICROSEG_NUM_CLASSES,
    remap_cityscapes_ids,
    remap_pathclear_ids,
)
from pathclear_pipeline import load_cityscapes_pair
from pathclear_taxonomy import (
    NUM_CLASSES as PATHCLEAR_NUM_CLASSES,
    remap_cityscapes_train_ids,
)

_GATE_NAME = "pathclear_active_classes"


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _load_fast_scnn(checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    custom = {}
    try:
        from pathclear_stages import ActiveClasses

        custom["PathClear>ActiveClasses"] = ActiveClasses
    except ImportError:  # pragma: no cover - gated model without stages module
        pass
    return keras.models.load_model(
        checkpoint, compile=False, custom_objects=custom
    )


def _fast_scnn_dataset(pairs, batch_size, raw_train_ids):
    images, masks = zip(*pairs)
    data = tf.data.Dataset.from_tensor_slices((list(images), list(masks)))
    loader = load_pair if raw_train_ids else load_cityscapes_pair
    data = data.map(loader, num_parallel_calls=2, deterministic=True)
    return data.batch(batch_size).prefetch(1)


def _confusion_to_report(confusion, total_valid):
    with np.errstate(invalid="ignore", divide="ignore"):
        true_support = confusion.sum(axis=1)
        pred_support = confusion.sum(axis=0)
        intersection = np.diag(confusion).astype(np.float64)
        union = true_support + pred_support - intersection
        per_class_iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        present = union > 0
        mean_iou = float(per_class_iou[present].mean()) if present.any() else 0.0
        pixel_accuracy = (
            float(intersection.sum() / confusion.sum()) if confusion.sum() else 0.0
        )
    return {
        "valid_pixels_scored": total_valid,
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "per_class_iou": {
            name: float(per_class_iou[index])
            for index, name in enumerate(MICROSEG_CLASS_NAMES)
        },
        "per_class_true_support": {
            name: int(true_support[index])
            for index, name in enumerate(MICROSEG_CLASS_NAMES)
        },
        "confusion_matrix_true_by_pred": confusion.tolist(),
    }


def main(argv=None):
    args = _arguments(argv)
    model = _load_fast_scnn(args.checkpoint)
    head_classes = model.output_shape[-1]
    if head_classes == BASELINE_NUM_CLASSES:
        head_kind = "baseline-19"
        predict_to_microseg = lambda ids: remap_cityscapes_ids(
            remap_cityscapes_train_ids(ids)
        )
    elif head_classes == PATHCLEAR_NUM_CLASSES:
        head_kind = "pathclear-10"
        predict_to_microseg = remap_pathclear_ids
    else:
        raise ValueError(
            f"Expected a 19-class baseline or ten-class PathClear head, "
            f"got {model.output_shape}."
        )
    try:
        gate = model.get_layer(_GATE_NAME)
        active_ids = tuple(gate.active_ids)
    except ValueError:
        gate = None
        active_ids = tuple(range(PATHCLEAR_NUM_CLASSES))

    images, masks = collect_pairs(args.split)
    pairs = list(zip(images, masks))
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise ValueError(f"No Cityscapes pairs for split '{args.split}'.")
    dataset = _fast_scnn_dataset(
        pairs, args.batch_size, raw_train_ids=head_kind == "baseline-19"
    )

    confusion = np.zeros(
        (MICROSEG_NUM_CLASSES, MICROSEG_NUM_CLASSES), dtype=np.int64
    )
    total_valid = 0
    for batch_images, batch_targets, batch_weights in dataset:
        logits = model(batch_images, training=False)
        if gate is not None:
            logits = gate(logits)
        predicted = tf.argmax(logits, axis=-1).numpy().astype(np.int64)
        predicted_microseg = predict_to_microseg(predicted)
        targets = batch_targets.numpy().astype(np.int64)
        weights = batch_weights.numpy()
        valid = weights > 0
        if head_kind == "baseline-19":
            remapped_targets = remap_cityscapes_ids(targets)
        else:
            valid &= targets != MICROSEG_IGNORE
            remapped_targets = remap_pathclear_ids(np.clip(targets, 0, 255))
        valid &= remapped_targets != MICROSEG_IGNORE
        valid &= predicted_microseg != MICROSEG_IGNORE
        total_valid += int(np.sum(valid))
        if not np.any(valid):
            continue
        true_ids = remapped_targets[valid].ravel()
        pred_ids = predicted_microseg[valid].ravel()
        np.add.at(confusion, (true_ids, pred_ids), 1)

    report = _confusion_to_report(confusion, total_valid)
    report.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_head": head_kind,
            "split": args.split,
            "evaluated_pairs": len(pairs),
            "cityscapes_root": str(CITYSCAPES_ROOT),
            "fast_scnn_active_ids": list(active_ids),
            "microseg_class_names": list(MICROSEG_CLASS_NAMES),
            "mapping_note": (
                "19-class predictions go Cityscapes trainId -> PathClear 10 "
                "(building now Boundary) -> MicroSeg 4; 10-class predictions "
                "go PathClear 10 -> MicroSeg 4 directly: 1 Sidewalk->0 "
                "Walkable, 0 Road->1 Non-walkable, 4 Boundary->2 Boundary, "
                "8 Crosswalk->3 Crosswalk. IDs 2, 3, 5, 6, 7, 9 stay "
                "ignored. Cityscapes has no Crosswalk truth, so any "
                "Crosswalk prediction is a false positive."
            ),
        }
    )
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()

