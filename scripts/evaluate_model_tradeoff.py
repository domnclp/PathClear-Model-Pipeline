"""Score MicroSeg and Fast-SCNN checkpoints on one shared protocol.

Both model families are judged on the classes they were both trained on,
taken from the original Cityscapes gtFine labelIds masks:

    Road (labelId 7), Sidewalk (8), Boundary (building 11, wall 12, fence 13)

Every other pixel (sky, vegetation, people, vehicles, void, ...) is excluded
from scoring for every model. Predictions are mapped into the same space:

    Fast-SCNN:  0 Road -> Road, 1 Sidewalk -> Sidewalk, 4 Boundary -> Boundary,
                8 Crosswalk -> Crosswalk, any other class -> Other
    MicroSeg:   Non-walkable -> Road, Walkable -> Sidewalk,
                Boundary -> Boundary, Crosswalk -> Crosswalk

Crosswalk and Other predictions on scored pixels are errors. Each model gets
the input preprocessing it was trained with (bilinear resize of the full
frame to its input size, divide by 255). Ground truth is resized once, by
nearest neighbor, to the 512x1024 evaluation grid; MicroSeg's 96x96 output
is upsampled to that grid by nearest neighbor, so every model is scored on
the same pixels. A secondary score scores MicroSeg on a 96x96 nearest-neighbor
ground truth (its native output grid).

Metrics over the three scored classes:
- mIoU: mean of per-class IoU (TP / (GT + predicted - TP)).
- Pixel accuracy: correctly labelled scored pixels / scored pixels.
- Road-as-sidewalk error rate: ground-truth Road pixels predicted Sidewalk /
  ground-truth Road pixels. This is the dangerous error for a mobility aid:
  telling the user a road surface is walkable sidewalk.
95% confidence intervals come from a bootstrap over frames.

Optional --kpv-split-manifest adds full-frame Crosswalk IoU on the real KPV
validation frames as a supplementary row.

Read-only with respect to every checkpoint (hashed before and after).
"""
import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import pathclear_stages  # noqa: F401  (registers ActiveClasses for deserialization)
from dataset_config import CITYSCAPES_ROOT
from microseg import model as _microseg_model  # noqa: F401
from pathclear_pipeline import sha256

EVAL_HEIGHT, EVAL_WIDTH = 512, 1024
SCORED = ("Road", "Sidewalk", "Boundary")
PREDICTED = SCORED + ("Crosswalk", "Other")
ROAD, SIDEWALK, BOUNDARY, CROSSWALK, OTHER = range(5)
IGNORE = 255

LABEL_IDS_TO_EVAL = {7: ROAD, 8: SIDEWALK, 11: BOUNDARY, 12: BOUNDARY, 13: BOUNDARY}
FAST_TO_EVAL = {0: ROAD, 1: SIDEWALK, 4: BOUNDARY, 8: CROSSWALK}
MICRO_TO_EVAL = {1: ROAD, 0: SIDEWALK, 2: BOUNDARY, 3: CROSSWALK}
FAST_CROSSWALK_ID, MICRO_CROSSWALK_ID = 8, 3


def _lookup(table, default):
    lookup = np.full(256, default, dtype=np.uint8)
    for source, target in table.items():
        lookup[source] = target
    return lookup


GT_LOOKUP = _lookup(LABEL_IDS_TO_EVAL, IGNORE)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="append", default=[], metavar="NAME=PATH",
                        help="Fast-SCNN checkpoint; repeatable.")
    parser.add_argument("--micro", action="append", default=[], metavar="NAME=PATH",
                        help="MicroSeg checkpoint; repeatable.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--kpv-split-manifest", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _named(specs, family):
    models = []
    for spec in specs:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Expected NAME=PATH, got {spec!r}.")
        models.append({"name": name, "family": family, "path": Path(path)})
    return models


class Model:
    def __init__(self, spec):
        self.name, self.family, self.path = spec["name"], spec["family"], spec["path"]
        self.sha256 = sha256(self.path)
        self.keras = keras.models.load_model(self.path, compile=False, safe_mode=False)
        self.height, self.width = self.keras.input_shape[1:3]
        try:
            self.gate = self.keras.get_layer(pathclear_stages.GATE_NAME)
        except ValueError:
            self.gate = None
        if self.family == "fast":
            self.to_eval = _lookup(FAST_TO_EVAL, OTHER)
            self.crosswalk_id = FAST_CROSSWALK_ID
        else:
            self.to_eval = _lookup(MICRO_TO_EVAL, OTHER)
            self.crosswalk_id = MICRO_CROSSWALK_ID

    def predict(self, rgb):
        """Return the native-resolution argmax for one uint8 RGB frame."""
        image = tf.image.resize(rgb, [self.height, self.width], method="bilinear") / 255.0
        logits = self.keras(image[None], training=False)
        if self.gate is not None and self.keras.layers[-1] is not self.gate:
            logits = self.gate(logits)
        return tf.argmax(logits[0], axis=-1).numpy().astype(np.uint8)

    def describe(self):
        return {
            "name": self.name,
            "family": self.family,
            "checkpoint": str(self.path),
            "sha256": self.sha256,
            "params": int(self.keras.count_params()),
            "file_bytes": self.path.stat().st_size,
            "input": [int(self.height), int(self.width)],
        }


def _nearest(mask, height, width):
    return tf.image.resize(mask[..., None], [height, width],
                           method="nearest").numpy()[..., 0].astype(np.uint8)


def _confusion(gt, pred):
    valid = gt != IGNORE
    confusion = np.zeros((len(SCORED), len(PREDICTED)), dtype=np.int64)
    np.add.at(confusion, (gt[valid].astype(np.int64), pred[valid].astype(np.int64)), 1)
    return confusion


def _metrics(confusion):
    true_pixels = confusion.sum(axis=1)
    predicted = confusion[:, : len(SCORED)].sum(axis=0)
    tp = np.diag(confusion[:, : len(SCORED)]).astype(np.float64)
    union = true_pixels + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    total = confusion.sum()
    return {
        "miou": float(iou.mean()),
        "pixel_accuracy": float(tp.sum() / total) if total else 0.0,
        "road_as_sidewalk": (float(confusion[ROAD, SIDEWALK] / true_pixels[ROAD])
                             if true_pixels[ROAD] else 0.0),
        "sidewalk_as_road": (float(confusion[SIDEWALK, ROAD] / true_pixels[SIDEWALK])
                             if true_pixels[SIDEWALK] else 0.0),
        "per_class_iou": {name: float(iou[i]) for i, name in enumerate(SCORED)},
        "road_predicted_as": {name: (float(confusion[ROAD, i] / true_pixels[ROAD])
                                     if true_pixels[ROAD] else 0.0)
                              for i, name in enumerate(PREDICTED)},
    }


def _bootstrap(per_frame, samples, seed=42):
    rng = np.random.default_rng(seed)
    draws = {"miou": [], "pixel_accuracy": [], "road_as_sidewalk": []}
    for _ in range(samples):
        picked = per_frame[rng.integers(0, len(per_frame), len(per_frame))].sum(axis=0)
        result = _metrics(picked)
        for key in draws:
            draws[key].append(result[key])
    return {key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            for key, values in draws.items()}


def _cityscapes_pairs(split, max_pairs):
    image_root = CITYSCAPES_ROOT / "leftImg8bit" / split
    pairs = []
    for image in sorted(image_root.glob("*/*_leftImg8bit.png")):
        mask = (CITYSCAPES_ROOT / "gtFine" / split / image.parent.name
                / image.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
        if not mask.exists():
            raise FileNotFoundError(f"Missing labelIds mask: {mask}")
        pairs.append((image, mask))
    return pairs[:max_pairs] if max_pairs > 0 else pairs


def evaluate_cityscapes(models, pairs, bootstrap):
    per_frame = {model.name: [] for model in models}
    native = {model.name: np.zeros((len(SCORED), len(PREDICTED)), np.int64)
              for model in models if model.family == "micro"}
    started = time.perf_counter()
    for index, (image_path, mask_path) in enumerate(pairs):
        rgb = tf.io.decode_png(tf.io.read_file(str(image_path)), channels=3)
        label_ids = tf.io.decode_png(tf.io.read_file(str(mask_path)), channels=1).numpy()[..., 0]
        gt_native = GT_LOOKUP[label_ids]
        gt = _nearest(gt_native, EVAL_HEIGHT, EVAL_WIDTH)
        for model in models:
            pred = model.to_eval[model.predict(rgb)]
            if model.name in native:
                gt_small = _nearest(gt_native, model.height, model.width)
                native[model.name] += _confusion(gt_small, pred)
            per_frame[model.name].append(_confusion(gt, _nearest(pred, EVAL_HEIGHT, EVAL_WIDTH)))
        if (index + 1) % 50 == 0:
            print(f"Cityscapes {index + 1}/{len(pairs)} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    results = {}
    for model in models:
        frames = np.stack(per_frame[model.name])
        confusion = frames.sum(axis=0)
        results[model.name] = {
            **_metrics(confusion),
            "ci95": _bootstrap(frames, bootstrap) if bootstrap > 0 else None,
            "confusion_gt_by_pred": confusion.tolist(),
        }
        if model.name in native:
            results[model.name]["native_output_grid"] = {
                "grid": [int(model.height), int(model.width)],
                **_metrics(native[model.name]),
            }
    return results


def evaluate_kpv(models, manifest):
    from kpv_loader import pairs_from_split_manifest

    pairs = pairs_from_split_manifest(manifest, split="validation", source="real")
    counts = {model.name: np.zeros(3, np.int64) for model in models}  # tp, fp, fn
    for image_path, mask_path in pairs:
        rgb = tf.io.decode_image(tf.io.read_file(image_path), channels=3,
                                 expand_animations=False)
        mask = tf.io.decode_image(tf.io.read_file(mask_path), channels=1,
                                  expand_animations=False)
        truth = tf.image.resize(mask, [EVAL_HEIGHT, EVAL_WIDTH],
                                method="nearest").numpy()[..., 0] > 127
        for model in models:
            pred = _nearest(model.predict(rgb), EVAL_HEIGHT, EVAL_WIDTH) == model.crosswalk_id
            counts[model.name] += [(pred & truth).sum(), (pred & ~truth).sum(),
                                   (~pred & truth).sum()]
    results = {}
    for name, (tp, fp, fn) in counts.items():
        results[name] = {
            "frames": len(pairs),
            "crosswalk_iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0,
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        }
    return results


def main():
    args = _arguments()
    specs = _named(args.fast, "fast") + _named(args.micro, "micro")
    if not specs:
        raise ValueError("Pass at least one --fast or --micro checkpoint.")
    models = [Model(spec) for spec in specs]
    pairs = _cityscapes_pairs(args.split, args.max_pairs)
    print(f"Scoring {len(models)} models on {len(pairs)} Cityscapes {args.split} frames",
          flush=True)
    report = {
        "protocol": {
            "dataset": f"Cityscapes gtFine {args.split} (labelIds)",
            "frames": len(pairs),
            "scored_classes": {"Road": [7], "Sidewalk": [8], "Boundary": [11, 12, 13]},
            "evaluation_grid": [EVAL_HEIGHT, EVAL_WIDTH],
            "note": __doc__,
        },
        "models": [model.describe() for model in models],
        "cityscapes": evaluate_cityscapes(models, pairs, args.bootstrap),
    }
    if args.kpv_split_manifest is not None:
        report["kpv_crosswalk_supplementary"] = evaluate_kpv(models, args.kpv_split_manifest)
    for model in models:
        if sha256(model.path) != model.sha256:
            raise AssertionError(f"Evaluation changed checkpoint weights: {model.path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in report["cityscapes"].items():
        print(f"{name}: mIoU={100 * result['miou']:.2f} "
              f"PA={100 * result['pixel_accuracy']:.2f} "
              f"road-as-sidewalk={100 * result['road_as_sidewalk']:.2f}%", flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
