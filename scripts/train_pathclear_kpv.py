"""Train the ten-class PathClear head with Cityscapes + KPV Crosswalk mix.

Starts from the shared-Boundary adaptation checkpoint (7-active gate),
removes the gate via enable_local_classes, and trains the classifier head
only (frozen features -- the fast probe for whether frozen epoch-19
features can represent stripes). Cityscapes and KPV streams are sampled at
a controlled ratio; every branch repeats during training so the realised
mix cannot drift with steps_per_epoch. Checkpoints on harmonic_flood:
harmonic mean of Cityscapes val mIoU and KPV full-frame val IoU, minus a
flood penalty when predicted Crosswalk coverage exceeds tolerance x true.
With --teacher-manifest, unmarked KPV training pixels take teacher labels
(see scripts/make_kpv_teacher_labels.py) instead of the horizon split.
With --unfreeze all, every layer except BatchNorm trains at
--learning-rate; BatchNorm stays in inference mode because batch-1
statistics would corrupt it. A start checkpoint that is already ungated
(ten raw logits, e.g. a previous KPV run) is used as is.
Refuses to overwrite a nonempty output dir; CPU requires --allow-cpu.
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from dataset_loader import collect_pairs
from kpv_loader import (
    attach_teacher_labels,
    make_kpv_dataset,
    pairs_from_split_manifest,
)
from pathclear_pipeline import (
    compile_pathclear,
    load_cityscapes_pair,
    remap_cityscapes_tensor,
    sha256,
)
from pathclear_stages import GATE_NAME, enable_local_classes
from pathclear_taxonomy import CITYSCAPES_SUPPORTED_IDS, NUM_CLASSES
from segmentation_metrics import make_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_MANIFEST = Path("/home/nacsi/microseg_runs/crosswalk_split.json")
CROSSWALK_ID = 8


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def cityscapes_dataset(images, masks, training, seed):
    data = tf.data.Dataset.from_tensor_slices((images, masks))
    if training:
        data = data.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    data = data.map(
        lambda image, mask: load_cityscapes_pair(image, mask),
        num_parallel_calls=2, deterministic=True,
    )
    # Repeat during training so the finite Cityscapes stream never ends
    # inside a mixed sampler (the sampler-drift bug: without this the
    # realised KPV share drifts with steps_per_epoch).
    if training:
        data = data.repeat()
    return data.batch(1).prefetch(1)


class Checks(keras.callbacks.Callback):
    def __init__(self, output, total_steps):
        super().__init__()
        self.output, self.total_steps = output, total_steps

    def on_train_begin(self, logs=None):
        self.started = time.perf_counter()
        self.frozen = {layer.name: layer.get_weights() for layer in self.model.layers
                       if layer.weights and not layer.trainable_weights}
        self.head = self.model.get_layer("classifier_logits").get_weights()

    def on_train_batch_end(self, batch, logs=None):
        if not all(np.isfinite(value) for value in (logs or {}).values()):
            raise FloatingPointError(f"Non-finite training metrics at batch {batch + 1}")
        if batch == 0 or (batch + 1) % 100 == 0 or batch + 1 == self.total_steps:
            progress = {"batch": batch + 1, "total_batches": self.total_steps,
                        "elapsed_seconds": time.perf_counter() - self.started,
                        "metrics": {key: float(value) for key, value in (logs or {}).items()}}
            write_json(self.output / "progress.json", progress)
            print(f"Training {batch + 1}/{self.total_steps}: " + json.dumps(progress["metrics"]), flush=True)

    def on_test_begin(self, logs=None):
        self.before_validation = self.model.get_weights()
        print("Validation started.", flush=True)

    def on_test_end(self, logs=None):
        for before, after in zip(self.before_validation, self.model.get_weights()):
            np.testing.assert_array_equal(before, after)
        print("Validation preserved all model weights.", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        if not all(np.isfinite(value) for value in (logs or {}).values()):
            raise FloatingPointError("Non-finite epoch metrics; model will not be saved.")

    def on_train_end(self, logs=None):
        for name, before in self.frozen.items():
            for before_weight, after_weight in zip(
                    before, self.model.get_layer(name).get_weights()):
                np.testing.assert_array_equal(before_weight, after_weight)
        print(f"Frozen layers unchanged ({len(self.frozen)}).", flush=True)
        head = self.model.get_layer("classifier_logits").get_weights()
        change = max(float(np.max(np.abs(before - after)))
                     for before, after in zip(self.head, head))
        if not np.isfinite(change) or change == 0:
            raise AssertionError("Head weights did not update.")
        print(f"Verified head weight change: {change:.8f}", flush=True)
        self.max_head_change = change


class KpvFullFrameMetrics(keras.callbacks.Callback):
    """Score Crosswalk across whole held-out KPV frames; floods become FPs.

    The KPV label contract marks only the crossing, so weighted validation
    metrics only ever see the marked quad. This recomputes the Crosswalk
    confusion matrix over every pixel: a prediction on an unmarked pixel
    counts as a false positive. Logs val_kpv_iou/precision/recall,
    val_kpv_pred_fraction and val_kpv_true_fraction.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        true_pos = false_pos = false_neg = total = marked = 0
        for images, targets, _ in self.dataset:
            logits = self.model.predict(images, verbose=0)
            pred = np.argmax(logits, axis=-1)
            is_cross = targets.numpy() == CROSSWALK_ID
            pred_cross = pred == CROSSWALK_ID
            true_pos += int(np.logical_and(pred_cross, is_cross).sum())
            false_pos += int(np.logical_and(pred_cross, ~is_cross).sum())
            false_neg += int(np.logical_and(~pred_cross, is_cross).sum())
            marked += int(is_cross.sum())
            total += int(is_cross.size)
        union = true_pos + false_pos + false_neg
        iou = true_pos / union if union else 0.0
        denom = true_pos + false_pos
        precision = true_pos / denom if denom else 0.0
        recall_denom = true_pos + false_neg
        recall = true_pos / recall_denom if recall_denom else 0.0
        if logs is not None:
            logs.update({
                "val_kpv_iou": float(iou),
                "val_kpv_precision": float(precision),
                "val_kpv_recall": float(recall),
                "val_kpv_pred_fraction": float((true_pos + false_pos) / total) if total else 0.0,
                "val_kpv_true_fraction": float(marked / total) if total else 0.0,
            })
        print(f"KPV full-frame: iou={iou:.4f} prec={precision:.4f} "
              f"rec={recall:.4f} pred_frac={(true_pos + false_pos) / total:.4f} "
              f"true_frac={marked / total:.4f}", flush=True)


class HarmonicFloodScore(keras.callbacks.Callback):
    """Combined checkpoint score: harmonic mean minus a flood penalty.

    score = H(w, city_miou, kpv_iou) - flood_weight * excess, where
    excess = max(0, pred_fraction / true_fraction - tolerance). The deadband
    (tolerance=2.0) means healthy runs pay nothing; only floods are charged.
    Must run after KpvFullFrameMetrics so the val_kpv_* logs exist.
    """

    NAME = "val_combined_score"

    def __init__(self, cityscapes_weight=0.5, flood_weight=1.0, flood_tolerance=2.0):
        super().__init__()
        if not 0.0 < cityscapes_weight < 1.0:
            raise ValueError("cityscapes_weight must be between 0 and 1.")
        self.cityscapes_weight = cityscapes_weight
        self.flood_weight = flood_weight
        self.flood_tolerance = flood_tolerance

    def harmonic(self, city, kpv):
        w = self.cityscapes_weight
        if city <= 0 or kpv <= 0:
            return 0.0
        return 1.0 / (w / city + (1.0 - w) / kpv)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        city = float(logs.get("val_mean_iou") or 0.0)
        kpv = float(logs.get("val_kpv_iou") or 0.0)
        pred = float(logs.get("val_kpv_pred_fraction") or 0.0)
        true = float(logs.get("val_kpv_true_fraction") or 0.0)
        excess = 0.0
        if true > 0:
            excess = max(0.0, pred / true - self.flood_tolerance)
        score = self.harmonic(city, kpv) - self.flood_weight * excess
        logs[self.NAME] = float(score)
        logs["val_kpv_flood_excess"] = float(excess)
        print(f"Combined score: {score:.4f} (city={city:.4f} kpv={kpv:.4f} "
              f"excess={excess:.4f})", flush=True)


def run(args, output):
    started = time.perf_counter()
    seed = args.seed
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus and not args.allow_cpu:
        raise RuntimeError("Expected the verified WSL GPU; stopping before training.")
    if not gpus:
        tf.config.set_visible_devices([], "GPU")
        print("CPU fallback: hiding GPUs for this process.", flush=True)
    print(f"Devices: {gpus or 'CPU'}; input=512x1024; batch=1", flush=True)

    import pathclear_stages  # noqa: F401 (registers ActiveClasses)
    staged = keras.models.load_model(args.start_checkpoint, compile=False)
    if any(layer.name == GATE_NAME for layer in staged.layers):
        model = enable_local_classes(staged)
        print("Gate removed: 10 raw logits (0..9, incl. slot 8 Crosswalk).", flush=True)
    elif staged.output_shape[-1] == NUM_CLASSES:
        model = staged
        print("Start checkpoint already ungated: 10 raw logits.", flush=True)
    else:
        raise ValueError(f"Unexpected start checkpoint output {staged.output_shape}.")
    if args.unfreeze == "all":
        for layer in model.layers:
            layer.trainable = not isinstance(layer, keras.layers.BatchNormalization)
    trainable = {weight.path.split("/")[0] for weight in model.trainable_weights}
    print(f"Unfreeze={args.unfreeze}; trainable layers={len(trainable)}; "
          f"learning_rate={args.learning_rate}", flush=True)
    compile_pathclear(model, learning_rate=args.learning_rate)

    train_images, train_masks = collect_pairs("train")
    val_images, val_masks = collect_pairs("val")
    city_train = cityscapes_dataset(train_images, train_masks, True, seed)
    city_val = cityscapes_dataset(val_images, val_masks, False, seed)

    manifest = Path(args.split_manifest)
    kpv_real_train = pairs_from_split_manifest(manifest, split="train", source="real")
    kpv_synth_train = pairs_from_split_manifest(
        manifest, split="train", source="synthetic")
    kpv_val_pairs = pairs_from_split_manifest(manifest, split="validation")
    print(f"KPV: real_train={len(kpv_real_train)} synth_train={len(kpv_synth_train)} "
          f"held_out_val={len(kpv_val_pairs)}", flush=True)
    if args.teacher_manifest is not None:
        kpv_real_train = attach_teacher_labels(kpv_real_train, args.teacher_manifest)
        kpv_synth_train = attach_teacher_labels(kpv_synth_train, args.teacher_manifest)
        print(f"Teacher labels: {args.teacher_manifest} "
              f"(weight {args.teacher_weight})", flush=True)

    kpv_real_ds = make_kpv_dataset(
        kpv_real_train, training=True, negative_weight=args.negative_weight,
        horizon_fraction=args.horizon_fraction, seed=seed,
        teacher_weight=args.teacher_weight)
    kpv_synth_ds = make_kpv_dataset(
        kpv_synth_train, training=True, negative_weight=args.negative_weight,
        horizon_fraction=args.horizon_fraction, seed=seed,
        teacher_weight=args.teacher_weight)
    kpv_train = tf.data.Dataset.sample_from_datasets(
        [kpv_real_ds, kpv_synth_ds],
        weights=[args.real_weight, 1.0 - args.real_weight],
        rerandomize_each_iteration=True,
    )
    train_ds = tf.data.Dataset.sample_from_datasets(
        [city_train, kpv_train],
        weights=[1.0 - args.crosswalk_weight, args.crosswalk_weight],
        rerandomize_each_iteration=True,
    )
    kpv_val_ds = make_kpv_dataset(
        kpv_val_pairs, training=False, negative_weight=0.0)

    steps = 3 if args.mode == "smoke" else args.steps_per_epoch
    epochs = 1 if args.mode == "smoke" else args.epochs
    if args.mode == "smoke":
        city_val = city_val.take(1)
        kpv_val_ds = kpv_val_ds.take(1)
    checks = Checks(output, steps * epochs)
    kpv_metrics = KpvFullFrameMetrics(kpv_val_ds)
    score = HarmonicFloodScore(cityscapes_weight=args.cityscapes_weight)
    callbacks = [kpv_metrics, score, checks,
                 keras.callbacks.ModelCheckpoint(
                     output / "best.keras", monitor=HarmonicFloodScore.NAME,
                     mode="max", save_best_only=True),
                 keras.callbacks.CSVLogger(output / "history.csv")]
    history = model.fit(train_ds, epochs=epochs, steps_per_epoch=steps,
                        validation_data=city_val, callbacks=callbacks, verbose=2)
    model.save(output / "last.keras")
    write_json(output / "history.json",
               {k: [float(v) for v in vals] for k, vals in history.history.items()})
    report = {
        "status": "complete", "mode": args.mode,
        "start_checkpoint": str(args.start_checkpoint),
        "completed_epochs": epochs, "steps_per_epoch": steps,
        "optimizer_iterations": int(model.optimizer.iterations),
        "elapsed_seconds": time.perf_counter() - started,
        "crosswalk_weight": args.crosswalk_weight,
        "real_weight": args.real_weight,
        "negative_weight": args.negative_weight,
        "horizon_fraction": args.horizon_fraction,
        "teacher_manifest": (None if args.teacher_manifest is None
                             else str(args.teacher_manifest)),
        "teacher_weight": args.teacher_weight,
        "unfreeze": args.unfreeze,
        "learning_rate": args.learning_rate,
        "cityscapes_weight": args.cityscapes_weight,
        "checkpoint_metric": HarmonicFloodScore.NAME,
        "max_head_change": getattr(checks, "max_head_change", None),
        "final_metrics": {k: float(v[-1]) for k, v in history.history.items()},
        "best_checkpoint": str(output / "best.keras"),
        "checkpoint_sha256": sha256(output / "best.keras"),
        "note": ("Ten-class Cityscapes+KPV head-only probe. Slots 7/9 unsupervised."
                 if args.unfreeze == "none" else
                 "Ten-class Cityscapes+KPV fine-tune, BatchNorm frozen. "
                 "Slots 7/9 unsupervised."),
    }
    write_json(output / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--crosswalk-weight", type=float, default=0.5)
    parser.add_argument("--real-weight", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=0.25)
    parser.add_argument("--horizon-fraction", type=float, default=0.45)
    parser.add_argument("--cityscapes-weight", type=float, default=0.5)
    parser.add_argument("--teacher-manifest", type=Path, default=None,
                        help="teacher_manifest.json; replaces the horizon split.")
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--unfreeze", choices=("none", "all"), default="none",
                        help="all: train every non-BatchNorm layer.")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    for name in ("crosswalk_weight", "real_weight"):
        if not 0.0 < getattr(args, name) < 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if not 0.0 <= args.teacher_weight <= 1.0:
        parser.error("--teacher-weight must be between 0 and 1.")
    if not 0.0 <= args.negative_weight <= 1.0:
        parser.error("--negative-weight must be between 0 and 1.")
    if not 0.0 <= args.horizon_fraction <= 1.0:
        parser.error("--horizon-fraction must be between 0 and 1.")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1.")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            try:
                run(args, output)
            except Exception as error:
                write_json(output / "failure.json",
                           {"status": "failed", "error": repr(error)})
                raise


if __name__ == "__main__":
    main()



