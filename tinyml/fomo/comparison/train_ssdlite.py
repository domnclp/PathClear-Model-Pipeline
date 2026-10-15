"""M4 SSDLite training on the FOMO v2 data with FOMO's optimizer, stopping and selection rules."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import common
from common import SELECTION_THRESHOLD, measure_detections
from config import ROOT, SETTINGS, output_directory
from dataset import load_arrays, validated_records
import ssdlite

# A larger budget than FOMO's 20 epochs does not favor M4: v2 early-stopped (best epoch 14 of 19),
# and the same patience rules end this run once validation F1 stops improving.
EPOCH_BUDGET = 40


def encode_all(records):
    """Targets for the original and the horizontally flipped image of every record."""
    labels = np.zeros((2, len(records), len(ssdlite.ANCHORS)), dtype=np.int8)
    deltas = np.zeros((2, len(records), len(ssdlite.ANCHORS), 4), dtype=np.float16)
    for index, record in enumerate(records):
        for flip in (0, 1):
            labels[flip, index], deltas[flip, index] = ssdlite.encode(record, bool(flip))
    return labels, deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sanity", action="store_true", help="12 train images, 30 updates, no accuracy claim")
    parser.add_argument("--epochs", type=int, default=EPOCH_BUDGET)
    parser.add_argument("--batch-size", type=int, default=SETTINGS.batch_size)
    parser.add_argument("--backbone-weights", default="imagenet")
    parser.add_argument("--allow-synthetic", action="store_true", help="Plumbing-only synthetic manifests")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    records, dataset_report = validated_records(args.manifest, args.allow_synthetic)
    directory = output_directory(args.output, "m4-ssdlite-sanity" if args.sanity else "m4-ssdlite-training")
    cache = ROOT / "artifacts" / "keras-cache"
    cache.mkdir(exist_ok=True)
    os.environ["KERAS_HOME"] = str(cache)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SETTINGS.seed)
    tf.config.experimental.enable_op_determinism()

    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    if args.sanity:
        train = [train[i] for i in np.linspace(0, len(train) - 1, min(12, len(train)), dtype=int)]
        validation = [validation[i] for i in np.linspace(0, len(validation) - 1, min(12, len(validation)), dtype=int)]
    x_train, _ = load_arrays(train)
    x_val, _ = load_arrays(validation)
    labels, deltas = encode_all(train)
    batch_size = min(8, args.batch_size) if args.sanity else args.batch_size
    positives = int((labels[0] > 0).sum())
    print(json.dumps({"train_images": len(train), "validation_images": len(validation),
                      "anchors": len(ssdlite.ANCHORS), "positive_anchors": positives,
                      "objects": sum(len(r["annotations"]) for r in train)}), flush=True)

    model = ssdlite.build_model(args.backbone_weights)
    optimizer = tf.keras.optimizers.Adam(SETTINGS.learning_rate)

    @tf.function(reduce_retracing=True)
    def train_step(images, target_labels, target_deltas):
        with tf.GradientTape() as tape:
            out = model(images, training=True)
            loss, class_loss, box_loss = ssdlite.detection_loss(
                target_labels, target_deltas, out["class_probabilities"], out["box_deltas"])
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return loss, class_loss, box_loss

    @tf.function(input_signature=[tf.TensorSpec((None, 160, 160, 3), tf.float32)], reduce_retracing=True)
    def infer(batch):
        out = model(batch, training=False)
        return out["class_probabilities"], out["box_deltas"]

    def detect(images, subset):
        found = []
        for i in range(0, len(images), batch_size):
            p, d = infer(images[i:i + batch_size].astype(np.float32) / 255)
            for j, (pi, di) in enumerate(zip(p.numpy(), d.numpy())):
                record = subset[i + j]
                found.append(ssdlite.decode(pi, di, record["width"], record["height"], SELECTION_THRESHOLD))
        return found

    rng = np.random.default_rng(SETTINGS.seed)
    metadata = {"contract": ssdlite.contract(), "synthetic": dataset_report["synthetic"], "sanity": args.sanity,
                "manifest": str(Path(args.manifest).resolve()),
                "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "settings": {k: v for k, v in asdict(SETTINGS).items()
                             if k in ("image_size", "alpha", "batch_size", "learning_rate", "patience",
                                      "lr_patience", "min_learning_rate", "seed", "match_radius")},
                "loss": {"negative_ratio": ssdlite.NEGATIVE_RATIO, "min_negatives": ssdlite.MIN_NEGATIVES,
                         "positive_iou": ssdlite.POSITIVE_IOU, "negative_iou": ssdlite.NEGATIVE_IOU,
                         "box_loss_weight": ssdlite.BOX_LOSS_WEIGHT,
                         "ignore": "negatives centered in FOMO target ignore cells"},
                "augmentation": "horizontal flip image and targets together",
                "batch_size": batch_size, "requested_epochs": args.epochs, "initialization": args.backbone_weights,
                "tensorflow": tf.__version__, "keras": tf.keras.__version__,
                "devices": [str(d) for d in tf.config.list_physical_devices()], "parameters": model.count_params(),
                "macs": common.count_macs(model), "train_images": len(train), "validation_images": len(validation),
                "positive_anchors_unflipped": positives,
                "checkpoint_selection": f"maximum validation centroid F1 at threshold {SELECTION_THRESHOLD}",
                "decoding": ssdlite.decoding_policy()}
    metadata["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in list(ROOT.glob("*.py")) + list(common.COMPARISON_DIR.glob("*.py"))}
    metadata["pretrained_weight_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.rglob("*.h5")}
    checkpoint = directory / "best.keras"
    (directory / "best.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history, best, best_epoch, wait, lr_wait = [], -1.0, 0, 0, 0
    started = time.perf_counter()
    epochs = 1 if args.sanity else args.epochs
    before = [w.numpy().copy() for w in model.trainable_weights] if args.sanity else None
    for epoch in range(epochs):
        losses = []
        if args.sanity:
            steps = [rng.choice(len(train), min(batch_size, len(train)), replace=False) for _ in range(30)]
        else:
            order = rng.permutation(len(train))
            steps = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        for batch in steps:
            flips = (rng.random(len(batch)) < 0.5).astype(int) if not args.sanity else np.zeros(len(batch), int)
            images = x_train[batch].astype(np.float32) / 255
            images[flips == 1] = images[flips == 1][:, :, ::-1]
            losses.append([float(v) for v in train_step(
                tf.constant(images), tf.constant(labels[flips, batch]),
                tf.constant(deltas[flips, batch].astype(np.float32)))])
        losses = np.asarray(losses)
        if not np.isfinite(losses).all():
            raise RuntimeError("Nonfinite loss, refusing checkpoint")
        result = measure_detections(validation, detect(x_val, validation), SELECTION_THRESHOLD)
        f1 = result["micro"]["f1"] or 0.0
        row = {"epoch": epoch + 1, "train_loss": float(losses[:, 0].mean()),
               "train_class_loss": float(losses[:, 1].mean()), "train_box_loss": float(losses[:, 2].mean()),
               "validation_f1": f1, "validation_precision": result["micro"]["precision"],
               "validation_recall": result["micro"]["recall"], "per_class": result["per_class"],
               "learning_rate": float(optimizer.learning_rate.numpy()),
               "elapsed_seconds": time.perf_counter() - started}
        if f1 > best + 1e-6:
            best, best_epoch, wait, lr_wait = f1, epoch + 1, 0, 0
            model.save(checkpoint)
        else:
            wait += 1
            lr_wait += 1
            if lr_wait >= SETTINGS.lr_patience:
                optimizer.learning_rate.assign(max(SETTINGS.min_learning_rate, row["learning_rate"] * 0.5))
                lr_wait = 0
        history.append(row)
        (directory / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in row.items() if k != "per_class"}), flush=True)
        if wait >= SETTINGS.patience:
            break
    max_change = (max(float(np.max(np.abs(w.numpy() - old))) for w, old in zip(model.trainable_weights, before))
                  if args.sanity else None)
    restored, _ = ssdlite.load(checkpoint)
    found = []
    for i in range(0, len(x_val), batch_size):
        p, d = restored(x_val[i:i + batch_size].astype(np.float32) / 255)
        found += [ssdlite.decode(pi, di, validation[i + j]["width"], validation[i + j]["height"], SELECTION_THRESHOLD)
                  for j, (pi, di) in enumerate(zip(p, d))]
    metrics = measure_detections(validation, found, SELECTION_THRESHOLD)
    passed = not args.sanity or (max_change > 0 and sum(map(len, found)) > 0)
    metadata.update({"sanity_gate_passed": passed, "duration_seconds": time.perf_counter() - started,
                     "history": history, "best_epoch": best_epoch, "completed_epochs": len(history),
                     "max_trainable_weight_change": max_change, "updates": int(optimizer.iterations.numpy()),
                     "metrics": metrics, "best_model_bytes": checkpoint.stat().st_size})
    (directory / "best.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(directory), "sanity_gate_passed": passed, "best_epoch": best_epoch,
                      "duration_seconds": metadata["duration_seconds"], "micro": metrics["micro"]}, indent=2), flush=True)
    if not passed:
        raise SystemExit("Sanity gate failed: inspect gradients/detections before full training.")


if __name__ == "__main__":
    main()
