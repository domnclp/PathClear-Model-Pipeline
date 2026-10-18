"""Write teacher labels for KPV frames from the pre-KPV PathClear checkpoint.

KPV masks mark only the crossing. Without a reference for the rest of the
frame, the old horizon split taught the head that everything above the
horizon is Boundary and everything below is Road, which erased Person,
Vehicle and Vegetation on pedestrian-view photos. This script runs the
seven-class start checkpoint (gate on, so it can only emit IDs 0-6) over
every KPV frame in the split manifest and saves its argmax at 512x1024 as
uint8 PNG, with pixels below --threshold softmax confidence set to Ignore
(255). `kpv_loader.attach_teacher_labels` reads the resulting manifest.

Read-only with respect to the checkpoint (hashed before and after).
"""
import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import pathclear_stages  # noqa: F401  (registers ActiveClasses for deserialization)
from kpv_loader import load_kpv_image, pairs_from_split_manifest
from pathclear_pipeline import sha256
from pathclear_taxonomy import IGNORE_LABEL

DEFAULT_TEACHER = Path("/home/nacsi/pathclear_runs/shared_bnd_epoch1_20261006/best.keras")
DEFAULT_SPLIT_MANIFEST = Path("/home/nacsi/microseg_runs/crosswalk_split.json")


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: first N frames.")
    return parser.parse_args()


def main():
    args = _arguments()
    if not 0.0 <= args.threshold < 1.0:
        raise ValueError("--threshold must be in [0, 1).")
    output = args.output_dir.resolve()
    (output / "labels").mkdir(parents=True, exist_ok=True)
    before_hash = sha256(args.teacher)
    model = keras.models.load_model(args.teacher, compile=False)
    if model.layers[-1].name != pathclear_stages.GATE_NAME:
        raise ValueError("Teacher must be the gated seven-class stage checkpoint.")

    pairs = []
    for split in ("train", "validation"):
        pairs += pairs_from_split_manifest(args.split_manifest, split=split)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    images = tf.data.Dataset.from_tensor_slices([p[0] for p in pairs])
    images = images.map(load_kpv_image, num_parallel_calls=2)
    images = images.batch(args.batch_size).prefetch(1)

    labels, confident, index = {}, [], 0
    for batch in images:
        probs = tf.nn.softmax(model(batch, training=False), axis=-1).numpy()
        argmax = probs.argmax(-1).astype(np.uint8)
        argmax[probs.max(-1) < args.threshold] = IGNORE_LABEL
        for label in argmax:
            image_path = pairs[index][0]
            label_path = output / "labels" / f"{index:05d}.png"
            tf.io.write_file(str(label_path), tf.io.encode_png(label[..., None]))
            labels[image_path] = str(label_path)
            confident.append(float((label != IGNORE_LABEL).mean()))
            index += 1
        if index % 200 < args.batch_size:
            print(f"Labelled {index}/{len(pairs)}", flush=True)

    if sha256(args.teacher) != before_hash:
        raise AssertionError("Teacher labelling changed checkpoint weights.")
    manifest = {
        "schema_version": 1,
        "teacher_checkpoint": str(args.teacher),
        "teacher_sha256": before_hash,
        "split_manifest": str(args.split_manifest),
        "threshold": args.threshold,
        "frames": len(labels),
        "mean_confident_fraction": float(np.mean(confident)),
        "labels": labels,
    }
    (output / "teacher_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(labels)} labels; mean confident fraction "
          f"{manifest['mean_confident_fraction']:.3f}", flush=True)


if __name__ == "__main__":
    main()
