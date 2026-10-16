"""Write the reviewed v2 data as an Ultralytics YOLO dataset for M1-M3.

Images are FOMO v2's cached 160 x 160 direct-stretch inputs, so YOLO trainers at imgsz=160 never
letterbox or resize them; labels are the same boxes, normalized, so they are valid in that space.
YOLO has no ignore regions: void/group areas become plain background (a recorded M1-M3 deviation).
"""
import argparse
import json

import numpy as np
from PIL import Image

import common  # noqa: F401
from config import CLASS_NAMES, output_directory
from dataset import load_arrays, validated_records

SPLITS = {"train": "train", "validation": "val"}


def yolo_lines(record):
    lines = []
    for annotation in record["annotations"]:
        x, y, w, h = annotation["bbox"]
        cx, cy = (x + w / 2) / record["width"], (y + h / 2) / record["height"]
        lines.append(f"{annotation['class_id']} {cx:.6f} {cy:.6f} {w / record['width']:.6f} {h / record['height']:.6f}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    records, _ = validated_records(args.manifest)
    directory = output_directory(args.output, "yolo-data")
    counts = {}
    for split, folder in SPLITS.items():
        subset = [r for r in records if r["split"] == split]
        images, _ = load_arrays(subset)
        (directory / "images" / folder).mkdir(parents=True)
        (directory / "labels" / folder).mkdir(parents=True)
        for record, pixels in zip(subset, images):
            Image.fromarray(np.asarray(pixels)).save(directory / "images" / folder / f"{record['id']}.png")
            (directory / "labels" / folder / f"{record['id']}.txt").write_text(
                "\n".join(yolo_lines(record)) + ("\n" if record["annotations"] else ""), encoding="utf-8")
        counts[folder] = {"images": len(subset), "boxes": sum(len(r["annotations"]) for r in subset)}
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    (directory / "data.yaml").write_text(
        f"path: {directory.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    (directory / "conversion.json").write_text(json.dumps(
        {"manifest": args.manifest, "counts": counts, "image_size": [160, 160],
         "image_source": "FOMO v2 uint8 cache (direct bilinear stretch)",
         "ignore_regions": "not representable in YOLO labels; treated as background"}, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"YOLO dataset: {directory}")


if __name__ == "__main__":
    main()
