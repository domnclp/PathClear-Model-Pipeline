"""Prove a manifest rebuilt on another machine is equivalent to the data FOMO v2 was scored on.

Checks, in order: conversion counts against the committed conversion record, split IDs and order
against the frozen split, calibration IDs against the v2 export, and finally that the frozen v2
INT8 model reproduces its recorded validation counts exactly on the rebuilt manifest.
"""
import argparse
import json

import numpy as np

import common  # noqa: F401
from analyze_baseline_v1 import measure
from config import ROOT, SETTINGS, output_directory
from dataset import load_arrays, validated_records

FROZEN = ROOT / "checkpoints" / "cityscapes_v2"
CONVERSION = ROOT / "artifacts" / "cityscapes-data-20261004" / "conversion.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Rebuilt center-target manifest")
    parser.add_argument("--output")
    args = parser.parse_args()
    records, report = validated_records(args.manifest)
    checks = {}
    conversion = json.loads(CONVERSION.read_text(encoding="utf-8"))
    checks["images"] = (len(records), conversion["converted_images"])
    checks["annotations"] = (report["annotations"], conversion["annotation_count"])
    checks["classes"] = (report["classes"], conversion["classes"])
    checks["empty_images"] = (report["images_without_relevant_objects"], conversion["images_without_relevant_objects"])
    splits = json.loads((FROZEN / "split_ids.json").read_text(encoding="utf-8"))
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    frozen_train = splits.get("train") or splits.get("train_ids")
    frozen_validation = splits.get("validation") or splits.get("validation_ids")
    checks["train_ids_in_order"] = ([r["id"] for r in train] == frozen_train, True)
    checks["validation_ids_in_order"] = ([r["id"] for r in validation] == frozen_validation, True)
    order = np.random.default_rng(SETTINGS.seed).permutation(len(train))[:SETTINGS.calibration_images]
    calibration = json.loads((FROZEN / "report.json").read_text(encoding="utf-8"))["calibration_ids"]
    checks["calibration_ids"] = ([train[i]["id"] for i in order] == calibration, True)

    from model import TFLiteModel
    model = TFLiteModel(FROZEN / "model.int8.tflite")
    images, _ = load_arrays(validation)
    grids = np.concatenate([model(images[i:i + 1].astype(np.float32) / 255) for i in range(len(images))])
    config = json.loads((FROZEN / "config.json").read_text(encoding="utf-8"))
    result, _ = measure(validation, grids, config["confidence_threshold"], decoder=config["decoder"])
    metrics = json.loads((FROZEN / "metrics.json").read_text(encoding="utf-8"))
    for key in ("tp", "fp", "fn", "ignored"):
        checks[f"v2_{key}"] = (result["micro"][key], metrics["micro"][key])
    passed = {name: actual == expected for name, (actual, expected) in checks.items()}
    summary = {"manifest": args.manifest, "all_passed": all(passed.values()),
               "checks": {name: {"actual": a, "expected": e, "passed": passed[name]} for name, (a, e) in checks.items()}}
    directory = output_directory(args.output, "rebuild-verification")
    (directory / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit("Rebuilt data is not equivalent to the FOMO v2 data; do not train comparison models on it.")


if __name__ == "__main__":
    main()
