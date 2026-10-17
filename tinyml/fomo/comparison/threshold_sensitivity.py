"""Secondary analysis: the pre-registered rule with the threshold sweep extended down to 0.05.

The primary rule sweeps 0.20-0.95. When a model's selected threshold sits at 0.20 with precision
well above the 70% floor, the sweep range itself limits its recall. This script reports, for every
model, what the same rule selects over 0.05-0.95. It never replaces the primary results.
"""
import argparse
import json

import numpy as np

import common  # noqa: F401
from common import PRECISION_FLOOR, select_operating_point, sweep
from config import ROOT, output_directory
from dataset import load_arrays, validated_records
from inference import decode_grid

EXTENDED = tuple(p / 100 for p in range(5, 96, 5))
ART = ROOT / "artifacts"
SAVED = {"M4": "m4-ssdlite-evaluation", "M1": "m1-espdet-evaluation-1200", "M3": "m3-yolo11n-evaluation", "M2": "m2-yolofastest-evaluation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    records, _ = validated_records(args.manifest)
    validation = [r for r in records if r["split"] == "validation"]
    from model import TFLiteModel
    fomo = TFLiteModel(ROOT / "checkpoints" / "cityscapes_v2" / "model.int8.tflite")
    images, _ = load_arrays(validation)
    detections = {"M0": [decode_grid(fomo(images[i:i + 1].astype(np.float32) / 255)[0], r["width"], r["height"],
                                     0.0, 0, "cells") for i, r in enumerate(validation)]}
    for model_id, folder in SAVED.items():
        saved = json.loads((ART / folder / "detections.json").read_text(encoding="utf-8"))
        if [s["id"] for s in saved] != [r["id"] for r in validation]:
            raise ValueError(f"{model_id} detections do not match the validation order")
        detections[model_id] = [s["detections"] for s in saved]
    summary = {}
    for model_id, found in detections.items():
        primary = select_operating_point(sweep(validation, found))
        extended = select_operating_point(sweep(validation, found, EXTENDED))
        summary[model_id] = {k: {"threshold": v["threshold"], **{m: v["micro"][m] for m in ("precision", "recall", "f1")}}
                             for k, v in (("primary_0.20_0.95", primary), ("extended_0.05_0.95", extended))}
        print(model_id, json.dumps(summary[model_id]))
    directory = output_directory(args.output, "threshold-sensitivity")
    (directory / "sensitivity.json").write_text(json.dumps(
        {"precision_floor": PRECISION_FLOOR, "extended_thresholds": list(EXTENDED), "models": summary,
         "status": "secondary analysis; primary pre-registered results unchanged"}, indent=2), encoding="utf-8")
    print(f"Report: {directory}")


if __name__ == "__main__":
    main()
