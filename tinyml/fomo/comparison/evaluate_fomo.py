"""Score frozen FOMO v2 (M0) through the same comparison scorer and desktop timing as other candidates."""
import argparse
import hashlib
import json
import time

import numpy as np

import common
from common import average_precision, measure_detections, select_operating_point, sweep
from config import ROOT, SETTINGS, output_directory
from dataset import load_arrays, validated_records
from inference import decode_grid

FROZEN = ROOT / "checkpoints" / "cityscapes_v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    records, _ = validated_records(args.manifest)
    records = [r for r in records if r["split"] == "validation"]
    config = json.loads((FROZEN / "config.json").read_text(encoding="utf-8"))
    decoder = config["decoder"]
    from model import TFLiteModel
    import tensorflow as tf
    path = FROZEN / "model.int8.tflite"
    model = TFLiteModel(path)
    images, _ = load_arrays(records)
    grids = np.concatenate([model(images[i:i + 1].astype(np.float32) / 255) for i in range(len(images))])
    # Threshold 0 keeps every winning cell; the shared scorer applies each sweep threshold afterwards.
    detections = [decode_grid(g, r["width"], r["height"], 0.0, 0, decoder) for g, r in zip(grids, records)]
    results = sweep(records, detections)
    selected = select_operating_point(results)
    if selected["threshold"] != config["confidence_threshold"]:
        raise RuntimeError("Shared rule no longer reproduces the frozen v2 operating point")
    inputs = [images[i:i + 1].astype(np.float32) / 255 for i in range(min(100, len(images)))]
    for i in range(20):
        model(inputs[i % len(inputs)])
    model_ms, total_ms = [], []
    for i in range(200):
        record = records[i % len(inputs)]
        start = time.perf_counter()
        grid = model(inputs[i % len(inputs)])[0]
        middle = time.perf_counter()
        decode_grid(grid, record["width"], record["height"], selected["threshold"], 0, decoder)
        end = time.perf_counter()
        model_ms.append((middle - start) * 1000)
        total_ms.append((end - start) * 1000)
    keras_model = tf.keras.models.load_model(FROZEN / "best.keras", compile=False)
    coarse = selected["coarse_size_recall"]
    row = {"model_id": "M0", "model_name": "FOMO v2 (frozen cityscapes_v2)", "input_size": SETTINGS.image_size,
           "params": keras_model.count_params(), "macs": common.count_macs(keras_model),
           "int8_bytes": path.stat().st_size, "precision": selected["micro"]["precision"],
           "recall": selected["micro"]["recall"], "f1": selected["micro"]["f1"],
           "recall_lt8px": (coarse["lt2"]["matched"] + coarse["2to8"]["matched"]) /
                           max(coarse["lt2"]["total"] + coarse["2to8"]["total"], 1),
           "recall_ge8px": coarse["ge8"]["recall"],
           "desktop_median_ms": float(np.median(total_ms)), "desktop_p95_ms": float(np.percentile(total_ms, 95)),
           "g1_runtime": None, "g2_flash_bytes": None, "g3_arena_bytes": None, "g4_loop_fps": None,
           "notes": f"threshold {selected['threshold']:.2f}; centroid-only output, no mAP"}
    report = {"split": "validation", "images": len(records), "independent_test": False,
              "model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "decoder": decoder, "selected": selected,
              "desktop_timing": {"model_call_ms_median": float(np.median(model_ms)),
                                 "model_call_ms_p95": float(np.percentile(model_ms, 95)),
                                 "model_plus_decode_ms_median": row["desktop_median_ms"],
                                 "model_plus_decode_ms_p95": row["desktop_p95_ms"],
                                 "warmups": 20, "measured_calls": 200, "distinct_inputs": len(inputs)},
              "sweep": [{"threshold": r["threshold"], **r["micro"]} for r in results], "results_row": row}
    directory = output_directory(args.output, "m0-fomo-evaluation")
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "results_row.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))
    print(f"Report: {directory}")


if __name__ == "__main__":
    main()
