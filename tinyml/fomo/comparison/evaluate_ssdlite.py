"""Score M4 with the pre-registered comparison rule and write one results row."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import common
from common import SELECTION_THRESHOLD, average_precision, measure_detections, select_operating_point, sweep
from config import SETTINGS, output_directory
from dataset import load_arrays, validated_records
import ssdlite


def timing(callable_, images, records, warmups=20, measured=200):
    """v2 desktop protocol: one thread, 20 warm-ups, 200 calls over up to 100 prepared inputs."""
    inputs = [images[i:i + 1].astype(np.float32) / 255 for i in range(min(100, len(images)))]
    for i in range(warmups):
        callable_(inputs[i % len(inputs)])
    model_ms, total_ms = [], []
    for i in range(measured):
        record = records[i % len(inputs)]
        start = time.perf_counter()
        p, d = callable_(inputs[i % len(inputs)])
        middle = time.perf_counter()
        ssdlite.decode(p[0], d[0], record["width"], record["height"], SELECTION_THRESHOLD)
        end = time.perf_counter()
        model_ms.append((middle - start) * 1000)
        total_ms.append((end - start) * 1000)
    return {"model_call_ms_median": float(np.median(model_ms)), "model_call_ms_p95": float(np.percentile(model_ms, 95)),
            "model_plus_decode_ms_median": float(np.median(total_ms)),
            "model_plus_decode_ms_p95": float(np.percentile(total_ms, 95)),
            "warmups": warmups, "measured_calls": measured, "distinct_inputs": len(inputs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True, help="model.int8.tflite for the comparison row; .keras for diagnosis")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    records, dataset_report = validated_records(args.manifest, args.allow_synthetic)
    records = [r for r in records if r["split"] == args.split]
    if not records:
        parser.error(f"No {args.split} records")
    forward, metadata = ssdlite.load(args.model)
    if metadata["synthetic"] and not args.allow_synthetic:
        parser.error("Synthetic model requires --allow-synthetic")
    images, _ = load_arrays(records)
    probabilities, deltas = [], []
    for i in range(0, len(images), 1 if args.model.endswith(".tflite") else SETTINGS.batch_size):
        p, d = forward(images[i:i + (1 if args.model.endswith(".tflite") else SETTINGS.batch_size)].astype(np.float32) / 255)
        probabilities.append(p)
        deltas.append(d)
    probabilities, deltas = np.concatenate(probabilities), np.concatenate(deltas)
    detections = [ssdlite.decode(p, d, r["width"], r["height"]) for p, d, r in zip(probabilities, deltas, records)]
    results = sweep(records, detections)
    selected = select_operating_point(results)
    fixed = measure_detections(records, detections, SELECTION_THRESHOLD)
    ap = average_precision(records, detections)
    model_path = Path(args.model)
    parameters = macs = None
    training_metadata_path = model_path.parent / "best.json"
    if model_path.suffix == ".keras":
        import tensorflow as tf
        keras_model = tf.keras.models.load_model(model_path, compile=False)
        parameters, macs = keras_model.count_params(), common.count_macs(keras_model)
    elif "source_model" in metadata:
        training_metadata_path = Path(metadata["source_model"]).parent / "best.json"
    if training_metadata_path.is_file():
        training = json.loads(training_metadata_path.read_text(encoding="utf-8"))
        parameters, macs = training.get("parameters", parameters), training.get("macs", macs)
    desktop = timing(forward, images, records)
    row = {"model_id": "M4", "model_name": "SSDLite-MobileNetV2 alpha 0.35", "input_size": ssdlite.IMAGE_SIZE,
           "params": parameters, "macs": macs, "int8_bytes": model_path.stat().st_size if model_path.suffix == ".tflite" else None,
           "precision": selected["micro"]["precision"] if selected else None,
           "recall": selected["micro"]["recall"] if selected else None,
           "f1": selected["micro"]["f1"] if selected else None,
           "recall_lt8px": None, "recall_ge8px": selected["coarse_size_recall"]["ge8"]["recall"] if selected else None,
           "desktop_median_ms": desktop["model_plus_decode_ms_median"], "desktop_p95_ms": desktop["model_plus_decode_ms_p95"],
           "g1_runtime": None, "g2_flash_bytes": None, "g3_arena_bytes": None, "g4_loop_fps": None,
           "notes": f"threshold {selected['threshold']:.2f}; mAP@0.5 {ap['mean']:.4f}" if selected and ap["mean"] is not None
                    else "no threshold reached the precision floor"}
    if selected:
        coarse = selected["coarse_size_recall"]
        small = coarse["lt2"]["matched"] + coarse["2to8"]["matched"]
        row["recall_lt8px"] = small / max(coarse["lt2"]["total"] + coarse["2to8"]["total"], 1)
    report = {"split": args.split, "images": len(records), "synthetic": dataset_report["synthetic"] or metadata["synthetic"],
              "independent_test": False, "model_path": str(model_path.resolve()),
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "model_bytes": model_path.stat().st_size,
              "runtime": "TFLite CPU, one thread" if model_path.suffix == ".tflite" else "TensorFlow compiled inference",
              "decoding": ssdlite.decoding_policy(),
              "rule": {"thresholds": list(common.SWEEP_THRESHOLDS), "precision_floor": common.PRECISION_FLOOR,
                       "selection": "Highest recall at or above precision floor; tie-break highest precision"},
              "selected": selected, "fixed_threshold_0_60": fixed, "map_0_5": ap, "desktop_timing": desktop,
              "timing_scope": "prepared normalized input; model call includes quantization and output copy; "
                              "decode adds box decoding and NMS; excludes camera and decision logic",
              "sweep": [{"threshold": r["threshold"], **r["micro"]} for r in results], "results_row": row}
    directory = output_directory(args.output, "m4-ssdlite-evaluation")
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "results_row.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (directory / "detections.json").write_text(json.dumps([{"id": r["id"], "detections": d}
                                                           for r, d in zip(records, detections)]), encoding="utf-8")
    np.save(directory / "probabilities.npy", probabilities.astype(np.float16))
    np.save(directory / "deltas.npy", deltas.astype(np.float16))
    print(json.dumps({"selected_threshold": selected["threshold"] if selected else None, "row": row}, indent=2))
    print(f"Report: {directory}")


if __name__ == "__main__":
    main()
