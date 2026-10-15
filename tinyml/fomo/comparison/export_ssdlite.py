"""Full INT8 TFLite export of M4 with FOMO v2's training-only calibration images."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

import common  # noqa: F401
from config import ROOT, SETTINGS, output_directory
from dataset import load_arrays, validated_records
import ssdlite

V2_REPORT = ROOT / "checkpoints" / "cityscapes_v2" / "report.json"


def calibration_records(records):
    """The v2 rule (seeded permutation of the train split), checked against v2's recorded IDs."""
    train = [r for r in records if r["split"] == "train"]
    order = np.random.default_rng(SETTINGS.seed).permutation(len(train))[:SETTINGS.calibration_images]
    chosen = [train[i] for i in order]
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True, help="Calibration uses only this manifest's train split")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not str(args.model).endswith(".keras"):
        parser.error("Export requires the .keras checkpoint")
    records, dataset_report = validated_records(args.manifest, args.allow_synthetic)
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model, compile=False)
    metadata = json.loads((Path(args.model).parent / "best.json").read_text(encoding="utf-8"))
    if metadata["contract"] != ssdlite.contract():
        raise ValueError("Checkpoint contract differs from current SSDLite settings")
    if metadata["synthetic"] and not args.allow_synthetic:
        parser.error("Synthetic-trained model requires --allow-synthetic")
    chosen = calibration_records(records)
    expected = json.loads(V2_REPORT.read_text(encoding="utf-8"))["calibration_ids"]
    same_calibration = [r["id"] for r in chosen] == expected
    if not same_calibration and not dataset_report["synthetic"]:
        raise ValueError("Calibration IDs differ from FOMO v2; the rebuilt manifest is not comparable")
    calibration, _ = load_arrays(chosen)

    def representative():
        for pixels in calibration:
            yield [pixels.astype(np.float32)[None] / 255.0]

    # Firmware invokes one frame at a time. A fixed batch of 1 keeps the reshape layers static,
    # avoiding SHAPE/PACK/STRIDED_SLICE operators that an unknown batch dimension introduces.
    fixed_input = tf.keras.Input((ssdlite.IMAGE_SIZE, ssdlite.IMAGE_SIZE, 3), batch_size=1, name="rgb_0_to_1")
    fixed = tf.keras.Model(fixed_input, model(fixed_input), name=model.name)
    converter = tf.lite.TFLiteConverter.from_keras_model(fixed)
    directory = output_directory(args.output, "m4-ssdlite-export")
    with tempfile.TemporaryDirectory(prefix="conversion-", dir=directory) as scratch:
        previous = tempfile.tempdir
        try:
            tempfile.tempdir = scratch
            float_binary = converter.convert()
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            binary = converter.convert()
        finally:
            tempfile.tempdir = previous
    (directory / "model.int8.tflite").write_bytes(binary)
    (directory / "model.float32.tflite").write_bytes(float_binary)
    interpreter = tf.lite.Interpreter(model_content=binary)
    operators = sorted({op["op_name"] for op in interpreter._get_ops_details()})
    adapter = ssdlite.TFLiteSSD(directory / "model.int8.tflite")
    sample = calibration[:1].astype(np.float32) / 255.0
    adapter(sample)
    latencies = []
    for _ in range(5):
        start = time.perf_counter()
        adapter(sample)
        latencies.append((time.perf_counter() - start) * 1000)
    probabilities, deltas = adapter(sample)
    reference = model(sample, training=False)
    frame = chosen[0]
    report = {"contract": metadata["contract"], "synthetic": dataset_report["synthetic"] or metadata["synthetic"],
              "sanity": metadata["sanity"], "source_model": str(Path(args.model).resolve()), "model_bytes": len(binary), "float_tflite_bytes": len(float_binary),
              "keras_checkpoint_bytes": Path(args.model).stat().st_size, "decoding": ssdlite.decoding_policy(),
              "calibration_train_images": len(chosen), "calibration_ids": [r["id"] for r in chosen],
              "calibration_matches_fomo_v2": same_calibration,
              "input_quantization": list(adapter.input["quantization"]),
              "class_output_quantization": list(adapter.classes["quantization"]),
              "box_output_quantization": list(adapter.boxes["quantization"]),
              "tflite_operators": operators,
              "pc_int8_invoke_ms_median": float(np.median(latencies)),
              "sample_max_probability_difference": float(np.max(np.abs(probabilities - reference["class_probabilities"].numpy()))),
              "sample_max_delta_difference": float(np.max(np.abs(deltas - reference["box_deltas"].numpy()))),
              "sample_detections": ssdlite.decode(probabilities[0], deltas[0], frame["width"], frame["height"], 0.6),
              "embedded_flash_requirement": None, "embedded_ram_requirement": None,
              "esp32_latency_ms": None, "esp32_fps": None,
              "limitations": "PC check only; operator support and memory must be verified on the ESP32 runtime."}
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("calibration_ids", "sample_detections")}, indent=2))
    print(f"Export: {directory}")


if __name__ == "__main__":
    main()
