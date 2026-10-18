"""Benchmark inference latency, params, and file size for doc comparison.

Loads the Fast-SCNN PathClear adaptation checkpoint and the MicroSeg s800
checkpoint, runs timed forward passes on CPU, and writes a JSON report of
params, file size, and latency. Accuracy numbers come from the separate
evaluation reports. No training, no weight changes.
"""
import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np

import pathclear_stages  # noqa: F401  (registers ActiveClasses for deserialization)
from microseg import model as _microseg_model  # noqa: F401

FAST_CKPT = Path("/home/nacsi/pathclear_runs/shared_bnd_epoch1_20261006/best.keras")
MICRO_CKPT = Path(
    "/home/nacsi/microseg_runs/pilot-neg025-s800-augoff-fixed-20261006/best.keras"
)
OUTPUT = Path("/home/nacsi/doc_benchmark.json")

WARMUP = 3
TIMED = 20


def load_predict_fn(path):
    model = keras.models.load_model(path, compile=False, safe_mode=False)
    input_shape = tuple(model.input_shape[1:])
    params = int(model.count_params())

    def predict(batch):
        return model(batch, training=False)

    predict(model_input := np.zeros((1, *input_shape), dtype=np.float32))
    return predict, model_input, params, input_shape


def time_forward(predict, sample):
    for _ in range(WARMUP):
        predict(sample)
    times = []
    for _ in range(TIMED):
        start = time.perf_counter()
        predict(sample)
        times.append((time.perf_counter() - start) * 1000.0)
    arr = np.array(times)
    return {
        "runs": TIMED,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "fps_mean": float(1000.0 / arr.mean()),
        "fps_median": float(1000.0 / np.median(arr)),
    }


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast-ckpt",
        type=Path,
        default=FAST_CKPT,
        help="Fast-SCNN PathClear checkpoint to benchmark.",
    )
    parser.add_argument(
        "--micro-ckpt", type=Path, default=MICRO_CKPT, help="MicroSeg checkpoint."
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help="Where to write the JSON report."
    )
    return parser.parse_args()


def main():
    args = _arguments()
    report = {}
    for name, path in (
        ("fast_scnn", args.fast_ckpt),
        ("microseg", args.micro_ckpt),
    ):
        predict, sample, params, input_shape = load_predict_fn(path)
        latency = time_forward(predict, sample)
        report[name] = {
            "checkpoint": str(path),
            "file_bytes": path.stat().st_size,
            "file_kb": round(path.stat().st_size / 1024, 1),
            "params": params,
            "input_shape": list(input_shape),
            "latency_cpu_ms": latency,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
