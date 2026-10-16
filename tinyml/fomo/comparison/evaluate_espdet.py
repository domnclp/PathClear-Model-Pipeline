"""Export, INT8-quantize (ESP-PPQ, classic esp32 target) and score M1 ESPDet-Pico.

Run with the .venv-espdet interpreter. The quantized graph runs in ESP-PPQ's TorchExecutor
simulator, Espressif's own PC path for evaluating .espdl models, and is decoded exactly as
esp-detection's eval_quantized_model.py does. Detections then go through the shared scorer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

FOMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import average_precision, select_operating_point, sweep  # noqa: E402
from config import ROOT, CLASS_NAMES, SETTINGS, output_directory  # noqa: E402
from dataset import load_arrays, validated_records  # noqa: E402

ESPDET_ROOT = Path(os.environ.get("ESPDET_ROOT", "C:/Projects/esp-detection")).resolve()
NMS_IOU = 0.7  # esp-detection firmware default (espdet_detect.hpp default_nms_thr).
SCORE_FLOOR = 0.05
FIRMWARE_TOP_K = 10  # esp-detection firmware keeps at most 10 boxes; reported, not applied.


def decode(executor, batch, torch):
    """esp-detection ppq_graph_inference with nc=2, then Ultralytics class-aware NMS."""
    from ultralytics.nn.modules.head import Detect
    from ultralytics.utils.nms import non_max_suppression
    outputs = executor(batch)
    bs = batch.shape[0]
    boxes = torch.cat([outputs[2 * i].view(bs, 4, -1) for i in range(3)], dim=-1)
    scores = torch.cat([outputs[2 * i + 1].view(bs, len(CLASS_NAMES), -1) for i in range(3)], dim=-1)
    head = Detect(nc=len(CLASS_NAMES), reg_max=1, end2end=False, ch=[32, 64, 128])
    head.stride = [8.0, 16.0, 32.0]
    y = head._inference(dict(boxes=boxes, scores=scores, feats=[outputs[i] for i in range(0, 6, 2)]))
    return non_max_suppression(y, conf_thres=SCORE_FLOOR, iou_thres=NMS_IOU, max_det=300)


def to_detections(rows, record):
    sx, sy = record["width"] / SETTINGS.image_size, record["height"] / SETTINGS.image_size
    found = []
    for x1, y1, x2, y2, confidence, class_id in rows.tolist():
        x1, x2 = np.clip([x1, x2], 0, SETTINGS.image_size) * sx
        y1, y2 = np.clip([y1, y2], 0, SETTINGS.image_size) * sy
        if x2 <= x1 or y2 <= y1:
            continue
        found.append({"centroid_x": float((x1 + x2) / 2), "centroid_y": float((y1 + y2) / 2),
                      "coarse_class_id": int(class_id), "class_name": CLASS_NAMES[int(class_id)],
                      "confidence": float(confidence), "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
    return sorted(found, key=lambda d: (-d["confidence"], d["coarse_class_id"], d["centroid_y"], d["centroid_x"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", required=True, help="Ultralytics best.pt")
    parser.add_argument("--target", default="esp32", help="ESP-PPQ target; esp32 = classic ESP32 (ESP32-CAM)")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    weights = Path(args.weights).resolve()
    records, dataset_report = validated_records(args.manifest, args.allow_synthetic)
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    directory = output_directory(args.output, "m1-espdet-evaluation")
    os.chdir(ESPDET_ROOT)
    sys.path.insert(0, str(ESPDET_ROOT))
    import torch
    import ultralytics.nn.tasks as tasks
    from nn.esp_tasks import custom_parse_model
    from deploy.export import Export
    from deploy.quantize import quant_espdet
    from esp_ppq.executor import TorchExecutor
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    tasks.parse_model = custom_parse_model
    # Compatibility shim: esp-detection's export_onnx returns (file, model); Ultralytics 8.4 expects the file only.
    import deploy.export as espdet_export
    original_export_onnx = espdet_export.ESP_Detect_Exporter.export_onnx

    def export_onnx(self, *a, **k):
        result = original_export_onnx(self, *a, **k)
        return result[0] if isinstance(result, tuple) else result

    espdet_export.ESP_Detect_Exporter.export_onnx = export_onnx

    # Same 100 calibration images as FOMO v2 and M4, selected by ID from the seeded train permutation.
    order = np.random.default_rng(SETTINGS.seed).permutation(len(train))[:SETTINGS.calibration_images]
    chosen = [train[i] for i in order]
    expected = json.loads((ROOT / "checkpoints" / "cityscapes_v2" / "report.json").read_text())["calibration_ids"]
    if [r["id"] for r in chosen] != expected and not dataset_report["synthetic"]:
        raise ValueError("Calibration IDs differ from FOMO v2")
    calib = directory / "calibration"
    calib.mkdir()
    from PIL import Image
    pixels, _ = load_arrays(chosen)
    for record, image in zip(chosen, pixels):
        Image.fromarray(np.asarray(image)).save(calib / f"{record['id']}.png")

    staged = directory / "best.pt"
    shutil.copy2(weights, staged)
    Export(str(staged), SETTINGS.image_size)
    onnx_path = staged.with_suffix(".onnx")
    espdl_path = directory / f"espdet_pico_160_160_pathclear_{args.target}.espdl"
    graph = quant_espdet(onnx_path=str(onnx_path), target=args.target, num_of_bits=8, device="cpu", batchsz=32,
                         imgsz=SETTINGS.image_size, calib_dir=str(calib), espdl_model_path=str(espdl_path))
    executor = TorchExecutor(graph=graph, device="cpu")

    images, _ = load_arrays(validation)
    detections, model_ms = [], []
    with torch.no_grad():
        for record, image in zip(validation, images):
            batch = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))[None].astype(np.float32) / 255)
            start = time.perf_counter()
            rows = decode(executor, batch, torch)[0]
            model_ms.append((time.perf_counter() - start) * 1000)
            detections.append(to_detections(rows, record))
    results = sweep(validation, detections)
    selected = select_operating_point(results)
    ap = average_precision(validation, detections)
    model = YOLO(str(staged)).model
    parameters = sum(p.numel() for p in model.parameters())
    gflops = get_flops(model, imgsz=SETTINGS.image_size)
    row = {"model_id": "M1", "model_name": "ESPDet-Pico", "input_size": SETTINGS.image_size, "params": parameters,
           "macs": int(round(gflops * 1e9 / 2)), "int8_bytes": espdl_path.stat().st_size,
           "precision": selected["micro"]["precision"] if selected else None,
           "recall": selected["micro"]["recall"] if selected else None,
           "f1": selected["micro"]["f1"] if selected else None,
           "recall_lt8px": None, "recall_ge8px": selected["coarse_size_recall"]["ge8"]["recall"] if selected else None,
           "desktop_median_ms": None, "desktop_p95_ms": None,
           "g1_runtime": None, "g2_flash_bytes": None, "g3_arena_bytes": None, "g4_loop_fps": None,
           "notes": (f"threshold {selected['threshold']:.2f}; mAP@0.5 {ap['mean']:.4f}; " if selected else "") +
                    "ESP-PPQ simulator, so no desktop latency comparable to TFLite"}
    if selected:
        coarse = selected["coarse_size_recall"]
        row["recall_lt8px"] = (coarse["lt2"]["matched"] + coarse["2to8"]["matched"]) / max(
            coarse["lt2"]["total"] + coarse["2to8"]["total"], 1)
    report = {"split": "validation", "images": len(validation), "independent_test": False,
              "weights": str(weights), "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "espdl": str(espdl_path), "espdl_sha256": hashlib.sha256(espdl_path.read_bytes()).hexdigest(),
              "ppq_target": args.target, "calibration_ids": [r["id"] for r in chosen],
              "decoding": {"decoder": "Ultralytics Detect (reg_max 1) + class-aware NMS", "nms_iou": NMS_IOU,
                           "score_floor": SCORE_FLOOR, "max_det": 300,
                           "firmware_top_k_not_applied": FIRMWARE_TOP_K},
              "runtime": "ESP-PPQ TorchExecutor INT8 simulation on CPU",
              "simulator_ms_median": float(np.median(model_ms)),
              "timing_caveat": "Simulator timing is not a deployment latency and is not comparable to TFLite rows.",
              "gflops_ultralytics": gflops, "selected": selected, "map_0_5": ap,
              "sweep": [{"threshold": r["threshold"], **r["micro"]} for r in results], "results_row": row}
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "results_row.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (directory / "detections.json").write_text(json.dumps([{"id": r["id"], "detections": d}
                                                           for r, d in zip(validation, detections)]), encoding="utf-8")
    print(json.dumps(row, indent=2))
    print(f"Report: {directory}")


if __name__ == "__main__":
    main()
