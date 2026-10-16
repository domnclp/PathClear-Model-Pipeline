"""Export, INT8-quantize (ESP-PPQ) and score Ultralytics-based candidates (M1 ESPDet-Pico, M3 YOLO11n).

Run with the .venv-espdet interpreter. Each model keeps its maker's deployment recipe:
- espdet: esp-detection's deploy/export.py and deploy/quantize.py (equalization) settings.
- yolo11n: Espressif's "How to deploy YOLO11n" tutorial (percentile calibration, TQT 800 steps, ESP32-S3).
The quantized graph runs in ESP-PPQ's TorchExecutor simulator and is decoded with Ultralytics' Detect head,
as esp-detection's eval_quantized_model.py does; detections then go through the shared scorer.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402,F401
from common import average_precision, select_operating_point, sweep  # noqa: E402
from config import ROOT, CLASS_NAMES, SETTINGS, output_directory  # noqa: E402
from dataset import load_arrays, validated_records  # noqa: E402

ESPDET_ROOT = Path(os.environ.get("ESPDET_ROOT", "C:/Projects/esp-detection")).resolve()
ARCHES = {"espdet": {"model_id": "M1", "name": "ESPDet-Pico", "reg_max": 1},
          "yolo11n": {"model_id": "M3", "name": "YOLO11n", "reg_max": 16}}
NMS_IOU = 0.7  # Ultralytics validation default and esp-detection firmware default_nms_thr.
SCORE_FLOOR = 0.05
FIRMWARE_TOP_K = 10  # esp-detection firmware keeps at most 10 boxes; reported, not applied.


def export_onnx(weights, arch):
    """esp-detection / esp-dl export: Detect head returns six raw tensors, opset 13, onnxsim."""
    import deploy.export as espdet_export
    from nn.modules import ESPDetect
    from ultralytics.nn.modules import Attention, Detect
    original = espdet_export.ESP_Detect_Exporter.export_onnx

    def export_file_only(self, *a, **k):  # Ultralytics 8.4 expects the file path, not (file, model).
        result = original(self, *a, **k)
        return result[0] if isinstance(result, tuple) else result

    espdet_export.ESP_Detect_Exporter.export_onnx = export_file_only
    model = espdet_export.ESP_YOLO(str(weights))
    head_type = ESPDetect if arch == "espdet" else Detect
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = espdet_export.ESP_Attention.forward.__get__(module)
        if isinstance(module, head_type):
            # Same body as ESPDetect.export_onnx_forward and esp-dl's ESP_Detect.forward.
            module.forward = ESPDetect.export_onnx_forward.__get__(module)
    model.export(format="onnx", simplify=True, opset=13, imgsz=SETTINGS.image_size)
    return weights.with_suffix(".onnx")


def quantize(arch, onnx_path, target, calib_dir, espdl_path):
    if arch == "espdet":
        from deploy.quantize import quant_espdet
        return quant_espdet(onnx_path=str(onnx_path), target=target, num_of_bits=8, device="cpu", batchsz=32,
                            imgsz=SETTINGS.image_size, calib_dir=str(calib_dir), espdl_model_path=str(espdl_path))
    import onnx
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx
    from onnxsim import simplify
    from torch.utils.data import DataLoader
    from deploy.quantize import CaliDataset
    model, check = simplify(onnx.load(str(onnx_path)))
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx.shape_inference.infer_shapes(model), str(onnx_path))
    setting = QuantizationSettingFactory.espdl_setting()
    setting.quantize_activation_setting.calib_algorithm = "percentile"
    setting.tqt_optimization = True
    tqt = setting.tqt_optimization_setting
    tqt.lr, tqt.steps, tqt.block_size = 1e-5, 800, 4
    tqt.is_scale_trainable, tqt.gamma, tqt.int_lambda = True, 0.0, 0.0
    tqt.collecting_device = "cpu"  # The tutorial uses cuda; this laptop has no CUDA build.
    loader = DataLoader(CaliDataset(str(calib_dir), img_shape=SETTINGS.image_size), batch_size=32, shuffle=False)
    return espdl_quantize_onnx(onnx_import_file=str(onnx_path), espdl_export_file=str(espdl_path),
                               calib_dataloader=loader, calib_steps=32,
                               input_shape=[1, 3, SETTINGS.image_size, SETTINGS.image_size], target=target,
                               num_of_bits=8, collate_fn=lambda b: b.to("cpu"), setting=setting, device="cpu",
                               error_report=True, skip_export=False, export_test_values=False, verbose=0)


def decode(executor, batch, reg_max, torch):
    """esp-detection ppq_graph_inference generalized to nc=2 and either box head, then class-aware NMS."""
    from ultralytics.nn.modules.head import Detect
    from ultralytics.utils.nms import non_max_suppression
    outputs = executor(batch)
    bs = batch.shape[0]
    boxes = torch.cat([outputs[2 * i].view(bs, 4 * reg_max, -1) for i in range(3)], dim=-1)
    scores = torch.cat([outputs[2 * i + 1].view(bs, len(CLASS_NAMES), -1) for i in range(3)], dim=-1)
    head = Detect(nc=len(CLASS_NAMES), reg_max=reg_max, end2end=False, ch=[64, 128, 256])
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
    parser.add_argument("--arch", choices=sorted(ARCHES), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", required=True, help="Ultralytics best.pt")
    parser.add_argument("--target", default="esp32", help="ESP-PPQ target; esp32 = classic ESP32 (ESP32-CAM)")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    spec = ARCHES[args.arch]
    weights = Path(args.weights).resolve()
    records, dataset_report = validated_records(args.manifest, args.allow_synthetic)
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    directory = output_directory(args.output, f"{args.arch}-evaluation")
    os.chdir(ESPDET_ROOT)
    sys.path.insert(0, str(ESPDET_ROOT))
    import torch
    import ultralytics.nn.tasks as tasks
    from esp_ppq.executor import TorchExecutor
    from nn.esp_tasks import custom_parse_model
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    tasks.parse_model = custom_parse_model  # Adds ESP blocks; standard YOLO11 modules parse unchanged.

    # Same 100 calibration images as FOMO v2 and M4, selected by ID from the seeded train permutation.
    order = np.random.default_rng(SETTINGS.seed).permutation(len(train))[:SETTINGS.calibration_images]
    chosen = [train[i] for i in order]
    expected = json.loads((ROOT / "checkpoints" / "cityscapes_v2" / "report.json").read_text())["calibration_ids"]
    if [r["id"] for r in chosen] != expected and not dataset_report["synthetic"]:
        raise ValueError("Calibration IDs differ from FOMO v2")
    calib = directory / "calibration"
    calib.mkdir()
    pixels, _ = load_arrays(chosen)
    for record, image in zip(chosen, pixels):
        Image.fromarray(np.asarray(image)).save(calib / f"{record['id']}.png")

    staged = directory / "best.pt"
    shutil.copy2(weights, staged)
    onnx_path = export_onnx(staged, args.arch)
    espdl_path = directory / f"{args.arch}_160_160_pathclear_{args.target}.espdl"
    executor = TorchExecutor(graph=quantize(args.arch, onnx_path, args.target, calib, espdl_path), device="cpu")

    images, _ = load_arrays(validation)
    detections, model_ms = [], []
    with torch.no_grad():
        for record, image in zip(validation, images):
            batch = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))[None].astype(np.float32) / 255)
            start = time.perf_counter()
            rows = decode(executor, batch, spec["reg_max"], torch)[0]
            model_ms.append((time.perf_counter() - start) * 1000)
            detections.append(to_detections(rows, record))
    results = sweep(validation, detections)
    selected = select_operating_point(results)
    ap = average_precision(validation, detections)
    model = YOLO(str(staged)).model
    parameters = sum(p.numel() for p in model.parameters())
    gflops = get_flops(model, imgsz=SETTINGS.image_size)
    row = {"model_id": spec["model_id"], "model_name": spec["name"], "input_size": SETTINGS.image_size,
           "params": parameters, "macs": int(round(gflops * 1e9 / 2)), "int8_bytes": espdl_path.stat().st_size,
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
    report = {"arch": args.arch, "split": "validation", "images": len(validation), "independent_test": False,
              "weights": str(weights), "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "espdl": str(espdl_path), "espdl_sha256": hashlib.sha256(espdl_path.read_bytes()).hexdigest(),
              "ppq_target": args.target, "calibration_ids": [r["id"] for r in chosen],
              "decoding": {"decoder": f"Ultralytics Detect (reg_max {spec['reg_max']}) + class-aware NMS",
                           "nms_iou": NMS_IOU, "score_floor": SCORE_FLOOR, "max_det": 300,
                           "firmware_top_k_not_applied": FIRMWARE_TOP_K if args.arch == "espdet" else None},
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
