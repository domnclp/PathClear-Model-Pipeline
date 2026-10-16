"""Export, INT8-quantize (ESP-PPQ) and score M2 YOLO-FastestV2 with its own pre/post-processing.

Run with the .venv-espdet interpreter. Export follows the repository's pytorch2onnx.py (export_onnx=True,
which folds sigmoid/softmax into two NHWC outputs). The INT8 graph runs in ESP-PPQ's TorchExecutor and is
decoded exactly like the repository's handel_preds, then its non_max_suppression (IoU 0.4, as in test.py),
then the shared scorer. Input is BGR uint8 / 255, the repository's own convention (cv2.imread).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402,F401
from common import average_precision, select_operating_point, sweep  # noqa: E402
from config import ROOT, CLASS_NAMES, SETTINGS, output_directory  # noqa: E402
from dataset import load_arrays, validated_records  # noqa: E402
import run_yolofastest  # noqa: E402,F401  (restores removed NumPy aliases)

YOLOFASTEST_ROOT = run_yolofastest.ROOT
NMS_IOU = 0.4  # Yolo-FastestV2 test.py
SCORE_FLOOR = 0.05


def count_macs(model, size, torch):
    """Multiply-accumulates of Conv2d and Linear layers, via forward hooks."""
    total = [0]

    def hook(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            kh, kw = module.kernel_size
            total[0] += output.numel() * kh * kw * module.in_channels // module.groups
        elif isinstance(module, torch.nn.Linear):
            total[0] += output.numel() * module.in_features

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for handle in handles:
        handle.remove()
    return int(total[0])


def decode(outputs, anchors, anchor_num, torch):
    """handel_preds on the export graph's already-activated NHWC outputs (one per stride level)."""
    rows = []
    for level, out in enumerate(outputs):
        out = out[0]  # H, W, C
        h, w = out.shape[:2]
        reg = out[..., :4 * anchor_num].reshape(h, w, anchor_num, 4)
        obj = out[..., 4 * anchor_num:5 * anchor_num].reshape(h, w, anchor_num, 1)
        cls = out[..., 5 * anchor_num:].reshape(h, w, 1, -1).repeat(1, 1, anchor_num, 1)
        hv, wv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
        grid = torch.stack((wv, hv), 2).repeat(1, 1, anchor_num).reshape(h, w, anchor_num, 2).float()
        stride = SETTINGS.image_size / h
        xy = ((reg[..., :2] * 2.0 - 0.5) + grid) * stride
        wh = (reg[..., 2:4] * 2) ** 2 * anchors[level]
        rows.append(torch.cat((xy, wh, obj, cls), -1).reshape(-1, 5 + cls.shape[-1]))
    return torch.cat(rows, 0)[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", required=True, help="Yolo-FastestV2 .pth checkpoint")
    parser.add_argument("--data", required=True, help="pathclear.data used for training")
    parser.add_argument("--target", default="esp32")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    weights, data = Path(args.weights).resolve(), Path(args.data).resolve()
    records, dataset_report = validated_records(args.manifest, args.allow_synthetic)
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    directory = output_directory(args.output, "yolofastest-evaluation")
    os.chdir(YOLOFASTEST_ROOT)
    sys.path.insert(0, str(YOLOFASTEST_ROOT))
    import onnx
    import torch
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor import TorchExecutor
    from onnxsim import simplify
    # The clone's model/ and utils/ are namespace packages (no __init__.py), which lose to FOMO's
    # model.py on sys.path; hide the FOMO folders while importing them.
    saved_path = sys.path[:]
    hidden = {common.FOMO_DIR.resolve(), common.COMPARISON_DIR.resolve()}
    sys.path = [p for p in sys.path if not p or Path(p).resolve() not in hidden]
    import model.detector
    import utils.utils as yf_utils
    sys.path = saved_path
    cfg = yf_utils.load_datafile(str(data))
    anchors = torch.tensor(np.array(cfg["anchors"], dtype=np.float32).reshape(2, cfg["anchor_num"], 2))

    # Official pytorch2onnx.py: Detector(classes, anchor_num, load_param=True, export_onnx=True), opset 11.
    detector = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, True)
    detector.load_state_dict(torch.load(str(weights), map_location="cpu"))
    detector.eval()
    onnx_path = directory / "model.onnx"
    torch.onnx.export(detector, torch.rand(1, 3, cfg["height"], cfg["width"]), str(onnx_path), export_params=True,
                      opset_version=11, do_constant_folding=True)
    simplified, check = simplify(onnx.load(str(onnx_path)))
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx.shape_inference.infer_shapes(simplified), str(onnx_path))

    order = np.random.default_rng(SETTINGS.seed).permutation(len(train))[:SETTINGS.calibration_images]
    chosen = [train[i] for i in order]
    expected = json.loads((ROOT / "checkpoints" / "cityscapes_v2" / "report.json").read_text())["calibration_ids"]
    if [r["id"] for r in chosen] != expected and not dataset_report["synthetic"]:
        raise ValueError("Calibration IDs differ from FOMO v2")

    def bgr(images):  # RGB uint8 cache -> BGR float NCHW / 255, the repository's training input.
        return torch.from_numpy(np.ascontiguousarray(images[..., ::-1].transpose(0, 3, 1, 2)).astype(np.float32) / 255)

    calibration, _ = load_arrays(chosen)
    # The official export bakes batch size 1 into ShuffleNetV2's channel-shuffle reshapes, so calibration
    # runs one image per step; 100 steps still cover all 100 calibration images, like every other model.
    batches = [bgr(calibration[i:i + 1]) for i in range(len(calibration))]
    espdl_path = directory / f"yolofastestv2_160_160_pathclear_{args.target}.espdl"
    graph = espdl_quantize_onnx(onnx_import_file=str(onnx_path), espdl_export_file=str(espdl_path),
                                calib_dataloader=batches, calib_steps=len(batches),
                                input_shape=[1, 3, cfg["height"], cfg["width"]], target=args.target, num_of_bits=8,
                                collate_fn=lambda b: b.to("cpu"), setting=QuantizationSettingFactory.espdl_setting(),
                                device="cpu", error_report=True, skip_export=False, export_test_values=False, verbose=0)
    executor = TorchExecutor(graph=graph, device="cpu")

    images, _ = load_arrays(validation)
    detections, model_ms = [], []
    with torch.no_grad():
        for record, image in zip(validation, images):
            start = time.perf_counter()
            outputs = executor(bgr(image[None]))
            rows = yf_utils.non_max_suppression(decode(outputs, anchors, cfg["anchor_num"], torch),
                                                conf_thres=SCORE_FLOOR, iou_thres=NMS_IOU)[0]
            model_ms.append((time.perf_counter() - start) * 1000)
            sx, sy = record["width"] / SETTINGS.image_size, record["height"] / SETTINGS.image_size
            found = []
            for x1, y1, x2, y2, confidence, class_id in rows.tolist():
                x1, x2 = np.clip([x1, x2], 0, SETTINGS.image_size) * sx
                y1, y2 = np.clip([y1, y2], 0, SETTINGS.image_size) * sy
                if x2 > x1 and y2 > y1:
                    found.append({"centroid_x": float((x1 + x2) / 2), "centroid_y": float((y1 + y2) / 2),
                                  "coarse_class_id": int(class_id), "class_name": CLASS_NAMES[int(class_id)],
                                  "confidence": float(confidence),
                                  "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
            detections.append(sorted(found, key=lambda d: (-d["confidence"], d["coarse_class_id"],
                                                           d["centroid_y"], d["centroid_x"])))
    results = sweep(validation, detections)
    selected = select_operating_point(results)
    ap = average_precision(validation, detections)
    plain = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, False)
    plain.load_state_dict(torch.load(str(weights), map_location="cpu"))
    plain.eval()
    row = {"model_id": "M2", "model_name": "YOLO-FastestV2", "input_size": SETTINGS.image_size,
           "params": sum(p.numel() for p in plain.parameters()), "macs": count_macs(plain, SETTINGS.image_size, torch),
           "int8_bytes": espdl_path.stat().st_size,
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
              "ppq_target": args.target, "quantization": "ESP-PPQ default espdl_setting (no official recipe)",
              "calibration_ids": [r["id"] for r in chosen], "input": "BGR uint8 / 255",
              "decoding": {"decoder": "Yolo-FastestV2 handel_preds on activated export outputs",
                           "nms": "repository non_max_suppression", "nms_iou": NMS_IOU, "score_floor": SCORE_FLOOR},
              "runtime": "ESP-PPQ TorchExecutor INT8 simulation on CPU",
              "simulator_ms_median": float(np.median(model_ms)), "selected": selected, "map_0_5": ap,
              "sweep": [{"threshold": r["threshold"], **r["micro"]} for r in results], "results_row": row}
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "results_row.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (directory / "detections.json").write_text(json.dumps([{"id": r["id"], "detections": d}
                                                           for r, d in zip(validation, detections)]), encoding="utf-8")
    print(json.dumps(row, indent=2))
    print(f"Report: {directory}")


if __name__ == "__main__":
    main()
