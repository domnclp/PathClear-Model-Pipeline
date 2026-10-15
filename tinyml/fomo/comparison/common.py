"""Shared, model-independent scoring for the standalone detector comparison.

Every candidate is reduced to the same detection dictionaries that FOMO's decoder emits
(original-frame centroid, class, confidence) and scored with the frozen v2 matching code.
"""
from collections import Counter
import math
from pathlib import Path
import sys

FOMO_DIR = Path(__file__).resolve().parent.parent
if str(FOMO_DIR) not in sys.path:
    sys.path.insert(0, str(FOMO_DIR))

import numpy as np  # noqa: E402

from analyze_baseline_v1 import FINE_BINS, assignments, size_bin  # noqa: E402
from config import CLASS_NAMES, SETTINGS  # noqa: E402
from evaluate import scores  # noqa: E402

COMPARISON_DIR = Path(__file__).resolve().parent
# Pre-registered in PLAN.md: identical sweep and precision floor for every candidate.
SWEEP_THRESHOLDS = tuple(p / 100 for p in range(20, 96, 5))
PRECISION_FLOOR = 0.70
SELECTION_THRESHOLD = 0.60  # Checkpoint selection threshold shared with FOMO v1/v2 training.


def coarse_sizes(result):
    """v2's coarse size groups; empty groups report None instead of dividing by zero."""
    groups = {"lt2": ("lt2",), "2to8": ("2to4", "4to8"), "ge8": ("8to16", "ge16")}
    coarse = {}
    for name, keys in groups.items():
        total = sum(result["size_recall"][k]["total"] for k in keys)
        matched = sum(result["size_recall"][k]["matched"] for k in keys)
        coarse[name] = dict(total=total, matched=matched, recall=matched / total if total else None)
    return coarse


def measure_detections(records, detections, threshold):
    """FOMO v2 `measure`, but taking decoded detections so any detector can be scored."""
    if len(records) != len(detections):
        raise ValueError("Record/detection counts must match")
    totals = {name: Counter(tp=0, fp=0, fn=0, ignored=0) for name in CLASS_NAMES}
    bins = {name: {c: Counter(total=0, matched=0) for c in CLASS_NAMES} for name in FINE_BINS}
    for record, found in zip(records, detections):
        kept = [d for d in found if d["confidence"] >= threshold]
        used, labeled = assignments(record, kept)
        for event in labeled:
            totals[event["detection"]["class_name"]][event["kind"]] += 1
        for index, annotation in enumerate(record["annotations"]):
            _, _, w, h = annotation["bbox"]
            side = min(w * SETTINGS.image_size / record["width"], h * SETTINGS.image_size / record["height"])
            name = CLASS_NAMES[annotation["class_id"]]
            bins[size_bin(side)][name].update(total=1, matched=int(index in used))
            if index not in used:
                totals[name]["fn"] += 1
    overall = Counter(tp=0, fp=0, fn=0, ignored=0)
    for counts in totals.values():
        overall.update(counts)
    size_results = {}
    for name, classes in bins.items():
        total = sum(c["total"] for c in classes.values())
        matched = sum(c["matched"] for c in classes.values())
        size_results[name] = dict(total=total, matched=matched, recall=matched / total if total else None,
                                  per_class={c: dict(v, recall=v["matched"] / v["total"] if v["total"] else None)
                                             for c, v in classes.items()})
    result = dict(threshold=threshold, micro=scores(overall), per_class={c: scores(v) for c, v in totals.items()},
                  size_recall=size_results)
    result["coarse_size_recall"] = coarse_sizes(result)
    return result


def sweep(records, detections, thresholds=SWEEP_THRESHOLDS):
    return [measure_detections(records, detections, t) for t in thresholds]


def select_operating_point(results, precision_floor=PRECISION_FLOOR):
    """Highest recall at or above the precision floor; tie-break highest precision (v2 rule)."""
    eligible = [r for r in results if r["micro"]["precision"] is not None and r["micro"]["precision"] >= precision_floor]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (r["micro"]["recall"], r["micro"]["precision"]))


def box_iou(box, boxes):
    """IoU of one [x1,y1,x2,y2] box against an (N, 4) array."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-12)


def nms(boxes, confidences, iou_threshold):
    """Greedy single-class NMS; returns kept indices in descending confidence."""
    order = np.argsort(-np.asarray(confidences), kind="stable")
    keep = []
    boxes = np.asarray(boxes, dtype=np.float64)
    while order.size:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        overlap = box_iou(boxes[best], boxes[order[1:]])
        order = order[1:][overlap <= iou_threshold]
    return keep


def average_precision(records, detections, iou_threshold=0.5):
    """Secondary box metric for box detectors: per-class all-point AP at IoU 0.5.

    Unmatched predictions whose centroid lies in an original void/group cell are ignored,
    mirroring the centroid evaluator; every labeled object stays in the denominator.
    """
    per_class = {}
    for class_id, name in enumerate(CLASS_NAMES):
        events, total = [], 0
        for record, found in zip(records, detections):
            truths = [a for a in record["annotations"] if a["class_id"] == class_id]
            total += len(truths)
            gt = np.array([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in truths)], dtype=np.float64)
            used = np.zeros(len(truths), dtype=bool)
            ignored = set(map(tuple, record.get("ignore_cells", [])))
            for d in sorted((d for d in found if d["coarse_class_id"] == class_id), key=lambda d: -d["confidence"]):
                if "bbox" not in d:
                    raise ValueError("average_precision needs box detections")
                x, y, w, h = d["bbox"]
                hit = False
                if len(truths):
                    overlaps = box_iou(np.array([x, y, x + w, y + h]), gt)
                    overlaps[used] = -1
                    best = int(np.argmax(overlaps))
                    if overlaps[best] >= iou_threshold:
                        used[best] = True
                        hit = True
                if not hit:
                    g = SETTINGS.grid_size
                    cell = (min(g - 1, int(d["centroid_y"] / record["height"] * g)),
                            min(g - 1, int(d["centroid_x"] / record["width"] * g)))
                    if cell in ignored:
                        continue
                events.append((d["confidence"], hit))
        events.sort(key=lambda e: -e[0])
        hits = np.array([e[1] for e in events], dtype=bool)
        tp, fp = np.cumsum(hits), np.cumsum(~hits)
        recall = tp / total if total else np.zeros_like(tp, dtype=float)
        precision = tp / np.maximum(tp + fp, 1)
        # All-point interpolation (VOC 2010+): area under the monotone precision envelope.
        r = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])
        p = np.concatenate([[1.0], precision, [0.0]])
        for i in range(len(p) - 2, -1, -1):
            p[i] = max(p[i], p[i + 1])
        steps = np.where(r[1:] != r[:-1])[0]
        per_class[name] = float(np.sum((r[steps + 1] - r[steps]) * p[steps + 1])) if total else None
    values = [v for v in per_class.values() if v is not None]
    return {"iou_threshold": iou_threshold, "per_class": per_class, "mean": float(np.mean(values)) if values else None}


def count_macs(model):
    """Multiply-accumulates of Conv2D, DepthwiseConv2D and Dense layers, recursing into submodels."""
    import tensorflow as tf
    total = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            total += count_macs(layer)
            continue
        shape = getattr(layer, "output", None)
        if shape is None:
            continue
        out = tuple(shape.shape)
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            kh, kw = layer.kernel_size
            total += out[1] * out[2] * out[3] * kh * kw
        elif isinstance(layer, tf.keras.layers.Conv2D):
            kh, kw = layer.kernel_size
            cin = layer.input.shape[-1]
            total += out[1] * out[2] * out[3] * kh * kw * cin // layer.groups
        elif isinstance(layer, tf.keras.layers.Dense):
            total += math.prod(out[1:]) * layer.input.shape[-1]
    return int(total)
