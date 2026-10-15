"""M4: SSDLite with a MobileNetV2 alpha 0.35 backbone, sharing FOMO v2's input contract.

Same 160 x 160 RGB [0,1] input, embedded 2*x-1 scaling, ImageNet initialization and
Person/Vehicle classes as FOMO v2. The difference under test is the head: SSDLite
anchor boxes with depthwise-separable predictors instead of FOMO's centroid grid.
Box decoding and NMS run outside the model, as they would in firmware.
"""
import json
import math
from pathlib import Path

import numpy as np

import common  # noqa: F401  (adds the FOMO directory to sys.path)
from common import nms
from config import CLASS_NAMES, SETTINGS

IMAGE_SIZE = SETTINGS.image_size
ALPHA = SETTINGS.alpha
# (feature-map size, anchor scale as a fraction of the stretched 160 x 160 input).
# The 20 x 20 level reuses FOMO's stride-8 cut point so small objects get the same resolution.
LEVELS = ((20, 0.05), (10, 0.15), (5, 0.32), (3, 0.55), (2, 0.80))
LAST_SCALE = 0.95
# Width/height ratios in the stretched input. A 2048 x 1024 frame squashed to a square doubles
# apparent height/width, so upright people appear roughly 4-5x taller than wide.
ASPECT_RATIOS = (1.0, 0.5, 0.25, 2.0)
ANCHORS_PER_CELL = len(ASPECT_RATIOS) + 1  # Plus one interpolated-scale square anchor.
VARIANCES = (0.1, 0.1, 0.2, 0.2)
POSITIVE_IOU = 0.5
NEGATIVE_IOU = 0.4
NEGATIVE_RATIO = 3  # Hard negatives kept per positive anchor.
MIN_NEGATIVES = 8  # Negatives kept for images with no positive anchors.
BOX_LOSS_WEIGHT = 1.0
NMS_IOU = 0.45
PRE_NMS_TOP_K = 400  # Per class, before NMS.
POST_NMS_MAX = 200  # Per class, after NMS; far above Cityscapes per-image object counts.
SCORE_FLOOR = 0.05  # Lowest confidence kept for the threshold sweep and mAP.
CUT_POINTS = {20: "block_6_expand_relu", 10: "block_13_expand_relu", 5: "out_relu"}
EXTRA_CHANNELS = {3: 128, 2: 64}
PREDICTOR_CHANNELS = None  # Predictors keep their input depth (SSDLite depthwise design).


def contract():
    return {"implementation": "pathclear-ssdlite-mobilenetv2-m4", "class_names": list(CLASS_NAMES),
            "background_class": 0, "object_class_offset": 1,
            "input": "RGB float32 [0,1], direct resize, NHWC", "image_size": IMAGE_SIZE, "alpha": ALPHA,
            "levels": [list(level) for level in LEVELS], "last_scale": LAST_SCALE,
            "aspect_ratios": list(ASPECT_RATIOS), "variances": list(VARIANCES),
            "anchors": int(anchors().shape[0]),
            "outputs": {"class_probabilities": f"{int(anchors().shape[0])}x{len(CLASS_NAMES)+1} softmax",
                        "box_deltas": f"{int(anchors().shape[0])}x4 SSD-encoded (cx, cy, w, h)"}}


def anchors():
    """(A, 4) normalized [cx, cy, w, h], level-major then row, column, anchor."""
    rows = []
    for index, (size, scale) in enumerate(LEVELS):
        following = LEVELS[index + 1][1] if index + 1 < len(LEVELS) else LAST_SCALE
        shapes = [(scale * math.sqrt(r), scale / math.sqrt(r)) for r in ASPECT_RATIOS]
        shapes.append((math.sqrt(scale * following),) * 2)
        for row in range(size):
            for col in range(size):
                cx, cy = (col + 0.5) / size, (row + 0.5) / size
                rows.extend((cx, cy, w, h) for w, h in shapes)
    return np.asarray(rows, dtype=np.float32)


ANCHORS = anchors()


def to_corners(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    return np.stack([boxes[..., 0] - boxes[..., 2] / 2, boxes[..., 1] - boxes[..., 3] / 2,
                     boxes[..., 0] + boxes[..., 2] / 2, boxes[..., 1] + boxes[..., 3] / 2], axis=-1)


def pairwise_iou(a, b):
    """IoU between (N, 4) and (M, 4) corner boxes."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def normalized_boxes(record, flip=False):
    """Annotations as normalized [cx, cy, w, h] in the stretched input, plus class IDs."""
    boxes, classes = [], []
    for annotation in record["annotations"]:
        x, y, w, h = annotation["bbox"]
        cx = (x + w / 2) / record["width"]
        boxes.append((1 - cx if flip else cx, (y + h / 2) / record["height"],
                      w / record["width"], h / record["height"]))
        classes.append(annotation["class_id"] + 1)
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.asarray(classes, dtype=np.int64)


def ignored_anchor_mask(ignore_cells, flip=False):
    """Anchors whose centers fall in FOMO's 20 x 20 ignore cells (void / unresolved groups)."""
    g = SETTINGS.grid_size
    cells = np.zeros((g, g), dtype=bool)
    for row, col in ignore_cells or []:
        cells[row, g - 1 - col if flip else col] = True
    rows = np.minimum((ANCHORS[:, 1] * g).astype(int), g - 1)
    cols = np.minimum((ANCHORS[:, 0] * g).astype(int), g - 1)
    return cells[rows, cols]


def encode(record, flip=False):
    """Per-anchor class target (-1 ignore, 0 background, 1.. object) and SSD box deltas."""
    boxes, classes = normalized_boxes(record, flip)
    labels = np.zeros(len(ANCHORS), dtype=np.int8)
    deltas = np.zeros((len(ANCHORS), 4), dtype=np.float32)
    ignore = ignored_anchor_mask(record.get("target_ignore_cells", record.get("ignore_cells")), flip)
    if not len(boxes):
        labels[ignore] = -1
        return labels, deltas
    iou = pairwise_iou(to_corners(ANCHORS), to_corners(boxes))
    best_truth = iou.argmax(axis=1)
    best_iou = iou[np.arange(len(ANCHORS)), best_truth]
    # Every object keeps its best anchor, even when its IoU is below the positive threshold.
    for truth in np.argsort(-iou.max(axis=0), kind="stable"):
        anchor = int(iou[:, truth].argmax())
        best_truth[anchor], best_iou[anchor] = truth, 2.0
    labels[:] = np.where(best_iou >= POSITIVE_IOU, classes[best_truth], 0)
    labels[(best_iou >= NEGATIVE_IOU) & (best_iou < POSITIVE_IOU)] = -1
    labels[ignore & (labels == 0)] = -1
    matched = boxes[best_truth]
    a = ANCHORS.astype(np.float64)
    deltas[:] = np.stack([(matched[:, 0] - a[:, 0]) / a[:, 2] / VARIANCES[0],
                          (matched[:, 1] - a[:, 1]) / a[:, 3] / VARIANCES[1],
                          np.log(matched[:, 2] / a[:, 2]) / VARIANCES[2],
                          np.log(matched[:, 3] / a[:, 3]) / VARIANCES[3]], axis=-1)
    deltas[labels <= 0] = 0
    return labels, deltas


def decode_boxes(deltas):
    """Inverse of `encode`: (A, 4) deltas -> normalized [cx, cy, w, h]."""
    d = np.asarray(deltas, dtype=np.float64)
    a = ANCHORS.astype(np.float64)
    cx = d[:, 0] * VARIANCES[0] * a[:, 2] + a[:, 0]
    cy = d[:, 1] * VARIANCES[1] * a[:, 3] + a[:, 1]
    w = np.exp(np.clip(d[:, 2] * VARIANCES[2], -10, 10)) * a[:, 2]
    h = np.exp(np.clip(d[:, 3] * VARIANCES[3], -10, 10)) * a[:, 3]
    return np.stack([cx, cy, w, h], axis=-1)


def decode(probabilities, deltas, frame_width, frame_height, score_floor=SCORE_FLOOR):
    """Per-class thresholded NMS; detections in FOMO's dictionary format plus `bbox`."""
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(ANCHORS), len(CLASS_NAMES) + 1) or np.asarray(deltas).shape != (len(ANCHORS), 4):
        raise ValueError("Unexpected SSDLite output shapes")
    corners = np.clip(to_corners(decode_boxes(deltas)), 0, 1)
    detections = []
    for class_id, name in enumerate(CLASS_NAMES):
        scores = probabilities[:, class_id + 1]
        candidates = np.where(scores >= score_floor)[0]
        candidates = candidates[np.argsort(-scores[candidates], kind="stable")[:PRE_NMS_TOP_K]]
        candidates = candidates[(corners[candidates, 2] > corners[candidates, 0]) &
                                (corners[candidates, 3] > corners[candidates, 1])]
        for index in candidates[nms(corners[candidates], scores[candidates], NMS_IOU)][:POST_NMS_MAX]:
            x1, y1, x2, y2 = corners[index]
            detections.append({"centroid_x": float((x1 + x2) / 2 * frame_width),
                               "centroid_y": float((y1 + y2) / 2 * frame_height),
                               "coarse_class_id": class_id, "class_name": name,
                               "confidence": float(scores[index]),
                               "bbox": [float(x1 * frame_width), float(y1 * frame_height),
                                        float((x2 - x1) * frame_width), float((y2 - y1) * frame_height)]})
    detections.sort(key=lambda d: (-d["confidence"], d["coarse_class_id"], d["centroid_y"], d["centroid_x"]))
    return detections


def decoding_policy():
    return {"decoder": "ssd_per_class_nms", "nms_iou": NMS_IOU, "pre_nms_top_k": PRE_NMS_TOP_K,
            "post_nms_max_per_class": POST_NMS_MAX, "score_floor": SCORE_FLOOR,
            "centroid": "decoded box center", "consumer": "PathClear decision logic"}


def build_model(backbone_weights="imagenet"):
    import tensorflow as tf
    layers = tf.keras.layers
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), alpha=ALPHA, include_top=False,
        weights="imagenet" if backbone_weights == "imagenet" else None)
    if backbone_weights not in (None, "random", "imagenet"):
        backbone.load_weights(backbone_weights)
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.momentum = 0.9  # Same backbone adjustment as FOMO v2.
    trunk = tf.keras.Model(backbone.input, [backbone.get_layer(CUT_POINTS[s]).output for s in (20, 10, 5)],
                           name="mobilenet_trunk")
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="rgb_0_to_1")
    features = list(trunk(layers.Rescaling(2.0, offset=-1.0, name="mobilenet_scaling")(inputs)))
    x = features[-1]
    for size, channels in EXTRA_CHANNELS.items():
        # SSDLite extra layer: 1x1 bottleneck, stride-2 depthwise 3x3, 1x1 expansion.
        x = layers.Conv2D(channels // 2, 1, use_bias=False, name=f"extra{size}_reduce")(x)
        x = layers.BatchNormalization(momentum=0.9, name=f"extra{size}_reduce_bn")(x)
        x = layers.ReLU(6.0, name=f"extra{size}_reduce_relu")(x)
        x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name=f"extra{size}_dw")(x)
        x = layers.BatchNormalization(momentum=0.9, name=f"extra{size}_dw_bn")(x)
        x = layers.ReLU(6.0, name=f"extra{size}_dw_relu")(x)
        x = layers.Conv2D(channels, 1, use_bias=False, name=f"extra{size}_expand")(x)
        x = layers.BatchNormalization(momentum=0.9, name=f"extra{size}_expand_bn")(x)
        x = layers.ReLU(6.0, name=f"extra{size}_expand_relu")(x)
        features.append(x)
    logits, boxes = [], []
    for (size, _), feature in zip(LEVELS, features):
        if feature.shape[1] != size:
            raise ValueError(f"Feature map {feature.shape} does not match level {size}")
        for kind, depth, out in (("class", len(CLASS_NAMES) + 1, logits), ("box", 4, boxes)):
            y = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name=f"{kind}{size}_dw")(feature)
            y = layers.BatchNormalization(momentum=0.9, name=f"{kind}{size}_dw_bn")(y)
            y = layers.ReLU(6.0, name=f"{kind}{size}_dw_relu")(y)
            y = layers.Conv2D(ANCHORS_PER_CELL * depth, 1, name=f"{kind}{size}_pw")(y)
            out.append(layers.Reshape((size * size * ANCHORS_PER_CELL, depth), name=f"{kind}{size}_flat")(y))
    probabilities = layers.Softmax(name="class_probabilities")(layers.Concatenate(axis=1, name="class_logits")(logits))
    deltas = layers.Concatenate(axis=1, name="box_deltas")(boxes)
    return tf.keras.Model(inputs, {"class_probabilities": probabilities, "box_deltas": deltas},
                          name="pathclear_ssdlite_mobilenetv2_035")


def detection_loss(labels, deltas, probabilities, predicted_deltas):
    """SSD multibox loss: CE with 3:1 hard-negative mining plus smooth-L1 on positives."""
    import tensorflow as tf
    labels = tf.cast(labels, tf.int32)
    positive = labels > 0
    negative = tf.equal(labels, 0)
    safe = tf.maximum(labels, 0)
    ce = -tf.math.log(tf.maximum(tf.gather(probabilities, safe, batch_dims=2), 1e-7))
    num_positive = tf.reduce_sum(tf.cast(positive, tf.int32), axis=1)
    num_negative = tf.reduce_sum(tf.cast(negative, tf.int32), axis=1)
    keep = tf.minimum(tf.maximum(NEGATIVE_RATIO * num_positive, MIN_NEGATIVES), num_negative)
    negative_ce = tf.where(negative, ce, tf.zeros_like(ce))
    # Rank-based selection keeps exactly `keep` negatives per image, even when losses tie.
    order = tf.argsort(negative_ce, axis=1, direction="DESCENDING", stable=True)
    rank = tf.argsort(order, axis=1, stable=True)
    hard = negative & (rank < keep[:, None])
    difference = tf.abs(predicted_deltas - deltas)
    smooth = tf.reduce_sum(tf.where(difference < 1.0, 0.5 * difference ** 2, difference - 0.5), axis=-1)
    total_positive = tf.maximum(tf.cast(tf.reduce_sum(num_positive), tf.float32), 1.0)
    class_loss = (tf.reduce_sum(tf.where(positive | hard, ce, 0.0))) / total_positive
    box_loss = tf.reduce_sum(tf.where(positive, smooth, 0.0)) / total_positive
    return class_loss + BOX_LOSS_WEIGHT * box_loss, class_loss, box_loss


class TFLiteSSD:
    """PC adapter for the full-INT8 export; returns dequantized (probabilities, deltas)."""

    def __init__(self, path):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        outputs = self.interpreter.get_output_details()
        by_depth = {int(o["shape"][-1]): o for o in outputs}
        if set(by_depth) != {len(CLASS_NAMES) + 1, 4} or self.input["dtype"] != np.int8:
            raise ValueError("Expected INT8 input and class/box outputs")
        self.classes, self.boxes = by_depth[len(CLASS_NAMES) + 1], by_depth[4]

    def __call__(self, batch):
        outputs = []
        for image in np.asarray(batch, dtype=np.float32):
            scale, zero = self.input["quantization"]
            self.interpreter.set_tensor(self.input["index"],
                                        np.clip(np.rint(image[None] / scale + zero), -128, 127).astype(np.int8))
            self.interpreter.invoke()
            outputs.append([self._dequantize(self.classes), self._dequantize(self.boxes)])
        probabilities = np.concatenate([o[0] for o in outputs])
        # Correct small rounding errors in quantized softmax sums, preserving class order.
        probabilities /= np.maximum(probabilities.sum(axis=-1, keepdims=True), 1e-8)
        return probabilities, np.concatenate([o[1] for o in outputs])

    def _dequantize(self, detail):
        scale, zero = detail["quantization"]
        return (self.interpreter.get_tensor(detail["index"]).astype(np.float32) - zero) * scale


def load(path):
    """Return (callable batch -> (probabilities, deltas), metadata) for .keras or .tflite."""
    path = Path(path)
    metadata_path = path.parent / ("report.json" if path.suffix == ".tflite" else "best.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["contract"] != contract():
        raise ValueError("Model contract differs from current SSDLite settings; do not reinterpret it.")
    if path.suffix == ".tflite":
        return TFLiteSSD(path), metadata
    import tensorflow as tf
    model = tf.keras.models.load_model(path, compile=False)

    @tf.function(input_signature=[tf.TensorSpec((None, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32)])
    def forward(x):
        out = model(x, training=False)
        return out["class_probabilities"], out["box_deltas"]

    return (lambda batch: tuple(t.numpy() for t in forward(np.asarray(batch, dtype=np.float32)))), metadata
