"""Comparison scoring and M4 SSDLite unit tests; synthetic data, no Cityscapes required."""
import math
import unittest

import numpy as np

import common
from common import average_precision, measure_detections, nms, select_operating_point
from analyze_baseline_v1 import measure
from inference import decode_grid
import ssdlite


def record(annotations, width=2048, height=1024, ignore_cells=None):
    return {"id": "r", "width": width, "height": height, "annotations": annotations,
            "ignore_cells": ignore_cells or []}


class SharedScoringTests(unittest.TestCase):
    def test_matches_fomo_v2_measure_on_fomo_grids(self):
        rng = np.random.default_rng(0)
        records, grids = [], []
        for _ in range(25):
            annotations = []
            for _ in range(rng.integers(0, 6)):
                w, h = rng.integers(10, 300, 2)
                annotations.append({"class_id": int(rng.integers(0, 2)),
                                    "bbox": [int(rng.integers(0, 2048 - w)), int(rng.integers(0, 1024 - h)), int(w), int(h)]})
            records.append(record(annotations, ignore_cells=[[int(a), int(b)] for a, b in rng.integers(0, 20, (3, 2))]))
            logits = rng.normal(size=(20, 20, 3)) + np.array([2.0, 0, 0])
            grids.append(np.exp(logits) / np.exp(logits).sum(-1, keepdims=True))
        for threshold in (0.3, 0.6):
            expected, _ = measure(records, grids, threshold, decoder="cells")
            detections = [decode_grid(g, r["width"], r["height"], 0.0, 0, "cells") for r, g in zip(records, grids)]
            actual = measure_detections(records, detections, threshold)
            self.assertEqual(actual["micro"], expected["micro"])
            self.assertEqual(actual["per_class"], expected["per_class"])
            self.assertEqual(actual["size_recall"], expected["size_recall"])

    def test_operating_point_rule(self):
        results = [{"threshold": t, "micro": {"precision": p, "recall": r}}
                   for t, p, r in ((0.2, 0.6, 0.9), (0.5, 0.72, 0.5), (0.7, 0.8, 0.5), (0.9, 0.95, 0.1))]
        self.assertEqual(select_operating_point(results)["threshold"], 0.7)
        self.assertIsNone(select_operating_point(results[:1]))

    def test_nms_keeps_highest_and_separate_boxes(self):
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
        self.assertEqual(nms(boxes, [0.5, 0.9, 0.4], 0.45), [1, 2])

    def test_perfect_boxes_give_unit_ap(self):
        r = record([{"class_id": 0, "bbox": [100, 100, 50, 120]}, {"class_id": 1, "bbox": [800, 400, 300, 150]}])
        detections = [{"coarse_class_id": a["class_id"], "class_name": ("person", "vehicle")[a["class_id"]],
                       "confidence": 0.9, "bbox": a["bbox"],
                       "centroid_x": a["bbox"][0] + a["bbox"][2] / 2, "centroid_y": a["bbox"][1] + a["bbox"][3] / 2}
                      for a in r["annotations"]]
        self.assertAlmostEqual(average_precision([r], [detections])["mean"], 1.0)
        self.assertEqual(measure_detections([r], [detections], 0.6)["micro"]["f1"], 1.0)


class SSDLiteTargetTests(unittest.TestCase):
    def test_anchor_layout(self):
        expected = sum(s * s for s, _ in ssdlite.LEVELS) * ssdlite.ANCHORS_PER_CELL
        self.assertEqual(ssdlite.ANCHORS.shape, (expected, 4))
        self.assertTrue(((ssdlite.ANCHORS[:, :2] > 0) & (ssdlite.ANCHORS[:, :2] < 1)).all())

    def test_encode_decode_round_trip(self):
        r = record([{"class_id": 0, "bbox": [400, 200, 60, 260]}, {"class_id": 1, "bbox": [1000, 500, 500, 250]}])
        labels, deltas = ssdlite.encode(r)
        decoded = ssdlite.decode_boxes(deltas)
        truths, classes = ssdlite.normalized_boxes(r)
        for class_id, truth in zip(classes, truths):
            matched = np.where(labels == class_id)[0]
            self.assertGreater(len(matched), 0)
            np.testing.assert_allclose(decoded[matched], np.repeat(truth[None], len(matched), 0), atol=1e-4)

    def test_tiny_object_keeps_best_anchor(self):
        labels, _ = ssdlite.encode(record([{"class_id": 0, "bbox": [1000, 500, 4, 8]}]))
        self.assertEqual(int((labels == 1).sum()), 1)

    def test_ignore_cells_only_mask_negatives_and_flip(self):
        r = record([], ignore_cells=[[0, 0]])
        labels, _ = ssdlite.encode(r)
        flipped, _ = ssdlite.encode(r, flip=True)
        centers = ssdlite.ANCHORS[:, :2]
        self.assertTrue((labels[(centers[:, 0] < 0.05) & (centers[:, 1] < 0.05)] == -1).all())
        self.assertTrue((flipped[(centers[:, 0] > 0.95) & (centers[:, 1] < 0.05)] == -1).all())
        self.assertEqual(int((labels == -1).sum()), int((flipped == -1).sum()))

    def test_flipped_targets_mirror_boxes(self):
        r = record([{"class_id": 1, "bbox": [100, 300, 400, 200]}])
        labels, deltas = ssdlite.encode(r, flip=True)
        box = ssdlite.decode_boxes(deltas)[labels == 2][0]
        self.assertAlmostEqual(box[0], 1 - (300 / 2048), places=4)

    def test_decode_format_matches_fomo_detections(self):
        r = record([{"class_id": 1, "bbox": [1000, 500, 500, 250]}])
        labels, deltas = ssdlite.encode(r)
        probabilities = np.zeros((len(ssdlite.ANCHORS), 3), dtype=np.float32)
        probabilities[:, 0] = 1
        probabilities[labels == 2] = [0.1, 0.0, 0.9]
        detections = ssdlite.decode(probabilities, deltas, 2048, 1024)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "vehicle")
        self.assertAlmostEqual(detections[0]["centroid_x"], 1250, delta=1)
        self.assertAlmostEqual(detections[0]["centroid_y"], 625, delta=1)


class SSDLiteModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(1)
        cls.tf = tf
        cls.model = ssdlite.build_model(None)

    def test_output_shapes_and_macs(self):
        out = self.model(np.zeros((2, 160, 160, 3), np.float32), training=False)
        self.assertEqual(tuple(out["class_probabilities"].shape), (2, len(ssdlite.ANCHORS), 3))
        self.assertEqual(tuple(out["box_deltas"].shape), (2, len(ssdlite.ANCHORS), 4))
        self.assertGreater(common.count_macs(self.model), 1_000_000)

    def test_loss_mines_three_negatives_per_positive(self):
        tf = self.tf
        labels = np.zeros((1, len(ssdlite.ANCHORS)), np.int8)
        labels[0, :4] = 1
        labels[0, 4:10] = -1
        probabilities = np.full((1, len(ssdlite.ANCHORS), 3), 1 / 3, np.float32)
        deltas = np.zeros((1, len(ssdlite.ANCHORS), 4), np.float32)
        total, class_loss, box_loss = ssdlite.detection_loss(tf.constant(labels), tf.constant(deltas),
                                                             tf.constant(probabilities), tf.constant(deltas))
        self.assertAlmostEqual(float(class_loss), (4 + 12) * math.log(3) / 4, places=4)
        self.assertEqual(float(box_loss), 0.0)

    def test_int8_export_round_trip(self):
        import tempfile
        from pathlib import Path
        tf = self.tf
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: ([np.random.default_rng(i).random((1, 160, 160, 3), np.float32)]
                                                    for i in range(4))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = converter.inference_output_type = tf.int8
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.tflite"
            path.write_bytes(converter.convert())
            probabilities, deltas = ssdlite.TFLiteSSD(path)(np.zeros((1, 160, 160, 3), np.float32))
        self.assertEqual(probabilities.shape, (1, len(ssdlite.ANCHORS), 3))
        self.assertEqual(deltas.shape, (1, len(ssdlite.ANCHORS), 4))
        np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
