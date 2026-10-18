"""Tests for kpv_loader: contract, horizon split, manifest, sampler safety."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from kpv_loader import (
    CROSSWALK_ID,
    HORIZON_FRACTION,
    HORIZON_ID,
    NEGATIVE_ID,
    attach_teacher_labels,
    load_kpv_pair,
    make_kpv_dataset,
    pairs_from_split_manifest,
)


def _write_jpeg(path, array):
    tf.io.write_file(str(path), tf.io.encode_jpeg(tf.convert_to_tensor(array)))


class LoadKpvPairTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rgb = np.zeros((8, 8, 3), np.uint8)
        rgb[..., 0] = 200
        self.image = self.tmp / "img.jpg"
        _write_jpeg(self.image, rgb)

    def _mask(self, name, top_value, bottom_value):
        mask = np.full((8, 8, 1), bottom_value, np.uint8)
        mask[:4] = top_value
        path = self.tmp / name
        _write_jpeg(path, mask)
        return str(path)

    def test_marked_becomes_crosswalk_full_weight(self):
        mask = self._mask("m1.jpg", 255, 0)
        image, targets, weights = load_kpv_pair(
            str(self.image), mask, negative_weight=0.0,
            horizon_fraction=0.0,
        )
        self.assertEqual(tuple(image.shape), (512, 1024, 3))
        self.assertEqual(tuple(targets.shape), (512, 1024))
        uniq, counts = np.unique(targets.numpy(), return_counts=True)
        by_id = dict(zip(uniq.tolist(), counts.tolist()))
        self.assertGreater(by_id.get(8, 0), 0)
        w = weights.numpy()
        self.assertAlmostEqual(float(w[targets.numpy() == 8].mean()), 1.0)
        self.assertAlmostEqual(float(w[targets.numpy() == 0].mean()), 0.0)

    def test_negative_weight_labels_unmarked_with_horizon(self):
        mask = self._mask("m2.jpg", 0, 0)  # nothing marked
        _, targets, weights = load_kpv_pair(
            str(self.image), mask, negative_weight=0.25,
            horizon_fraction=0.5,
        )
        t = targets.numpy()
        self.assertEqual(int(t[10, 500]), HORIZON_ID)
        self.assertEqual(int(t[400, 500]), NEGATIVE_ID)
        self.assertNotIn(8, np.unique(t).tolist())
        self.assertNotIn(255, np.unique(t).tolist())
        self.assertTrue(bool((weights.numpy() == 0.25).all()))

    def test_horizon_zero_disables_split(self):
        mask = self._mask("m3.jpg", 0, 0)
        _, targets, _ = load_kpv_pair(
            str(self.image), mask, negative_weight=0.25, horizon_fraction=0.0
        )
        self.assertTrue(bool((targets.numpy() == NEGATIVE_ID).all()))

    def test_rejects_identical_horizon_and_negative(self):
        mask = self._mask("m4.jpg", 0, 0)
        with self.assertRaises(ValueError):
            load_kpv_pair(
                str(self.image), mask, negative_weight=0.25,
                horizon_id=0, negative_id=0,
            )

    def test_rejects_out_of_range_weight(self):
        mask = self._mask("m5.jpg", 0, 0)
        with self.assertRaises(ValueError):
            load_kpv_pair(str(self.image), mask, negative_weight=1.5)

    def test_default_horizon_fraction_matches_design(self):
        self.assertAlmostEqual(HORIZON_FRACTION, 0.45)
        self.assertEqual(HORIZON_ID, 4)
        self.assertEqual(NEGATIVE_ID, 0)
        self.assertEqual(CROSSWALK_ID, 8)


class ManifestTest(unittest.TestCase):
    def _manifest(self, tmp, root):
        manifest = {
            "schema_version": 1,
            "dataset_root": str(root.resolve()),
            "items": [
                {"image": "a/i1.jpg", "mask": "a/m1.jpg",
                 "split": "train", "source": "real"},
                {"image": "a/i2.jpg", "mask": "a/m2.jpg",
                 "split": "train", "source": "synthetic"},
                {"image": "a/i3.jpg", "mask": "a/m3.jpg",
                 "split": "validation", "source": "real"},
            ],
        }
        path = tmp / "split.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_reads_split_and_source_filters(self):
        tmp = Path(tempfile.mkdtemp())
        root = tmp / "kw"
        (root / "a").mkdir(parents=True)
        path = self._manifest(tmp, root)
        all_train = pairs_from_split_manifest(path, root=root, split="train")
        self.assertEqual(len(all_train), 2)
        real_train = pairs_from_split_manifest(
            path, root=root, split="train", source="real")
        self.assertEqual(len(real_train), 1)
        val = pairs_from_split_manifest(path, root=root, split="validation")
        self.assertEqual(len(val), 1)

    def test_rejects_bad_source_and_root(self):
        tmp = Path(tempfile.mkdtemp())
        root = tmp / "kw"
        (root / "a").mkdir(parents=True)
        path = self._manifest(tmp, root)
        with self.assertRaises(ValueError):
            pairs_from_split_manifest(path, root=root, split="train",
                                      source="bogus")
        with self.assertRaises(ValueError):
            pairs_from_split_manifest(path, root=tmp / "elsewhere",
                                      split="train")


class SamplerSafetyTest(unittest.TestCase):
    def test_training_stream_repeats(self):
        tmp = Path(tempfile.mkdtemp())
        rgb = np.zeros((8, 8, 3), np.uint8)
        img = tmp / "img.jpg"
        _write_jpeg(img, rgb)
        mask = np.zeros((8, 8, 1), np.uint8)
        m = tmp / "m.jpg"
        _write_jpeg(m, mask)
        ds = make_kpv_dataset([(str(img), str(m))], training=True,
                              batch_size=1, negative_weight=0.25)
        batches = list(ds.take(3))
        self.assertEqual(len(batches), 3)
        ds_eval = make_kpv_dataset([(str(img), str(m))], training=False,
                                   batch_size=1)
        self.assertEqual(len(list(ds_eval)), 1)


class TeacherLabelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.image = self.tmp / "img.jpg"
        _write_jpeg(self.image, np.zeros((8, 8, 3), np.uint8))
        mask = np.zeros((8, 8, 1), np.uint8)
        mask[4:] = 255  # bottom half marked as crossing
        self.mask = self.tmp / "m.jpg"
        _write_jpeg(self.mask, mask)
        teacher = np.full((8, 8, 1), 6, np.uint8)  # Vegetation outside
        teacher[0, :] = 255                        # low-confidence row
        teacher[6:, :4] = 2                        # Person on the crossing
        self.teacher = self.tmp / "t.png"
        tf.io.write_file(str(self.teacher), tf.io.encode_png(teacher))

    def test_teacher_replaces_horizon_split_and_keeps_occluders(self):
        _, targets, weights = load_kpv_pair(
            str(self.image), str(self.mask), teacher_path=str(self.teacher),
            teacher_weight=0.5,
        )
        t, w = targets.numpy(), weights.numpy()
        self.assertEqual(int(t[200, 500]), 6)           # unmarked -> teacher
        self.assertAlmostEqual(float(w[200, 500]), 0.5)
        self.assertAlmostEqual(float(w[10, 500]), 0.0)  # teacher 255 ignored
        self.assertEqual(int(t[300, 900]), CROSSWALK_ID)
        self.assertAlmostEqual(float(w[300, 900]), 1.0)
        self.assertEqual(int(t[500, 100]), 2)           # Person kept on crossing
        self.assertAlmostEqual(float(w[500, 100]), 0.5)
        self.assertNotIn(HORIZON_ID, np.unique(t).tolist())

    def test_attach_teacher_labels_and_dataset(self):
        manifest = self.tmp / "teacher.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "labels": {str(self.image): str(self.teacher)},
        }))
        triples = attach_teacher_labels([(str(self.image), str(self.mask))], manifest)
        self.assertEqual(triples[0][2], str(self.teacher))
        batch = next(iter(make_kpv_dataset(triples, training=False)))
        self.assertIn(CROSSWALK_ID, np.unique(batch[1].numpy()).tolist())
        with self.assertRaises(ValueError):
            attach_teacher_labels([("missing.jpg", str(self.mask))], manifest)


if __name__ == "__main__":
    unittest.main()

