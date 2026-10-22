"""Tests for the CVAT Segmentation mask 1.1 -> PathClear mask converter."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pathclear_taxonomy import CLASS_IDS, IGNORE_LABEL
from scripts.convert_cvat_masks import (
    build_color_table,
    convert_export,
    convert_mask,
    parse_labelmap,
)

# The PathClear CVAT project's raw label configuration.
CVAT_LABELS = [
    {"name": "Road", "color": "#804080"},
    {"name": "Sidewalk", "color": "#f423e8"},
    {"name": "Person", "color": "#dc143c"},
    {"name": "Vehicle", "color": "#00008e"},
    {"name": "Boundary", "color": "#be9999"},
    {"name": "Obstacle", "color": "#fad200"},
    {"name": "Vegetation", "color": "#6b8e23"},
    {"name": "Stairs", "color": "#ff8000"},
    {"name": "Crosswalk", "color": "#ffffff"},
    {"name": "Uneven Pavement", "color": "#00c8c8"},
]
ROAD, CROSSWALK, BACKGROUND = (128, 64, 128), (255, 255, 255), (0, 0, 0)
UNKNOWN = (37, 37, 37)  # e.g. an antialiased edge pixel

LABELMAP_TXT = """# label:color_rgb:parts:actions
background:0,0,0::
Road:128,64,128::
Sidewalk:244,35,232::
Crosswalk:255,255,255::
Uneven Pavement:0,200,200::
"""


def _tmp():
    return Path(tempfile.mkdtemp())


class LabelmapTest(unittest.TestCase):
    def test_cvat_json_config_maps_every_pathclear_class(self):
        path = _tmp() / "labels.json"
        path.write_text(json.dumps(CVAT_LABELS))
        table = build_color_table(parse_labelmap(path))
        for item in CVAT_LABELS:
            color = tuple(int(item["color"][i:i + 2], 16) for i in (1, 3, 5))
            self.assertEqual(table[color], CLASS_IDS[item["name"]], item["name"])
        self.assertEqual(table[BACKGROUND], IGNORE_LABEL)

    def test_labelmap_txt_background_is_ignore_not_road(self):
        path = _tmp() / "labelmap.txt"
        path.write_text(LABELMAP_TXT)
        table = build_color_table(parse_labelmap(path))
        self.assertEqual(table[BACKGROUND], IGNORE_LABEL)
        self.assertEqual(table[ROAD], CLASS_IDS["Road"])
        self.assertEqual(table[(0, 200, 200)], CLASS_IDS["Uneven Pavement"])

    def test_rejects_unknown_label_name(self):
        with self.assertRaises(ValueError):
            build_color_table([("Pothole", (1, 2, 3))])

    def test_rejects_one_color_for_two_classes(self):
        with self.assertRaises(ValueError):
            build_color_table([("Road", (9, 9, 9)), ("Sidewalk", (9, 9, 9))])

    def test_rejects_class_drawn_in_background_black(self):
        with self.assertRaises(ValueError):
            build_color_table([("Road", BACKGROUND)])


class ConvertMaskTest(unittest.TestCase):
    def test_background_two_classes_and_unknown_color(self):
        table = build_color_table([("Road", ROAD), ("Crosswalk", CROSSWALK)])
        rgb = np.array([[BACKGROUND, ROAD], [CROSSWALK, UNKNOWN]], dtype=np.uint8)
        mask, unknown = convert_mask(rgb, table)
        np.testing.assert_array_equal(
            mask, [[IGNORE_LABEL, CLASS_IDS["Road"]], [CLASS_IDS["Crosswalk"], IGNORE_LABEL]])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(unknown, 1)

    def test_all_background_never_becomes_road(self):
        table = build_color_table([("Road", ROAD)])
        mask, _ = convert_mask(np.zeros((4, 6, 3), np.uint8), table)
        self.assertTrue((mask == IGNORE_LABEL).all())


class ConvertExportTest(unittest.TestCase):
    def _export(self):
        root = _tmp()
        (root / "SegmentationClass").mkdir()
        (root / "labelmap.txt").write_text(LABELMAP_TXT)
        rgb = np.zeros((4, 8, 3), np.uint8)       # 32 px, background
        rgb[2:, :] = ROAD                          # 16 px Road
        rgb[2:, :2] = CROSSWALK                    # 4 of them Crosswalk
        rgb[0, 0] = UNKNOWN                        # 1 unknown px
        Image.fromarray(rgb).save(root / "SegmentationClass" / "frame_a.png")
        # CVAT can also write palette PNGs; the converter must read them too.
        Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=8).save(
            root / "SegmentationClass" / "frame_b.png")
        return root

    def test_end_to_end_writes_valid_single_channel_masks(self):
        export, output = self._export(), _tmp() / "masks"
        report = convert_export(export, output)
        self.assertEqual(report["masks"], 2)
        for name in ("frame_a.png", "frame_b.png"):
            with Image.open(output / name) as image:
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.size, (8, 4))  # original resolution
                mask = np.array(image)
            self.assertTrue(set(np.unique(mask)) <= {CLASS_IDS["Road"],
                                                     CLASS_IDS["Crosswalk"], IGNORE_LABEL})
            self.assertEqual(int((mask == CLASS_IDS["Road"]).sum()), 12)
            self.assertEqual(int((mask == CLASS_IDS["Crosswalk"]).sum()), 4)
            self.assertEqual(int((mask == IGNORE_LABEL).sum()), 16)
        first = report["files"][0]
        self.assertEqual(first["unknown_color_pixels"], 1)
        self.assertAlmostEqual(first["ignored_fraction"], 0.5)
        self.assertEqual(report["class_pixels"]["Road"], 24)
        self.assertTrue((output / "conversion_report.json").is_file())

    def test_refuses_nonempty_output(self):
        export, output = self._export(), _tmp()
        (output / "existing.png").write_bytes(b"x")
        with self.assertRaises(FileExistsError):
            convert_export(export, output)


if __name__ == "__main__":
    unittest.main()
