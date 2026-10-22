"""Convert a CVAT "Segmentation mask 1.1" export into PathClear training masks.

CVAT writes RGB color PNGs (SegmentationClass/*.png) in which unannotated
pixels are background (0,0,0). Read naively, background would become ID 0
(Road), and prepare_targets() cannot catch that because 0 is a valid ID. This
converter maps colors to IDs explicitly:

- Each labelmap color maps to its PathClear ID through CLASS_IDS, matched
  by label name (case and spacing are normalized; unknown names fail).
- Background (0,0,0) and any color not in the labelmap become Ignore (255).
- Output is an 8-bit single-channel PNG per mask with only 0-9 and 255,
  at the original resolution, written with no resampling.
- Every output is checked with prepare_targets(), and a JSON report lists
  per-class pixel counts, ignored fractions and unknown-color counts.

The labelmap is read from the export's labelmap.txt by default, or from
--labelmap (labelmap.txt format, or the CVAT project's raw label JSON).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pathclear_taxonomy import CLASS_IDS, CLASS_NAMES, IGNORE_LABEL

BACKGROUND_NAME = "background"


def _normalized(name):
    return " ".join(name.strip().lower().replace("_", " ").split())


_NAME_IDS = {_normalized(name): index for name, index in CLASS_IDS.items()}


def _hex_color(value):
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a #rrggbb color, got {value!r}.")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def parse_labelmap(path):
    """Return [(label name, (r, g, b)), ...] from labelmap.txt or CVAT label JSON."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return [(item["name"], _hex_color(item["color"])) for item in json.loads(text)]
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(":")
        color = rest.split(":", 1)[0]
        channels = tuple(int(part) for part in color.split(","))
        if len(channels) != 3 or not all(0 <= c <= 255 for c in channels):
            raise ValueError(f"Bad color in labelmap line: {line!r}")
        entries.append((name, channels))
    if not entries:
        raise ValueError(f"No labels found in {path}.")
    return entries


def build_color_table(entries):
    """Map each (r, g, b) to a PathClear ID; background maps to Ignore."""
    table = {(0, 0, 0): IGNORE_LABEL}
    for name, color in entries:
        key = _normalized(name)
        if key == BACKGROUND_NAME:
            class_id = IGNORE_LABEL
        elif key in _NAME_IDS:
            class_id = _NAME_IDS[key]
        else:
            raise ValueError(f"CVAT label {name!r} is not a PathClear class "
                             f"({', '.join(CLASS_NAMES)}).")
        if color in table and table[color] != class_id:
            raise ValueError(f"Color {color} is used for two different classes "
                             f"(IDs {table[color]} and {class_id}).")
        table[tuple(color)] = class_id
    if table[(0, 0, 0)] != IGNORE_LABEL:
        raise ValueError("A PathClear class uses black (0,0,0), which CVAT reserves "
                         "for background; give that label another color.")
    return table


def convert_mask(rgb, table):
    """Return (uint8 ID mask, unknown-color pixel count) for an HxWx3 array."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Expected an HxWx3 uint8 RGB mask, got {rgb.shape} {rgb.dtype}.")
    codes = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) \
        | rgb[..., 2].astype(np.int32)
    unique, inverse = np.unique(codes, return_inverse=True)
    lookup = np.full(len(unique), IGNORE_LABEL, dtype=np.uint8)
    unknown = 0
    counts = np.bincount(inverse.ravel(), minlength=len(unique))
    for index, code in enumerate(unique):
        color = (int(code) >> 16 & 255, int(code) >> 8 & 255, int(code) & 255)
        if color in table:
            lookup[index] = table[color]
        else:
            unknown += int(counts[index])
    return lookup[inverse].reshape(rgb.shape[:2]), unknown


def _validate(mask):
    """Run the training contract check from pathclear_pipeline."""
    from pathclear_pipeline import prepare_targets

    prepare_targets(mask)


def convert_export(export_dir, output_dir, labelmap=None):
    """Convert every SegmentationClass PNG; return the report dict."""
    export_dir, output_dir = Path(export_dir), Path(output_dir)
    mask_dir = export_dir / "SegmentationClass"
    sources = sorted(mask_dir.glob("*.png"))
    if not sources:
        raise FileNotFoundError(f"No PNG masks in {mask_dir}.")
    labelmap = Path(labelmap) if labelmap else export_dir / "labelmap.txt"
    table = build_color_table(parse_labelmap(labelmap))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files, totals = [], np.zeros(256, dtype=np.int64)
    for source in sources:
        with Image.open(source) as image:
            rgb = np.array(image.convert("RGB"))
        mask, unknown = convert_mask(rgb, table)
        _validate(mask)
        target = output_dir / source.name
        Image.fromarray(mask, mode="L").save(target)
        counts = np.bincount(mask.ravel(), minlength=256)
        totals += counts
        files.append({
            "source": str(source),
            "mask": str(target),
            "size": [int(mask.shape[1]), int(mask.shape[0])],
            "class_pixels": {CLASS_NAMES[i]: int(counts[i]) for i in range(len(CLASS_NAMES))},
            "ignored_fraction": float(counts[IGNORE_LABEL] / mask.size),
            "unknown_color_pixels": unknown,
        })
    total = int(totals.sum())
    report = {
        "export_dir": str(export_dir),
        "labelmap": str(labelmap),
        "color_table": {",".join(map(str, color)): int(class_id)
                        for color, class_id in table.items()},
        "masks": len(files),
        "class_pixels": {CLASS_NAMES[i]: int(totals[i]) for i in range(len(CLASS_NAMES))},
        "ignored_fraction": float(totals[IGNORE_LABEL] / total),
        "unknown_color_pixels": sum(item["unknown_color_pixels"] for item in files),
        "files": files,
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True,
                        help="Unzipped CVAT export containing SegmentationClass/.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labelmap", type=Path,
                        help="labelmap.txt or CVAT label JSON; default <export>/labelmap.txt.")
    args = parser.parse_args()
    report = convert_export(args.export_dir, args.output_dir, args.labelmap)
    print(f"Converted {report['masks']} masks; ignored {100 * report['ignored_fraction']:.1f}%")
    for name, pixels in report["class_pixels"].items():
        print(f"  {name}: {pixels}")
    if report["unknown_color_pixels"]:
        print(f"WARNING: {report['unknown_color_pixels']} pixels had colors not in the "
              f"labelmap and were set to Ignore (255). Check for antialiasing or a stale "
              f"labelmap.")


if __name__ == "__main__":
    main()
