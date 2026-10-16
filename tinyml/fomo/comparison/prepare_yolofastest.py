"""Lay out the prepare_yolo.py dataset the way YOLO-FastestV2 reads it, without copying pixels.

YOLO-FastestV2 reads a text list of image paths and expects each label at the image path with a .txt
extension. Images and labels are hard-linked from the YOLO dataset, so both trainers see identical bytes.
"""
import argparse
import json
import os
from pathlib import Path

import common  # noqa: F401
from config import output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-data", required=True, help="Directory written by prepare_yolo.py")
    parser.add_argument("--output")
    args = parser.parse_args()
    source = Path(args.yolo_data).resolve()
    directory = output_directory(args.output, "yolofastest-data")
    counts = {}
    for split in ("train", "val"):
        folder = directory / split
        folder.mkdir()
        paths = []
        for image in sorted((source / "images" / split).glob("*.png")):
            if "." in image.stem:
                raise ValueError(f"YOLO-FastestV2 derives label paths by splitting on '.': {image.name}")
            target = folder / image.name
            os.link(image, target)
            os.link(source / "labels" / split / f"{image.stem}.txt", folder / f"{image.stem}.txt")
            paths.append(target.as_posix())
        if any("." in p.rsplit("/", 1)[0] for p in paths):
            raise ValueError("A directory name contains '.', which breaks YOLO-FastestV2's label lookup")
        (directory / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        counts[split] = len(paths)
    (directory / "pathclear.names").write_text("person\nvehicle\n", encoding="utf-8")
    (directory / "layout.json").write_text(json.dumps({"source": str(source), "counts": counts,
                                                       "method": "hard links"}, indent=2), encoding="utf-8")
    print(json.dumps(counts))
    print(f"YOLO-FastestV2 dataset: {directory}")


if __name__ == "__main__":
    main()
