"""Ultralytics-based candidates with their official recipes (see PLAN.md for each budget rule).

M1 espdet: esp-detection's train.py settings, from scratch.
M3 yolo11n: the standard Ultralytics custom-data recipe from COCO-pretrained yolo11n.pt.
Run with the .venv-espdet interpreter. ESPDET_ROOT points at a pinned esp-detection checkout.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

FOMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FOMO_DIR))
from config import ROOT, SETTINGS, output_directory  # noqa: E402

ESPDET_ROOT = Path(os.environ.get("ESPDET_ROOT", "C:/Projects/esp-detection")).resolve()
RECIPES = {
    # esp-detection train.py defaults (epochs set per PLAN.md budget rule).
    "espdet": {"model": "cfg/models/espdet_pico.yaml", "batch": 128, "optimizer": "auto", "close_mosaic": 30,
               "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.1, "rect": False},
    # Ultralytics defaults for fine-tuning a pretrained detector on custom data.
    "yolo11n": {"model": "yolo11n.pt", "batch": 16, "optimizer": "auto"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=sorted(RECIPES), required=True)
    parser.add_argument("--data", required=True, help="data.yaml written by prepare_yolo.py")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--close-mosaic", type=int, help="Override only for a timing run (0 keeps mosaic on)")
    # Ultralytics resume only finishes an interrupted run's original schedule; it cannot extend a cap.
    parser.add_argument("--resume", help="last.pt of an interrupted run")
    parser.add_argument("--output")
    args = parser.parse_args()
    args.data = str(Path(args.data).resolve())  # Before changing into the esp-detection root.
    directory = output_directory(args.output, f"{args.arch}-training")
    recipe = dict(RECIPES[args.arch])
    model_source = recipe.pop("model")
    if args.close_mosaic is not None:
        recipe["close_mosaic"] = args.close_mosaic
    # Pretrained downloads stay inside the ignored artifacts area.
    weights_dir = ROOT / "artifacts" / "ultralytics-weights"
    weights_dir.mkdir(exist_ok=True)
    os.chdir(ESPDET_ROOT if args.arch == "espdet" else weights_dir)
    sys.path.insert(0, str(ESPDET_ROOT))
    import torch
    import ultralytics
    from ultralytics import YOLO
    if args.arch == "espdet":
        import ultralytics.nn.tasks as tasks
        from nn.esp_tasks import custom_parse_model
        tasks.parse_model = custom_parse_model
    head = (ESPDET_ROOT / ".git" / "HEAD").read_text().strip()
    ref = ESPDET_ROOT / ".git" / head.split(" ", 1)[1] if head.startswith("ref:") else None
    record = {"arch": args.arch, "espdet_commit": ref.read_text().strip() if ref and ref.is_file() else head,
              "ultralytics": ultralytics.__version__, "torch": torch.__version__, "python": sys.version.split()[0],
              "data": args.data, "recipe": {"model": model_source, "epochs": args.epochs, "imgsz": SETTINGS.image_size,
                                            "device": "cpu", "seed": SETTINGS.seed, **recipe},
              "resume": args.resume}
    (directory / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    started = time.perf_counter()
    if args.resume:
        YOLO(args.resume).train(resume=True)
    else:
        YOLO(model_source).train(data=args.data, epochs=args.epochs, imgsz=SETTINGS.image_size, device="cpu",
                                 seed=SETTINGS.seed, project=str(directory), name="run", exist_ok=False, **recipe)
    record["duration_seconds"] = time.perf_counter() - started
    (directory / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
