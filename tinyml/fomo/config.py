"""The separate standalone experiment contract. No Fast-SCNN imports."""
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("person", "vehicle")  # Cityscapes Phase 1, explicitly requested on 2026-10-04.
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
GRID_CUT_POINTS = {20: "block_6_expand_relu", 40: "block_3_expand_relu"}
CITYSCAPES_CLASS_MAP = {24: 0, 25: 0, 26: 1, 27: 1, 28: 1, 31: 1, 32: 1, 33: 1}
# The detector comparison branch shares this data pipeline; both remain confined to tinyml/fomo/.
WRITABLE_BRANCHES = ("tinyml-fomo", "fomo-model-comparison")


@dataclass(frozen=True)
class Settings:
    image_size: int = 160  # RGB frame is stretched to this square for the first experiment.
    crop_size: int = 96  # Legacy visualization helper/checkpoint metadata only; no model handoff.
    stride: int = 8  # Each output cell covers 8 input pixels: 20 x 20 cells.
    alpha: float = 0.35  # MobileNetV2 channel width; smaller usually uses less memory.
    head_channels: int = 32  # Features used by the small grid classification head.
    batch_size: int = 32  # Small grid model fits a larger batch than Fast-SCNN.
    learning_rate: float = 0.001  # Size of each optimizer update.
    epochs: int = 20  # Maximum full passes; validation early stopping can end the run sooner.
    object_weight: float = 20.0  # Urban scenes have many centers; balance against background cells.
    patience: int = 5  # Stop after this many epochs without improving validation F1.
    lr_patience: int = 2  # Halve learning rate after this many epochs without validation F1 improvement.
    min_learning_rate: float = 0.00001  # Lower bound for learning-rate reductions.
    confidence_threshold: float = 0.60  # Minimum winning object probability to return a detection.
    max_detections: int = 0  # Uncapped module output; PathClear decision logic selects warnings.
    decoder: str = "cell_nms"  # Confidence-ordered, same-class suppression in input coordinates.
    nms_radius_pixels: float = 8.0  # Fixed input-space suppression distance across output grids.
    seed: int = 42  # Makes sample order and initialization repeatable in this environment.
    match_radius: float = 8.0  # Evaluation tolerance, in 160 x 160 detector-input pixels.
    calibration_images: int = 100  # Training images used to estimate INT8 ranges.

    @property
    def grid_size(self):
        return self.image_size // self.stride


SETTINGS = Settings()
# Paths and splits are supplied by --manifest; there is no assumed local dataset.
# Generated outputs default to a fresh directory under ROOT / "artifacts".


def contract(grid_size=SETTINGS.grid_size):
    if grid_size not in GRID_CUT_POINTS:
        raise ValueError("Supported output grids are 20 and 40")
    result = {"implementation": "local-fomo-inspired-cityscapes-v2", "class_names": list(CLASS_NAMES),
            "background_channel": 0, "object_channel_offset": 1,
            "input": "RGB float32 [0,1], direct resize, NHWC",
            "output": f"{grid_size}x{grid_size}x{NUM_CLASSES+1} softmax; background, person, vehicle",
            "image_size": SETTINGS.image_size, "grid_size": grid_size,
            "alpha": SETTINGS.alpha, "head_channels": SETTINGS.head_channels,
            "crop_size": SETTINGS.crop_size,  # Historical metadata retained to load frozen v2 checkpoints.
            "collision_policy": "same-class centers share a positive cell; mixed-class cells ignored"}
    if grid_size != 20:
        result.update(implementation="local-fomo-inspired-cityscapes-grid-v3", cut_point=GRID_CUT_POINTS[grid_size])
    return result


def assert_branch():
    """Check before any pipeline-generated writes; never switch branches in code."""
    repo = ROOT.parent.parent
    base = ["git", "--no-optional-locks", "-c", f"safe.directory={repo.as_posix()}", "-c", "core.safecrlf=false"]
    branch = subprocess.check_output(base + ["branch", "--show-current"], cwd=repo, text=True).strip()
    if branch not in WRITABLE_BRANCHES:
        raise RuntimeError(f"Refusing writes on branch {branch!r}; use one of {WRITABLE_BRANCHES}.")
    changed = subprocess.check_output(base + ["diff", "--name-only", "HEAD"], cwd=repo, text=True)
    outside = [p for p in changed.splitlines() if not p.startswith("tinyml/fomo/")]
    untracked = subprocess.check_output(base + ["ls-files", "--others", "--exclude-standard"],
                                        cwd=repo, text=True)
    outside += [p for p in untracked.splitlines() if not p.startswith("tinyml/fomo/")]
    if outside:
        raise RuntimeError(f"Unexpected changes outside tinyml/fomo: {outside}")


def output_directory(path=None, label="run"):
    from datetime import datetime, timezone
    assert_branch()
    directory = Path(path).resolve() if path else ROOT / "artifacts" / (
        label + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    # Confine every generated artifact to the nested ignored output area.
    if not directory.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Output must be inside tinyml/fomo/artifacts/.")
    directory.mkdir(parents=True, exist_ok=False)
    return directory
