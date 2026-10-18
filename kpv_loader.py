"""Kaggle FPVCrosswalk2025 supervision for the ten-class PathClear head.

Geometry matches dataset_loader (512x1024, bilinear RGB resize,
nearest-neighbor mask resize). Marked mask pixels (>127) become PathClear
Crosswalk (ID 8) at full sample weight. Unmarked pixels become weak negative
evidence with the horizon split proven on MicroSeg: rows above
``horizon_fraction * 512`` take ``horizon_id`` (Boundary, ID 4: facades and
sky are the lateral anchor, matching the shared Boundary map), rows below
take ``negative_id`` (Road, ID 0). Both are never Crosswalk, and that is
exactly the pressure the negative evidence applies: without it the model is
free to paint Crosswalk anywhere, because ignored pixels never enter the
loss. ``horizon_fraction`` of 0.0 labels every unmarked pixel
``negative_id``; ``negative_weight`` of 0.0 restores the old contract where
unmarked pixels are Ignore (255) with zero weight.

Teacher labels (``teacher_path``) replace the horizon split. They are the
pre-KPV checkpoint's confident argmax over IDs 0-6 (see
``scripts/make_kpv_teacher_labels.py``), so unmarked pixels keep a full-scene
reference: people stay Person and trees stay Vegetation instead of being
taught as Boundary. Inside the marked crossing, confident teacher Person or
Vehicle pixels (``TEACHER_OCCLUDER_IDS``) keep the teacher label, because a
cyclist on a zebra is still a cyclist. Low-confidence teacher pixels (255)
are ignored.
"""
import json
from pathlib import Path

import tensorflow as tf

from dataset_config import KAGGLE_CROSSWALK_ROOT
from dataset_loader import IMAGE_HEIGHT, IMAGE_WIDTH
from pathclear_pipeline import prepare_targets
from pathclear_taxonomy import IGNORE_LABEL

CROSSWALK_ID = 8
HORIZON_ID = 4
NEGATIVE_ID = 0
HORIZON_FRACTION = 0.45
TEACHER_WEIGHT = 0.5
TEACHER_OCCLUDER_IDS = (2, 3)

AUTOTUNE = tf.data.AUTOTUNE


def _unit_interval(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be between 0 and 1.") from error
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return number


def load_kpv_image(image_path):
    """Decode and resize one KPV RGB frame to the 512x1024 training geometry."""
    raw_image = tf.io.read_file(image_path)
    image = tf.io.decode_image(raw_image, channels=3, expand_animations=False)
    image = tf.image.resize(
        image, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    return tf.cast(image, tf.float32) / 255.0


def _teacher_targets(positive, teacher_path, teacher_weight):
    teacher = tf.io.decode_png(tf.io.read_file(teacher_path), channels=1)
    teacher = tf.image.resize(
        teacher,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    teacher = tf.cast(tf.squeeze(teacher, axis=-1), tf.int32)
    occluder = tf.zeros_like(positive)
    for class_id in TEACHER_OCCLUDER_IDS:
        occluder |= teacher == class_id
    crossing = positive & ~occluder
    targets = tf.where(crossing, tf.constant(CROSSWALK_ID, tf.int32), teacher)
    weights = tf.where(
        crossing, tf.constant(1.0, tf.float32), tf.constant(teacher_weight, tf.float32)
    )
    return targets, weights


def load_kpv_pair(
    image_path,
    mask_path,
    negative_weight=0.25,
    horizon_fraction=HORIZON_FRACTION,
    horizon_id=HORIZON_ID,
    negative_id=NEGATIVE_ID,
    teacher_path=None,
    teacher_weight=TEACHER_WEIGHT,
):
    """Load one KPV image/mask pair as PathClear targets plus sample weights.

    With ``teacher_path`` the horizon-split arguments are unused and
    unmarked pixels take the teacher label at ``teacher_weight``.
    """
    negative_weight = _unit_interval("negative_weight", negative_weight)
    teacher_weight = _unit_interval("teacher_weight", teacher_weight)
    horizon_fraction = _unit_interval("horizon_fraction", horizon_fraction)
    for name, value in (("horizon_id", horizon_id), ("negative_id", negative_id)):
        if not isinstance(value, int) or not 0 <= value <= 9:
            raise ValueError(f"{name} must be an integer PathClear class ID 0..9.")
    if horizon_id == negative_id:
        raise ValueError(
            "horizon_id and negative_id must be different classes for the "
            "horizon split to carry information."
        )

    image = load_kpv_image(image_path)

    # KPV masks ship as JPEG, so decode_image (not decode_png) is required.
    raw_mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_image(raw_mask, channels=1, expand_animations=False)
    mask = tf.image.resize(
        mask,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    mask = tf.cast(tf.squeeze(mask, axis=-1), tf.int32)

    positive = mask > 127
    crosswalk = tf.fill(tf.shape(mask), CROSSWALK_ID)
    if teacher_path is not None:
        targets_raw, weights_raw = _teacher_targets(positive, teacher_path, teacher_weight)
    elif negative_weight > 0:
        if horizon_fraction <= 0:
            negative = tf.fill(tf.shape(mask), negative_id)
        else:
            horizon_row = int(round(IMAGE_HEIGHT * horizon_fraction))
            rows = tf.range(IMAGE_HEIGHT)[:, None]
            negative_column = tf.where(
                rows < horizon_row,
                tf.constant(horizon_id, tf.int32),
                tf.constant(negative_id, tf.int32),
            )
            negative = tf.broadcast_to(negative_column, tf.shape(mask))
        targets_raw = tf.where(positive, crosswalk, negative)
        weights_raw = tf.where(
            positive,
            tf.constant(1.0, tf.float32),
            tf.constant(negative_weight, tf.float32),
        )
    else:
        targets_raw = tf.where(
            positive, crosswalk, tf.fill(tf.shape(mask), IGNORE_LABEL)
        )
        weights_raw = None

    targets, supervised = prepare_targets(targets_raw)
    weights = supervised if weights_raw is None else supervised * weights_raw
    return image, targets, weights


def attach_teacher_labels(pairs, teacher_manifest_path):
    """Return (image, mask, teacher_label) triples for every KPV pair.

    Fails loudly when any frame lacks a teacher label, so a stale manifest
    can never silently fall back to the horizon split.
    """
    manifest = json.loads(Path(teacher_manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported teacher manifest: {teacher_manifest_path}")
    labels = manifest["labels"]
    missing = [image for image, _ in pairs if image not in labels]
    if missing:
        raise ValueError(
            f"{len(missing)} KPV frames have no teacher label, e.g. {missing[0]}."
        )
    return [(image, mask, labels[image]) for image, mask in pairs]


def pairs_from_split_manifest(manifest_path, root=None, split="train", source=None):
    """Read (image, mask) pairs from a crosswalk_split.json manifest.

    Paths in the manifest are relative to the dataset root; absolute paths
    and ``..`` escapes are rejected. ``source`` optionally restricts to
    ``real`` or ``synthetic``.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported Crosswalk split manifest: {manifest_path}")
    root = Path(KAGGLE_CROSSWALK_ROOT if root is None else root)
    if Path(manifest["dataset_root"]).resolve() != root.resolve():
        raise ValueError(
            f"Manifest dataset root {manifest['dataset_root']} does not match {root}."
        )
    if source is not None and source not in ("real", "synthetic"):
        raise ValueError("source must be 'real', 'synthetic', or None.")
    pairs = []
    for item in manifest.get("items", []):
        if item.get("split") != split:
            continue
        if source is not None and item.get("source") != source:
            continue
        image_relative, mask_relative = Path(item["image"]), Path(item["mask"])
        if (
            image_relative.is_absolute()
            or mask_relative.is_absolute()
            or ".." in image_relative.parts
            or ".." in mask_relative.parts
        ):
            raise ValueError(f"Crosswalk paths must be relative to {root}: {item!r}")
        pairs.append((str(root / image_relative), str(root / mask_relative)))
    if not pairs:
        raise ValueError(
            f"No Crosswalk pairs for split={split!r} source={source!r} "
            f"in {manifest_path}."
        )
    return pairs


def make_kpv_dataset(
    pairs,
    training=False,
    batch_size=1,
    negative_weight=0.25,
    horizon_fraction=HORIZON_FRACTION,
    seed=None,
    teacher_weight=TEACHER_WEIGHT,
):
    """Build a batched KPV dataset; repeated when training for mixed sampling.

    ``pairs`` holds (image, mask) pairs, or (image, mask, teacher_label)
    triples from ``attach_teacher_labels`` to train on teacher labels.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not pairs:
        raise ValueError("KPV pairs must not be empty.")
    columns = tuple(list(column) for column in zip(*pairs))
    if len(columns) not in (2, 3):
        raise ValueError("KPV pairs must be (image, mask) or (image, mask, teacher).")
    dataset = tf.data.Dataset.from_tensor_slices(columns)
    if training:
        dataset = dataset.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda image, mask, *teacher: load_kpv_pair(
            image,
            mask,
            negative_weight=negative_weight,
            horizon_fraction=horizon_fraction,
            teacher_path=teacher[0] if teacher else None,
            teacher_weight=teacher_weight,
        ),
        num_parallel_calls=AUTOTUNE,
    )
    if training:
        # Repeat so the smaller source is never exhausted inside a mixed
        # sampler; without this the realised mix drifts with steps_per_epoch.
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

