"""Preparation helpers only. This module never starts training or saves a model."""
import hashlib
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from dataset_loader import load_pair, IMAGE_HEIGHT, IMAGE_WIDTH
from fast_scnn import build_fast_scnn
from pathclear_taxonomy import NUM_CLASSES, IGNORE_LABEL, cityscapes_lookup
from segmentation_metrics import make_metrics

BASELINE = Path(__file__).resolve().parent / "checkpoints/baseline-5epochs/best.keras"


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def remap_cityscapes_tensor(mask):
    mask = tf.cast(mask, tf.int32)
    valid = (mask >= 0) & (mask < 256)
    mapped = tf.gather(tf.constant(cityscapes_lookup(), dtype=tf.int32), tf.clip_by_value(mask, 0, 255))
    return tf.where(valid, mapped, IGNORE_LABEL)


def prepare_targets(mask):
    """Return safe class IDs and weights; always keep both together."""
    mask = tf.convert_to_tensor(mask)
    if not mask.dtype.is_integer:
        raise TypeError("PathClear masks must contain integer IDs.")
    mask = tf.cast(mask, tf.int32)
    valid = (mask >= 0) & (mask < NUM_CLASSES)
    tf.debugging.assert_equal(
        tf.reduce_all(valid | (mask == IGNORE_LABEL)), True,
        message="PathClear masks must contain only 0..9 or 255",
    )
    return tf.where(valid, mask, 0), tf.cast(valid, tf.float32)


def load_cityscapes_pair(image_path, mask_path):
    """Reuse baseline geometry/normalization, restoring void before remapping."""
    image, safe_train_ids, weights = load_pair(image_path, mask_path)
    train_ids = tf.where(weights > 0, safe_train_ids, IGNORE_LABEL)
    targets, weights = prepare_targets(remap_cityscapes_tensor(train_ids))
    return image, targets, weights


def make_loss():
    # Unlike the historical baseline, normalize by valid pixels only.
    # This prevents adding ignored pixels from diluting the loss.
    return keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="mean_with_sample_weight",
    )


def compile_pathclear(model, learning_rate=0.001):
    """Configure future training/evaluation; does not perform optimizer updates."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=make_loss(), weighted_metrics=make_metrics(NUM_CLASSES),
    )


def transfer_baseline(checkpoint=BASELINE):
    """Strict name/shape transfer, with a fresh 10-channel classifier.

    All reused layers are frozen, including BatchNorm, for a future initial
    classifier-training stage. No optimizer state is transferred.
    An unexpected mismatch is an error, never a random-backbone fallback.
    """
    checkpoint = Path(checkpoint).resolve()
    before_hash = sha256(checkpoint)
    source = keras.models.load_model(checkpoint, compile=False)
    expected_input = (None, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    if source.input_shape != expected_input or source.output_shape != (*expected_input[:3], 19):
        raise ValueError("Expected the existing 512x1024, 19-class Cityscapes baseline.")
    target = build_fast_scnn(input_shape=expected_input[1:], num_classes=NUM_CLASSES)
    source_layers = {layer.name: layer for layer in source.layers}
    if set(source_layers) != {layer.name for layer in target.layers}:
        raise ValueError("Baseline and target layer names differ; review architecture before transfer.")
    transferred, reinitialized, unweighted = [], [], []
    for layer in target.layers:
        old = source_layers[layer.name]
        if type(old) is not type(layer):
            raise ValueError(f"Layer type mismatch: {layer.name}")
        original, current = old.get_weights(), layer.get_weights()
        old_shapes, new_shapes = [list(w.shape) for w in original], [list(w.shape) for w in current]
        if layer.name == "classifier_logits":
            if old_shapes != [[1, 1, 128, 19], [19]] or new_shapes != [[1, 1, 128, 10], [10]]:
                raise ValueError("Unexpected classifier structure; review transfer explicitly.")
            reinitialized.append({"name": layer.name, "source_shapes": old_shapes,
                                  "target_shapes": new_shapes, "reason": "19 outputs replaced by 10; fresh classifier"})
            continue
        if old_shapes != new_shapes:
            raise ValueError(f"Unexpected weight mismatch: {layer.name}: {old_shapes} vs {new_shapes}")
        layer.set_weights(original)
        layer.trainable = False
        for before, after in zip(original, layer.get_weights()):
            np.testing.assert_array_equal(before, after, err_msg=f"Transfer failed: {layer.name}")
        if original:
            transferred.append({"name": layer.name, "weight_tensors": len(original),
                                "parameters": sum(int(w.size) for w in original)})
        else:
            unweighted.append(layer.name)
    if not transferred or sha256(checkpoint) != before_hash:
        raise AssertionError("No weights transferred or source checkpoint changed.")
    return target, {
        "baseline": str(checkpoint), "baseline_sha256": before_hash,
        "transferred_layers": transferred, "transferred_layer_count": len(transferred),
        "transferred_weight_tensor_count": sum(row["weight_tensors"] for row in transferred),
        "transferred_parameters": sum(row["parameters"] for row in transferred),
        "reinitialized_layers": reinitialized, "layers_without_weights": unweighted,
        "exact_weight_equality": True, "feature_layers_frozen": True,
        "optimizer_state_transferred": False, "output_shape": list(target.output_shape),
        "training_performed": False, "model_saved": False,
    }
