# My training pipelines

## Fast-SCNN segmentation

```text
Cityscapes -> dataset loader -> 19-class Fast-SCNN baseline
-> PathClear class remapping + custom Philippine dataset
-> fine-tuning -> held-out evaluation -> TFLite deployment
```

I retain the existing Fast-SCNN code and the frozen baseline checkpoint.
`dataset_config.py` selects the external data root; `dataset_loader.py` pairs RGB images with trainId masks.
It resizes RGB with bilinear interpolation, masks with nearest-neighbor interpolation, and normalizes RGB to 0..1.
Void pixels receive zero sample weights.
`fast_scnn.py` produces per-pixel logits at 512 x 1024.

I use `pathclear_taxonomy.py` as the executable source of truth for the ten classes and Cityscapes mapping.
`configs/pathclear_classes.yaml` records those same zero-based IDs for the handoff; it is not loaded by the training scripts.
Cityscapes supports only the first seven PathClear classes.
Stairs, Crosswalk, and Uneven pavement require reviewed local annotations.
The existing Cityscapes adaptation experiment is documented in [TRAINING.md](TRAINING.md) and [the stage report](PATHCLEAR_CITYSCAPES_STAGE_2026-09-29.md).
I distinguish that experiment from Philippine fine-tuning and local accuracy.

I use `train.py` for the original baseline and the existing PathClear training/continuation scripts for their documented stages.
I evaluate on held-out data and preserve route/session separation before considering deployment.
`scripts/export_pathclear_tflite.py` and `scripts/verify_pathclear_tflite_bundle.py` provide the existing export and verification path.
I follow their command help and [TRAINING.md](TRAINING.md), including deployment-bundle checks, rather than treating a successful conversion as device readiness.

My first infrastructure check from the project directory in WSL is:

```bash
bash run_wsl.sh tests/test_pipeline.py
```

This reads one batch and runs inference with random model weights; it does not evaluate a trained checkpoint or train.
I keep checkpoints, generated bundles, and evaluation outputs local.
No training is authorized by this repository preparation.

## Experimental TinyML/FOMO

```text
PathClear images / reviewed converted annotations
-> resize/preprocess -> FOMO-compatible object labels
-> TinyML/FOMO training -> quantization
-> deployment experiment on ESP32 / ESP32-S3 candidate
```

I isolate this work in `tinyml/` and plan to use the `tinyml-fomo` branch after the initial repository push.
FOMO remains experimental and does not replace Fast-SCNN.
I first agree object definitions, annotation format, input geometry, target board, and an independent test split.
I transform annotations with the same resize/crop geometry as images and check whether small or adjacent objects remain distinguishable.
I then measure quantized accuracy, memory use, camera integration, and latency on the actual board.
ESP32 and ESP32-S3 are candidates, not verified deployment targets for this project.
The FOMO pipeline lives on the `tinyml-fomo` branch, starting from [tinyml/fomo/README.md](../tinyml/fomo/README.md).
I start with [tinyml/README.md](../tinyml/README.md) before selecting tooling or collecting annotations.
