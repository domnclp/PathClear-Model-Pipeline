# My standalone detector comparison

This folder holds the comparison described in [PLAN.md](PLAN.md).
Shared scoring lives in `common.py`; it reuses the frozen v2 matching code, and a unit test proves it reproduces v2's own `measure()` counts exactly.
Each candidate gets its own model, training, export, and evaluation scripts.

## M4: SSDLite-MobileNetV2 alpha 0.35

| Item | Value |
|---|---|
| Input | 160 x 160 RGB [0,1], v2 direct bilinear resize, embedded `2*x-1` scaling |
| Backbone | MobileNetV2 alpha 0.35, ImageNet weights (same SHA-256 as v2) |
| Feature levels | 20 x 20 (`block_6_expand_relu`, v2's cut point), 10 x 10, 5 x 5, plus SSDLite extras 3 x 3 and 2 x 2 |
| Anchors | 2,690: four width/height ratios (1, 0.5, 0.25, 2) plus one interpolated square per cell |
| Head | Depthwise 3 x 3 + pointwise predictors per level (SSDLite) |
| Loss | Softmax cross-entropy with 3:1 hard-negative mining, smooth-L1 box loss |
| Ignore policy | Negatives centered in v2's target ignore cells contribute no loss |
| Training | Adam 0.001, batch 32, seed 42, horizontal flips, v2's patience rules, 40-epoch budget |
| Selection | Validation centroid F1 at 0.60, as for v1/v2 |
| Export | Full INT8, batch fixed at 1, v2's 100 calibration images (checked by ID) |
| Decoding | Per-class NMS at IoU 0.45, box center used as the centroid |

I measured 616,399 parameters and 36.6 million MACs, against FOMO v2's 20,787 parameters and 16.3 million MACs with the same counter.
The INT8 graph uses only `ADD`, `CONCATENATION`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `MUL`, `RESHAPE`, and `SOFTMAX`.
A synthetic smoke export produced a 1,013,864-byte INT8 file: 617 KB of weights, 268 KB of per-channel quantization parameters, and flatbuffer structure.
These synthetic figures prove the plumbing only, not accuracy.

The 40-epoch budget does not favor M4: v2 early-stopped at epoch 19 of 20 (best epoch 14), and the same patience rules stop M4 when validation F1 stops improving.

## Results so far (Cityscapes validation, development data, 2026-10-15)

Both rows use the same rebuilt data, shared scorer, INT8 models, pre-registered threshold rule, and desktop timing on this laptop ([results.csv](results.csv)).

| | M0 FOMO v2 | M4 SSDLite |
|---|---:|---:|
| Selected threshold | 0.70 | 0.20 (lowest swept value) |
| Precision / recall / F1 | 71.89% / 38.15% / 49.85% | 77.66% / 51.68% / 62.06% |
| Person recall / Vehicle recall | 32.66% / 41.64% | 34.23% / 62.79% |
| Recall below 8 px / at least 8 px | 31.32% / 62.81% | 45.29% / 74.75% |
| Parameters | 20,787 | 616,399 (29.7x) |
| MACs | 16.3 M | 36.6 M (2.2x) |
| INT8 file | 70,736 B | 1,014,816 B (14.3x) |
| Desktop model + decode, median / p95 | 2.66 / 3.18 ms | 4.77 / 5.47 ms (1.8x) |

M4 is more accurate than FOMO v2 on this validation split.
These are desktop figures; G1–G4 remain unmeasured, so no ESP32-CAM conclusion follows yet.
The validation split selected both checkpoints and both thresholds, so neither row is an independent test.

### Decisions and evidence

- **Data equivalence:** the rebuilt manifest reproduces the committed conversion counts, split order, calibration IDs, and frozen v2's exact validation counts (3,860 TP, 1,509 FP, 6,259 FN, 1,072 ignored); see `artifacts/rebuild-verification/`.
  Claude performed the visual reviews of the rebuilt data and targets; the receipts say so.
- **Anchors unchanged:** on the training split only 19.0% of objects have a best-anchor IoU of at least 0.5 with the predefined anchors.
  Three data-derived alternatives reached 14.9–18.6%, so I kept the predefined set rather than tune it.
  The limit is the 8-pixel anchor spacing against objects whose median person size is 2.3 x 10.8 input pixels; FOMO's centroid matching does not depend on box overlap.
- **Sanity gate:** the 30-update M4 sanity run produced no detections at 0.60 (object probability at positive anchors median 0.33 versus 0.28 at background), so the gate is recorded as failed in `artifacts/m4-ssdlite-sanity/`.
  The synthetic run had already shown convergence; full training then reached validation F1 0.21 at 0.60 by epoch 23 and early-stopped at epoch 28 after 21 minutes on the CPU.

## Environment on this laptop

I use Python 3.13.5 with the pinned packages from `../requirements.txt` in the repository `.venv`.
Native Windows TensorFlow 2.21.0 has no GPU support, so training runs on the CPU.
The pinned environment in `../requirements.txt` was tested with Python 3.11.15; I record this difference in every run's metadata.

## Commands

I run these from the repository root in Git Bash.
Cityscapes needs `gtFine_trainvaltest` and `leftImg8bit_trainvaltest` extracted side by side (on this laptop: `C:/dataset/leftImg8bit_trainvaltest/{leftImg8bit,gtFine}`).

```bash
PY=.venv/Scripts/python.exe
A=tinyml/fomo/artifacts

# 1. Rebuild the v2 data on this machine, reviewing previews before each receipt.
$PY -B tinyml/fomo/prepare_cityscapes.py --source C:/dataset/leftImg8bit_trainvaltest --output $A/cityscapes-data-rebuild
$PY -B tinyml/fomo/inspect_dataset.py --manifest $A/cityscapes-data-rebuild/manifest.json --review-note "<actual findings>"
$PY -B tinyml/fomo/refine_targets.py --manifest $A/cityscapes-data-rebuild/manifest.json --grid-size 20 --output $A/cityscapes-center-targets-rebuild
$PY -B tinyml/fomo/inspect_dataset.py --manifest $A/cityscapes-center-targets-rebuild/manifest.json --review-note "<actual findings>"

# 2. Prove equivalence: counts, split order, calibration IDs, and exact v2 validation counts.
$PY -B tinyml/fomo/comparison/verify_rebuild.py --manifest $A/cityscapes-center-targets-rebuild/manifest.json

# 3. M4 sanity gate, training, export, and evaluation.
$PY -B tinyml/fomo/comparison/train_ssdlite.py --manifest $A/cityscapes-center-targets-rebuild/manifest.json --sanity
$PY -B tinyml/fomo/comparison/train_ssdlite.py --manifest $A/cityscapes-center-targets-rebuild/manifest.json --output $A/m4-ssdlite-training
$PY -B tinyml/fomo/comparison/export_ssdlite.py --model $A/m4-ssdlite-training/best.keras --manifest $A/cityscapes-center-targets-rebuild/manifest.json --output $A/m4-ssdlite-export
$PY -B tinyml/fomo/comparison/evaluate_ssdlite.py --manifest $A/cityscapes-center-targets-rebuild/manifest.json --model $A/m4-ssdlite-export/model.int8.tflite --output $A/m4-ssdlite-evaluation

# Tests
$PY -B -m unittest discover -s tinyml/fomo/comparison -p 'test_*.py'
```

Every writer refuses existing output folders and paths outside `tinyml/fomo/artifacts/`.
