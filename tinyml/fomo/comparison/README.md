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

## Results (Cityscapes validation, development data, final 2026-10-17)

All five models use the same rebuilt data, shared scorer, INT8 models, pre-registered threshold rule, and desktop timing on this laptop ([results.csv](results.csv)).

| | M0 FOMO v2 | M1 ESPDet-Pico | M2 YOLO-FastestV2 | M3 YOLO11n | M4 SSDLite |
|---|---:|---:|---:|---:|---:|
| Selected threshold | 0.70 | 0.20 (floor) | 0.30 | 0.20 (floor) | 0.20 (floor) |
| Precision / recall / F1 | 71.89 / 38.15 / **49.85%** | 84.64 / 37.75 / **52.21%** | 70.16 / 43.13 / **53.42%** | 88.57 / 41.49 / **56.50%** | 77.66 / 51.68 / **62.06%** |
| F1 with sweep extended to 0.05 | 49.85% | 59.42% (0.10) | 53.42% | 66.89% (0.05) | 62.06% |
| Person / vehicle recall | 32.66 / 41.64% | 29.99 / 42.70% | 30.32 / 51.29% | 30.62 / 48.41% | 34.23 / 62.79% |
| Recall below 8 px / at least 8 px | 31.32 / 62.81% | 26.99 / 76.62% | 37.94 / 61.85% | 30.50 / 81.18% | 45.29 / 74.75% |
| mAP@0.5 (secondary) | — | 0.2555 | 0.0264 | 0.2776 | 0.1603 |
| Parameters | 20,787 | 361,266 (17.4x) | 237,401 (11.4x) | 2,605,126 (125.3x) | 616,399 (29.7x) |
| MACs | 16.3 M | 44.2 M (2.7x) | 21.3 M (1.3x) | 203.5 M (12.5x) | 36.6 M (2.2x) |
| INT8 file | 70,736 B | 496,160 B (7.0x) | 420,368 B (5.9x) | 2,810,368 B (39.7x) | 1,014,816 B (14.3x) |
| Training budget | frozen v2 | official 1,200 epochs, best 1,195 | official 300, best checkpoint 180 | official 100, best 99 | 40-epoch cap, early stop at 28 (best 23) |
| Desktop model + decode, median / p95 | 2.66 / 3.18 ms | simulator only | simulator only | simulator only | 4.77 / 5.47 ms (1.8x) |

M3 YOLO11n (Ultralytics 100-epoch recipe from COCO weights, best epoch 99, so possibly under-trained) scored 88.57% precision, 41.49% recall, and 56.50% F1 at threshold 0.20.
Its person / vehicle recall is 30.62% / 48.41%, recall below 8 px / at least 8 px is 30.50% / 81.18%, and mAP@0.5 is 0.2776.
It has 2,605,126 parameters, 203.5 M MACs (12.5x FOMO v2), and a 2,810,368-byte `.espdl` file (39.7x).

M2 YOLO-FastestV2 (official 300-epoch recipe, 75 minutes; highest-AP checkpoint epoch 180) scored 70.16% precision, 43.13% recall, and 53.42% F1 at threshold 0.30, inside the sweep range.
Its person / vehicle recall is 30.32% / 51.29%, recall below 2 px / 2–8 px / at least 8 px is 25.11% / 44.46% / 61.85%, and mAP@0.5 is only 0.0264: its centers are good but its boxes are poorly localized.
It has 237,401 parameters, 21.3 M MACs (1.3x FOMO v2), and a 420,368-byte `.espdl` file (5.9x).
It is the closest challenger to FOMO v2: 3.6 F1 points higher for 1.3x the compute.
Its evaluation calibrates one image per step (100 steps, all 100 images) because the official export fixes batch size 1 inside ShuffleNetV2's channel-shuffle reshapes.

Under both the primary rule and the extended sweep, FOMO v2 has the lowest F1 of the five candidates and by far the lowest cost; M4 is the most accurate under the primary rule and M3 under the extended sweep.
M1's 300-epoch cap gave 82.81% precision, 22.61% recall, and 35.52% F1 with only 3.78% person recall; its best epoch was 291, so the PLAN.md rule triggered the unchanged 1,200-epoch recipe.
That run took 12.9 h and raised float mAP50 from 0.175 to 0.250 and person recall to 29.99%; its best epoch was 1,195, so ESPDet-Pico was still improving at the end of the official schedule.
The table reports the 1,200-epoch M1 (`artifacts/m1-espdet-evaluation-1200/`); the 300-epoch result remains in `artifacts/m1-espdet-evaluation/`.
Two exports of the same M1 weights produced `.espdl` files of 497,376 and 497,424 bytes with identical scores, so ESP-PPQ quantization is not byte-for-byte deterministic.

### Threshold-range sensitivity (secondary, `threshold_sensitivity.py`)

M1, M3, and M4 all selected 0.20, the lowest value in the pre-registered sweep, so I reran the same rule over 0.05–0.95 (`artifacts/threshold-sensitivity-final/`).
M0 and M4 keep their thresholds (M4's precision falls below 70% at 0.15).
The 1,200-epoch M1 moves to threshold 0.10 with 74.66% precision, 49.35% recall, and 59.42% F1 (the 300-epoch M1 moved to 48.99%): its confidence scores run lower than the other models', so the sweep floor limits its primary F1.
M3 also selected 0.20 under the primary rule; over 0.05–0.95 it moves to 0.05 with 72.34% precision, 62.21% recall, and 66.89% F1, the highest F1 of any candidate.
0.05 is also the decoding score floor, so M3's extended optimum may lie lower still.
The primary results above remain the registered ones; this analysis is reported alongside them.
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

## M1: ESPDet-Pico (Espressif esp-detection)

| Item | Value |
|---|---|
| Source | [espressif/esp-detection](https://github.com/espressif/esp-detection) at commit `28de5902a357`, cloned to `C:/Projects/esp-detection` (outside this repository; AGPL-3.0) |
| Environment | `.venv-espdet`: Python 3.11.9, PyTorch 2.2.0 CPU, Ultralytics 8.4.177, NumPy 1.24.4, ONNX 1.17.0, ESP-PPQ |
| Model | `cfg/models/espdet_pico.yaml`, from scratch, nc overridden 1 -> 2; 361,266 parameters, 44.2 M MACs at 160 x 160 |
| Data | `prepare_yolo.py`: FOMO v2's cached 160 x 160 stretched images and the same boxes, so Ultralytics never letterboxes |
| Recipe | Official `train.py` settings at imgsz 160, seed 42; a 300-epoch cap under the PLAN.md rule (one epoch 65 s), then the unchanged official 1,200 epochs when the rule triggered |
| INT8 | ESP-PPQ 8-bit for target `esp32` (no SIMD kernels), FOMO v2's 100 calibration images; final `.espdl` file is 496,160 bytes (exports vary by tens of bytes) |
| Scoring | ESP-PPQ `TorchExecutor` INT8 simulation, esp-detection's decoder, Ultralytics class-aware NMS at IoU 0.7 (firmware default), then the shared scorer |

Recorded deviations and caveats:

- YOLO labels cannot express FOMO's ignore regions, so void and group areas train as background.
- Ultralytics selects `best.pt` by its own box fitness, not centroid F1.
- esp-detection's firmware keeps at most 10 boxes per frame; scoring is uncapped like M0 and M4.
- `evaluate_espdet.py` (which produced the M1 artifacts) and its generalization `evaluate_ultralytics.py` (M3, and M1 via `--arch espdet`) shim one Ultralytics 8.4 API change (`export_onnx` return value); the exported graph is unchanged.
  Rerunning M1 through `evaluate_ultralytics.py --arch espdet` reproduced the identical sweep, selected point, and mAP (`artifacts/m1-espdet-evaluation-regression/`); only the `.espdl` size moved by 64 bytes.
- The simulator's timing is not a deployment latency, so M1 has no desktop latency column; G4 on the board decides speed.

### Training-budget sensitivity (epoch-50 snapshot)

Midway through the 300-epoch run I scored `last.pt` after epoch 50 through the full INT8 path (`artifacts/m1-espdet-decode-check-e50/`).
It reached 73.00% precision, 23.01% recall, and 34.99% F1 at threshold 0.20, with mAP@0.5 0.1185 against Ultralytics' float validation mAP50 of 0.110, which also confirms the INT8 decoding.
Float validation mAP50 rose from 0.103 at epoch 50 to 0.130 at epoch 75 while validation loss fell with training loss, so the model was not overfitting.
A 50-epoch cap would leave M0 and M4 unchanged (they early-stopped at epochs 19 and 28) but would stop this from-scratch model while it was still improving.
I therefore kept the pre-registered 300-epoch cap; this snapshot comes from a 300-epoch schedule (learning rate still high, mosaic still on), so it is not a true 50-epoch result.
The budget mattered: F1 went from 34.99% (epoch-50 snapshot) to 35.52% (300 epochs) to 52.21% (1,200 epochs), and person recall from 3.78% at 300 epochs to 29.99% at 1,200.

## Phase 3 pre-board gate checks (2026-10-16)

**G1 runtime support (static).** Every operator in each INT8 graph is on its runtime's supported list:

| Model | Runtime | Operators |
|---|---|---|
| M0 FOMO v2 | TFLite Micro | ADD, CONV_2D, DEPTHWISE_CONV_2D, MUL, SOFTMAX |
| M4 SSDLite | TFLite Micro | ADD, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, MUL, RESHAPE, SOFTMAX |
| M1 ESPDet-Pico | ESP-DL | Conv, Relu, Split, Add, Concat, MaxPool, Resize |
| M3 YOLO11n | ESP-DL | Conv, Sigmoid, Mul, Split, Add, Concat, MaxPool, Reshape, Transpose, MatMul, Softmax, Resize |
| M2 YOLO-FastestV2 | ESP-DL | Conv, Relu, MaxPool, Concat, Reshape, Transpose, Gather, Resize, Sigmoid, Softmax |

ESP-DL lists plain `esp32` as a target without PIE vector instructions, so its operators run as C kernels ([operator support](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md)).
A static pass does not prove the build links, allocates, or runs; the board build confirms G1.

**G3 RAM (estimate, `estimate_memory.py`).** Peak INT8 activation memory with a greedy first-fit planner, weights excluded:

| Model | Peak activations | Largest tensor |
|---|---:|---:|
| M0 FOMO v2 | 384,000 B | 307,200 B |
| M4 SSDLite | 384,000 B | 307,200 B |
| M3 YOLO11n | 307,200 B | 102,400 B |
| M2 YOLO-FastestV2 | 307,200 B | 153,600 B |
| M1 ESPDet-Pico | 204,800 B | 102,400 B |

Peak activation memory follows the 160 x 160 input and early layers, not model size: FOMO and SSDLite share the MobileNetV2 stem, whose 80 x 80 x 48 expansion tensor is the largest buffer of any model.
Every candidate exceeds the classic ESP32's free internal SRAM and fits easily in the AI-Thinker board's 4 MB PSRAM, so this estimate does not separate the candidates; runtime scratch and persistent allocations still need board measurement.

**G2 flash (not yet built).** INT8 model files range from 70,736 B (M0) to 2,810,368 B (M3) against 4 MB of flash shared with the application, camera, and Wi-Fi code.
`idf.py size` needs ESP-IDF, which this laptop does not have (see open items).

## Open items for Angel and the team

- **ESP-IDF install (disk space):** drive C: had 12 GB free (98% full) on 2026-10-16; ESP-IDF needs about 4–5 GB, so I did not install it. G2 builds wait for that decision.
- **G4 threshold:** 3 FPS remains a working value pending team and Ms. Capule's confirmation.
- **Licensing for Commercialization:** ESPDet-Pico and YOLO11n are AGPL-3.0 (Ultralytics); YOLO-FastestV2 has no license file.
- **Hardware:** Phase 4 needs a working ESP32-CAM; Phase 5 needs recorded, labeled walking footage.

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
