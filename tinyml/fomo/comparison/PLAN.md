# My standalone detector comparison plan

I drafted this plan on 2026-10-15.
It compares the frozen FOMO v2 checkpoint against four alternative detectors for the **no-phone fallback mode**, where FOMO alone runs on the AI-Thinker ESP32-CAM.
In the primary phone-connected mode, FOMO and Fast-SCNN both run on the phone; that mode is outside this comparison.
No results exist yet. Every figure below is either a verified v2 value or explicitly marked as unmeasured.

## Question

Which object detector best fits the ESP32-CAM fallback mode, given its flash, RAM, and speed limits?
I compare fit for this hardware, not accuracy alone.
FOMO v2 currently reaches 71.89% precision, 38.15% recall, and 49.85% F1 on Cityscapes validation, so a larger detector may well be more accurate.

## Candidates

| ID | Model | Reason for inclusion | Status |
|---|---|---|---|
| M0 | FOMO v2 (`checkpoints/cityscapes_v2/`) | Baseline, used unchanged | Frozen; 20,787 parameters; 70,736-byte INT8 file |
| M1 | ESPDet-Pico (Espressif ESP-DL) | Tiny detector built for Espressif chips | Phases 1–2 done 2026-10-17 (1,200-epoch recipe); see `README.md` |
| M2 | YOLO-Fastest v2 | Smallest common YOLO-family detector | Phases 1–2 done 2026-10-16; see `README.md` |
| M3 | YOLOv8n or YOLO11n at 160 x 160 | Standard lightweight YOLO | YOLO11n; phases 1–2 done 2026-10-16 |
| M4 | SSDLite-MobileNetV2, alpha 0.35 | Same backbone family as FOMO, box head instead of centroid grid | Phases 1–2 done 2026-10-15; see `README.md` |

## Pre-registered decision rule

I commit this rule before collecting any candidate results.

Hard gates, each recorded as pass or fail with evidence:

- **G1 Runtime support:** every operator runs under the chosen runtime (TFLite Micro or ESP-DL) on the classic ESP32.
- **G2 Flash:** firmware plus model fits the ESP32-CAM application partition (`idf.py size`).
- **G3 RAM:** tensor arena plus camera frame buffers fit in 520 KB internal SRAM and 4 MB PSRAM, measured with `RecordingMicroInterpreter` and ESP-IDF heap counters.
- **G4 Speed:** complete capture-to-result loop reaches at least **3 FPS** (fresh frames divided by elapsed wall time) on the board. **3 FPS is a working value, pending team confirmation** (set 2026-10-15).

Ranking among models that pass all gates:

1. Micro F1 at the shared operating point: highest recall among thresholds 0.20:0.05:0.95 with overall precision >= 70%, the same rule used to select v2.
2. Tie-break: lower median loop latency, then lower peak RAM.
3. Reported design factor, not a measured score: whether the same model can serve both the phone and the fallback mode.

If a competitor passes the gates and ranks above FOMO, I report that result unchanged.

## M1 ESPDet-Pico training budget (pre-registered 2026-10-15, before any M1 training)

ESPDet-Pico trains from scratch with Espressif's official recipe (`esp-detection/train.py`: 1,200 epochs, batch 128, mosaic 1.0, copy-paste 0.1, close-mosaic 30, optimizer auto), at 160 x 160 on Person and Vehicle.
Ultralytics early stopping (default patience 100 on its own validation fitness) stays enabled.

1. I time one full training epoch on this laptop before choosing the budget.
2. If 1,200 epochs fit in about 8 hours (one epoch of about 24 s or less), I run the official 1,200-epoch budget unchanged.
3. Otherwise I cap the run at 300 epochs and record the cap as a deviation from the official recipe.
4. If the capped run's best epoch is 250 or later, I resume training toward 1,200 epochs until early stopping or epoch 1,200.
5. Whatever the budget, final scoring uses the shared centroid scorer and threshold rule above on INT8 outputs.

Timing result (2026-10-15): one epoch took 58.7 s of training plus validation, 0.018 h in total, so 1,200 epochs would need about 21.7 h.
Under step 3 I therefore cap M1 at 300 epochs (about 5.4 h) and record the cap as a deviation.
Ultralytics resume cannot extend a finished schedule, so if step 4 triggers I rerun the unchanged official 1,200-epoch recipe.

Outcome (2026-10-16): the 300-epoch run's best epoch was 291 by Ultralytics fitness, so step 4 triggered.
Float validation mAP50 rose only from 0.172 (epoch 232) to 0.175 (epoch 300).
By team decision, the 1,200-epoch rerun is scheduled last, after M3 and M2, because it is the longest job; it remains required, and the 300-epoch figures stand until it finishes.
Rerun (2026-10-17): the unchanged 1,200-epoch recipe took 12.9 h; its best epoch was 1,195 (float mAP50 0.250, mAP50-95 0.118), so the model was still improving at the official schedule's end.
The 1,200-epoch result replaces the 300-epoch figures as M1's primary result; there is no further extension beyond the official recipe.

## M3 YOLO11n training and quantization (pre-registered 2026-10-16, before any M3 training)

M3 is YOLO11n (Ultralytics), chosen over YOLOv8n because Espressif ships an official YOLO11n deployment path in ESP-DL.
It follows the standard Ultralytics custom-data recipe: COCO-pretrained `yolo11n.pt`, 100 epochs, batch 16, patience 100, optimizer auto, default augmentation, imgsz 160, seed 42, CPU.
COCO pretraining includes person and car classes, a detection-specific head start that M0 and M4 (ImageNet classification weights) do not have; I report this rather than remove the model's standard initialization.

1. I time one epoch; if 100 epochs fit in about 8 hours I run them unchanged.
2. If they do not fit, I stop and ask before choosing a budget.
3. If the best epoch is 90 or later, I report M3 as possibly under-trained; there is no automatic extension, since 100 is the official default.

Outcome (2026-10-16): one timed epoch took 72 s, but the 100-epoch run took 7.0 h (about 253 s per epoch, partly from a concurrent smoke test).
The best epoch by Ultralytics fitness was 99 (float mAP50 0.272, mAP50-95 0.139), so under step 3 I report M3 as possibly under-trained.

Export and INT8 follow Espressif's "How to deploy YOLO11n" tutorial: Detect forward overridden to six raw outputs, ONNX opset 13 with onnxsim, ESP-PPQ `espdl_setting` with percentile activation calibration and TQT optimization (800 steps, the tutorial's ESP32-S3 setting), on CPU, for target `esp32`, calibrated on FOMO v2's 100 images.
Scoring reuses the M1 path: TorchExecutor INT8 simulation, Ultralytics DFL decoding, class-aware NMS at IoU 0.7, shared centroid scorer and threshold rule.

## M2 YOLO-FastestV2 training and quantization (pre-registered 2026-10-16, before any M2 training)

M2 is [dog-qiuqiu/Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2) at commit `ac2a5e3060bf`, cloned to `C:/Projects/Yolo-FastestV2`; the repository has no license file.
It follows the official `data/coco.data` recipe: ImageNet-pretrained ShuffleNetV2 backbone, 300 epochs, MultiStep learning-rate drops at 150 and 250, batch 128, SGD at 0.001 with the built-in 5-epoch warm-up, built-in augmentation, 3 anchors per level.
Changes: 2 classes, 160 x 160 input, and six anchors from the official `genanchors.py` on our training labels (1.82,5.18, 4.52,11.45, 5.99,29.58, 12.40,17.52, 19.71,42.19, 38.16,69.54).
`run_yolofastest.py` restores NumPy aliases removed since 2022 and sets seed 42; the clone's code is unmodified, and its training writes checkpoints to the clone's `weights/` folder.

1. I time one epoch; if 300 epochs fit in about 8 hours I run them unchanged, otherwise I stop and ask.
2. The official script saves a checkpoint every 10 epochs with its own validation AP and has no best-checkpoint logic or early stopping; I select the saved checkpoint with the highest AP.
3. If that checkpoint is from epoch 250 or later, I report M2 as possibly under-trained.

Timing (2026-10-16): one training epoch took 40 s (23 batches of 128), so 300 epochs plus the 30 built-in evaluations fit in about 4–5 h; under step 1 I run the official 300 epochs unchanged.
`run_yolofastest.py` also shims `Tensor.clamp_` for integer tensors with float bounds (`utils/loss.py`), which torch 1.9 allowed and torch 2.2 rejects; casting the bounds reproduces the original result.

Outcome (2026-10-16): the 300 epochs took 75 minutes; the highest saved AP was 0.0351 at epoch 180, so under step 3 M2 is not flagged as under-trained.

M2 has no official ESP-DL recipe, so INT8 uses ESP-PPQ's default `espdl_setting` for target `esp32` with FOMO v2's 100 calibration images, and the official BGR uint8 / 255 preprocessing.
Its decoding uses the repository's `handel_preds` and `non_max_suppression` with NMS IoU 0.4 (its `test.py` value), then the shared scorer.

Ultralytics selects its checkpoint by its own box fitness, not the centroid F1 used for M0 and M4; I report this difference rather than change either tool.

## Fairness controls

- Data: the v2 Cityscapes conversion and the 2,975 / 500 image IDs in `checkpoints/cityscapes_v2/split_ids.json`; no BDD100K or other external detection data.
- Classes: Person and Vehicle only, with the v2 Cityscapes ID mapping; Obstacle remains untrained for every model.
- Input: 160 x 160 RGB with the v2 direct bilinear resize for the primary comparison; each model's native input size as a secondary result.
- Quantization: full INT8 using the same 100 training calibration images recorded in `checkpoints/cityscapes_v2/report.json`.
- Matching: v2 centroid matching, same class, one-to-one, 8 input pixels; box models are scored by box center. Box models also report mAP@0.5 as a secondary column.
- Desktop timing: one TFLite CPU thread, 20 warm-ups, 200 measured calls, as in the v2 benchmarks.
- The frozen checkpoint directory remains untouched; every new run writes a fresh dated folder under `tinyml/fomo/artifacts/`.

## Phases

| Phase | Work | Done when |
|---|---|---|
| 0 Setup | Confirm the 3 FPS gate and this rule with the team; candidate list M0–M4 confirmed 2026-10-15 | Rule committed before results |
| 1 Train and export | Train or fine-tune M1–M4 on the shared data; export INT8 | Each model has weights, config, hashes, and manifest |
| 2 Desktop metrics | Parameters, MACs, INT8 size, precision/recall/F1, size-bucket recall, desktop latency | One row per model in `results.csv` |
| 3 Static gates | Operator support, flash size, arena size | G1–G3 recorded with evidence |
| 4 On-device benchmark | `ESP32_BENCHMARK_PROTOCOL.md` for each model passing G1–G3 | Measured FPS, latency, RAM, failures; G4 recorded |
| 5 Independent footage | Gate-passing models on recorded ESP32-CAM walking footage | Independent precision/recall/F1 |
| 6 Conclusion | Decision matrix, accuracy-versus-latency and memory-versus-budget figures, Chapter 3 paragraph | Thesis-ready table and figures |

## Known blockers

- No working firmware exists for any model; `firmware/firmware.ino` is unmeasured scaffolding and its stated FPS is not evidence.
- One ESP32-CAM is damaged; phase 4 needs a working board.
- ESPDet-Pico may require an ESP32-S3; if so, it fails G1 on the ESP32-CAM, and an optional S3 run is recorded separately.
- Phase 5 needs walking footage that has not been captured yet.

## Results template

`results.csv` columns:
`model_id,model_name,input_size,params,macs,int8_bytes,precision,recall,f1,recall_lt8px,recall_ge8px,desktop_median_ms,desktop_p95_ms,g1_runtime,g2_flash_bytes,g3_arena_bytes,g4_loop_fps,notes`
