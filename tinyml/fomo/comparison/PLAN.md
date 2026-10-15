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
| M1 | ESPDet-Pico (Espressif ESP-DL) | Tiny detector built for Espressif chips | Classic-ESP32 support unverified |
| M2 | YOLO-Fastest v2 | Smallest common YOLO-family detector | Not started |
| M3 | YOLOv8n or YOLO11n at 160 x 160 | Standard lightweight YOLO | Not started |
| M4 | SSDLite-MobileNetV2, alpha 0.35 | Same backbone family as FOMO, box head instead of centroid grid | Not started |

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
