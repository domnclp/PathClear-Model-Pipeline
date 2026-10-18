# PathClear KPV Crosswalk handoff: Fast-SCNN 10-class probe results
Status handoff | 13 October 2026 | Prepared for AI continuity and project-group onboarding

> **Update, 18 October 2026: the results below are superseded.** The head-only probe described here has been replaced:
> - KPV frames are now labelled outside the crossing with the pre-KPV checkpoint's confident predictions (`scripts/make_kpv_teacher_labels.py`, `--teacher-manifest` in the trainer), replacing the horizon split that erased Person/Vehicle/Vegetation on pedestrian-view photos.
> - The backbone is fine-tuned with BatchNorm frozen (`--unfreeze all --learning-rate 1e-4`). The final checkpoint is fine-tune epoch 9 (`/home/nacsi/pathclear_runs/kpv_ft_final4/best.keras`, backup `checkpoints/kpv_ft_ep9/`): KPV Crosswalk IoU 68.9%, Cityscapes shared-class mIoU 85.4%.
> - MicroSeg is kept only as the thesis comparison baseline. Current numbers, figures and method: `docs/thesis/MODEL_TRADEOFF_RESULTS.md`.
> - The mid-run WSL crashes were WSL updating itself (`ERROR_INSTALL_ALREADY_RUNNING`), not the trainer.
> - Next: a held-out, labelled test set from the deployment camera (CVAT), then on-device TFLite latency.

## Purpose of this document
This handoff records the import of KPV (Kaggle FPVCrosswalk2025) supervision into the Crosswalk class (ID 8) of the ten-class Fast-SCNN PathClear checkpoint. It covers the measured results, the cost to the existing Cityscapes classes, the visual findings, and the operational lessons needed to continue without repeating failed approaches. Read it together with `PATHCLEAR_HANDOFF_2026-09-22.md` (baseline history) and `PATHCLEAR_10CLASS_PREPARATION.md` (class design).

MicroSeg has been dropped. Fast-SCNN is the only model carried forward, so this document makes no cross-model comparisons.

## Current position
- The mobile pivot replaced the abandoned ESP32-CAM path. Inference moves to a mobile device running Fast-SCNN.
- The shared Boundary map (Cityscapes building/wall/fence -> Boundary, ID 4) is merged in `pathclear_taxonomy.py`, `configs/pathclear_classes.yaml`, and both test suites; 149 tests pass.
- KPV Crosswalk was imported as a 10-epoch head-only probe on top of the shared-Boundary adaptation checkpoint.
- Result: 39.11% full-frame Crosswalk IoU on the 58 real KPV validation frames (precision 0.439, recall 0.782). The seven Cityscapes classes went from 68.78% to 67.13% mIoU (-1.65 pp).

## The ten classes (IDs 0-9)
| ID | Class | Supervision source |
| --- | --- | --- |
| 0 | Road | Cityscapes |
| 1 | Sidewalk | Cityscapes |
| 2 | Person | Cityscapes |
| 3 | Vehicle | Cityscapes |
| 4 | Boundary | shared map (building+wall+fence -> Boundary) |
| 5 | Obstacle | Cityscapes |
| 6 | Vegetation | Cityscapes |
| 7 | Stairs | local-only (unsupervised in current runs) |
| 8 | Crosswalk | KPV masks (this handoff) |
| 9 | Uneven Pavement | local-only (unsupervised in current runs) |

IGNORE_LABEL is 255. IDs 0-6 are the Cityscapes-supported set. IDs 7-9 are local-only slots activated by `enable_local_classes` when a local dataset provides them. In the KPV runs only slot 8 receives training signal. Slots 7 and 9 remain unsupervised, and the trainer removes the class gate.

## KPV loader contract (`kpv_loader.py`)
- Geometry matches the Cityscapes loader: 512x1024, bilinear RGB resize, nearest-neighbor mask resize. KPV masks ship as JPEG, so `decode_image` (not `decode_png`) is required.
- Marked mask pixels (>127) become Crosswalk (ID 8) at full weight. Unmarked pixels become weak negatives split at a horizon line: rows above 0.45 x 512 take Boundary (ID 4), rows below take Road (ID 0).
- `negative_weight` is 0.25, so a false Crosswalk positive costs one quarter as much as a miss. This asymmetry is deliberate (recall-first for a mobility aid) and is the main lever for the over-prediction visible in panels.
- `pairs_from_split_manifest` reads `crosswalk_split.json` (schema version 1, root must match `KAGGLE_CROSSWALK_ROOT`, relative paths only). The split has 242 real + 3000 synthetic training pairs and 58 real validation frames.
- The canonical manifest currently lives at `/home/nacsi/microseg_runs/crosswalk_split.json`, a leftover from the MicroSeg work. Move it somewhere under `pathclear_runs` (and update `DEFAULT_SPLIT_MANIFEST` in the trainer) before the MicroSeg run directories are cleaned up.
- Tests: `tests/test_kpv_loader.py`, 9/9 pass.

## Trainer (`scripts/train_pathclear_kpv.py`)
- Starts from the shared-Boundary adaptation checkpoint (`shared_bnd_epoch1_20261006/best.keras`), removes the gate via `enable_local_classes`, freezes all feature layers, and trains only `classifier_logits` (a 1x1 convolution, i.e. a linear decision per frozen feature vector).
- Mixed sampler: the Cityscapes and KPV streams each repeat, so the 50/50 mix cannot drift with `steps_per_epoch`.
- Checkpoints on `val_combined_score`: the harmonic mean of Cityscapes val mIoU and KPV full-frame val IoU, minus a flood penalty when predicted Crosswalk coverage exceeds 2x true coverage. Flood epochs are common, and selecting the best checkpoint on this score keeps them out of the saved model.
- Flags: `--mode {smoke,full}`, `--epochs`, `--steps-per-epoch` (800), `--allow-cpu` (GPU stays the default gate), `--start-checkpoint`, `--output-dir` (refuses nonempty). The `KpvFullFrameMetrics` and `HarmonicFloodScore` callbacks log per-epoch `KPV full-frame` and `Combined score` lines to `run.log`.

## Verified 10-epoch run (`/home/nacsi/pathclear_runs/kpv_full_10ep`)
The run made 8000 optimizer updates in 3114 s of CPU wall clock and was deterministic (epoch 1 matched the earlier 1-epoch run exactly). The best checkpoint is epoch 8 (`best.keras`, SHA256 8218d310b9e5f547f4ceb64d7c208c9876decb472ab948558b8451faff1ddc8c). Epochs here are 1-indexed; `history.csv` and `visualization_report.json` are 0-indexed, so epoch 8 is their row `epoch=7`.

| Metric | Epoch 1 | Best (epoch 8) |
| --- | --- | --- |
| KPV full-frame Crosswalk IoU (58 frames) | 30.98% | 39.11% |
| KPV precision / recall | 0.375 / 0.642 | 0.439 / 0.782 |
| KPV predicted / true Crosswalk coverage | 35.3% / 20.6% | 36.6% / 20.6% |
| Flood excess | 0.0 | 0.0 |
| Combined score | 0.4017 | 0.4695 |
| Cityscapes val mIoU as logged (8 classes, see below) | 57.12% | 58.73% |
| Cityscapes val pixel accuracy | 88.41% | 89.59% |

Epochs 7 and 10 flooded (excess 0.087 and 0.158), with recall surging toward 0.88 while precision held near 0.40. The final-epoch state is worse than the best-epoch state, so always evaluate `best.keras`, never `last.keras`.

The 39.11% is a validation number, not a held-out test result: the same 58 frames drove best-epoch selection. Score the checkpoint on frames outside the split before publishing it as a test figure.

## Cost to the Cityscapes classes
The logged `val_mean_iou` of 58.73% is misleading. The Keras MeanIoU counts every class with a nonzero union, and the ungated KPV head predicts Crosswalk on Cityscapes, where there is no Crosswalk ground truth. Crosswalk therefore enters the mean as an eighth class at 0 IoU. A per-class rescore of `best.keras` on all 500 Cityscapes val frames (`/home/nacsi/kpv_best_cityscapes_per_class.json`) reproduces 58.73% exactly as the 8-class mean. The like-for-like figure over IDs 0-6 is 67.13%.

| Class | Start (shared_bnd_epoch1) | After KPV (best epoch 8) | Change |
| --- | --- | --- | --- |
| Road | 94.29% | 92.05% | -2.24 pp |
| Sidewalk | 62.57% | 62.09% | -0.48 pp |
| Person | 47.81% | 47.66% | -0.15 pp |
| Vehicle | 78.44% | 76.72% | -1.72 pp |
| Boundary | 81.77% | 78.79% | -2.98 pp |
| Obstacle | 30.42% | 27.16% | -3.26 pp |
| Vegetation | 86.18% | 85.41% | -0.77 pp |
| **mIoU, IDs 0-6** | **68.78%** | **67.13%** | **-1.65 pp** |

Importing KPV cost 1.65 pp on the existing classes even though the backbone is frozen, because retraining the shared 1x1 head moves every class's decision boundary. Road loses the most in absolute pixels, mainly to Crosswalk false positives. On Cityscapes val the head labels 1,899,085 pixels Crosswalk (0.86% of scored pixels): 1,703,571 true Road, 105,439 Boundary, 69,493 Sidewalk, and about 21k other. Some of the Road hits may be real painted crossings that Cityscapes labels Road. The rest is the same spill seen on KPV frames.

## Deployment cost (`scripts/benchmark_doc_models.py`)
Measured on CPU with the KPV `best.keras` (`/home/nacsi/doc_benchmark_kpv.json`, 3 warm-up + 20 timed batch-1 calls):

| Item | Fast-SCNN 10-class + KPV |
| --- | --- |
| Parameters | 1,159,546 |
| File size | 5.03 MB (5,153,842 B) |
| Input | 512 x 1024 |
| CPU latency median | 443.4 ms (2.26 fps); min 415.9, max 471.3 |

This latency was measured on a desktop CPU in WSL running eager `model(x, training=False)`, not on a phone. The mobile pivot needs an on-device TFLite measurement (`scripts/export_pathclear_tflite.py`) before latency claims are published. The benchmark script still loads a MicroSeg checkpoint and defaults `--fast-ckpt` to the pre-KPV checkpoint. Reproduce the table with `--fast-ckpt /home/nacsi/pathclear_runs/kpv_full_10ep/best.keras`, or trim the MicroSeg half out of the script.

## Visualization (`scripts/visualize_kpv_run.py`)
The script renders `training_curves.png`, with four panels (Cityscapes val mIoU, KPV full-frame val IoU, combined score, flood excess) and the best epoch marked. It also renders per-frame prediction panels for the first N validation frames: input image, horizon-split ground truth, prediction, and a Crosswalk overlay (green hit / red missed / blue spurious). The checkpoint is hashed before and after rendering to confirm the script is read-only. Copies for the 10-epoch run live in `visuals/kpv_full_10ep/`, with `visualization_report.json` (best-epoch metrics plus per-frame precision/recall). Note that the curves' Cityscapes mIoU panel plots the logged 8-class value.

### Why the prediction looks like an archipelago (diagnosed, three stacked causes)
1. **Frozen stripe-blind features.** The backbone was trained on Cityscapes, where zebra stripes were just Road, so in the frozen feature space stripes and asphalt are nearly identical. The only trainable part is a 1x1 linear classifier over those features. The Crosswalk-vs-Road boundary is therefore thin and unstable, and pixels at the margin flip almost randomly into fragmented islands. This is the expected ceiling of a head-only probe, not a bug.
2. **Asymmetric loss makes over-prediction cheap.** With negative_weight 0.25, calling a road pixel Crosswalk costs one quarter of missing a real one. That bought the recall jump (0.64 -> 0.78) and shows up as blue spill onto surrounding road. Setting the negative weight back to 1.0 shrinks the islands but collapses recall. It is a dial, and recall-first was chosen deliberately for a mobility aid.
3. **1/8-resolution logits upsampled bilinearly.** Fast-SCNN classifies on a 64x128 grid, then upsamples x8, so each decision covers an 8x8-pixel cell. Edge cells straddle stripe/road mixtures and flip cell-by-cell into blobby islands.

The KPV ground truth is a filled polygon over the whole crossing area, not per-stripe labels. By the label contract, one solid predicted region over the crossing is correct; the real errors are spill outside the polygon and gaps inside it.

The probe answered its question. Frozen epoch-19 features CAN represent stripes (39.11% full-frame IoU, clearly above chance), but the archipelago is the visual signature of where that representation runs out. Unfreezing the backbone (low-LR fine-tuning) is the only real fix for cause 1 and is the planned next experiment.

## Operational lessons (WSL environment)
- Background `nohup ... &` jobs across PowerShell->WSL invocations are unreliable. One launch produced no log; another was killed when the whole WSL distro was torn down mid-epoch (`dmesg` showed a kernel reboot and ext4 unmount, no OOM). Run training in the FOREGROUND via `bash run_wsl.sh ...`. The tooling leaves it running, and progress can be polled through `progress.json`, `history.csv`, and `grep "Combined score" run.log`.
- PowerShell quoting frequently mangles nested WSL commands. The workaround is to write temp probe scripts and strip CRLF. `wsl -e grep/cat` directly is more reliable than `wsl -e bash -lc '...'` for one-liners. From Git Bash, set `MSYS_NO_PATHCONV=1` or Linux paths get rewritten to Windows paths.
- `--allow-cpu` exists on the trainer; the GPU gate remains the default.

## File index
- `kpv_loader.py`, `tests/test_kpv_loader.py` - KPV supervision contract (9/9 pass).
- `scripts/train_pathclear_kpv.py` - mixed-sampler head-only trainer with flood-penalized checkpointing.
- `scripts/visualize_kpv_run.py` - curves + prediction panels (read-only with respect to the checkpoint).
- `scripts/benchmark_doc_models.py` - latency/params/size report; accepts `--fast-ckpt/--micro-ckpt/--output`.
- `scripts/generate_label_train_ids.py` + `tests/test_generate_label_train_ids.py` - missing Cityscapes trainIds masks.
- Reports: `/home/nacsi/kpv_best_cityscapes_per_class.json` (per-class Cityscapes rescore of KPV best), `/home/nacsi/doc_benchmark_kpv.json` (cost), `shared_bnd_epoch1_20261006/validation_per_class.json` (start-point per-class).
- Runs: `/home/nacsi/pathclear_runs/kpv_full_10ep` (best), `kpv_full_ep1` (1-epoch determinism check), `shared_bnd_epoch1_20261006` (shared-Boundary adaptation start point). Full test suite: 149 pass.

## What remains next
1. Unfreeze the backbone (low-LR fine-tune) from `kpv_full_10ep/best.keras` to attack the archipelago at its cause. Watch flood excess, because the flood penalty may need retuning at feature-level learning rates. Keep BatchNorm layers in inference mode, and track the IoUs of IDs 0-6 against the 68.78% start rather than the logged 8-class mean.
2. Fix the logged Cityscapes metric so checkpoint selection uses the IDs 0-6 mean instead of the 8-class mean that includes Crosswalk at 0. With the current metric, a single Crosswalk pixel predicted anywhere on Cityscapes val drops the Cityscapes term by about one eighth, however small the spill is.
3. Score Crosswalk on frames outside the 58-frame selection split, and measure latency on-device, before publishing either number.
4. Consider 50-100 annotated field frames (same camera as deployment) to close the precision gap.
5. Decide a canonical checkpoint backup location; weights have vanished once already. Relocate `crosswalk_split.json` out of `microseg_runs` at the same time.
6. Slots 7 (Stairs) and 9 (Uneven Pavement) remain unsupervised; a dataset for them is still missing.
