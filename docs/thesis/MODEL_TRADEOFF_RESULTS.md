# Model tradeoff: MicroSeg (custom CNN) vs Fast-SCNN

Results for the thesis comparison of the two PathClear segmentation models, evaluated 17 October 2026. All numbers below come from the reports in this folder and can be regenerated with the commands at the end.

## Models compared
| Model | Role | Checkpoint (SHA256 prefix) |
| --- | --- | --- |
| MicroSeg s800 | Custom CNN, 96x96 input, 4 classes (Walkable, Non-walkable, Boundary, Crosswalk) | `microseg_runs/pilot-neg025-s800-augoff-fixed-20261006/best.keras` (`936b19b1`) |
| Fast-SCNN, before KPV | Fast-SCNN, 512x1024 input, 10-class PathClear head, Cityscapes only | `pathclear_runs/shared_bnd_epoch1_20261006/best.keras` (`e8b28e2e`) |
| Fast-SCNN, final | Same network after KPV Crosswalk import: teacher-labelled KPV frames, backbone fine-tuned (BatchNorm frozen, lr 1e-4), best epoch 9 | `pathclear_runs/kpv_ft_final4/best.keras` (`7962e45a`) |

The "before KPV" row is included as a reference point, to show that adding the Crosswalk class did not cost the Fast-SCNN accuracy on the shared classes.

## Evaluation protocol
- **Data:** the 500 frames of the Cityscapes `gtFine` validation split, scored against the original `labelIds` annotations.
- **Scored classes:** only the classes both models were trained on: **Road** (labelId 7), **Sidewalk** (8), and **Boundary** (building 11, wall 12, fence 13). Every other pixel (sky, vegetation, people, vehicles, void) is excluded for every model.
- **Prediction mapping:** Fast-SCNN Road/Sidewalk/Boundary map directly. MicroSeg Non-walkable maps to Road and Walkable to Sidewalk. A Crosswalk prediction, or any other class (for example Fast-SCNN Person or Vegetation), on a scored pixel counts as an error.
- **Preprocessing and resolution:** each model receives the input it was trained with (bilinear resize of the full frame to its input size, divided by 255). Ground truth is resized once, by nearest neighbour, to a 512x1024 grid. MicroSeg's 96x96 output is upsampled to that grid by nearest neighbour, so both models are scored on the same pixels. Scoring MicroSeg on a 96x96 ground truth instead gives the same result to within 0.02 pp (mIoU 69.46%, pixel accuracy 90.20%, road-as-sidewalk 4.45%), so the resolution handling does not drive the gap.
- **Metrics:**
  - *mIoU*: mean over the three classes of TP / (ground truth + predicted − TP).
  - *Pixel accuracy*: correctly labelled scored pixels / all scored pixels.
  - *Road-as-sidewalk error rate*: ground-truth Road pixels predicted as Sidewalk / all ground-truth Road pixels. This is the most dangerous error for a mobility aid, because it tells the user a road surface is walkable sidewalk.
- **Uncertainty:** 95% confidence intervals from 1,000 bootstrap resamples over frames.

## Results

**Table 1. Main comparison (Cityscapes val, 500 frames).**

| Model | mIoU (%) | Pixel accuracy (%) | Road-as-sidewalk error (%) |
| --- | --- | --- | --- |
| MicroSeg (custom CNN) | 69.44 [68.46, 70.44] | 90.21 [89.67, 90.74] | 4.45 [4.03, 4.86] |
| Fast-SCNN, before KPV | 83.19 [81.99, 84.47] | 93.54 [93.02, 94.04] | 1.69 [1.41, 1.99] |
| **Fast-SCNN, final** | **85.44 [84.42, 86.47]** | **94.38 [93.98, 94.76]** | **1.84 [1.52, 2.17]** |

Brackets are 95% bootstrap confidence intervals. Figure: `model_tradeoff.png` / `model_tradeoff.pdf`.

**Table 2. Per-class IoU and where errors go.**

| Model | Road IoU | Sidewalk IoU | Boundary IoU | Sidewalk-as-road error | Road predicted as Boundary / Crosswalk / other |
| --- | --- | --- | --- | --- | --- |
| MicroSeg | 86.33 | 31.15 | 90.83 | 46.58 | 1.03 / 0.21 / 0.00 |
| Fast-SCNN, before KPV | 94.86 | 64.29 | 90.41 | 17.22 | 0.10 / 0.00 / 0.66 |
| Fast-SCNN, final | 95.70 | 69.12 | 91.51 | 12.13 | 0.09 / 0.03 / 0.52 |

All values in %. "Other" means Fast-SCNN classes outside the shared set (Person, Vehicle, Obstacle, Vegetation).

**Table 3. Deployment cost.**

| Item | MicroSeg | Fast-SCNN, final | Ratio |
| --- | --- | --- | --- |
| Parameters | 61,924 | 1,159,546 | 18.7x |
| Model file (weights, no optimizer state) | 0.99 MB | 5.14 MB | 5.2x |
| Input resolution | 96 x 96 | 512 x 1024 | 56.9x pixels |
| CPU latency, median per frame (3 runs) | 168 ms (range 147-324) | 478 ms (range 443-732) | 2.3-3.3x |

Latency was measured on a desktop CPU under WSL with eager Keras calls (3 warm-up + 20 timed batch-1 calls per run). Repeated runs varied by up to 2x on this machine, so only the order of magnitude and the ratio should be quoted. Phone latency was not measured.

**Table 4. Supplementary: Crosswalk on real KPV frames.**

| Model | Crosswalk IoU (%) | Precision | Recall |
| --- | --- | --- | --- |
| MicroSeg | 76.12 | 0.813 | 0.923 |
| Fast-SCNN, before KPV | 0.00 (no Crosswalk class trained) | – | – |
| Fast-SCNN, final | 68.90 | 0.797 | 0.836 |

58 real KPV validation frames, full-frame, scored on the same 512x1024 grid.

## Qualitative figures
Both figures use the shared label space from Table 1. MicroSeg's Non-walkable class (which also covers sky) is drawn in the Road color, and its Walkable class in the Sidewalk color. Grey "Other" marks Fast-SCNN classes that MicroSeg does not have (people, vehicles, vegetation, obstacles).

- **`qualitative_cityscapes.png` / `.pdf`:** four Cityscapes val frames spaced evenly through the split (not hand-picked). Columns: input, ground truth, each model's prediction, and each model's error map, where red marks road predicted as sidewalk and blue marks any other error. MicroSeg's predictions are visibly blocky (96x96 output) and its sidewalk regions are too wide or missing. Per frame the ranking varies: on `munster_000173_000019` Fast-SCNN has the higher road-as-sidewalk rate (5.7% vs 0.0%), which is why Table 1 reports the full 500-frame score with intervals.
- **`qualitative_photos.png` / `.pdf`:** the three real pedestrian photos (unlabelled, so not scored). Both models find the painted crossings in the close-up and at the intersection. Fast-SCNN keeps the cyclists and the walker's legs out of Crosswalk; MicroSeg's crossing region covers the cyclists. In the residential photo, both models label the sidewalk on the right as road, which matches the high sidewalk-as-road rates in Table 2 and shows that sidewalk detection on pedestrian-view images is still a weak point for both.

## Interpretation
- **Accuracy:** Fast-SCNN is clearly ahead on all three metrics: +16.0 pp mIoU, +4.2 pp pixel accuracy, and 2.4x fewer road-as-sidewalk errors (1.84% vs 4.45%). None of the confidence intervals overlap. Most of the mIoU gap is Sidewalk. MicroSeg labels 46.6% of true sidewalk as road and reaches only 31.2% sidewalk IoU, against 69.1% for Fast-SCNN.
- **Safety-relevant error:** in absolute terms, MicroSeg calls about 1 in 22 road pixels sidewalk, Fast-SCNN about 1 in 54.
- **Effect of the Crosswalk import:** the final Fast-SCNN scores slightly higher than the pre-KPV one on mIoU (+2.3 pp) and pixel accuracy (+0.8 pp); the intervals overlap at their edges, so this is evidence of no loss rather than a firm gain. Road-as-sidewalk is 1.84% vs 1.69% with overlapping intervals, so there is no detectable change on that metric. Adding the Crosswalk class therefore did not cost accuracy on the shared classes.
- **Cost:** MicroSeg is about 19x smaller in parameters, 5x smaller on disk, and roughly 2-3x faster per frame on CPU. It also only sees a 96x96 view of the scene.
- **Crosswalk:** MicroSeg retains a higher Crosswalk IoU (76.1% vs 68.9%). Both numbers are optimistic (see Limitations).
- **Tradeoff in one sentence:** Fast-SCNN buys a 16-point mIoU gain and less than half the road-as-sidewalk error rate for roughly 2-3x the CPU latency and 5x the model size; MicroSeg is the choice only when that latency or size budget cannot be met.

## Limitations
1. **Validation data was also used for checkpoint selection.** Cityscapes test labels are not public, so the val split is the standard reporting set. However, both training runs chose their best epoch partly on Cityscapes val mIoU, which can make all rows slightly optimistic. The KPV Crosswalk numbers are more affected: both models chose their best epoch on these same 58 frames.
2. **Cityscapes only.** Cityscapes is driver's-eye footage from German cities. The target use is a pedestrian's handheld view. Results on the three real pedestrian photos were qualitatively consistent with Table 1, but they are unlabelled and not scored here.
3. **Class mapping.** The shared space is coarse. MicroSeg's Non-walkable class was trained to also cover sky, which is excluded here, and Fast-SCNN can lose scored pixels to classes MicroSeg does not have ("Other" in Table 2, at most 0.66% of road).
4. **Latency is not on-device.** The latency figures are desktop CPU and noisy; deployment claims need a TFLite measurement on the target phone.

## Reproduction
```bash
# Accuracy (Tables 1, 2, 4): about 15 minutes on CPU
bash run_wsl.sh scripts/evaluate_model_tradeoff.py \
  --micro microseg_s800=/home/nacsi/microseg_runs/pilot-neg025-s800-augoff-fixed-20261006/best.keras \
  --fast fast_scnn_pre_kpv=/home/nacsi/pathclear_runs/shared_bnd_epoch1_20261006/best.keras \
  --fast fast_scnn_kpv_ft_ep9=/home/nacsi/pathclear_runs/kpv_ft_final4/best.keras \
  --kpv-split-manifest /home/nacsi/microseg_runs/crosswalk_split.json \
  --output docs/thesis/tradeoff_eval_20261017.json

# Figure
bash run_wsl.sh scripts/plot_model_tradeoff.py docs/thesis/tradeoff_eval_20261017.json \
  --labels "microseg_s800=MicroSeg (custom CNN)" "fast_scnn_pre_kpv=Fast-SCNN, before KPV" \
           "fast_scnn_kpv_ft_ep9=Fast-SCNN, final (KPV fine-tuned)" \
  --output docs/thesis/model_tradeoff

# Qualitative figures
bash run_wsl.sh scripts/plot_model_qualitative.py \
  --micro microseg_s800=/home/nacsi/microseg_runs/pilot-neg025-s800-augoff-fixed-20261006/best.keras \
  --fast fast_scnn_kpv_ft_ep9=/home/nacsi/pathclear_runs/kpv_ft_final4/best.keras \
  --cityscapes 4 --images visuals/kpv_full_10ep/unlabeled/originals/*.png \
  --output-dir docs/thesis

# Cost (Table 3)
bash run_wsl.sh scripts/benchmark_doc_models.py \
  --fast-ckpt /home/nacsi/pathclear_runs/kpv_ft_final4/inference.keras \
  --output docs/thesis/doc_benchmark_ft_ep9.json
```

Files in this folder: `tradeoff_eval_20261017.json` (full accuracy report, including confusion matrices), `doc_benchmark_kpv.json`, `doc_benchmark_ft_ep9.json`, `doc_benchmark_ft_ep9_run2.json` (latency runs), `model_tradeoff.png` / `.pdf` (figure). The epoch-9 checkpoint is backed up at `checkpoints/kpv_ft_ep9/best.keras`.
