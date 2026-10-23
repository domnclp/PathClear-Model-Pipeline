# My Cityscapes FOMO baseline

> **Note (2026-10-23):** MicroSeg was later dropped.
> Standalone mode is FOMO + ToF only.
> References to MicroSeg below are historical.

I am training a separate standalone Person/Vehicle baseline on `tinyml-fomo`.
All experiment code and outputs stay under `tinyml/fomo/`.
I keep the Fast-SCNN pipeline, original Cityscapes files, and earlier model checkpoints unchanged.
I use [my finalized standalone architecture](STANDALONE_ARCHITECTURE.md) for current system decisions.
The original baseline and sanity reports describe earlier experiments; their classifier-handoff language is historical.
I record the current decoder diagnosis and measured changes in [DIAGNOSIS_REPORT.md](DIAGNOSIS_REPORT.md).
I record the separate 20 x 20 versus 40 x 40 model comparison in [GRID_RESOLUTION_REPORT.md](GRID_RESOLUTION_REPORT.md).
I freeze the current 20 x 20 model as `baseline_v1` and record its threshold, error, loss-balance, and controlled v2 investigation in [BASELINE_V1_V2_REPORT.md](BASELINE_V1_V2_REPORT.md).
I keep the planned factors in [V2_EXPERIMENT_PLAN.md](V2_EXPERIMENT_PLAN.md) and physical measurement instructions in [ESP32_BENCHMARK_PROTOCOL.md](ESP32_BENCHMARK_PROTOCOL.md).
The v2 inference candidate explicitly uses `--decoder cells --threshold 0.80`; it reuses the saved baseline weights and does not change the default NMS/0.60 reference policy.
After the controlled loss-weight experiments and equal-rule threshold comparison, my strongest balanced candidate uses weight 10 with `--decoder cells --threshold 0.70` and the separate `v2-weight10-export-20261004/model.int8.tflite` export.
I distinguish this trained candidate from the earlier inference-only candidate and preserve `baseline_v1` as the frozen reference.
I preserve the selected weight-10 model, exact inference settings, hashes, and reproduction instructions in [my immutable Cityscapes v2 checkpoint](checkpoints/cityscapes_v2/README.md).
The local annotated tag `fomo-cityscapes-v2` identifies its frozen source and model files.

## What I am doing

```text
Cityscapes RGB images + individual-object masks
  -> boxes around individual people and vehicles
  -> 160 x 160 RGB inputs and 20 x 20 center targets
  -> local FOMO-inspired training
  -> Cityscapes validation metrics and prediction previews
  -> float32 and INT8 TFLite export
  -> later measurement on the physical ESP32-CAM
```

Fast-SCNN answers what class each pixel belongs to.
FOMO roughly answers where relevant objects are located.
MicroSeg is my lightweight custom CNN for scene/navigation semantic segmentation.
FOMO and MicroSeg independently consume camera frames and send their separate outputs to PathClear decision logic.
I do not feed one model into the other and do not include a third CNN or secondary classifier in the baseline.
Logical independence allows alternating execution or different frame rates on constrained hardware.
I have not implemented MicroSeg or the combined firmware here.

## My implementation choice

This is a **local FOMO-inspired implementation**, using the published FOMO architecture.
It is not an official Edge Impulse training run, and I do not claim identical loss, decoding, or evaluation behavior.
I use the early layers of MobileNetV2 to extract visual features, ending at `block_6_expand_relu`.
That describes the default 20 x 20 baseline.
The explicit `--grid-size 40` experiment ends at `block_3_expand_relu`, retaining the same 160 x 160 camera input.
Its shallower trunk produces a 40 x 40 grid and has 6,339 parameters instead of 20,787.
This comparison changes feature depth as well as spatial resolution.
A 32-channel convolutional head then predicts background, Person, or Vehicle at each grid location.
The model has 20,787 parameters, including nontrainable parameters.
I initialize the feature extractor from public Keras ImageNet weights so the first baseline starts with learned visual features.
The grid head and all retained backbone layers are trained on the converted Cityscapes data.

The public input remains RGB `float32` in `[0,1]`, shaped `160 x 160 x 3`.
A rescaling layer inside the model converts this to MobileNetV2's expected `[-1,1]` range.
The model output is `20 x 20 x 3` probabilities, with internal channels `background=0`, `person=1`, and `vehicle=2`.
Public detection IDs remain `person=0` and `vehicle=1`.
There is no Obstacle output in this phase.
The earlier three-class synthetic checkpoints remain historical artifacts and deliberately have a different contract.

Edge Impulse remains the official FOMO workflow and a useful reference comparison.
Using it would introduce its training environment and data-upload decisions; it is not required to run this local baseline.
Local code gives me explicit control over conversion, data splits, and evaluation, while requiring me to document differences from the official implementation.
Both approaches require hardware measurements before claiming ESP32 suitability.

Sources: [FOMO design and options](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo), [Keras MobileNet documentation](https://keras.io/api/applications/mobilenet/), [Cityscapes instance encoding](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py), and [official class IDs](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py).

## Why I convert the annotations

The Fast-SCNN loader uses `*_gtFine_labelTrainIds.png`, which labels pixel classes.
This FOMO converter reads `*_gtFine_instanceIds.png`, which distinguishes individual objects.
For a pixel value of at least 1000, `value // 1000` is the original semantic class ID and `value % 1000` is the individual instance number.
For example, `26000` and `26001` are two different cars; instance number zero is valid.
A plain `26` denotes an unresolved car group and does not identify individual cars.
I never invent one detection box for such a group.

| Original Cityscapes label IDs | Public FOMO class |
|---|---|
| Person 24, rider 25 | Person 0 |
| Car 26, truck 27, bus 28, train 31, motorcycle 32, bicycle 33 | Vehicle 1 |
| Other classes | Excluded from object annotations |

I find the smallest rectangle enclosing the visible pixels of each retained instance.
The saved box is `[x, y, width, height]` in original-image pixels, including both extreme pixel coordinates.
Its center becomes the grid target.
An occluded object's box follows its visible mask, rather than guessing hidden extent.
Unresolved groups and official void regions supply ignored cells, with zero training loss.
Ordinary known surfaces such as road, sidewalk, and buildings remain background.
A known individual center takes precedence over a void/group cell.

Two same-class centers in one cell supply one positive target, because the class is still known.
Two different classes in one cell are ambiguous, so that cell receives no training loss.
I retain **all individual annotations** for evaluation, including centers sharing cells.
The inspection report measures collisions rather than silently dropping difficult objects.
These collisions limit how well this small grid can count crowded street scenes.
The original connected-component decoder also merged nearby activations into one result.
I retain that decoder as `--decoder connected` for reproducing the historical baseline.
My current experimental default is `--decoder cell_nms`, which keeps confident cell centers and suppresses immediate same-class neighbors rather than merging an entire connected chain.
`--decoder cells` retains every positive cell as a diagnostic with higher recall and more duplicate/false detections.

## Prepared data and review

The converter preserves all 2,975 official training images and 500 official validation images.
The 1,525 test images have withheld object labels and are not converted into training examples.
I preserve source-image references instead of duplicating the full-resolution dataset.
Small RGB input arrays occupy about 267 MB for all 3,475 images.
The generated manifest records original image/mask paths, source size/timestamps, a pixel hash, instance IDs, boxes, ignored cells, groups, and cache locations.
Windows paths in the manifest are translated when read from WSL.

Inspection checks missing/broken files, invalid boxes and IDs, duplicate content across splits, group leakage, cached inputs, counts, and grid collisions.
`--verify-sources` also rereads every original image, verifies its pixel hash, and compares the resized pixels with the prepared cache.
A group represents a source city and recording sequence.
I retain the official train/validation split and do not fabricate a separate labeled test set.

The inspector saves annotation previews at visible resolution.
Green boxes are people, cyan boxes are vehicles, and red dots are their centers.
After opening and checking previews, `--review-note` creates a review receipt tied to the manifest's SHA-256 digest.
Training refuses an unreviewed or changed manifest.
This is a full structural audit plus sample visual review, not a claim that every source annotation was manually relabeled.
The converter marks the manifest unreviewed until that review step is recorded.

## Training choices

The important settings live in `config.py`.
Dataset paths are supplied by `--source` and `--manifest`; model/output paths are arguments.
I use batch size 32, learning rate 0.001, a maximum of 20 epochs, and seed 42.
Object-center cells receive 20 times the background weight because urban images contain many centers.
Weights are normalized per image, and ignored cells always receive zero weight.
The original converter conservatively ignores a whole cell if any source pixel in that cell is unknown.
The diagnostic center-mask variant adds `target_ignore_cells` based on the source label at each output-cell center.
`load_arrays()` uses that field for training targets when present, while evaluation always uses the original `ignore_cells` field.
Known positive centers still override unknown regions, and mixed-class conflicts remain ignored.
I keep source boxes, RGB caches, split assignments, and evaluation masks unchanged in the variant.
I use horizontal flips during baseline training, applying the same flip to images, targets, and ignored cells.

The sanity run uses up to 12 training images, up to 12 validation images, batch size 8, and 30 updates.
It checks finite loss, changed trainable weights, checkpoint save/reload, and actual decoded detections at confidence 0.60.
A failed sanity gate ends with an error instead of proceeding to baseline training.
The full baseline is a separate explicit command.

I select the best checkpoint by **validation centroid F1 at threshold 0.60**, not by the final epoch.
New runs record the decoding policy used for checkpoint selection.
The frozen original checkpoint was selected using the connected decoder; changing its postprocessing does not retrain or reselect that checkpoint.
I halve the learning rate after two epochs without F1 improvement, with a minimum of 0.00001.
I stop after five epochs without improvement.
Each run uses a fresh output directory and records settings, software versions, source hashes, initialization hashes, losses, F1, elapsed time, best epoch, and metrics.
The previous run's checkpoints cannot be overwritten by reusing its directory.

## Commands

I run these PowerShell commands from the repository root.
The existing Windows environment supports conversion, tests, inspection, evaluation, and export.
The existing WSL launcher provides GPU training without changing the Fast-SCNN environment or launcher.
Public pretrained weights and Keras cache files stay in the ignored `tinyml/fomo/artifacts/keras-cache/` directory.

```powershell
# Verify the branch and existing work before writing.
git -c safe.directory=C:/Users/Asus/OneDrive/Desktop/Code/PathClear branch --show-current
git -c safe.directory=C:/Users/Asus/OneDrive/Desktop/Code/PathClear status

# Conversion creates a new timestamped artifact directory and prints its path.
& .\.venv\Scripts\python.exe tinyml\fomo\prepare_cityscapes.py --source C:\datasets\cityscapes

# Current verified dataset. A new conversion prints its own manifest path.
$manifestPath = 'tinyml/fomo/artifacts/cityscapes-data-20261004/manifest.json'
& .\.venv\Scripts\python.exe tinyml\fomo\inspect_dataset.py --manifest $manifestPath --verify-sources --samples 16

# For a NEW conversion, only after opening its previews and checking the labels:
# & .\.venv\Scripts\python.exe tinyml\fomo\inspect_dataset.py --manifest $manifestPath --samples 0 --review-note 'Describe the samples checked and findings'

& .\.venv\Scripts\python.exe -m unittest discover -s tinyml\fomo -p 'test_*.py'
wsl -d Ubuntu -- bash ./run_wsl.sh -u tinyml/fomo/train.py --manifest $manifestPath --sanity
wsl -d Ubuntu -- bash ./run_wsl.sh -u tinyml/fomo/train.py --manifest $manifestPath --epochs 20 --batch-size 32
```

Training prints the new run directory, containing `best.keras`, `best.json`, `history.json`, and prediction previews.
The supplied `--output` directory must be new and inside `tinyml/fomo/artifacts/`.
Once training finishes, these commands evaluate and export its checkpoint:

```powershell
$checkpointPath = Read-Host 'Path to the completed run best.keras'
& .\.venv\Scripts\python.exe tinyml\fomo\evaluate.py --model $checkpointPath --manifest $manifestPath --split validation
$imagePath = Read-Host 'Path to one RGB camera or Cityscapes image'
& .\.venv\Scripts\python.exe tinyml\fomo\inference.py --model $checkpointPath --image $imagePath
& .\.venv\Scripts\python.exe tinyml\fomo\export.py --model $checkpointPath --manifest $manifestPath
$int8Path = Read-Host 'Path to the exported model.int8.tflite'
& .\.venv\Scripts\python.exe tinyml\fomo\evaluate.py --model $int8Path --manifest $manifestPath --split validation
& .\.venv\Scripts\python.exe tinyml\fomo\inference.py --model $int8Path --image $imagePath
```

I keep `best.json` beside `best.keras` and the export `report.json` beside `model.int8.tflite`.
They describe the model contract, which loading checks before interpreting outputs.
`model.float32.tflite` is also exported for a size comparison; the current command-line TFLite adapter expects the INT8 model.
Generated datasets, caches, and experimental models stay ignored by Git.
I make a narrow exception for the canonical `checkpoints/cityscapes_v2/` files and explicitly preserve selected experiment reports and previews in Git.

## Evaluation meaning

I match predicted centers to labeled centers of the same class, highest-confidence prediction first, with at most one match per object.
The initial distance tolerance is eight pixels in the resized `160 x 160` image.
I count unmatched predictions in ordinary labeled regions as false positives.
Unmatched predictions in source group/void cells are reported as ignored because those regions do not provide reliable individual labels.
Known individual objects remain in the recall denominator even if their centers collide or lie in ignored regions.
This is my documented local metric, not the official Cityscapes benchmark or an exact reproduction of Edge Impulse scoring.

The report includes per-class and overall precision, recall, F1, confidence summaries, matched-center errors, and desktop latency.
I report all detections for model evaluation and also retain `legacy_top3_micro` to quantify the removed three-detection cap.
The baseline module now returns all detections; the top-three metric is a historical comparison, not a required output limit.
Gray boxes in prediction previews are ground truth; colored dots and labels are predicted centers and confidence.
Center error covers correctly matched objects only, so I read it alongside recall.
I do not use bounding-box overlap as the primary metric because the model does not predict object extents.
I also avoid grid accuracy, which can look high simply because most cells are background.

The same validation split selects checkpoints and supplies the reported baseline metrics.
These are development-validation results, not an untouched final test or evidence of Philippine-scene performance.
Threshold diagnostics at 0.30, 0.60, and 0.80 show confidence tradeoffs; the primary selection threshold remains 0.60.

## Independent FOMO output

`predict()` returns detections and the model-call time.
Each detection contains `centroid_x`, `centroid_y`, `coarse_class_id`, `class_name`, and `confidence`.
Coordinates are continuous **original camera-frame pixel coordinates**, measured from the upper-left image edge.
For a cell center `(column+0.5, row+0.5)`, I map back with `x = (column+0.5)/20*frame_width` and `y = (row+0.5)/20*frame_height`.
This mapping assumes the current full-frame direct resize, with no letterboxing or cropping.
The default decoder sorts winning foreground cells by confidence and suppresses same-class candidates at Chebyshev distance at most eight detector-input pixels from a retained candidate.
That is one output cell on the 20 x 20 grid or two on the 40 x 40 grid.
I keep this physical suppression distance fixed across the grid comparison.
It does not suppress a different class merely because it is adjacent.
Equal-confidence ties use class, row, and column order for reproducibility.
This is a local postprocessing experiment, not an assertion of equivalence to Edge Impulse's decoder.
It still cannot separate two objects represented by the same cell, and it can retain several peaks from one large activation region.
The initial threshold remains 0.60 and `--max-detections 0` means uncapped output.
I expose an explicit nonzero cap only for measured buffer/decision experiments.

The CLI returns a FOMO module identifier, detector taxonomy, decoding policy, image dimensions, detections, and desktop model-call timing.
It creates no classifier crops and never invokes MicroSeg.
The legacy `crop_around_centroid()` helper remains available only for inspecting historical experiments; neither `predict()` nor the CLI calls it.
The historical `crop_size` tensor-contract metadata remains unchanged so preserved two-class checkpoints can still load.
That field is not a current pipeline requirement.
I resolve camera orientation into stored pixels before annotation; the loader does not silently rotate EXIF-oriented images.
The future firmware must attach capture timestamps and frame identifiers before combining model outputs in decision logic.

## Export and hardware limits

Export produces float32 and full INT8 TFLite files, then invokes the INT8 model on the PC.
Calibration uses 100 reproducibly sampled training images and never uses validation images to set integer ranges.
The export report records Keras checkpoint size, both TFLite sizes, input/output scales, sample prediction differences, and PC timing.
I evaluate the exported INT8 model across the same full validation split to measure any detection-quality change.

The TFLite file size is model payload, not total firmware flash.
Actual firmware also needs the runtime and supported operators.
Peak RAM must include inference arenas, input/output tensors, camera buffers, MicroSeg masks, detection storage, and decision logic.
The RGB float input alone takes 307,200 bytes, while an INT8 input takes 76,800 bytes.
Desktop timing excludes capture, decoding, MicroSeg, and decision logic and cannot establish ESP32 FPS.
ESP32-CAM AI Thinker compatibility remains unmeasured until the exported model runs on the physical board.
I test FOMO alone, MicroSeg alone, and both together before choosing a deployment schedule.
My preferred upgrade is ESP32-S3 with 8 MB PSRAM, with XIAO ESP32-S3 Sense as the compact target and DevKitC N8R8 acceptable for development.
I use the Sense camera setup and retain the VL53L1X sensor over I2C.
The architecture document records the benchmark matrix and fallback order.

## Reproducing the diagnosis

I can compare saved probabilities without retraining or editing any checkpoint:

```powershell
& .\.venv\Scripts\python.exe tinyml\fomo\diagnose.py --manifest $manifestPath --evaluation tinyml/fomo/artifacts/cityscapes-evaluation-20261004
& .\.venv\Scripts\python.exe tinyml\fomo\diagnose.py --manifest $manifestPath --evaluation tinyml/fomo/artifacts/cityscapes-int8-evaluation-20261004
```

Each command verifies record order and reproduces the original connected-decoder metrics before comparing alternatives.
It reports object-size recall, ignored predictions, old top-three losses, and ideal annotation-grid recovery at 20 x 20 and 40 x 40.
The 40 x 40 result is a geometry diagnostic, not a trained-model score or an accuracy ceiling.
I regenerate ignore masks from original annotations before any future finer-grid training rather than resizing the cached 20 x 20 ignore-cell list.
All comparisons use the same previously consulted development-validation data, so improvements require independent evaluation before deployment claims.
Unknown source centers are not automatically relabeled as background or masked out at runtime.
The center-mask variant restores supervision only where the source center label is known.
I still need reviewed negatives and a defined camera field of view for genuinely unknown regions.

I create and inspect a separate training-mask variant with these commands:

```powershell
& .\.venv\Scripts\python.exe tinyml\fomo\refine_targets.py --manifest $manifestPath
# Use the NEW manifest path printed above. Open its target-*.png previews before recording review.
$centerManifestPath = Read-Host 'Path to the new center-target manifest.json'
& .\.venv\Scripts\python.exe tinyml\fomo\inspect_dataset.py --manifest $centerManifestPath --samples 0
# After visual inspection, record the specific examples checked with --review-note.
wsl -d Ubuntu -- bash ./run_wsl.sh -u tinyml/fomo/train.py --manifest $centerManifestPath --epochs 20 --batch-size 32
```

Training refuses the unreviewed variant until its manifest-bound review receipt exists.
This creates a new checkpoint directory and preserves the old dataset and checkpoints.

## Finer-grid experiment

I regenerate the 40 x 40 training ignore mask from original source-mask pixels rather than stretching the 20 x 20 mask.
Each record declares `target_grid_size`; loading refuses a different requested training grid.
I keep the original 20 x 20 evaluation ignore mask, matching radius, threshold, image inputs, classes, and data split fixed for both models.
The model loader reads the supported grid from checked metadata, allowing old and new models to coexist without changing global settings.

```powershell
& .\.venv\Scripts\python.exe tinyml\fomo\refine_targets.py --manifest $manifestPath --grid-size 40
# Inspect the NEW manifest and target previews, then record a manifest-bound review receipt.
$fineManifestPath = Read-Host 'Path to the reviewed 40x40 manifest.json'
& .\.venv\Scripts\python.exe tinyml\fomo\train.py --manifest $fineManifestPath --grid-size 40 --sanity
wsl -d Ubuntu -- bash ./run_wsl.sh -u tinyml/fomo/train.py --manifest $fineManifestPath --grid-size 40 --epochs 20 --batch-size 32
```

I use the original evaluation manifest when evaluating the new checkpoint with `evaluate.py`.
Export reads the saved model grid automatically and accepts the reviewed fine-grid manifest for training-only calibration.
`benchmark_grids.py --model <20-grid-int8> --model <40-grid-int8> --manifest <original-manifest>` compares the two exports on identical inputs with interleaved CPU calls.
It reports model size, timing, and tensor buffer sizes; it leaves physical-board peak RAM and FPS unmeasured.

## Terms I need

| Term | Meaning here |
|---|---|
| Instance annotation | Pixels identifying one individual object |
| Bounding box | Smallest rectangle enclosing the visible instance pixels |
| Centroid | Here, the center of that rectangle |
| Epoch | One pass through the training images |
| Batch | Images used together for one weight update |
| Learning rate | Size of an optimizer update |
| Training loss | Weighted penalty for wrong training-grid probabilities |
| Validation loss | The same penalty on images excluded from weight updates |
| Precision | Fraction of scored detections matched to real labeled objects |
| Recall | Fraction of labeled objects matched by a detection |
| F1 | A score balancing precision and recall |
| Confidence threshold | Minimum object score kept by the decoder |
| Overfitting | Improving on training scenes without improving on other scenes |
| Quantization | Replacing float values with smaller integers for deployment |

I retain completed original results in `CITYSCAPES_BASELINE_REPORT.md` and current diagnostic results in `DIAGNOSIS_REPORT.md`.
BDD100K and Philippine Obstacle training remain later phases and are not implemented here.
