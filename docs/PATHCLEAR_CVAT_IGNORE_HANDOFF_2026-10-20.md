# PathClear handoff: CVAT masks and ignore label (2026-10-20)

## Confirmed in the code
- `IGNORE_LABEL = 255` (`pathclear_taxonomy.py`, `configs/pathclear_classes.yaml`).
- `pathclear_pipeline.prepare_targets()`: IDs 0-9 get weight 1, 255 gets weight 0; asserts masks contain only 0-9 or 255.
- `make_loss()`: `SparseCategoricalCrossentropy(from_logits=True, reduction="mean_with_sample_weight")`, so loss is averaged over labeled pixels only.
- Metrics (`segmentation_metrics.make_metrics`) are weighted, so ignored pixels are excluded from mIoU, Dice and pixel accuracy.
- `docs/PATHCLEAR_ANNOTATION_GUIDE.md`: masks start at 255; sky, facades, severe blur and uncertain pixels stay 255; 255 is not background or an 11th class.
- Model input is fixed at 512x1024 (`transfer_baseline()`); ESP32-CAM frames (800x400) are resized up to 1024x512 (same 2:1 shape).

## Annotation decisions (CVAT)
- Use the Polygon tool (not Polyline) for region masks.
- Overlaps are resolved by CVAT z-order. Draw order, bottom to top:
  1. Road, Sidewalk, Vegetation
  2. Crosswalk, Uneven Pavement, Stairs
  3. Boundary
  4. Obstacle
  5. Vehicle, Person
- Leave far-side buildings and distant vehicle clusters unannotated (they become 255). Proposed rule: skip anything past the far curb and any person or vehicle under about 20 px tall. Do label nearby objects even if motion-blurred.

## Rider decision (2026-10-20)
- Riders are labeled Vehicle together with the vehicle they ride (manuscript and annotation guide updated).
- TODO (code): `pathclear_taxonomy.py` `CITYSCAPES_TO_PATHCLEAR` still maps Cityscapes rider (trainId 12) -> 2 (Person). Change to `12: 3` so the Cityscapes stage matches; update any tests that assert the old mapping.

## CVAT project layout
- Segmentation task: polygons only, one polygon per object (pole+sign, each pillar, each vehicle) so boxes can be derived later.
- FOMO task: a separate CVAT task on the same images with rectangles for Person / Vehicle / Obstacle. Keep rectangles out of the segmentation task, because the Segmentation mask export fills rectangles as solid class regions.

## Converter status

CVAT "Segmentation mask 1.1" exports RGB color PNGs with unannotated pixels as (0,0,0).
Risk: background 0 collides with Road (ID 0), and the `prepare_targets()` assert will NOT catch it.
The converter must:
1. Read the CVAT labelmap (color -> label name) and map each color to its ID 0-9 via `CLASS_IDS`.
2. Map background (0,0,0) and any unknown color -> 255.
3. Write 8-bit single-channel PNGs with only 0-9 and 255, original resolution, no antialiasing.
4. Validate each output against `prepare_targets()` and report per-class pixel counts plus the ignored fraction.
5. Add a unit test in `tests/` (e.g. a tiny synthetic mask with background, two classes and an unknown color).

## Open decision for the team
- Building facades: the Cityscapes stage maps building -> Boundary (`CITYSCAPES_TO_PATHCLEAR`, trainId 2 -> 4), but the annotation guide leaves facades at 255. Pick one rule and state it in Chapter 3.

## For the manuscript (dataset section)
Add a sentence along the lines of: "Unannotated regions, including sky, building facades beyond the far curb, and distant objects below approximately 20 pixels in height, were assigned the ignore label (255) and excluded from the loss and evaluation metrics."
