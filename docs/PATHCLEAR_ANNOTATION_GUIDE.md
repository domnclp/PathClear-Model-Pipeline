# PathClear annotation guide

Date: 2026-09-29.
Status: fixed class IDs with draft annotation boundaries to check on the first local pilot.
Codex proposed preparing the pipeline while waiting for labeled local data; I confirmed that sequence on 2026-09-29.
I supplied the ten class IDs below.
I will check the detailed boundary rules against real images before expanding annotation.

## What I label

I assign one class to each visible pixel at the original image resolution.
I label visible surfaces, not an object's hidden extent, distance, movement, or estimated danger.
I keep segmentation labels separate from walking-corridor and obstruction-event decisions.
I start each mask at 255, then label the pixels I can identify.
I use 255 for unsupported content and uncertain regions; I never use Road as a background default.

| ID | Class | My draft inclusion rule | My exclusion rule |
|---|---|---|---|
| 0 | Road | Vehicle roadway, including ordinary painted lane markings | I give visible crosswalk markings ID 8 and clearly damaged paved areas ID 9. |
| 1 | Sidewalk | Recognizable pedestrian pavement, paved footpaths, curb ramps, and the visible curb edge | I do not infer that an unmarked roadway shoulder is a sidewalk just because people walk there. |
| 2 | Person | Visible pedestrians, including carried clothing/bags where inseparable from the person silhouette | I label riders as Vehicle together with the vehicle they ride (decision 2026-10-20). |
| 3 | Vehicle | Cars, jeepneys, buses, trucks, trains, motorcycles, bicycles, recognizable motorized or pedal tricycles, and the riders on them | I do not infer whether the vehicle is moving; parked vehicles use the same ID. |
| 4 | Boundary | Walls, fences, gates, and railings that form a visible barrier | I ignore building facades rather than relabel all buildings as Boundary. |
| 5 | Obstacle | Discrete objects such as poles, traffic lights/signs, bollards, cones, bins, boxes, loose debris, and distinguishable freestanding stall structures | I use a more specific class for people, vehicles, vegetation, barriers, stairs, or damaged pavement. |
| 6 | Vegetation | Trees, visible trunks, shrubs, grass, and plants, including plants growing into a path | I do not change vegetation to Obstacle based on its location. |
| 7 | Stairs | Visible stair treads and risers; an identifiable isolated constructed step can be included | I label a flat landing by its surface class, a handrail as Boundary, and an ordinary curb as Sidewalk. |
| 8 | Crosswalk | Visible paint belonging to an identifiable pedestrian crossing, including stripes and other crossing paint | I leave unpainted gaps as Road; lane lines, arrows, and stop lines remain Road. |
| 9 | Uneven Pavement | Clearly visible broken, sunken, raised, missing, or displaced paved surface, including identifiable pothole interiors | I exclude harmless texture, stains, shadows, ordinary paving joints, loose objects, and unsupported guesses about height. |
| 255 | Ignore | Sky, building facades, unpaved terrain, content outside these definitions, severe blur, and genuinely uncertain pixels | I do not count ignored pixels as background or as an eleventh class. |

## How I resolve overlaps

I label the visible object first: a person standing on a crosswalk remains Person; the covered crosswalk is not drawn through them.
I use the specific object class before the generic Obstacle class.
I label visible stair surfaces Stairs, even when painted; I review damaged stairs separately instead of switching them unpredictably to Uneven Pavement.
I label a clearly damaged pavement patch Uneven Pavement, even inside a crossing; I label the remaining visible crossing paint Crosswalk.
I keep intact roadway and sidewalk pixels around damage in their own classes.
I do not fill the entire rectangle around a pothole, crack, vehicle, or person.
I keep uncertain border pixels at 255 rather than inventing exact boundaries through blur or occlusion.

My Crosswalk definition is a paint class, not a filled pedestrian-crossing zone.
If I need the whole crossing zone for navigation, I will derive or annotate that separately rather than quietly change this mask convention.
My semantic masks lose the underlying Road/Sidewalk identity wherever I label Uneven Pavement; they cannot encode both labels in one pixel.

## Difficult Philippine street examples

- I label a rider and the tricycle, motorcycle, or bicycle they ride together as one Vehicle region.
- I distinguish freestanding stall objects from building facades; I leave an ambiguous attached frontage at 255 for review.
- I label a container as Obstacle and its plant as Vegetation where both are distinguishable.
- I keep a puddle or wet reflection at 255 when it obscures the underlying surface; water alone is not evidence of uneven pavement.
- I label an open drain or unpaved ditch 255 under this taxonomy and record it as a coverage gap; I do not silently expand Uneven Pavement into every ground hazard.
- I record animals and other unsupported content as gaps rather than forcing them into one of the ten classes.

## My first annotation review

I start with roughly 20-30 diverse frames as a workflow pilot, not as a promise of sufficient training data.
I deliberately include stairs, crossing paint, and pavement damage, plus intact surfaces, ordinary lane markings, shadows, and wet patches that could be confused with them.
I include multiple examples and viewpoints for each class across independent routes, rather than relying on one image with a few pixels.
I ask a second reviewer to independently inspect a small shared subset and resolve disagreements before I annotate many more frames.
I keep disputed masks unreviewed and record the reason in the manifest notes.
I inspect all masks overlaid on the original image, checking alignment, missing thin objects, and incorrect IDs.
I freeze a version of these rules with the reviewed pilot; later rule changes require reviewing affected masks.

## Mask format and collection record

I export each label mask as an 8-bit, single-channel PNG containing only IDs 0-9 and 255.
I can also use an indexed PNG only when the stored palette indices are exactly those IDs.
I never use RGB color previews as training masks, JPEG masks, antialiased label edges, or interpolated fractional IDs.
I retain original image dimensions and do not resize source annotations.
I use nearest-neighbor interpolation if a later loader resizes masks.

I add records to `datasets/pathclear_pilot/manifest.json` only when the image exists.
Paths are relative to that manifest, with forward slashes.
This example shows the record format and is not a collected sample:

```json
{
  "sample_id": "route01_session01_frame000120",
  "image": "images/route01_session01_frame000120.png",
  "mask": null,
  "route_id": "route01",
  "session_id": "route01_session01",
  "split": "train",
  "camera": "Actual device, mount height and angle; note any substitute camera",
  "source": "clips/route01_session01.mp4 at 00:00:04.000",
  "reviewed": false,
  "reviewer": "",
  "notes": "Capture conditions, annotation questions and unsupported hazards"
}
```

I replace `mask: null` with the relative PNG path when annotation is available.
I set `reviewed` to true and record the reviewer only after visual review.
I keep original clips, timestamps, mounting notes, and selected frames together.
I use train, val, test, or temporarily unassigned for `split`.
I assign each route and session to one split, keeping repeats of the same physical location together.
I avoid adjacent-frame and scene leakage even if that prevents an exact percentage split.
I use the test set only after choosing the final model; checking file validity and class coverage does not run predictions or tune the model.

## What the validator establishes

I run `validate_pathclear_pilot.py` to check file existence, image/mask dimensions, mask encoding, IDs, exact duplicate image files, and route/session separation.
I inspect its per-class pixel and frame counts in each split, including all three new classes.
Exact file hashes do not detect every re-encoded duplicate or visually similar scene, so I also review route assignments and selected frames.
The script cannot verify semantic label correctness, real reviewer work, or whether the dataset is large and varied enough.
Passing its coverage checks does not authorize training.
