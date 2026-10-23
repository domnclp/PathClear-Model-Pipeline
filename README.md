# PathClear

> **Note (2026-10-23):** I keep MicroSeg only as the thesis comparison baseline (see [docs/thesis/MODEL_TRADEOFF_RESULTS.md](docs/thesis/MODEL_TRADEOFF_RESULTS.md)).
> It is not deployed.

I use Fast-SCNN for semantic segmentation and keep the TinyML experiment separate.
The 19-class Cityscapes baseline, ten-class adaptation, local-label validation, and TFLite tools share the core modules below.

```text
PathClear/
  fast_scnn.py                 Model architecture
  train.py                     Original baseline training entry point
  dataset_loader.py            Cityscapes pairing and preprocessing
  dataset_config.py            External dataset location
  pathclear_taxonomy.py         Class IDs and remapping
  pathclear_pipeline.py         Safe targets and strict baseline transfer
  pathclear_stages.py           Adaptation stages and custom Keras objects
  segmentation_metrics.py      Weighted segmentation metrics
  configs/                     Class handoff specification
  scripts/                     Training, validation, evaluation and export tools
  tests/                       Regression tests and explicit dataset smoke check
  microseg/                    Four-class custom CNN experiment (taxonomy, model, loader)
  tinyml/                      Separate experimental training/preprocessing/export/ESP32 scaffold
  docs/                        Setup, training, annotation and experiment records
  run_wsl.sh                   Existing Linux TensorFlow environment launcher
```

## My first checks

I run these commands from the repository root with the existing WSL environment:

```bash
bash run_wsl.sh -m unittest discover -s tests -p 'test_*.py'
bash run_wsl.sh tests/test_pipeline.py
bash run_wsl.sh -m scripts.validate_pathclear_pilot datasets/pathclear_pilot/manifest.json
```

The unit tests use synthetic inputs and temporary model files.
The explicit smoke check reads one real Cityscapes batch and runs inference with random weights.
Neither command starts a dataset training run.
The pilot validator may reject incomplete or unreviewed local data; that is the intended training gate.

I invoke tools as modules, for example `python -m scripts.train_pathclear_cityscapes --help`, from an environment with the existing dependencies installed.
`run_wsl.sh` also accepts script paths and historical bare filenames so recorded experiment commands remain usable.
I retain the separate Windows and WSL environments rather than reinstalling TensorFlow during this reorganization.

I start with [dataset setup](docs/DATASET_SETUP.md), [pipeline responsibilities](docs/TRAINING_PIPELINE.md), [training commands](docs/TRAINING.md), and [WSL GPU setup](docs/WSL_GPU.md).
I follow the [local dataset plan](docs/LOCAL_DATASET_PLAN.md) and [annotation guide](docs/PATHCLEAR_ANNOTATION_GUIDE.md) before local fine-tuning.
The FOMO pipeline lives on the `tinyml-fomo` branch, starting from [tinyml/fomo/README.md](https://github.com/slvr123/PathClear-Model-Pipeline/blob/tinyml-fomo/tinyml/fomo/README.md).
The separate [MicroSeg custom CNN plan](docs/MICROSEG_CUSTOM_CNN_PLAN.md) describes one four-class lightweight segmentation model trained from Cityscapes and Kaggle Crosswalk batches. The [MicroSeg onboarding handoff](docs/MICROSEG_ONBOARDING.md) summarizes the current status and next decisions. A fresh five-epoch GroupNormalization pilot no longer produced all-Non-walkable inference predictions: Cityscapes validation reached 0.313 four-class mean IoU, while held-out positive-only Crosswalk recall peaked at 92.6% and ended at 76.0%. The Crosswalk metric cannot measure false positives because unmarked KPV pixels are ignored; Boundary IoU is also still low. MicroSeg uses GroupNormalization, while Fast-SCNN retains its BatchNormalization behavior. Do not scale training until Boundary semantics and Crosswalk-negative validation are addressed. The pipeline supports sparse, reviewed Cityscapes ignore masks; use `scripts.select_cityscapes_candidates` to rank possible scenes and the [annotation guide](docs/MICROSEG_CITYSCAPES_ANNOTATION.md) with `scripts.annotate_cityscapes_ignore` to verify them. A 50-image audit of the original rejected queue found 14 missed crossings, prompting an adaptive-contrast/transverse-band selector revision. The revised queue has 1,127 new training candidates after carrying prior review progress. Deployment is not implemented.

I leave checkpoints, archived deployment sources, datasets, and generated outputs in their existing locations.
This reorganization does not authorize additional training.
