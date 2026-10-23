# PathClear

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
The FOMO pipeline lives on the `tinyml-fomo` branch, starting from [tinyml/fomo/README.md](tinyml/fomo/README.md).

I leave checkpoints, archived deployment sources, datasets, and generated outputs in their existing locations.
This reorganization does not authorize additional training.
