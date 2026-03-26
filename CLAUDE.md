# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Custom implementation of YOLO computer vision object detection models for the Siftr personal inventory mobile app. Model architectures are primarily based on YOLOv9, cutting-edge research, and custom optimizations. There is a heavy focus on optimizations for running on resource constrained, hardware accelerated mobile phones - specifically newer iOS and Android devices.

Uses Hydra for configuration management and PyTorch Lightning for training. The local virtual environment (.venv) was created and is managed by uv.

## Technical Considerations

Every effort should be made to maximize compatibility with accelerated mobile phone hardware. When making implementation decisions, consider if the choice will run on Apple's ANE, GPU and Android device manufacturer's equivalents.

## Common Commands

### Install

```bash
uv pip install [package_name]
```

### Run (CLI entry point: `yolo/lazy.py`)

```bash
python yolo/lazy.py task=train dataset=mock model=v9-c
python yolo/lazy.py task=validation dataset=mock
python yolo/lazy.py task=inference task.data.source=path/to/images
```

### Test

```bash
pytest --cov=yolo                      # all tests with coverage
pytest tests/test_model/               # test a specific directory
pytest tests/test_tools/test_solver.py # single test file
pytest -k "test_name"                  # single test by name
```

### Lint / Format

```bash
pre-commit run --all-files
```

- **black** (line length 120) for formatting
- **isort** (black profile) for import sorting

## Architecture

### Configuration System (Hydra + OmegaConf)

- Entry config: `yolo/config/config.yaml` — sets defaults for task, dataset, model, general
- Dataclass schema: `yolo/config/config.py`
- Model configs: `yolo/config/model/` (v9-c, v9-s, v9-m, v9-t, v7, rd-9c) — YAML-defined layer architectures
- Dataset configs: `yolo/config/dataset/` (coco, mock, dev)
- Task configs: `yolo/config/task/` (train, validation, inference)
- Override any config value via CLI: `python yolo/lazy.py task.data.batch_size=8 model=v9-s`

### Core Modules

- **`yolo/model/yolo.py`** — `YOLO` class: dynamically builds model architecture from YAML config by composing layers defined in `module.py`
- **`yolo/model/module.py`** — All layer implementations (Conv, Pool, ELAN, ADown, Detection, Segmentation, etc.)
- **`yolo/tools/solver.py`** — Lightning modules: `InferenceModel` → `ValidateModel` → `TrainModel` (inheritance chain). Handles training loops, mAP validation, and inference with NMS
- **`yolo/tools/data_loader.py`** — `YoloDataset` (training/val) and `StreamDataLoader` (inference from files/dirs/webcam)
- **`yolo/tools/loss_functions.py`** — Loss computation including auxiliary head losses for v9
- **`yolo/tools/data_augmentation.py`** — Augmentation pipeline with composable transforms

### Key Design Patterns

- Model architectures are entirely config-driven: YAML files define layer types, channels, and connections via `source` tags
- The converter system (`Vec2Box`, `Anc2Box`) bridges model output vectors to bounding boxes, selected by model name
- Public API exports in `yolo/__init__.py`: `create_model`, `create_converter`, `Config`, `Vec2Box`, `Anc2Box`

### Test Fixtures

Session-scoped fixtures in `tests/conftest.py` provide pre-configured objects (configs, models, dataloaders, trainers). Tests use the `mock` dataset by default.

## Commit Style

This project uses [gitmoji](https://gitmoji.dev/) prefixes in commit messages.

## Rules to Live By

- Plans for significant updates requiring multiple phases should be saved to this project's local ./.claude/plans folder. Where the updates are major or when requested, a sub-folder for the plan should be created so other artifacts, future updates, decision logs, diagrams, etc can be saved along with the plan.
- When significant work has been performed on a plan, the markdown document should be updated with progress, any deviations, and brief notes useful for future reference.
- When asked to add notes or updates to a markdown document, be very brief. It is not necessary to include large code chunks or information that can be easily found with a basic search. Do include terse references to files containing important code.
