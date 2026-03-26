# CoreML (.mlpackage) Export Feature

## Requirements

- Export YOLO models (v9-c, v9-s, etc.) to Apple's `.mlpackage` format using coremltools
- Two export modes:
  1. **iOS export** (`task=export`) — standalone `.mlpackage` artifact with `ImageType` input, ready to drop into an Xcode project for on-device inference
  2. **macOS inference** (`task.fast_inference=coreml`) — export + load back into the Python pipeline for CoreML-accelerated inference on Mac
- FP16 precision for optimal Apple Neural Engine performance
- macOS-only (CoreML is an Apple platform technology)
- `coremltools` remains an optional dependency (imported dynamically)

## Plan

### iOS Export (primary use case)

Standalone `export_coreml()` function and `task=export` CLI command:
- Uses `ct.ImageType(scale=1/255.0, color_layout=RGB)` so the model accepts raw camera frames (uint8 [0,255]) directly in Swift — no preprocessing needed
- Targets `iOS16` minimum deployment
- Sets model metadata (author, description) for Xcode visibility
- Strips auxiliary heads for clean, minimal export

### macOS Inference (secondary use case)

Integrated into existing `FastModelLoader` as `fast_inference=coreml`:
- Uses `ct.TensorType` to match the existing Python pipeline (float tensors in [0,1])
- Targets `macOS13` minimum deployment
- On first use, exports and caches the `.mlpackage`; subsequent runs load from cache
- Reconstructs `{"Main": [...]}` output dict so PostProcess/NMS pipeline works unchanged

**Conversion path:** `torch.jit.trace` -> `ct.convert()` (stable path; `torch.export` support in coremltools is still beta as of v9.0)

## Implementation Details

### Files changed

| File | Change |
|------|--------|
| `yolo/utils/deploy_utils.py` | Added `export_coreml()`, `CoreMLExportWrapper`, `_patch_anchor2vec_for_coreml()`, `_load_coreml_model()`, `_create_coreml_model()`, updated validation and dispatch |
| `yolo/config/task/export.yaml` | New task config for CLI export |
| `yolo/config/task/inference.yaml` | Updated comment to list `coreml` |
| `yolo/config/config.py` | Added `ExportConfig` dataclass, fixed `fast_inference` type: `Optional[None]` -> `Optional[str]` |
| `yolo/lazy.py` | Added `task=export` routing |
| `yolo/__init__.py` | Exported `export_coreml` in public API |
| `tests/test_utils/test_deploy_utils.py` | New test file with 5 tests (wrapper + conversion) |

### CoreMLExportWrapper

The YOLO model's `forward()` returns a dict with optional kwargs — JIT tracing requires tensor-in, tuple-out. `CoreMLExportWrapper` wraps the model and flattens the 9 output tensors (3 detection heads x 3 tensors each: class logits, anchor distribution, vector box).

### Anchor2Vec Patch

`einops.rearrange` in the `Anchor2Vec` module generates dynamic `int()` casts in the JIT trace graph that coremltools cannot convert. `_patch_anchor2vec_for_coreml()` replaces it with an equivalent `unflatten` + `permute` using fixed `reg_max` constants that trace cleanly.

## Learnings

1. **einops and coremltools don't mix** — `einops.rearrange` decomposes tensor dimensions using dynamic shape queries (`tensor.shape[i]`) which produce `int()` cast ops in the traced graph. coremltools fails on these with `TypeError: only 0-dimensional arrays can be converted to Python scalars`. The fix is to replace with standard PyTorch reshape ops using known constants.

2. **torch.jit.trace is still the reliable path** — Despite deprecation warnings, `torch.jit.trace` remains the most battle-tested conversion path for coremltools 9.0. `torch.export` support exists but is beta with ~70% op parity.

3. **FP16 halves model size** — v9-c goes from 103MB (FP32 .pt) to 49MB (FP16 .mlpackage) with no architecture changes.

4. **PyTorch version mismatch warning** — coremltools 9.0 is tested up to PyTorch 2.7.0. Our project uses PyTorch 2.11.0. Conversion works but emits a warning. Monitor for issues if upgrading either dependency.

5. **ImageType vs TensorType** — `ct.ImageType` with `scale=1/255.0` bakes normalization into the CoreML model. This is critical for iOS deployment because it lets Swift pass raw `CVPixelBuffer` from the camera without manual preprocessing. `ct.TensorType` is better for Python-side inference where the pipeline already handles normalization.

## Usage

### CLI: Export for iOS deployment
```bash
# Default output: v9-c.mlpackage
python yolo/lazy.py task=export model=v9-c

# Custom output path
python yolo/lazy.py task=export model=v9-c task.output_path=weights/v9-c-ios.mlpackage

# Different model variant
python yolo/lazy.py task=export model=v9-s
```

The resulting `.mlpackage` can be dragged directly into Xcode. In Swift, load it with:
```swift
let model = try YOLOv9c(configuration: .init())
let prediction = try model.prediction(image: pixelBuffer)
```

The model accepts RGB images (640x640) and handles normalization internally.

### CLI: macOS inference with CoreML backend
```bash
python yolo/lazy.py task=inference task.fast_inference=coreml model=v9-c
```

The `.mlpackage` is created on first run and cached for subsequent runs.

### Programmatic export
```python
from yolo import export_coreml
from hydra import compose, initialize

with initialize(config_path="yolo/config", version_base=None):
    cfg = compose(config_name="config", overrides=["model=v9-c"])

export_coreml(
    model_cfg=cfg.model,
    weight_path="weights/v9-c.pt",
    class_num=80,
    image_size=(640, 640),
    output_path="v9-c.mlpackage",
)
```

### Output tensor mapping

The 9 output tensors map to 3 detection heads (strides 8, 16, 32):

| Output | Head | Tensor | Shape (640x640 input) |
|--------|------|--------|-----------------------|
| `output_0` | stride 8 | class logits | (1, 80, 80, 80) |
| `output_1` | stride 8 | anchor dist | (1, 16, 4, 80, 80) |
| `output_2` | stride 8 | vector box | (1, 4, 80, 80) |
| `output_3` | stride 16 | class logits | (1, 80, 40, 40) |
| `output_4` | stride 16 | anchor dist | (1, 16, 4, 40, 40) |
| `output_5` | stride 16 | vector box | (1, 4, 40, 40) |
| `output_6` | stride 32 | class logits | (1, 80, 20, 20) |
| `output_7` | stride 32 | anchor dist | (1, 16, 4, 20, 20) |
| `output_8` | stride 32 | vector box | (1, 4, 20, 20) |

### Install dependency
```bash
uv pip install coremltools
```
