# CoreML Pipeline Export: Implementation & Fixes

**Date:** 2026-03-26
**Status:** Complete — pipeline exports and runs in iOS project

## What We Built

CoreML Pipeline model (ML Program + NonMaximumSuppression) that the iOS app consumes directly. Replaces the standalone 9-tensor ML Program export.

### Files Modified

| File | Change |
|------|--------|
| `yolo/utils/deploy_utils.py` | `CoreMLPipelineExportWrapper`, `_build_nms_spec()`, `_export_coreml_pipeline()`, refactored `export_coreml()` with `pipeline` flag |
| `yolo/config/task/export.yaml` | Added `pipeline`, `iou_threshold`, `confidence_threshold` |
| `yolo/config/config.py` | Added fields to `ExportConfig` dataclass |
| `yolo/lazy.py` | Passes new config params to `export_coreml()` |
| `tests/test_utils/test_deploy_utils.py` | 6 new tests for pipeline wrapper, NMS spec, end-to-end |

### Key Components

- **`CoreMLPipelineExportWrapper`** — Bakes Vec2Box decoding + sigmoid into traced model. Outputs `(N, 80)` confidence + `(N, 4)` xywh-relative coordinates. Uses `flatten(1,2)` instead of `reshape` to avoid dynamic shape ops.
- **`_build_nms_spec()`** — Builds NMS protobuf with FLOAT32 arrays, flexible output shapes via `shapeRange`, class labels.
- **`_export_coreml_pipeline()`** — Traces wrapper, converts detector at FP32, builds NMS spec, assembles pipeline, patches image/threshold inputs, copies NMS output types to pipeline outputs, passes `weights_dir`.

## Issues Resolved During Implementation

### 1. Dynamic `int()` casts break coremltools
**Symptom:** `TypeError: only 0-dimensional arrays can be converted to Python scalars` during `ct.convert`.
**Cause:** `B, C, h, w = cls.shape` followed by `reshape(B, h*w, C)` generates `int()` ops in the JIT trace that coremltools can't convert.
**Fix:** Use `cls.permute(0, 2, 3, 1).flatten(1, 2)` — no shape extraction needed.

### 2. NMS spec version too low for FLOAT16
**Symptom:** `FLOAT16 dataType only valid in specification version >=7. This model has version 5.`
**Fix:** Set `spec.specificationVersion = 7` (later inherited from detector spec dynamically).

### 3. NMS requires FLOAT32 or DOUBLE — not FLOAT16
**Symptom:** `element data type of 'confidence' and 'coordinates' must be either MultiArray<DOUBLE> or MultiArray<FLOAT32>`
**Fix:** Convert detector with `compute_precision=ct.precision.FLOAT32` instead of FLOAT16. This is the cleanest solution — no type patching needed.

### 4. Detector output type patching breaks ML Program internal consistency
**Symptom:** `Model output 'raw_confidence' has a different type than its corresponding return value to main`
**Cause:** Patching the detector spec's output description to FLOAT32 while the ML Program's `main` function still returns FLOAT16.
**Fix:** Don't patch — use FP32 precision at conversion time (see #3).

### 5. Pipeline output types must match NMS output types
**Symptom:** `Type of pipeline output 'confidence' does not match type produced in pipeline input.`
**Fix:** After assembling pipeline, copy NMS output types to pipeline output descriptions:
```python
nms_outputs = {out.name: out for out in nms_spec.description.output}
for pipe_out in pipeline_spec.description.output:
    if pipe_out.name in nms_outputs:
        pipe_out.type.CopyFrom(nms_outputs[pipe_out.name].type)
```

### 6. NMS outputs need `shapeRange`, not fixed `shape`
**Symptom:** `If shape information is provided for confidence output, two dimensions must be specified using either shape (deprecated) or allowedShapes.`
**Fix:** Use only `shapeRange.sizeRanges` (flexible first dim, fixed second dim). Do not set `shape` at all on NMS outputs.

### 7. Missing `weights_dir` causes `outputSchema` metadata error
**Symptom:** `missingMetadataField(named: "outputSchema")` in Xcode. Also `Could not open weight.bin` warning during export.
**Fix:** Pass `weights_dir` from the detector model when creating the pipeline MLModel:
```python
ct.models.MLModel(pipeline_spec, weights_dir=detector_model.weights_dir)
```

## Lessons for Next Time

1. **CoreML NMS only supports FLOAT32/DOUBLE.** Don't use FP16 precision for the detector stage in a pipeline. The flat (non-pipeline) export can still use FP16.
2. **Pipeline type matching is strict.** Every connection between stages must match exactly in data type and shape. When in doubt, `CopyFrom` the upstream output type to the downstream input.
3. **NMS outputs are variable-length.** Use `shapeRange` with `upperBound = -1` for the detection count dimension. Never set a fixed `shape` on NMS outputs.
4. **`weights_dir` is required for ML Program pipelines.** Without it, the saved mlpackage won't contain the actual weight data and Xcode can't compile it.
5. **Avoid dynamic shape ops in JIT traces for coremltools.** Use `flatten`, `permute`, `view` with constants — never unpack `.shape` into Python ints.
6. **NMS spec version must match the detector.** Inherit it: `nms_spec.specificationVersion = detector_spec.specificationVersion`.

## Non-Square Input Support

- YOLO is fully convolutional — any dimensions divisible by stride (32) work.
- Override at export time: `'task.image_size=[384,640]'` (quote brackets for zsh).
- 384x640 is ~40% fewer pixels than 640x640, significant mobile speedup.
- Best accuracy requires training at the target resolution, but inference-only reshape often works acceptably.
