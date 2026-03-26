import json
import platform
from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor

from yolo.config.config import Config, ModelConfig
from yolo.model.yolo import create_model
from yolo.utils.bounding_box_utils import generate_anchors
from yolo.utils.logger import logger


def _patch_anchor2vec_for_coreml(model, reg_max=16):
    """Replace einops rearrange in Anchor2Vec with standard PyTorch ops.

    einops.rearrange generates dynamic int() casts in the JIT trace that
    coremltools cannot convert. This replaces the forward method with an
    equivalent reshape+permute using fixed constants that trace cleanly.
    """
    from yolo.model.module import Anchor2Vec

    for module in model.modules():
        if isinstance(module, Anchor2Vec):
            # Capture reg_max as a closure constant to avoid dynamic shape ops
            _reg_max = reg_max

            def _make_forward(r):
                def _coreml_safe_forward(self, anchor_x):
                    # Reshape (B, 4*R, h, w) -> (B, 4, R, h, w) -> (B, R, 4, h, w)
                    anchor_x = anchor_x.unflatten(1, (4, r)).permute(0, 2, 1, 3, 4).contiguous()
                    vector_x = anchor_x.softmax(dim=1)
                    vector_x = self.anc2vec(vector_x)[:, 0]
                    return anchor_x, vector_x

                return _coreml_safe_forward

            module.forward = _make_forward(_reg_max).__get__(module, Anchor2Vec)


class CoreMLExportWrapper(torch.nn.Module):
    """Wraps YOLO model for CoreML export: tensor-in, flat-tuple-out.

    JIT tracing requires simple tensor inputs and tuple/tensor outputs.
    This wrapper calls the YOLO model and flattens the nested output dict
    into a flat tuple of 9 tensors (3 detection heads x 3 tensors each):
        (cls_0, anc_0, vec_0, cls_1, anc_1, vec_1, cls_2, anc_2, vec_2)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        main_outputs = output["Main"]
        flat = []
        for head in main_outputs:
            for tensor in head:
                flat.append(tensor)
        return tuple(flat)


class CoreMLPipelineExportWrapper(torch.nn.Module):
    """Wraps YOLO model for CoreML Pipeline export with NMS.

    Bakes Vec2Box decoding + sigmoid into the traced model so the output is
    ready for CoreML's NonMaximumSuppression layer:
        confidence (N, num_classes) — sigmoid scores
        coordinates (N, 4) — xywh relative [0,1]
    """

    def __init__(self, model, image_size, strides):
        super().__init__()
        self.model = model
        H, W = image_size
        self.image_w = float(W)
        self.image_h = float(H)
        anchor_grid, scaler = generate_anchors((W, H), strides)
        self.register_buffer("anchor_grid", anchor_grid)
        self.register_buffer("scaler", scaler)

    def forward(self, x):
        output = self.model(x)
        main_outputs = output["Main"]

        all_cls = []
        all_vec = []
        for head in main_outputs:
            cls, _anc, vec = head
            # (B, C, h, w) -> (B, h, w, C) -> (B, h*w, C) — no dynamic shape ops
            cls = cls.permute(0, 2, 3, 1).flatten(1, 2)
            # (B, 4, h, w) -> (B, h, w, 4) -> (B, h*w, 4)
            vec = vec.permute(0, 2, 3, 1).flatten(1, 2)
            all_cls.append(cls)
            all_vec.append(vec)

        preds_cls = torch.cat(all_cls, dim=1)  # (1, N, 80)
        preds_box = torch.cat(all_vec, dim=1)  # (1, N, 4)

        # Vec2Box decoding
        pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        boxes_xyxy = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)

        # Sigmoid confidence
        confidence = preds_cls.sigmoid()

        # xyxy absolute -> xywh relative
        x1 = boxes_xyxy[..., 0:1]
        y1 = boxes_xyxy[..., 1:2]
        x2 = boxes_xyxy[..., 2:3]
        y2 = boxes_xyxy[..., 3:4]
        cx = (x1 + x2) / 2.0 / self.image_w
        cy = (y1 + y2) / 2.0 / self.image_h
        w = (x2 - x1) / self.image_w
        h = (y2 - y1) / self.image_h
        coordinates = torch.cat([cx, cy, w, h], dim=-1)

        # Squeeze batch dim
        return confidence.squeeze(0), coordinates.squeeze(0)


def _build_nms_spec(num_anchors, num_classes, class_list, iou_threshold=0.45, confidence_threshold=0.25):
    """Build a CoreML NonMaximumSuppression model spec (protobuf).

    Args:
        num_anchors: Total number of anchors (N).
        num_classes: Number of detection classes.
        class_list: List of class name strings.
        iou_threshold: Default IoU threshold.
        confidence_threshold: Default confidence threshold.

    Returns:
        coremltools Model spec for the NMS layer.
    """
    import coremltools as ct
    from coremltools.proto import Model_pb2

    spec = Model_pb2.Model()
    spec.specificationVersion = 7

    # Inputs
    conf_input = spec.description.input.add()
    conf_input.name = "raw_confidence"
    conf_input.type.multiArrayType.shape.append(num_anchors)
    conf_input.type.multiArrayType.shape.append(num_classes)
    conf_input.type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.FLOAT32

    coord_input = spec.description.input.add()
    coord_input.name = "raw_coordinates"
    coord_input.type.multiArrayType.shape.append(num_anchors)
    coord_input.type.multiArrayType.shape.append(4)
    coord_input.type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.FLOAT32

    iou_input = spec.description.input.add()
    iou_input.name = "iouThreshold"
    iou_input.type.doubleType.SetInParent()

    conf_thresh_input = spec.description.input.add()
    conf_thresh_input.name = "confidenceThreshold"
    conf_thresh_input.type.doubleType.SetInParent()

    # Outputs — FLOAT32, with flexible first dimension (variable detections after NMS)
    conf_output = spec.description.output.add()
    conf_output.name = "confidence"
    ma = conf_output.type.multiArrayType
    ma.dataType = Model_pb2.ArrayFeatureType.FLOAT32
    ma.shapeRange.sizeRanges.add().lowerBound = 0
    ma.shapeRange.sizeRanges[-1].upperBound = -1
    ma.shapeRange.sizeRanges.add().lowerBound = num_classes
    ma.shapeRange.sizeRanges[-1].upperBound = num_classes

    coord_output = spec.description.output.add()
    coord_output.name = "coordinates"
    ma = coord_output.type.multiArrayType
    ma.dataType = Model_pb2.ArrayFeatureType.FLOAT32
    ma.shapeRange.sizeRanges.add().lowerBound = 0
    ma.shapeRange.sizeRanges[-1].upperBound = -1
    ma.shapeRange.sizeRanges.add().lowerBound = 4
    ma.shapeRange.sizeRanges[-1].upperBound = 4

    # NMS layer config
    nms = spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = "raw_confidence"
    nms.coordinatesInputFeatureName = "raw_coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = iou_threshold
    nms.confidenceThreshold = confidence_threshold
    nms.pickTop.perClass = True

    # Class labels
    for label in class_list:
        nms.stringClassLabels.vector.append(label)

    return spec


class FastModelLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.compiler = cfg.task.fast_inference
        self.class_num = cfg.dataset.class_num

        self._validate_compiler()
        if cfg.weight == True:
            cfg.weight = Path("weights") / f"{cfg.model.name}.pt"
        if self.compiler == "coreml":
            self.model_path = f"{Path(cfg.weight).stem}.mlpackage"
        else:
            self.model_path = f"{Path(cfg.weight).stem}.{self.compiler}"

    def _validate_compiler(self):
        if self.compiler not in ["onnx", "trt", "coreml", "deploy"]:
            logger.warning(f":warning: Compiler '{self.compiler}' is not supported. Using original model.")
            self.compiler = None
        if self.cfg.device == "mps" and self.compiler == "trt":
            logger.warning(":red_apple: TensorRT does not support MPS devices. Using original model.")
            self.compiler = None
        if self.compiler == "coreml" and platform.system() != "Darwin":
            logger.warning(":warning: CoreML export is only supported on macOS. Using original model.")
            self.compiler = None

    def load_model(self, device):
        if self.compiler == "onnx":
            return self._load_onnx_model(device)
        elif self.compiler == "trt":
            return self._load_trt_model().to(device)
        elif self.compiler == "coreml":
            return self._load_coreml_model(device)
        elif self.compiler == "deploy":
            self.cfg.model.model.auxiliary = {}
        return create_model(self.cfg.model, class_num=self.class_num, weight_path=self.cfg.weight).to(device)

    def _load_onnx_model(self, device):
        from onnxruntime import InferenceSession

        def onnx_forward(self: InferenceSession, x: Tensor):
            x = {self.get_inputs()[0].name: x.cpu().numpy()}
            model_outputs, layer_output = [], []
            for idx, predict in enumerate(self.run(None, x)):
                layer_output.append(torch.from_numpy(predict).to(device))
                if idx % 3 == 2:
                    model_outputs.append(layer_output)
                    layer_output = []
            if len(model_outputs) == 6:
                model_outputs = model_outputs[:3]
            return {"Main": model_outputs}

        InferenceSession.__call__ = onnx_forward

        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider"]
        try:
            ort_session = InferenceSession(self.model_path, providers=providers)
            logger.info(":rocket: Using ONNX as MODEL frameworks!")
        except Exception as e:
            logger.warning(f"🈳 Error loading ONNX model: {e}")
            ort_session = self._create_onnx_model(providers)
        return ort_session

    def _create_onnx_model(self, providers):
        from onnxruntime import InferenceSession
        from torch.onnx import export

        model = create_model(self.cfg.model, class_num=self.class_num, weight_path=self.cfg.weight).eval()
        dummy_input = torch.ones((1, 3, *self.cfg.image_size))
        export(
            model,
            dummy_input,
            self.model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f":inbox_tray: ONNX model saved to {self.model_path}")
        return InferenceSession(self.model_path, providers=providers)

    def _load_trt_model(self):
        from torch2trt import TRTModule

        try:
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(self.model_path))
            logger.info(":rocket: Using TensorRT as MODEL frameworks!")
        except FileNotFoundError:
            logger.warning(f"🈳 No found model weight at {self.model_path}")
            model_trt = self._create_trt_model()
        return model_trt

    def _create_trt_model(self):
        from torch2trt import torch2trt

        model = create_model(self.cfg.model, class_num=self.class_num, weight_path=self.cfg.weight).eval()
        dummy_input = torch.ones((1, 3, *self.cfg.image_size)).cuda()
        logger.info(f"♻️ Creating TensorRT model")
        model_trt = torch2trt(model.cuda(), [dummy_input])
        torch.save(model_trt.state_dict(), self.model_path)
        logger.info(f":inbox_tray: TensorRT model saved to {self.model_path}")
        return model_trt

    def _load_coreml_model(self, device):
        import coremltools as ct

        def coreml_forward(self, x: Tensor):
            import numpy as np

            input_name = self.get_spec().description.input[0].name
            prediction = self.predict({input_name: x.cpu().numpy()})
            model_outputs, layer_output = [], []
            for idx in range(9):
                key = f"output_{idx}"
                layer_output.append(torch.from_numpy(np.array(prediction[key])).to(device))
                if idx % 3 == 2:
                    model_outputs.append(layer_output)
                    layer_output = []
            return {"Main": model_outputs}

        try:
            coreml_model = ct.models.MLModel(self.model_path)
            logger.info(":rocket: Using CoreML as MODEL framework!")
        except Exception as e:
            logger.warning(f"🈳 Error loading CoreML model: {e}")
            coreml_model = self._create_coreml_model()

        coreml_model.__class__.__call__ = coreml_forward
        return coreml_model

    def _create_coreml_model(self):
        import coremltools as ct

        # Strip auxiliary heads for clean export (same as "deploy" mode)
        self.cfg.model.model.auxiliary = {}
        model = create_model(self.cfg.model, class_num=self.class_num, weight_path=self.cfg.weight).eval()
        _patch_anchor2vec_for_coreml(model, reg_max=model.reg_max)
        wrapper = CoreMLExportWrapper(model).eval()
        dummy_input = torch.rand(1, 3, *self.cfg.image_size)

        logger.info(":gear: Tracing model for CoreML export...")
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, dummy_input)

        logger.info(":gear: Converting to CoreML mlprogram format (FP16)...")
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=(1, 3, *self.cfg.image_size))],
            outputs=[ct.TensorType(name=f"output_{i}") for i in range(9)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS13,
        )
        coreml_model.save(self.model_path)
        logger.info(f":inbox_tray: CoreML model saved to {self.model_path}")
        return coreml_model


def export_coreml(
    model_cfg: ModelConfig,
    weight_path: str = True,
    class_num: int = 80,
    image_size: tuple = (640, 640),
    output_path: Optional[str] = None,
    class_list: Optional[List[str]] = None,
    pipeline: bool = True,
    iou_threshold: float = 0.45,
    confidence_threshold: float = 0.25,
):
    """Export a YOLO model to an iOS-optimized CoreML .mlpackage.

    When pipeline=True (default), produces a Pipeline model:
        Stage 1: ML Program (detector) — image → raw_confidence, raw_coordinates
        Stage 2: NonMaximumSuppression — post-NMS confidence + coordinates
    This matches the structure iOS apps expect for object detection.

    When pipeline=False, produces a standalone ML Program with 9 raw tensors
    (backward-compatible with FastModelLoader).

    Args:
        model_cfg: Hydra model configuration.
        weight_path: Path to .pt weights, or True for auto-download.
        class_num: Number of detection classes.
        image_size: Input image dimensions (H, W).
        output_path: Where to save the .mlpackage. Defaults to {model_name}.mlpackage.
        class_list: List of class label strings.
        pipeline: If True, export as Pipeline with NMS. If False, raw 9-tensor export.
        iou_threshold: Default IoU threshold for NMS.
        confidence_threshold: Default confidence threshold for NMS.

    Returns:
        Path to the saved .mlpackage.
    """
    import coremltools as ct

    if platform.system() != "Darwin":
        raise RuntimeError("CoreML export is only supported on macOS.")

    if output_path is None:
        output_path = f"{model_cfg.name}.mlpackage"

    # Strip auxiliary heads for clean export
    model_cfg.model.auxiliary = {}
    model = create_model(model_cfg, class_num=class_num, weight_path=weight_path).eval()
    _patch_anchor2vec_for_coreml(model, reg_max=model.reg_max)

    if pipeline:
        return _export_coreml_pipeline(
            model, model_cfg, class_num, image_size, output_path, class_list, iou_threshold, confidence_threshold
        )
    else:
        return _export_coreml_flat(model, model_cfg, class_num, image_size, output_path, class_list)


def _export_coreml_flat(model, model_cfg, class_num, image_size, output_path, class_list):
    """Export as standalone ML Program with 9 raw tensors (backward compat)."""
    import coremltools as ct

    wrapper = CoreMLExportWrapper(model).eval()
    dummy_input = torch.rand(1, 3, *image_size)

    logger.info(":gear: Tracing model for CoreML export...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, dummy_input)

    logger.info(":gear: Converting to iOS-optimized CoreML mlprogram (FP16, ImageType)...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, *image_size),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name=f"output_{i}") for i in range(9)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )

    coreml_model.author = "YOLO"
    coreml_model.short_description = f"YOLO {model_cfg.name} object detection ({class_num} classes, {image_size[0]}x{image_size[1]})"

    coreml_model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "objectDetector"
    coreml_model.user_defined_metadata["task"] = "detect"
    coreml_model.user_defined_metadata["Confidence threshold"] = "0.25"
    coreml_model.user_defined_metadata["IoU threshold"] = "0.45"
    coreml_model.user_defined_metadata["date"] = "2024-12-17T10:49:53.367568"

    coreml_model.user_defined_metadata["imgsz"] = "[640, 384]"
    coreml_model.user_defined_metadata["stride"] = "32"
    coreml_model.user_defined_metadata["batch"] = "1"
    if class_list:
        names_dict = {str(i): name for i, name in enumerate(class_list)}
        coreml_model.user_defined_metadata["names"] = json.dumps(names_dict)
        spec = coreml_model._spec
        for label in class_list:
            spec.description.classLabels.stringClassLabels.vector.append(label)
    coreml_model.save(output_path)
    logger.info(f":inbox_tray: iOS-optimized CoreML model saved to {output_path}")
    return Path(output_path)


def _export_coreml_pipeline(
    model, model_cfg, class_num, image_size, output_path, class_list, iou_threshold, confidence_threshold
):
    """Export as Pipeline (ML Program → NonMaximumSuppression)."""
    import coremltools as ct

    strides = model_cfg.anchor.strides
    wrapper = CoreMLPipelineExportWrapper(model, image_size, strides).eval()
    dummy_input = torch.rand(1, 3, *image_size)

    logger.info(":gear: Tracing model for CoreML Pipeline export...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, dummy_input)

    # Calculate total anchors
    H, W = image_size
    num_anchors = sum((W // s) * (H // s) for s in strides)

    # NMS requires FLOAT32 inputs, so the detector must output FLOAT32.
    # Use FLOAT32 precision for the pipeline detector (FP16 is used in the flat export).
    logger.info(":gear: Converting detector to CoreML mlprogram (FP32, ImageType)...")
    detector_model = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, *image_size),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="raw_confidence"),
            ct.TensorType(name="raw_coordinates"),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS16,
    )
    detector_spec = detector_model._spec

    # Build NMS spec
    logger.info(":gear: Building NonMaximumSuppression spec...")
    nms_spec = _build_nms_spec(
        num_anchors=num_anchors,
        num_classes=class_num,
        class_list=class_list or [],
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )

    # Inherit spec version from the detector
    nms_spec.specificationVersion = detector_spec.specificationVersion

    # Assemble Pipeline
    logger.info(":gear: Assembling Pipeline model...")

    # Define pipeline input/output features
    input_features = [
        ("image", ct.models.datatypes.Array(3, *image_size)),
        ("iouThreshold", ct.models.datatypes.Double()),
        ("confidenceThreshold", ct.models.datatypes.Double()),
    ]
    output_features = [
        ("confidence", ct.models.datatypes.Array(class_num)),
        ("coordinates", ct.models.datatypes.Array(4)),
    ]

    pipeline = ct.models.pipeline.Pipeline(input_features, output_features)
    pipeline.add_model(detector_spec)
    pipeline.add_model(nms_spec)

    pipeline_spec = pipeline.spec

    # Patch first input to ImageType (Pipeline defaults to ArrayType)
    image_input = pipeline_spec.description.input[0]
    image_input.type.ClearField("multiArrayType")
    image_type = image_input.type.imageType
    image_type.width = W
    image_type.height = H
    image_type.colorSpace = ct.proto.FeatureTypes_pb2.ImageFeatureType.RGB

    # Mark threshold inputs as optional
    for inp in pipeline_spec.description.input:
        if inp.name in ("iouThreshold", "confidenceThreshold"):
            inp.type.isOptional = True

    # Patch pipeline output types to match NMS output types exactly
    nms_outputs = {out.name: out for out in nms_spec.description.output}
    for pipe_out in pipeline_spec.description.output:
        if pipe_out.name in nms_outputs:
            pipe_out.type.CopyFrom(nms_outputs[pipe_out.name].type)

    # Set metadata
    pipeline_spec.description.metadata.author = "YOLO"
    pipeline_spec.description.metadata.shortDescription = (
        f"YOLO {model_cfg.name} object detection ({class_num} classes, {image_size[0]}x{image_size[1]})"
    )

    user_defined = pipeline_spec.description.metadata.userDefined
    user_defined["com.apple.coreml.model.preview.type"] = "objectDetector"
    user_defined["task"] = "detect"
    user_defined["Confidence threshold"] = str(confidence_threshold)
    user_defined["IoU threshold"] = str(iou_threshold)
    user_defined["imgsz"] = json.dumps(list(image_size))
    user_defined["stride"] = str(max(strides))
    user_defined["batch"] = "1"
    if class_list:
        names_dict = {str(i): name for i, name in enumerate(class_list)}
        user_defined["names"] = json.dumps(names_dict)

    # Pass weights_dir from detector so ML Program weights are included in the pipeline mlpackage
    pipeline_model = ct.models.MLModel(pipeline_spec, weights_dir=detector_model.weights_dir)
    pipeline_model.save(output_path)
    logger.info(f":inbox_tray: CoreML Pipeline model saved to {output_path}")
    return Path(output_path)
