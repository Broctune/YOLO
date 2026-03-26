import platform

import pytest
import torch

from yolo.utils.deploy_utils import CoreMLExportWrapper, CoreMLPipelineExportWrapper


class MockDetectionHead(torch.nn.Module):
    """Minimal mock that mimics YOLO detection output structure.

    Uses deterministic convolutions instead of randn so the model is
    traceable and convertible by coremltools.
    """

    def __init__(self, num_classes=80, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        # Convolutions to produce each output type at each scale
        self.cls_convs = torch.nn.ModuleList([torch.nn.Conv2d(3, num_classes, 1) for _ in range(3)])
        self.anc_convs = torch.nn.ModuleList([torch.nn.Conv2d(3, reg_max * 4, 1) for _ in range(3)])
        self.vec_convs = torch.nn.ModuleList([torch.nn.Conv2d(3, 4, 1) for _ in range(3)])
        self.pools = torch.nn.ModuleList(
            [torch.nn.AdaptiveAvgPool2d(s) for s in [80, 40, 20]]
        )

    def forward(self, x):
        heads = []
        for i in range(3):
            pooled = self.pools[i](x)
            cls = self.cls_convs[i](pooled)
            anc_flat = self.anc_convs[i](pooled)
            anc = anc_flat.reshape(x.shape[0], self.reg_max, 4, pooled.shape[2], pooled.shape[3])
            vec = self.vec_convs[i](pooled)
            heads.append((cls, anc, vec))
        return {"Main": heads}


class TestCoreMLExportWrapper:
    def test_wrapper_output_count(self):
        mock_model = MockDetectionHead()
        wrapper = CoreMLExportWrapper(mock_model)
        x = torch.rand(1, 3, 640, 640)
        output = wrapper(x)
        assert isinstance(output, tuple)
        assert len(output) == 9

    def test_wrapper_output_shapes(self):
        mock_model = MockDetectionHead()
        wrapper = CoreMLExportWrapper(mock_model)
        x = torch.rand(1, 3, 640, 640)
        output = wrapper(x)
        # Head 0 (stride 8): 80x80
        assert output[0].shape == (1, 80, 80, 80)  # cls
        assert output[1].shape == (1, 16, 4, 80, 80)  # anc
        assert output[2].shape == (1, 4, 80, 80)  # vec
        # Head 1 (stride 16): 40x40
        assert output[3].shape == (1, 80, 40, 40)
        assert output[4].shape == (1, 16, 4, 40, 40)
        assert output[5].shape == (1, 4, 40, 40)
        # Head 2 (stride 32): 20x20
        assert output[6].shape == (1, 80, 20, 20)
        assert output[7].shape == (1, 16, 4, 20, 20)
        assert output[8].shape == (1, 4, 20, 20)

    def test_wrapper_is_traceable(self):
        mock_model = MockDetectionHead()
        wrapper = CoreMLExportWrapper(mock_model)
        x = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, x)
        output = traced(x)
        assert isinstance(output, tuple)
        assert len(output) == 9

    def test_wrapper_with_real_model(self, model):
        wrapper = CoreMLExportWrapper(model)
        device = next(model.parameters()).device
        x = torch.rand(1, 3, 640, 640).to(device)
        output = wrapper(x)
        assert isinstance(output, tuple)
        assert len(output) == 9


class TestCoreMLPipelineExportWrapper:
    IMAGE_SIZE = (640, 640)
    STRIDES = [8, 16, 32]
    NUM_ANCHORS = (640 // 8) ** 2 + (640 // 16) ** 2 + (640 // 32) ** 2  # 8400

    def _make_wrapper(self):
        mock_model = MockDetectionHead()
        return CoreMLPipelineExportWrapper(mock_model, self.IMAGE_SIZE, self.STRIDES).eval()

    def test_pipeline_wrapper_output_shapes(self):
        wrapper = self._make_wrapper()
        x = torch.rand(1, 3, *self.IMAGE_SIZE)
        with torch.no_grad():
            confidence, coordinates = wrapper(x)
        assert confidence.shape == (self.NUM_ANCHORS, 80)
        assert coordinates.shape == (self.NUM_ANCHORS, 4)

    def test_pipeline_wrapper_traceable(self):
        wrapper = self._make_wrapper()
        x = torch.rand(1, 3, *self.IMAGE_SIZE)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, x)
        confidence, coordinates = traced(x)
        assert confidence.shape == (self.NUM_ANCHORS, 80)
        assert coordinates.shape == (self.NUM_ANCHORS, 4)

    def test_pipeline_wrapper_confidence_range(self):
        wrapper = self._make_wrapper()
        x = torch.rand(1, 3, *self.IMAGE_SIZE)
        with torch.no_grad():
            confidence, _ = wrapper(x)
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0

    def test_pipeline_wrapper_coordinates_range(self):
        wrapper = self._make_wrapper()
        x = torch.rand(1, 3, *self.IMAGE_SIZE)
        with torch.no_grad():
            _, coordinates = wrapper(x)
        # Coordinates should be roughly in [0, 1] range (relative)
        # Allow some tolerance for edge anchors
        assert coordinates.min() >= -0.5
        assert coordinates.max() <= 1.5


class TestBuildNmsSpec:
    def test_build_nms_spec(self):
        from yolo.utils.deploy_utils import _build_nms_spec

        class_list = ["cat", "dog", "bird"]
        spec = _build_nms_spec(
            num_anchors=8400,
            num_classes=3,
            class_list=class_list,
            iou_threshold=0.45,
            confidence_threshold=0.25,
        )

        nms = spec.nonMaximumSuppression
        assert nms.iouThreshold == 0.45
        assert nms.confidenceThreshold == 0.25
        assert nms.confidenceInputFeatureName == "raw_confidence"
        assert nms.coordinatesInputFeatureName == "raw_coordinates"
        assert nms.confidenceOutputFeatureName == "confidence"
        assert nms.coordinatesOutputFeatureName == "coordinates"
        assert nms.pickTop.perClass is True
        assert list(nms.stringClassLabels.vector) == class_list

        # Verify inputs/outputs exist
        input_names = [inp.name for inp in spec.description.input]
        assert "raw_confidence" in input_names
        assert "raw_coordinates" in input_names
        assert "iouThreshold" in input_names
        assert "confidenceThreshold" in input_names

        output_names = [out.name for out in spec.description.output]
        assert "confidence" in output_names
        assert "coordinates" in output_names


@pytest.mark.skipif(platform.system() != "Darwin", reason="CoreML only supported on macOS")
class TestCoreMLConversion:
    def test_coreml_trace_and_convert(self):
        ct = pytest.importorskip("coremltools")

        mock_model = MockDetectionHead()
        wrapper = CoreMLExportWrapper(mock_model)
        dummy_input = torch.rand(1, 3, 640, 640)

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, dummy_input)

        coreml_model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=(1, 3, 640, 640))],
            outputs=[ct.TensorType(name=f"output_{i}") for i in range(9)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS13,
        )
        assert coreml_model is not None

    def test_pipeline_end_to_end(self, tmp_path):
        """Full pipeline export → load → verify structure."""
        ct = pytest.importorskip("coremltools")
        from yolo.utils.deploy_utils import _build_nms_spec

        mock_model = MockDetectionHead()
        image_size = (640, 640)
        strides = [8, 16, 32]
        num_classes = 80
        H, W = image_size
        num_anchors = sum((W // s) * (H // s) for s in strides)
        class_list = [f"class_{i}" for i in range(num_classes)]

        wrapper = CoreMLPipelineExportWrapper(mock_model, image_size, strides).eval()
        dummy_input = torch.rand(1, 3, *image_size)

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, dummy_input)

        detector_model = ct.convert(
            traced,
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
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS16,
        )

        nms_spec = _build_nms_spec(num_anchors, num_classes, class_list)

        input_features = [
            ("image", ct.models.datatypes.Array(3, *image_size)),
            ("iouThreshold", ct.models.datatypes.Double()),
            ("confidenceThreshold", ct.models.datatypes.Double()),
        ]
        output_features = [
            ("confidence", ct.models.datatypes.Array(num_classes)),
            ("coordinates", ct.models.datatypes.Array(4)),
        ]

        pipeline = ct.models.pipeline.Pipeline(input_features, output_features)
        pipeline.add_model(detector_model._spec)
        pipeline.add_model(nms_spec)

        pipeline_spec = pipeline.spec

        # Patch image input
        image_input = pipeline_spec.description.input[0]
        image_input.type.ClearField("multiArrayType")
        image_type = image_input.type.imageType
        image_type.width = W
        image_type.height = H
        image_type.colorSpace = ct.proto.FeatureTypes_pb2.ImageFeatureType.RGB

        for inp in pipeline_spec.description.input:
            if inp.name in ("iouThreshold", "confidenceThreshold"):
                inp.type.isOptional = True

        output_path = str(tmp_path / "test_pipeline.mlpackage")
        ct.models.MLModel(pipeline_spec).save(output_path)

        # Reload and verify
        loaded = ct.models.MLModel(output_path)
        spec = loaded.get_spec()
        assert spec.WhichOneof("Type") == "pipeline"
        assert len(spec.pipeline.models) == 2
