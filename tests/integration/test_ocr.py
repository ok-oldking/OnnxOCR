from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from onnxocr.onnx_paddleocr import ONNXPaddleOcr


pytestmark = pytest.mark.integration
SAMPLE_IMAGE = (
    Path(__file__).resolve().parents[2]
    / "onnxocr"
    / "test_images"
    / "715873facf064583b44ef28295126fa7.jpg"
)


@pytest.mark.parametrize(
    ("runtime", "options"),
    [
        ("onnxruntime", {"use_angle_cls": True, "use_openvino": False}),
        ("openvino", {"use_angle_cls": True, "use_openvino": True, "use_dml": False}),
    ],
)
def test_ocr_extracts_text_from_sample_image(runtime, options):
    pytest.importorskip(runtime)
    image = cv2.imread(str(SAMPLE_IMAGE))
    assert image is not None

    result = ONNXPaddleOcr(**options).ocr(image)

    assert result
    assert result[0]
