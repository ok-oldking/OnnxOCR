from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("openvino")

from onnxocr.onnx_paddleocr import ONNXPaddleOcr


pytestmark = pytest.mark.integration
LONG_TEXT_IMAGE = (
    Path(__file__).resolve().parents[2] / "onnxocr" / "test_images" / "long_text.png"
)


def test_openvino_handles_long_text_image():
    image = cv2.imread(str(LONG_TEXT_IMAGE))
    assert image is not None

    result = ONNXPaddleOcr(
        use_angle_cls=True, use_dml=False, use_openvino=True
    ).ocr(image)

    assert result
    assert result[0]
