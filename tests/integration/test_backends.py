from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from onnxocr.onnx_paddleocr import ONNXPaddleOcr


pytestmark = pytest.mark.integration
IMAGE_DIR = Path(__file__).resolve().parents[2] / "onnxocr" / "test_images"
IMAGE_PATHS = sorted(
    path
    for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".png"}
)


def get_texts(result):
    if not result or not result[0]:
        return []
    return [
        line[1][0]
        for line in result[0]
        if len(line) >= 2 and isinstance(line[1], tuple)
    ]


@pytest.fixture(scope="module")
def onnx_model():
    pytest.importorskip("onnxruntime")
    return ONNXPaddleOcr(
        use_openvino=False, force_static_shape=True, det_limit_side_len=960
    )


@pytest.fixture(scope="module")
def openvino_cpu_model():
    pytest.importorskip("openvino")
    return ONNXPaddleOcr(
        use_openvino=True,
        use_npu=False,
        force_static_shape=True,
        det_limit_side_len=960,
    )


@pytest.fixture(scope="module")
def openvino_npu_model():
    pytest.importorskip("openvino")
    return ONNXPaddleOcr(use_openvino=True, use_npu=True)


@pytest.mark.parametrize("image_path", IMAGE_PATHS, ids=lambda path: path.name)
def test_cpu_backends_extract_text(image_path, onnx_model, openvino_cpu_model):
    image = cv2.imread(str(image_path))
    assert image is not None

    onnx_texts = get_texts(onnx_model.ocr(image))
    openvino_texts = get_texts(openvino_cpu_model.ocr(image))

    assert onnx_texts
    assert openvino_texts


@pytest.mark.npu
@pytest.mark.parametrize("image_path", IMAGE_PATHS, ids=lambda path: path.name)
def test_openvino_npu_extracts_text(image_path, openvino_npu_model):
    image = cv2.imread(str(image_path))
    assert image is not None

    texts = get_texts(openvino_npu_model.ocr(image))

    assert texts
