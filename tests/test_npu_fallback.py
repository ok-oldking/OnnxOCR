import numpy as np

from onnxocr.onnx_paddleocr import ONNXPaddleOcr


class Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(('info', message))

    def warning(self, message):
        self.messages.append(('warning', message))


def stub_model_initialization(monkeypatch):
    init_calls = []

    def initialize(self, params):
        init_calls.append(params.use_npu)
        self.args = params

    monkeypatch.setattr(ONNXPaddleOcr, '_initialize_models', initialize)
    return init_calls


def test_npu_initialization_runs_ocr_smoke_test(monkeypatch):
    init_calls = stub_model_initialization(monkeypatch)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    ocr_calls = []
    logger = Logger()

    monkeypatch.setattr(
        ONNXPaddleOcr, '_npu_test_frame', staticmethod(lambda: frame)
    )

    def ocr(self, image, **kwargs):
        ocr_calls.append(image)
        return [[['box', ('okscript', 0.99)]]]

    monkeypatch.setattr(ONNXPaddleOcr, 'ocr', ocr)

    model = ONNXPaddleOcr(logger=logger, use_npu=True)

    assert init_calls == [True]
    assert len(ocr_calls) == 1
    assert ocr_calls[0] is frame
    assert model.args.use_npu is True
    assert any(
        level == 'info'
        and 'onnxocr init finished' in message
        and 'use_npu=True' in message
        and 'okscript' in message
        for level, message in logger.messages
    )


def test_failed_npu_smoke_test_reinitializes_without_npu(monkeypatch):
    init_calls = stub_model_initialization(monkeypatch)
    logger = Logger()
    monkeypatch.setattr(
        ONNXPaddleOcr,
        '_npu_test_frame',
        staticmethod(lambda: np.zeros((8, 12, 3), dtype=np.uint8)),
    )

    def failing_ocr(self, image, **kwargs):
        raise RuntimeError('NPU unavailable')

    monkeypatch.setattr(ONNXPaddleOcr, 'ocr', failing_ocr)

    model = ONNXPaddleOcr(logger=logger, use_npu=True)

    assert init_calls == [True, False]
    assert model.args.use_npu is False
    assert any(
        level == 'warning'
        and 'falling back to CPU' in message
        and 'NPU unavailable' in message
        for level, message in logger.messages
    )
    assert any(
        level == 'info'
        and 'onnxocr init finished' in message
        and 'use_npu=False' in message
        and 'NPU unavailable' in message
        for level, message in logger.messages
    )


def test_npu_disabled_skips_smoke_test(monkeypatch):
    init_calls = stub_model_initialization(monkeypatch)
    logger = Logger()
    ocr_calls = []
    monkeypatch.setattr(
        ONNXPaddleOcr,
        'ocr',
        lambda self, image, **kwargs: ocr_calls.append(image),
    )

    ONNXPaddleOcr(logger=logger, use_npu=False)

    assert init_calls == [False]
    assert ocr_calls == []
    assert any(
        level == 'info'
        and 'onnxocr init finished' in message
        and 'use_npu=False' in message
        for level, message in logger.messages
    )


def test_npu_test_frame_contains_rendered_text():
    frame = ONNXPaddleOcr._npu_test_frame()

    assert frame.shape == (160, 640, 3)
    assert frame.dtype == np.uint8
    assert frame.min() < 255
