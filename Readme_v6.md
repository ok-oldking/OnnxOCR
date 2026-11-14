# OnnxOCR PP-OCRv6 small

ONNX-based text detection and recognition with the official PP-OCRv6 **small**
models and selectable ONNX Runtime or OpenVINO inference. The models and their
character dictionary are included in the package; inference needs no model
download or PaddlePaddle installation. Models are licensed under Apache-2.0.

This is the v6 distribution of [ok-oldking/OnnxOCR](https://github.com/ok-oldking/OnnxOCR),
a fork of [jingsongliujing/OnnxOCR](https://github.com/jingsongliujing/OnnxOCR).

## Install

Use one OnnxOCR model distribution per environment: `onnxocr-ppocrv5` and
`onnxocr-ppocrv6` both install the `onnxocr` import package. To migrate an
existing environment, uninstall v5 before installing v6:

```bash
python -m pip uninstall onnxocr-ppocrv5
python -m pip install numpy opencv-python-headless "onnxocr-ppocrv6[onnx]"
```

Alternatively, install `"onnxocr-ppocrv6[openvino]"` for OpenVINO CPU or a
compatible Intel NPU. NumPy and OpenCV are application-managed dependencies,
matching the v5 distribution. Both inference backends are optional; select at
least one. Python 3.8+ is supported, subject to backend wheel availability.

## Use

```python
import cv2
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

ocr = ONNXPaddleOcr(use_openvino=False)
result = ocr.ocr(cv2.imread("image.png"))
print(result)
```

The default detector and recognizer are PP-OCRv6 small, with its matching
18,708-entry character dictionary. The decoder adds CTC blank and space.
Enable OpenVINO with `use_openvino=True` and optional NPU selection with
`use_npu=True`. The optional `use_angle_cls=True` reuses the bundled v5-era
0/180-degree text-line classifier; the detector and recognizer remain v6 small.
The existing inference thresholds and preprocessing are retained.

The wheel contains only the small detector/recognizer, dictionary, provenance
manifest, inference configurations, optional classifier, and rendering font.
V6 tiny/medium and the v5 detector/recognizer are excluded. The API and OCR
result format are the same as v5.

## Build and release

Building requires Python 3.11+; CI uses Python 3.12.

```bash
python -m pip install -e ".[build]"
python scripts/download_ppocrv6.py --tiers small
python scripts/build_ppocrv6.py
python scripts/check_ppocrv6.py
python -m twine check dist/ppocrv6/*
```

The independent version is in `packaging/ppocrv6.toml`. The build script stages
the shared source and selected assets without rewriting the v5 project.
Both a source distribution and a wheel are emitted into `dist/ppocrv6/`.

To release version `0.0.1`, push tag `ppocrv6-v0.0.1`. The
`.github/workflows/publish-v6.yml` workflow checks the tag against the v6
profile, downloads pinned official small models, builds and validates the
distributions, installs and tests the wheel, and publishes through PyPI
Trusted Publishing. A manual workflow dispatch builds and validates artifacts
without publishing.

CI smoke-tests both CPU backends. To run those checks locally, install the
development dependencies (`python -m pip install -e ".[dev,build]"`) and run
`python scripts/check_ppocrv6.py --smoke --backends onnxruntime openvino`.

Before the first release, register a PyPI pending publisher with:

| Setting | Value |
| --- | --- |
| Project | `onnxocr-ppocrv6` |
| GitHub owner | `ok-oldking` |
| Repository | `OnnxOCR` |
| Workflow filename | `publish-v6.yml` |
| Environment | `pypi` |

See [PyPI's new-project Trusted Publishing instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
