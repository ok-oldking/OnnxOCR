# OnnxOCR PP-OCRv5

ONNX-based PP-OCRv5 text detection and recognition with selectable ONNX
Runtime and OpenVINO inference backends. This project is a fork of
[jingsongliujing/OnnxOCR](https://github.com/jingsongliujing/OnnxOCR) and is
published on PyPI as `onnxocr-ppocrv5`.

The package includes the PP-OCRv5 detection, angle-classification, recognition,
and character-dictionary assets. It performs inference only; training and model
conversion are outside the scope of this package.

For Chinese documentation, see [Readme_cn.md](Readme_cn.md).

## Features

- **Two inference backends:** run the same PP-OCRv5 pipeline with either ONNX
  Runtime or OpenVINO.
- **Intel NPU acceleration:** OpenVINO can use a compatible Intel NPU and falls
  back to OpenVINO CPU when the NPU or its driver is unavailable.
- **CPU inference:** ONNX Runtime CPU and OpenVINO CPU are both supported without
  requiring specialized accelerator hardware.
- **Complete OCR pipeline:** bundled models provide text detection, optional
  angle classification, and text recognition.
- **Purpose-specific installations:** separate `onnx`, `openvino`, `dev`, and
  `build` profiles let users install only the backend or tooling they need.
- **Optimized model caching:** optimized ONNX Runtime models and OpenVINO cache
  data are reused across runs to reduce repeated initialization work.
- **Annotated output:** detected boxes and recognized text can be rendered to an
  output image with the included `sav2Img` helper.

## Requirements

- Python 3.8 or newer
- One inference backend: ONNX Runtime or OpenVINO
- An image represented as a NumPy array, normally loaded with OpenCV
- OpenVINO NPU execution additionally requires compatible Intel NPU hardware
  and drivers

`pyclipper`, `shapely`, and Pillow are installed automatically. NumPy and OpenCV
are intentionally not declared as mandatory package dependencies. This avoids
forcing a particular NumPy version or OpenCV distribution on applications that
already manage those packages. Install compatible NumPy and OpenCV packages for
your environment before using OCR.

The inference engines are separate optional dependency profiles so an
installation does not need to carry both large runtimes.

## Dependency profiles

The package defines independent optional dependency profiles in
`pyproject.toml`:

| Profile | Installed packages | Intended use |
| --- | --- | --- |
| Base | `pyclipper`, `shapely`, Pillow | OCR preprocessing and post-processing |
| `onnx` | `onnxruntime` | ONNX Runtime inference |
| `openvino` | `openvino` | OpenVINO CPU or Intel NPU inference |
| `dev` | NumPy, OpenCV headless, `onnxruntime`, `openvino`, `pytest` | Complete development and test environment for both backends |
| `build` | `build`, `twine` | Building and validating release artifacts |

The `dev` profile already includes both inference backends and all Python
packages required by the test suite. Add `build` when the same environment also
needs to create release artifacts:

```bash
python -m pip install -e ".[dev,build]"
```

NumPy and OpenCV remain application-managed for normal package installations.
They are included unpinned in `dev` only, with headless OpenCV selected for
consistent local and CI test environments.

## Installation

Upgrade pip first so it can resolve wheels for the selected Python version:

```bash
python -m pip install --upgrade pip
```

Install NumPy and one OpenCV distribution if the application does not already
provide them. Headless OpenCV is appropriate for servers and containers:

```bash
python -m pip install numpy opencv-python-headless
```

Desktop applications may use `opencv-python`, while applications that need the
additional OpenCV modules may use `opencv-contrib-python`. Avoid installing
multiple OpenCV distributions in the same environment because they all provide
the `cv2` module.

### ONNX Runtime

Install the `onnx` profile for CPU inference with ONNX Runtime:

```bash
python -m pip install "onnxocr-ppocrv5[onnx]"
```

Use `use_openvino=False` when constructing the OCR pipeline. This is also the
current default, but setting it explicitly makes the chosen backend clear:

```python
import cv2

from onnxocr.onnx_paddleocr import ONNXPaddleOcr

image = cv2.imread("image.jpg")
if image is None:
    raise FileNotFoundError("image.jpg")

ocr = ONNXPaddleOcr(
    use_openvino=False,
    use_angle_cls=False,
)
result = ocr.ocr(image)
print(result)
```

### OpenVINO CPU

Install the `openvino` profile:

```bash
python -m pip install "onnxocr-ppocrv5[openvino]"
```

Select OpenVINO and disable the NPU probe to run on CPU only:

```python
import cv2

from onnxocr.onnx_paddleocr import ONNXPaddleOcr

image = cv2.imread("image.jpg")
if image is None:
    raise FileNotFoundError("image.jpg")

ocr = ONNXPaddleOcr(
    use_openvino=True,
    use_npu=False,
    use_angle_cls=False,
)
result = ocr.ocr(image)
print(result)
```

### OpenVINO NPU

The same `openvino` profile supports Intel NPU execution. Enable the NPU probe
with `use_npu=True`:

```python
ocr = ONNXPaddleOcr(
    use_openvino=True,
    use_npu=True,
    use_angle_cls=True,
)
```

The pipeline validates the NPU driver, initializes the NPU models, and runs an
OCR smoke test against an OpenCV-generated `okscript` image. If NPU setup or
the first inference fails, it reinitializes the complete pipeline with
`use_npu=False`. The initialization completion log records the smoke-test or
fallback result. Enabling angle classification loads the bundled classification
model in addition to detection and recognition.

### Install both backends

Install both profiles when comparing backends or running the full CPU integration
suite:

```bash
python -m pip install "onnxocr-ppocrv5[onnx,openvino]"
```

Installing `onnxocr-ppocrv5` without an extra installs the declared
post-processing dependencies, but it cannot perform model inference until an
inference backend, NumPy, and an OpenCV distribution are available.

## Result format and saving an annotated image

`ocr.ocr(image)` returns one result list for the input image. Each detected line
contains its four-point polygon followed by `(text, confidence)`:

```text
[
    [
        [box_points, (recognized_text, confidence)],
        ...
    ]
]
```

Use `sav2Img` to write a copy with boxes and recognized text:

```python
from onnxocr.onnx_paddleocr import sav2Img

sav2Img(image, result, "ocr-result.jpg")
```

## Development

Clone the repository, create a virtual environment, and install the project in
editable mode with the development tools and the backend you need.

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs both supported inference backends and everything needed to run the
test suite. The equivalent requirements-file command is:

```bash
python -m pip install -r requirements-dev.txt
```

Add the independent `build` profile only when packaging distributions:

```bash
python -m pip install -e ".[dev,build]"
```

## Testing

The default command runs only the fast unit tests. Integration tests are excluded
by the configuration in `pyproject.toml`:

```bash
pytest
```

Run the CPU integration tests with the development profile:

```bash
python -m pip install -e ".[dev]"
pytest -m "integration and not npu"
```

Run the hardware-specific OpenVINO NPU tests only on a configured NPU machine:

```bash
python -m pip install -e ".[dev]"
pytest -m "integration and npu"
```

A skipped integration test normally means its backend is not installed. An NPU
test can also fail or fall back to CPU when the required hardware or driver is
unavailable.

## Building distributions

The `build` profile is independent of the inference backends because building a
distribution does not run model inference. Install it, build both distribution
formats, and validate their metadata:

```bash
python -m pip install -e ".[build]"
python -m build
python -m twine check dist/*
```

For development, tests, and packaging in one environment, combine the profiles:

```bash
python -m pip install -e ".[dev,build]"
```

The `onnx` and `openvino` extras are recorded in wheel metadata; neither backend
is bundled inside the wheel.
