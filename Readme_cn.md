# OnnxOCR PP-OCRv5

基于 ONNX 模型的 PP-OCRv5 文字检测与识别库，支持分别选择 ONNX Runtime
和 OpenVINO 推理后端。本项目 fork 自
[jingsongliujing/OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)，并以
`onnxocr-ppocrv5` 的名称发布到 PyPI。

安装包内包含 PP-OCRv5 的文字检测、方向分类、文字识别模型以及字符字典。
本项目只负责模型推理，不包含训练和模型转换流程。

## 功能特性

- **支持两种推理后端：** 同一套 PP-OCRv5 流程可分别使用 ONNX Runtime 或
  OpenVINO 执行。
- **Intel NPU 加速：** OpenVINO 可使用兼容的 Intel NPU；当 NPU 或驱动不可用时，
  会回退到 OpenVINO CPU。
- **CPU 推理：** 同时支持 ONNX Runtime CPU 和 OpenVINO CPU，无需专用加速硬件。
- **完整 OCR 流程：** 内置模型可完成文字检测、可选的文字方向分类和文字识别。
- **按用途分组安装：** 通过独立的 `onnx`、`openvino`、`dev` 和 `build` 依赖组，
  只需安装实际使用的推理引擎或工具。
- **优化模型缓存：** 会复用 ONNX Runtime 优化模型和 OpenVINO 缓存，减少重复的
  初始化工作。
- **标注结果输出：** 可使用内置 `sav2Img` 辅助函数，将文字框和识别文本绘制到
  输出图像。

## 环境要求

- Python 3.8 或更高版本
- 至少安装一个推理后端：ONNX Runtime 或 OpenVINO
- 输入图像为 NumPy 数组，通常使用 OpenCV 读取
- 如需使用 OpenVINO NPU，还需要兼容的 Intel NPU 硬件和驱动

基础安装会自动安装 `pyclipper`、`shapely` 和 Pillow。NumPy 和 OpenCV
不作为强制包依赖，以避免强制应用使用特定的 NumPy 版本或 OpenCV 发行包。
使用 OCR 前，请根据运行环境自行安装兼容的 NumPy 和 OpenCV。

推理引擎体积较大，因此在包配置中拆分为两个独立的可选依赖组。

## 依赖组

`pyproject.toml` 中定义了下列相互独立的可选依赖组：

| 依赖组 | 安装的包 | 用途 |
| --- | --- | --- |
| 基础依赖 | `pyclipper`、`shapely`、Pillow | OCR 预处理和后处理 |
| `onnx` | `onnxruntime` | ONNX Runtime 推理 |
| `openvino` | `openvino` | OpenVINO CPU 或 Intel NPU 推理 |
| `dev` | NumPy、无界面版 OpenCV、`onnxruntime`、`openvino`、`pytest` | 同时覆盖两个后端的完整开发和测试环境 |
| `build` | `build`、`twine` | 构建和验证发布文件 |

`dev` 依赖组已经包含两个推理后端和测试套件需要的所有 Python 包。
如果同一环境还需要构建发布包，再组合 `build` 依赖组：

```bash
python -m pip install -e ".[dev,build]"
```

普通安装中的 NumPy 和 OpenCV 仍由应用环境管理。它们只在 `dev` 中以
无版本限制的方式安装，并选用无界面版 OpenCV，以便本地和 CI 测试环境保持一致。

## 安装

建议先升级 pip，以便正确解析当前 Python 版本对应的 wheel：

```bash
python -m pip install --upgrade pip
```

如果应用环境尚未提供 NumPy 和 OpenCV，请先安装。服务器和容器环境通常适合使用
无界面版 OpenCV：

```bash
python -m pip install numpy opencv-python-headless
```

桌面应用可以选择 `opencv-python`；如果需要 OpenCV 扩展模块，可以选择
`opencv-contrib-python`。不要在同一环境中同时安装多个 OpenCV 发行包，
因为它们都会提供 `cv2` 模块。

### ONNX Runtime

安装 `onnx` 依赖组，使用 ONNX Runtime 进行 CPU 推理：

```bash
python -m pip install "onnxocr-ppocrv5[onnx]"
```

创建 OCR 实例时设置 `use_openvino=False`。这也是当前的默认值，但显式指定
可以让后端选择更清晰：

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

安装 `openvino` 依赖组：

```bash
python -m pip install "onnxocr-ppocrv5[openvino]"
```

开启 OpenVINO 并关闭 NPU 探测，即可固定使用 CPU：

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

Intel NPU 执行同样使用 `openvino` 依赖组。设置 `use_npu=True` 后，程序会探测
并尝试使用 NPU：

```python
ocr = ONNXPaddleOcr(
    use_openvino=True,
    use_npu=True,
    use_angle_cls=True,
)
```

程序会检查 NPU 驱动并优先尝试 NPU。如果没有可用的 NPU，会回退到 OpenVINO
CPU。启用 `use_angle_cls=True` 后，除检测和识别模型之外，还会加载内置的方向
分类模型。

### 同时安装两个后端

如果需要比较两个后端，或运行完整的 CPU 集成测试，可同时安装：

```bash
python -m pip install "onnxocr-ppocrv5[onnx,openvino]"
```

如果不指定可选依赖组，`onnxocr-ppocrv5` 只会安装声明的后处理依赖；
在推理后端、NumPy 和任一 OpenCV 发行包可用之前，无法执行模型推理。

## 返回结果和保存标注图像

`ocr.ocr(image)` 针对输入图像返回一个结果列表。每行文字包含四点多边形坐标，
以及 `(text, confidence)` 形式的识别文本和置信度：

```text
[
    [
        [box_points, (recognized_text, confidence)],
        ...
    ]
]
```

使用 `sav2Img` 可将文字框和识别文本保存到新图像：

```python
from onnxocr.onnx_paddleocr import sav2Img

sav2Img(image, result, "ocr-result.jpg")
```

## 开发环境

克隆仓库后，创建虚拟环境，并以可编辑模式安装项目、开发工具和所需的推理后端。

PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux 或 macOS：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

该命令会同时安装两个支持的推理后端和运行测试套件所需的全部依赖。
也可使用等价的 requirements 文件命令：

```bash
python -m pip install -r requirements-dev.txt
```

只有需要构建发布包时，才添加独立的 `build` 依赖组：

```bash
python -m pip install -e ".[dev,build]"
```

## 测试

默认命令只运行快速单元测试。`pyproject.toml` 中的配置会默认排除集成测试：

```bash
pytest
```

使用开发依赖组可运行 CPU 集成测试：

```bash
python -m pip install -e ".[dev]"
pytest -m "integration and not npu"
```

只应在已配置 NPU 的机器上运行硬件相关的 OpenVINO NPU 测试：

```bash
python -m pip install -e ".[dev]"
pytest -m "integration and npu"
```

集成测试被跳过通常表示对应的推理后端未安装。如果缺少所需的 NPU 硬件或驱动，
NPU 测试也可能失败或回退到 CPU。

## 构建发布包

`build` 依赖组与推理后端相互独立，因为构建发布包不会执行模型推理。
安装该依赖组后，构建 wheel 和源码包，并检查包元数据：

```bash
python -m pip install -e ".[build]"
python -m build
python -m twine check dist/*
```

如果需要在同一环境中进行开发、测试和构建，可组合所有相关依赖组：

```bash
python -m pip install -e ".[dev,build]"
```

`onnx` 和 `openvino` 可选依赖组会写入 wheel 元数据，但推理后端本身不会打包到
wheel 文件内。
