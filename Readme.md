### 这是 [https://github.com/jingsongliujing/OnnxOCR](https://github.com/jingsongliujing/OnnxOCR) 的fork 

* 发布了到PIP上
* 仅依赖无版本限制的 pyclipper, shapely, pillow
* 仅打包了"models/ppocrv5/det/det.onnx" "models/ppocrv5/rec/rec.onnx"

```
pip install onnxocr-ppocrv5
```


```python
model = ONNXPaddleOcr(use_angle_cls=False)
```

### 测试

```bash
pip install -r requirements-dev.txt
pytest
pytest -m "integration and not npu"
pytest -m "integration and npu"
```

默认的 `pytest` 命令只运行快速单元测试。集成测试会加载 OCR 模型和推理后端，
其中 `npu` 标记用于显式验证 OpenVINO NPU 路径。
