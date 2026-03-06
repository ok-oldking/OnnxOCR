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
