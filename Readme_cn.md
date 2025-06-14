### 这是 [https://github.com/jingsongliujing/OnnxOCR](https://github.com/jingsongliujing/OnnxOCR) 的fork 

* 发布了到PIP上
* 支持dml
* 仅依赖无版本限制的 pyclipper, shapely, pillow
* onnx可以根据自己需要安装gpu或directml
* 仅打包了"models/ppocrv5/det/det.onnx" "models/ppocrv5/rec/rec.onnx", 没有cls模型

```
pip install onnxocr-ppocrv5
pip install onnxocr-ppocrv4
```
