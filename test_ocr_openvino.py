import cv2
import time
from onnxocr.onnx_paddleocr import ONNXPaddleOcr,sav2Img
import sys
import time
#固定到onnx路径·
# sys.path.append('./paddle_to_onnx/onnx')

model = ONNXPaddleOcr(use_angle_cls=True, use_dml=False, use_openvino=True)


img = cv2.imread('./onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg')
s = time.time()
result = model.ocr(img)
e = time.time()

print("result:", result)
for box in result[0]:
    print(box)

print("total time: {:.3f} result count {}".format(e - s, len(result[0])))



img = cv2.imread('./onnxocr/test_images/long_text.png')
s = time.time()
result = model.ocr(img)
e = time.time()


print("long_text total time: {:.3f} result count {}".format(e - s, len(result[0])))

# sav2Img(img, result,name=str(time.time())+'.jpg')