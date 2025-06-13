import cv2
import time
from onnxocr.onnx_paddleocr import ONNXPaddleOcr,sav2Img
import sys
import time
#固定到onnx路径·
# sys.path.append('./paddle_to_onnx/onnx')

model = ONNXPaddleOcr(use_angle_cls=False, use_gpu=False, use_dml=True)


img = cv2.imread('./onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg')
model.ocr(img) #warm up
s = time.time()
result = model.ocr(img)
e = time.time()


for box in result[0]:
    print(box)

print("result:", len(result[0]))
print("total time: {:.3f}".format(e - s))

sav2Img(img, result,name=str(time.time())+'.jpg')