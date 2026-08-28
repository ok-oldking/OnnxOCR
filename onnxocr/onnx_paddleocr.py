import logging
import time

from .predict_system import TextSystem
from .utils import infer_args as init_args
from .utils import str2bool, draw_ocr
import argparse
import sys


class ONNXPaddleOcr(TextSystem):
    def __init__(self, logger=None, **kwargs):
        # Init logger
        if logger is None:
            logger = logging.getLogger('onnxocr')
            if not logger.handlers:
                logger.setLevel(logging.INFO)
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        self.logger = logger

        # 默认参数
        parser = init_args()
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        params = argparse.Namespace(**inference_args_dict)

        # params.rec_image_shape = "3, 32, 320"
        params.rec_image_shape = "3, 48, 320"

        # 根据传入的参数覆盖更新默认参数
        params.__dict__.update(**kwargs)

        # Pass logger through params for sub-components
        params.logger = logger

        logger.info(f'use_openvino: {params.use_openvino}')
        logger.info(f'use_npu: {getattr(params, "use_npu", False)}')
        logger.info(f'use_angle_cls: {params.use_angle_cls}')
        logger.info(f'det_model_dir: {params.det_model_dir}')
        logger.info(f'rec_model_dir: {params.rec_model_dir}')
        logger.info(f'rec_image_shape: {params.rec_image_shape}')
        if params.use_angle_cls:
            logger.info(f'cls_model_dir: {params.cls_model_dir}')

        start = time.time()
        if getattr(params, "use_npu", False):
            test_frame = self._npu_test_frame()
            try:
                self._initialize_models(params)
                test_result = self.ocr(test_frame)
                init_result = f'use_npu=True, test_ocr={test_result!r}'
            except Exception as error:
                logger.warning(
                    f'NPU OCR test failed, falling back to CPU: {error}'
                )
                params.use_npu = False
                self._initialize_models(params)
                init_result = f'use_npu=False (NPU test failed: {error!r})'
        else:
            self._initialize_models(params)
            init_result = 'use_npu=False'

        logger.info(
            f'onnxocr init finished, result: {init_result}, '
            f'cost: {time.time() - start:.2f}s'
        )

    def _initialize_models(self, params):
        super().__init__(params)

    @staticmethod
    def _npu_test_frame():
        import cv2
        import numpy as np

        frame = np.full((160, 640, 3), 255, dtype=np.uint8)
        cv2.putText(
            frame,
            'okscript',
            (24, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        return frame

    def ocr(self, img, det=True, rec=True, cls=True):
        # if cls == True and self.use_angle_cls == False:
        #     print(
        #         "Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process"
        #     )

        if det and rec:
            ocr_res = []
            dt_boxes, rec_res = self.__call__(img, cls)
            tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            dt_boxes = self.text_detector(img)
            tmp_res = [box.tolist() for box in dt_boxes]
            ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []

            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res_tmp = self.text_classifier(img)
                if not rec:
                    cls_res.append(cls_res_tmp)
            rec_res = self.text_recognizer(img)
            ocr_res.append(rec_res)

            if not rec:
                return cls_res
            return ocr_res


def sav2Img(org_img, result, name="draw_ocr.jpg"):
    # 显示结果
    from PIL import Image

    result = result[0]
    # image = Image.open(img_path).convert('RGB')
    # 图像转BGR2RGB
    image = org_img[:, :, ::-1]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores)
    im_show = Image.fromarray(im_show)
    im_show.save(name)


if __name__ == "__main__":
    import cv2

    model = ONNXPaddleOcr(use_angle_cls=True)

    img = cv2.imread(
        "/data2/liujingsong3/fiber_box/test/img/20230531230052008263304.jpg"
    )
    s = time.time()
    result = model.ocr(img)
    e = time.time()
    print("total time: {:.3f}".format(e - s))
    print("result:", result)
    for box in result[0]:
        print(box)

    sav2Img(img, result)
