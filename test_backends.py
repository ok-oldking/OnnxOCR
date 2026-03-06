import cv2
import glob
import os
import time
import numpy as np
import shutil
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img

def get_texts(result):
    if not result or not result[0]:
         return []
    texts = []
    for line in result[0]:
        if len(line) >= 2 and isinstance(line[1], tuple):
            texts.append(line[1][0])
    return sorted(texts)

def main():
    save_dir = "result_img"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    print("Initializing ONNX (CPU) model...")
    model_onnx = ONNXPaddleOcr(use_openvino=False, force_static_shape=True, det_limit_side_len=960)

    print("Initializing OpenVINO (CPU) model...")
    model_ov_cpu = ONNXPaddleOcr(use_openvino=True, use_npu=False, force_static_shape=True, det_limit_side_len=960)

    try:
        print("Initializing OpenVINO (NPU) model...")
        model_ov_npu = ONNXPaddleOcr(use_openvino=True, use_npu=True)
    except Exception as e:
        print("NPU init skipped or failed:", e)
        model_ov_npu = None

    test_images = glob.glob('onnxocr/test_images/*.*')
    valid_images = [f for f in test_images if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"\nFound {len(valid_images)} test images. Running tests...")
    
    total_images = len(valid_images)
    passed_images = 0

    print("-" * 60)
    for img_path in valid_images:
        img_name = os.path.basename(img_path)
        img_base_name = os.path.splitext(img_name)[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        print(f"Testing {img_name} ({img.shape})... ", end="")
        
        try:
            res_onnx = model_onnx.ocr(img)
            texts_onnx = get_texts(res_onnx)
            sav2Img(img, res_onnx, os.path.join(save_dir, f"{img_base_name}_ONNX.jpg"))
            
            res_ov_cpu = model_ov_cpu.ocr(img)
            texts_ov_cpu = get_texts(res_ov_cpu)
            sav2Img(img, res_ov_cpu, os.path.join(save_dir, f"{img_base_name}_OV_CPU.jpg"))
            
            results = {
                "ONNX": texts_onnx,
                "OV_CPU": texts_ov_cpu,
                "ONNX_BOX_COUNT": len(res_onnx[0]) if res_onnx and res_onnx[0] else 0,
                "OV_CPU_BOX_COUNT": len(res_ov_cpu[0]) if res_ov_cpu and res_ov_cpu[0] else 0,
            }
            
            if model_ov_npu is not None:
                res_ov_npu = model_ov_npu.ocr(img)
                results["OV_NPU"] = get_texts(res_ov_npu)
                results["OV_NPU_BOX_COUNT"] = len(res_ov_npu[0]) if res_ov_npu and res_ov_npu[0] else 0
                sav2Img(img, res_ov_npu, os.path.join(save_dir, f"{img_base_name}_OV_NPU.jpg"))
            
            # Compare lengths and contents
            mismatches = []
            base_texts = results["ONNX"]
            base_len = len(base_texts)
            
            for backend, texts in results.items():
                if backend == "ONNX" or "BOX_COUNT" in backend: continue
                
                if len(texts) != base_len:
                    mismatches.append(f"{backend} count ({len(texts)}) != ONNX count ({base_len})")
                else:
                    diff = [t for t in texts if t not in base_texts]
                    if diff:
                        mismatches.append(f"{backend} has different bounding box texts parsed vs ONNX")
            
            if not mismatches:
                print(f"OK (Extracted {base_len} texts)")
                passed_images += 1
            else:
                print("MISMATCH")
                for err in mismatches:
                    print(f"   -> {err}")
                    
        except Exception as e:
            print("ERROR during inference:", e)

    print("-" * 60)
    print(f"Test Complete: {passed_images}/{total_images} images produced absolutely identical text extraction results across all tested backends.")

if __name__ == "__main__":
    main()
