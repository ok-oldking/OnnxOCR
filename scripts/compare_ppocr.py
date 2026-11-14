"""Compare bundled v5 with downloaded v6 through the existing OCR pipeline.

Run from an environment containing this project, OpenCV, NumPy, Pillow, and
the selected backend. Download v6 first with download_ppocrv6.py.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import statistics
import sys
import time
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from onnxocr.onnx_paddleocr import ONNXPaddleOcr


def model_options(name):
    if name == "v5":
        folder = ROOT / "onnxocr/models/ppocrv5"
        dictionary = folder / "ppocrv5_dict.txt"
    else:
        folder = ROOT / "onnxocr/models/ppocrv6" / name[3:]
        dictionary = folder / "ppocrv6_dict.txt"
    return {
        "det_model_dir": str(folder / "det/det.onnx"),
        "rec_model_dir": str(folder / "rec/rec.onnx"),
        "rec_char_dict_path": str(dictionary),
    }


def normalized(text):
    # Report recognition excluding whitespace, with Unicode NFC normalization.
    return "".join(unicodedata.normalize("NFC", text).split())


def edit_distance(expected, actual):
    row = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        following = [i]
        for j, right in enumerate(actual, 1):
            following.append(min(following[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = following
    return row[-1]


def synthetic_cases(output):
    """Controlled smoke set, not a representative real-world accuracy dataset."""
    latin = Path("C:/Windows/Fonts/arial.ttf")
    japanese = Path("C:/Windows/Fonts/msgothic.ttc")
    chinese = ROOT / "onnxocr/fonts/simfang.ttf"
    texts = {
        "en": ["PaddleOCR version 6", "Invoice TOTAL 1234.56", "Serial AB0123456789", "OpenVINO CPU benchmark"],
        "zh": ["新一代文字识别模型", "订单编号：20261114001", "中华人民共和国", "总金额：1234.56元"],
        "ja": ["日本語の文字認識", "東京都渋谷区", "カタカナとひらがな", "合計金額：1234円"],
        "fr": ["Crème brûlée et café", "Élève à l'école française", "Numéro de série 12345", "Noël à Genève"],
        "de": ["Grüße aus München", "Straße, Öl und Käse", "Öffnungszeiten täglich", "Rechnungsbetrag 1234,56"],
    }
    cases = []
    output.mkdir(parents=True, exist_ok=True)
    for language, strings in texts.items():
        font_path = chinese if language == "zh" else japanese if language == "ja" else latin
        if not font_path.is_file():
            raise FileNotFoundError(f"Synthetic {language} cases require font {font_path}")
        font = ImageFont.truetype(str(font_path), 36)
        for index, text in enumerate(strings):
            left, top, right, bottom = font.getbbox(text)
            clean = Image.new("RGB", (right - left + 24, bottom - top + 20), "white")
            ImageDraw.Draw(clean).text((12 - left, 10 - top), text, font=font, fill="black")
            for style in ("clean", "blur", "small"):
                image = clean.copy()
                if style == "blur":
                    image = image.filter(ImageFilter.GaussianBlur(1.1))
                elif style == "small":
                    image = image.resize((image.width // 2, image.height // 2), Image.Resampling.BILINEAR)
                path = output / f"{language}_{index}_{style}.png"
                image.save(path)
                cases.append({
                    "id": path.stem, "language": language, "style": style,
                    "text": text, "font": str(font_path),
                    "image": cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
    return cases


def measure(name, backend, images, cases, repeats, output):
    options = model_options(name)
    for path in options.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {path}; run scripts/download_ppocrv6.py first")
    logger = logging.getLogger("ppocr-comparison")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.ERROR)
    start = time.perf_counter()
    model = ONNXPaddleOcr(
        logger=logger, use_openvino=backend == "openvino", use_npu=False,
        use_angle_cls=False, force_static_shape=False, det_limit_side_len=960,
        **options,
    )
    init_seconds = time.perf_counter() - start
    # Make sure the dictionary matches the export, including blank and space.
    if backend == "openvino":
        classes = int(model.text_recognizer.session.outputs[0].partial_shape[-1].get_length())
    else:
        classes = model.text_recognizer.session.get_outputs()[0].shape[-1]
    if classes != len(model.text_recognizer.postprocess_op.character):
        raise ValueError(f"{name}: output classes {classes} do not match dictionary")
    print(f"{backend}/{name}: initialized in {init_seconds:.2f}s; warming all {len(images)} images", flush=True)
    for _, image in images:
        model.ocr(image)
    timings = defaultdict(list)
    predictions = {}
    for repeat in range(repeats):
        start_pass = time.perf_counter()
        for path, image in images:
            start = time.perf_counter()
            prediction = model.ocr(image)
            timings[path].append((time.perf_counter() - start) * 1000)
            predictions[path] = prediction[0]
        print(f"{backend}/{name}: pass {repeat + 1}/{repeats}: {time.perf_counter() - start_pass:.2f}s", flush=True)
    recognition = []
    for case in cases:
        text, confidence = model.ocr(case["image"], det=False)[0][0]
        expected, actual = normalized(case["text"]), normalized(text)
        recognition.append({
            **{key: value for key, value in case.items() if key != "image"},
            "prediction": text, "confidence": confidence,
            "exact": expected == actual, "edit_distance": edit_distance(expected, actual),
            "characters": len(expected),
        })
    all_times = [value for times in timings.values() for value in times]
    language_metrics = {}
    for language in sorted({case["language"] for case in recognition}):
        rows = [case for case in recognition if case["language"] == language]
        language_metrics[language] = {
            "cases": len(rows), "exact": sum(row["exact"] for row in rows),
            "cer": sum(row["edit_distance"] for row in rows) / sum(row["characters"] for row in rows),
        }
    result = {
        "model": name, "backend": backend, "init_seconds": init_seconds,
        "model_bytes": sum(Path(options[key]).stat().st_size for key in ("det_model_dir", "rec_model_dir")),
        "asset_sha256": {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in options.items()},
        "recognition_classes": classes,
        "settings": {"angle_classifier": False, "npu": False, "static_shape": False,
                     "det_limit_side_len": 960, "det_limit_type": "max",
                     "det_db_thresh": 0.3, "det_db_box_thresh": 0.5, "det_db_unclip_ratio": 1.6,
                     "rec_image_shape": "3,48,320", "rec_batch_num": 6, "drop_score": 0.5,
                     "warmup": "one complete image sweep", "timed_passes": repeats,
                     "timing_excludes": "model initialization, disk image reading, synthetic recognition"},
        "summary": {"mean_ms": statistics.mean(all_times), "median_ms": statistics.median(all_times),
                    "p95_ms": float(np.percentile(all_times, 95)),
                    "total_lines": sum(len(value) for value in predictions.values()),
                    "synthetic_exact": sum(case["exact"] for case in recognition),
                    "synthetic_cases": len(recognition), "languages": language_metrics},
        "images": [{"path": path, "timings_ms": timings[path], "prediction": predictions[path]}
                   for path, _ in images],
        "synthetic_recognition": recognition,
    }
    (output / f"{backend}_{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del model
    gc.collect()
    return result


def write_summary(output):
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(output.glob("*.json"))
               if path.name != "environment.json"]
    model_order = {name: index for index, name in enumerate(("v5", "v6_tiny", "v6_small", "v6_medium"))}
    results.sort(key=lambda result: (result["backend"], model_order[result["model"]]))
    lines = ["# Local PP-OCRv5 / PP-OCRv6 comparison", "",
             "All models use this repository's existing pipeline and identical v5 preprocessing, thresholds, and batching.",
             "These are integration measurements, not the official PaddleOCR pipeline benchmark.",
             "Repository images have no ground truth: detected line counts and confidence are not accuracy.",
             "Synthetic recognition uses 60 labeled line crops (English, Chinese, Japanese, French, German; clean, blurred, half-size).",
             "Exact match and CER use Unicode NFC and ignore whitespace. The synthetic set is only a controlled smoke test.", "",
             "| Backend | Model | DET+REC MB | Mean ms/image | Median ms | P95 ms | Lines | Synthetic exact |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        summary = result["summary"]
        lines.append(f"| {result['backend']} | {result['model']} | {result['model_bytes'] / 1e6:.1f} | "
                     f"{summary['mean_ms']:.1f} | {summary['median_ms']:.1f} | {summary['p95_ms']:.1f} | "
                     f"{summary['total_lines']} | {summary['synthetic_exact']}/{summary['synthetic_cases']} |")
    lines += ["", "V6 tiny does not support Japanese; the shared 60-case total includes 12 Japanese cases.",
              "", "| Backend | Model | Language | Exact lines | CER |", "| --- | --- | --- | ---: | ---: |"]
    for result in results:
        for language, metrics in result["summary"]["languages"].items():
            lines.append(f"| {result['backend']} | {result['model']} | {language} | "
                         f"{metrics['exact']}/{metrics['cases']} | {metrics['cer']:.1%} |")
    lines += ["", "Raw JSON files include every prediction, box, confidence, timing, labeled synthetic case, and asset hash.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=["onnxruntime", "openvino"], default=["onnxruntime", "openvino"])
    parser.add_argument("--models", nargs="+", choices=["v5", "v6_tiny", "v6_small", "v6_medium"], default=["v5", "v6_tiny", "v6_small", "v6_medium"])
    parser.add_argument("--image-dir", type=Path, default=ROOT / "onnxocr/test_images")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ppocrv6")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    images = []
    image_manifest = []
    for path in sorted(args.image_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read {path}")
        images.append((path.name, image))
        image_manifest.append({"path": path.name, "shape": list(image.shape),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not images:
        parser.error("No images found")
    environment = {"created_utc": datetime.now(timezone.utc).isoformat(),
                   "platform": platform.platform(), "processor": platform.processor(),
                   "python": platform.python_version(),
                   "versions": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", *args.backends)},
                   "opencv": cv2.__version__, "images": image_manifest,
                   "command": sys.argv}
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    cases = synthetic_cases(args.output / "synthetic")
    for backend in args.backends:
        for name in args.models:
            measure(name, backend, images, cases, args.repeats, args.output)
            write_summary(args.output)
