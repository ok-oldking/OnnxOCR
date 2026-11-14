"""Validate v6 release artifacts and optionally smoke-test the installed wheel.

--smoke requires NumPy, OpenCV, pyclipper, shapely, Pillow and the chosen backend.
"""

import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(output):
    profile = tomllib.loads((ROOT / "packaging/ppocrv6.toml").read_text(encoding="utf-8"))
    name, version = profile["project"]["name"], profile["project"]["version"]
    wheels = list(output.glob(f"onnxocr_ppocrv6-{version}-*.whl"))
    sdists = list(output.glob(f"onnxocr_ppocrv6-{version}.tar.gz"))
    require(len(wheels) == len(sdists) == 1, "Expected one v6 wheel and one sdist for this version")
    assets = {f"onnxocr/{asset}" for asset in profile["tool"]["setuptools"]["package-data"]["onnxocr"]}
    with zipfile.ZipFile(wheels[0]) as wheel, tarfile.open(sdists[0], "r:gz") as sdist:
        entries = set(wheel.namelist())
        require(assets <= entries, f"Wheel is missing assets: {assets - entries}")
        models = {path for path in entries if path.endswith(".onnx")}
        require(models == {path for path in assets if path.endswith(".onnx")}, "Wheel has unexpected model files")
        metadata_paths = [path for path in entries if path.endswith(".dist-info/METADATA")]
        require(len(metadata_paths) == 1, "Wheel must have one METADATA file")
        metadata = Parser().parsestr(wheel.read(metadata_paths[0]).decode("utf-8"))
        require(metadata["Name"] == name and metadata["Version"] == version, "Wrong v6 wheel identity")
        prefix = f"onnxocr_ppocrv6-{version}/"
        members = {member.name: member for member in sdist.getmembers() if member.isfile()}
        source_assets = {path[len(prefix):] for path in members if path.startswith(prefix)}
        require(assets <= source_assets, "Source distribution is missing model assets")
        require({path for path in source_assets if path.endswith(".onnx")} == models, "Source distribution has unexpected models")
        # Check every shipped Python module and data asset survives the sdist.
        for path in assets | {path for path in entries if path.startswith("onnxocr/") and path.endswith(".py")}:
            source = sdist.extractfile(members[prefix + path])
            require(source.read() == wheel.read(path), f"Wheel/sdist contents differ: {path}")
        manifest = json.loads(wheel.read("onnxocr/models/ppocrv6/small/manifest.json"))
        for task in ("det", "rec"):
            digest = hashlib.sha256(wheel.read(f"onnxocr/models/ppocrv6/small/{task}/{task}.onnx")).hexdigest()
            require(digest == manifest["assets"][task]["sha256"], f"Wrong {task} hash in wheel")
        dictionary = wheel.read("onnxocr/models/ppocrv6/small/ppocrv6_dict.txt")
        require(hashlib.sha256(dictionary).hexdigest() == manifest["dictionary_sha256"], "Wrong dictionary hash in wheel")
        require(len(dictionary.decode("utf-8").splitlines()) == 18708, "Wrong v6 small dictionary size")
        print(f"Validated {name} {version}: wheel and sdist contain the small pair, dictionary, classifier and font.")
    return wheels[0]


SMOKE_CODE = r'''
import logging
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
import onnxocr
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

target = Path(sys.argv[1]).resolve()
assert Path(onnxocr.__file__).resolve().is_relative_to(target), onnxocr.__file__
logger = logging.getLogger("v6-wheel-check")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.ERROR)
model = ONNXPaddleOcr(logger=logger, use_openvino=sys.argv[2] == "openvino", use_npu=False, use_angle_cls=True)
for path in (model.args.det_model_dir, model.args.rec_model_dir, model.args.rec_char_dict_path):
    assert Path(path).resolve().is_relative_to(target / "onnxocr/models/ppocrv6/small"), path
assert Path(model.args.cls_model_dir).is_file()
assert len(model.text_recognizer.postprocess_op.character) == 18710
frame = np.full((160, 640, 3), 255, dtype=np.uint8)
cv2.putText(frame, "PaddleOCR 6", (24, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4, cv2.LINE_AA)
result = model.ocr(frame)
texts = [line[1][0] for line in result[0]]
assert "paddleocr6" in "".join(texts).replace(" ", "").lower(), texts
print("Installed v6 wheel passed", sys.argv[2], "OCR and optional angle classification:", texts)
'''


def smoke(wheel, backends):
    build_root = (ROOT / "build").resolve()
    require(build_root.parent == ROOT.resolve(), "Smoke directory must be inside the repository")
    build_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v6-wheel-check-", dir=build_root) as temporary:
        target = Path(temporary) / "installed"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel.resolve())], check=True)
        for backend in backends:
            subprocess.run([sys.executable, "-I", "-c", SMOKE_CODE, str(target), backend], cwd=temporary, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist/ppocrv6")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backends", nargs="+", choices=["onnxruntime", "openvino"], default=["onnxruntime"])
    args = parser.parse_args()
    wheel = check(args.dist)
    if args.smoke:
        smoke(wheel, args.backends)
