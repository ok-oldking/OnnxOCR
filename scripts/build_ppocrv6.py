"""Build the v6 small sdist and wheel in an isolated staging directory.

Requires Python 3.11+, build, tomli-w and PyYAML. Download small models first.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

import tomli_w

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "packaging/ppocrv6.toml"


def build(output):
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    profile = tomllib.loads(PROFILE.read_text(encoding="utf-8"))
    metadata["project"].update(profile["project"])
    metadata["tool"]["setuptools"]["package-data"] = profile["tool"]["setuptools"]["package-data"]
    metadata["tool"]["setuptools"]["include-package-data"] = False
    assets = metadata["tool"]["setuptools"]["package-data"]["onnxocr"]
    package = ROOT / "onnxocr"
    for asset in assets:
        if not (package / asset).is_file():
            raise FileNotFoundError(f"Missing {asset}; run scripts/download_ppocrv6.py --tiers small")
    # Verify downloaded model bytes against the pinned official LFS hashes.
    from download_ppocrv6 import REVISIONS, sha256

    manifest = json.loads((package / "models/ppocrv6/small/manifest.json").read_text(encoding="utf-8"))
    for task, revision in REVISIONS["small"].items():
        asset = manifest["assets"][task]
        model = package / f"models/ppocrv6/small/{task}/{task}.onnx"
        if asset["revision"] != revision or sha256(model) != asset["sha256"]:
            raise ValueError(f"V6 small {task} does not match its pinned download manifest")
    dictionary = package / "models/ppocrv6/small/ppocrv6_dict.txt"
    if sha256(dictionary) != manifest["dictionary_sha256"]:
        raise ValueError("V6 dictionary does not match its download manifest")
    build_root = (ROOT / "build").resolve()
    if build_root.parent != ROOT.resolve():
        raise ValueError("Staging directory must be inside the repository")
    build_root.mkdir(exist_ok=True)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ppocrv6-", dir=build_root) as temporary:
        stage = Path(temporary)
        target = stage / "onnxocr"
        target.mkdir()
        for source in package.glob("*.py"):
            shutil.copy2(source, target / source.name)
        for asset in assets:
            destination = target / asset
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package / asset, destination)
        settings = profile["tool"]["onnxocr"]
        (target / "_model_defaults.py").write_text(
            '"""Bundled defaults for the PP-OCRv6 small distribution."""\n\n'
            f'MODEL_DIRECTORY = {settings["model-directory"]!r}\n'
            f'CHARACTER_DICTIONARY = {settings["character-dictionary"]!r}\n',
            encoding="utf-8",
        )
        for name in ("LICENSE", metadata["project"]["readme"]):
            shutil.copy2(ROOT / name, stage / name)
        (stage / "pyproject.toml").write_text(tomli_w.dumps(metadata), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "build", str(stage), "--outdir", str(output)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "dist/ppocrv6")
    build(parser.parse_args().outdir)
