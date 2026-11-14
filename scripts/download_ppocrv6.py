"""Download pinned official PP-OCRv6 ONNX exports without changing v5 defaults.

Requires PyYAML (python -m pip install pyyaml). Models are Apache-2.0 licensed.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "tiny": {
        "det": "2ba1506c0380b8f0b03dd142459aac66d4421f6c",
        "rec": "2612ab37152ae0a677521bae4e1e3d4fb4cf7c30",
    },
    "small": {
        "det": "28fe5895c24fd108c19eb3e8479f4ab385fbfc62",
        "rec": "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
    },
    "medium": {
        "det": "61323801669c338b7891481ec7bac61ce31b576a",
        "rec": "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target, expected_sha256=None):
    if target.is_file() and expected_sha256 and sha256(target) == expected_sha256:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if expected_sha256 and sha256(temporary) != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {target}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def download_tier(tier, model_root):
    directory = model_root / tier
    manifest = {"tier": tier, "license": "Apache-2.0", "assets": {}}
    for task, revision in REVISIONS[tier].items():
        repo = f"PaddlePaddle/PP-OCRv6_{tier}_{task}_onnx"
        base = f"https://huggingface.co/{repo}/resolve/{revision}"
        with urlopen(
            f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true",
            timeout=60,
        ) as response:
            metadata = json.load(response)
        asset = next(f for f in metadata["siblings"] if f["rfilename"] == "inference.onnx")
        expected_hash = asset["lfs"]["sha256"]
        print(f"Downloading {repo} ({asset['size'] / 1e6:.1f} MB)", flush=True)
        model = directory / task / f"{task}.onnx"
        config = directory / task / "inference.yml"
        download(f"{base}/inference.onnx", model, expected_hash)
        download(f"{base}/inference.yml", config)
        settings = yaml.safe_load(config.read_text(encoding="utf-8"))
        manifest["assets"][task] = {
            "repo": repo,
            "revision": revision,
            "sha256": sha256(model),
            "bytes": model.stat().st_size,
            "config_sha256": sha256(config),
        }
        if task == "rec":
            # PaddleX's embedded dictionary excludes the CTC blank and trailing
            # space. Our CTCLabelDecode adds those two tokens itself.
            characters = settings["PostProcess"]["character_dict"]
            if any("\n" in char or "\r" in char for char in characters):
                raise ValueError("Dictionary contains a line separator")
            dictionary = directory / "ppocrv6_dict.txt"
            dictionary.write_text("\n".join(characters) + "\n", encoding="utf-8")
            manifest["dictionary_characters"] = len(characters)
            manifest["dictionary_sha256"] = sha256(dictionary)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Ready: {directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", nargs="+", choices=REVISIONS, default=list(REVISIONS))
    parser.add_argument("--model-root", type=Path, default=ROOT / "onnxocr/models/ppocrv6")
    args = parser.parse_args()
    for selected_tier in args.tiers:
        download_tier(selected_tier, args.model_root)
