"""Keep the legacy v5 distribution aligned with its bundled v5 assets."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        if self.distribution.get_name() == "onnxocr-ppocrv5":
            defaults = Path(self.build_lib) / "onnxocr" / "_model_defaults.py"
            defaults.write_text(
                '"""Bundled defaults for the PP-OCRv5 distribution."""\n\n'
                'MODEL_DIRECTORY = "ppocrv5"\n'
                'CHARACTER_DICTIONARY = "ppocrv5_dict.txt"\n',
                encoding="utf-8",
            )


setup(cmdclass={"build_py": BuildPy})
