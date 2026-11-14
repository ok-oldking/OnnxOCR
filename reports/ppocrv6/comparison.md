# PP-OCRv6 compared with this project's PP-OCRv5

Measured on 14 November 2026 using the existing OnnxOCR pipeline. All three
official v6 ONNX detection/recognition pairs are downloaded locally under
`onnxocr/models/ppocrv6/{tiny,small,medium}/`. The source checkout now defaults
to v6 small; the legacy v5 distribution continues to bundle and select v5.

## Result and recommendation

V6 **small** is the strongest candidate for an accuracy-focused upgrade here.
It fixes the tested Japanese sentence and French accents, and reads all 60
synthetic cases exactly, versus v5's 48. It is 11% faster with ONNX Runtime,
but 22% slower with OpenVINO CPU. That improvement needs confirmation on
labeled images from the application's actual workload before changing defaults.

| Model | DET+REC size | ONNX Runtime mean ms/image | OpenVINO CPU mean ms/image | Synthetic exact lines, either backend |
| --- | ---: | ---: | ---: | ---: |
| Bundled v5 | 21.3 MB | 753.4 | 198.5 | 48/60 |
| V6 tiny | 6.2 MB | 370.1 | 115.6 | 45/60 |
| V6 small | 31.0 MB | 670.1 | 242.9 | 60/60 |
| V6 medium | 138.6 MB | 1248.7 | 664.8 | 60/60 |

V6 **tiny** is the speed/size option: 2.04× faster on ONNX Runtime, 1.72× on
OpenVINO, and 71% smaller than bundled v5. It does not support Japanese;
excluding the 12 Japanese cases, it scores 45/48 versus v5's 37/48.

V6 **medium** has higher published benchmark accuracy, but provides no extra
synthetic exact matches here. It is 6.5× larger than v5 and takes 1.66× as long
on ONNX Runtime / 3.35× on OpenVINO in this pipeline. Evaluate it when difficult
real text warrants the additional cost.

OpenVINO CPU is faster than ONNX Runtime for every tested model. These
measurements use each backend's default precision and thread configuration;
they are observations on this machine, not a controlled comparison of engines
at identical thread counts or precision. Synthetic predictions are identical
between the two backends for all four models.

Validation completed: eight model/backend runs, each with all 30 images and
three timed passes plus 60 synthetic recognition cases; output dictionary
class counts match; the existing unit suite passes (7 passed, 63 integration
tests deselected). No inference code or package dependencies were changed.

## What the release changes

PP-OCRv6 shipped with PaddleOCR 3.7.0 on 11 June 2026. The model family has
three sizes; “v6” is a model generation, not a Python package version.
Small and medium cover 50 languages; tiny covers 49 and excludes Japanese.
The following are the authors' in-house benchmark results, not measurements
on this repository's images:

| Model | Detection Hmean | Recognition accuracy |
| --- | ---: | ---: |
| PP-OCRv5 mobile | 75.2% | 73.7% |
| PP-OCRv6 tiny | 80.6% | 73.5% |
| PP-OCRv6 small | 84.1% | 81.3% |
| PP-OCRv6 medium | 86.2% | 83.2% |

Sources: [official release](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0)
and [official technical documentation](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html).
The release headline comparing v6 medium with v5 **server** is a different
baseline from this repository's bundled mobile-sized v5 pair.

## Local measurements

See [summary.md](summary.md) for all measured latency, model-size, line-count,
and recognition results, and the adjacent JSON files for every output.
[environment.json](environment.json) records versions, image hashes and shapes,
and the benchmark command. Each model JSON includes model/dictionary SHA-256
hashes. Downloaded model directories also contain their source revisions,
official inference configuration, and hash manifests.

The machine identifies as an Intel Core Ultra 9 386H with 16 logical CPUs,
Windows 11, Python 3.12.15, ONNX Runtime 1.24.4, and OpenVINO 2026.4.1.

Method:

- 30 top-level repository sample images, decoded before timing.
- One complete warmup sweep, then three full timed sweeps per model/backend.
- Timing includes detection, cropping, recognition, and all pre/postprocessing;
  excludes model loading, file I/O, and the separate synthetic recognition test.
- All models use dynamic shapes, maximum detection side 960, recognition
  shape `3,48,320`, recognition batch 6, and the same v5 detection thresholds
  (`0.3`, box `0.5`, unclip `1.6`) and confidence cutoff `0.5`.
- CPU only, one model running at a time, backend default threading/precision,
  angle classifier disabled, no document rotation or unwarping.
- 60 labeled synthetic recognition-only crops: 4 strings in each of English,
  Chinese, Japanese, French and German, each rendered clean, blurred, and at
  half size. Text and generated PNGs are included. Exact match and character
  error rate (CER) normalize to Unicode NFC and ignore whitespace.

The synthetic set is a small controlled smoke test. It does not establish
production accuracy. The repository images have no ground-truth annotations,
so more detected lines or higher confidence cannot be interpreted as improved
accuracy. The Korean example is outside the documented v6 language coverage.

This is a comparison of model replacements in the current pipeline. Official
v6 detector defaults use threshold `0.2`, box threshold `0.4` (tiny) / `0.45`
(small/medium), unclip `1.4`, and 3000 candidates; this pipeline keeps its
existing v5 defaults, including 1000 candidates. Results therefore do not
reproduce the official PaddleOCR pipeline or its advertised speedups.

## Examples from the existing images

These are visually checked examples from the ONNX Runtime outputs, not a
complete labeled evaluation:

| Image / visible text | V5 | V6 small | V6 medium |
| --- | --- | --- | --- |
| [Japanese sentence](../../onnxocr/test_images/japan_1.jpg): `だいじょうぶ。未来は元気だよ。` | `だいょう。未来は元気よ。` | Exact | Exact |
| [French sign](../../onnxocr/test_images/french_0.jpg): `LES ARTS DÉCORATIFS` | `LES ARTS DECORATIFS` | Exact | Exact |
| Same French image: pedestrian pictograms | One becomes a leading `公` before `Musée` | Two become extra `穴` lines | No such `穴` lines, but an extra `R` elsewhere |

The difficult superscript in `Mairie du 1er` is still wrong with all four
models. V6 tiny corrupts the Japanese sentence, consistent with its documented
language coverage. V6 therefore improves some visible text while still
producing errors and false positives on graphics.

## Reproduce and use

```powershell
.\.venv\Scripts\python.exe -m pip install pyyaml
.\.venv\Scripts\python.exe scripts/download_ppocrv6.py
.\.venv\Scripts\python.exe scripts/compare_ppocr.py
```

The downloader uses pinned official PaddlePaddle Hugging Face revisions and
verifies the ONNX files against the repository LFS SHA-256 values. Models are
Apache-2.0 licensed. PyYAML is only needed to extract the official dictionary.
Small and medium share an 18,708-character dictionary; tiny has 6,904 entries.
The decoder adds CTC blank and space, yielding 18,710 and 6,906 output classes.
V5 has 18,385 output classes. Reusing the v5 dictionary would decode v6
incorrectly. The benchmark checks class count against the correct dictionary.

Use the explicit model and dictionary paths shown in the
[project README](../../Readme.md#try-pp-ocrv6-and-compare-with-v5).
Downloaded v6 assets are excluded from Git and the v5 wheel; this work does
not change the v5 distribution's model defaults or release version.
