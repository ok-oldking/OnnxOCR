# Local PP-OCRv5 / PP-OCRv6 comparison

All models use this repository's existing pipeline and identical v5 preprocessing, thresholds, and batching.
These are integration measurements, not the official PaddleOCR pipeline benchmark.
Repository images have no ground truth: detected line counts and confidence are not accuracy.
Synthetic recognition uses 60 labeled line crops (English, Chinese, Japanese, French, German; clean, blurred, half-size).
Exact match and CER use Unicode NFC and ignore whitespace. The synthetic set is only a controlled smoke test.

| Backend | Model | DET+REC MB | Mean ms/image | Median ms | P95 ms | Lines | Synthetic exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| onnxruntime | v5 | 21.3 | 753.4 | 759.0 | 1208.8 | 524 | 48/60 |
| onnxruntime | v6_tiny | 6.2 | 370.1 | 307.5 | 843.1 | 532 | 45/60 |
| onnxruntime | v6_small | 31.0 | 670.1 | 604.3 | 1177.1 | 513 | 60/60 |
| onnxruntime | v6_medium | 138.6 | 1248.7 | 1138.7 | 2365.8 | 521 | 60/60 |
| openvino | v5 | 21.3 | 198.5 | 141.9 | 498.9 | 524 | 48/60 |
| openvino | v6_tiny | 6.2 | 115.6 | 97.0 | 240.3 | 532 | 45/60 |
| openvino | v6_small | 31.0 | 242.9 | 169.3 | 667.5 | 513 | 60/60 |
| openvino | v6_medium | 138.6 | 664.8 | 458.2 | 1895.5 | 521 | 60/60 |

V6 tiny does not support Japanese; the shared 60-case total includes 12 Japanese cases.

| Backend | Model | Language | Exact lines | CER |
| --- | --- | --- | ---: | ---: |
| onnxruntime | v5 | de | 7/12 | 3.6% |
| onnxruntime | v5 | en | 12/12 | 0.0% |
| onnxruntime | v5 | fr | 6/12 | 3.9% |
| onnxruntime | v5 | ja | 11/12 | 1.0% |
| onnxruntime | v5 | zh | 12/12 | 0.0% |
| onnxruntime | v6_tiny | de | 12/12 | 0.0% |
| onnxruntime | v6_tiny | en | 12/12 | 0.0% |
| onnxruntime | v6_tiny | fr | 9/12 | 1.5% |
| onnxruntime | v6_tiny | ja | 0/12 | 42.4% |
| onnxruntime | v6_tiny | zh | 12/12 | 0.0% |
| onnxruntime | v6_small | de | 12/12 | 0.0% |
| onnxruntime | v6_small | en | 12/12 | 0.0% |
| onnxruntime | v6_small | fr | 12/12 | 0.0% |
| onnxruntime | v6_small | ja | 12/12 | 0.0% |
| onnxruntime | v6_small | zh | 12/12 | 0.0% |
| onnxruntime | v6_medium | de | 12/12 | 0.0% |
| onnxruntime | v6_medium | en | 12/12 | 0.0% |
| onnxruntime | v6_medium | fr | 12/12 | 0.0% |
| onnxruntime | v6_medium | ja | 12/12 | 0.0% |
| onnxruntime | v6_medium | zh | 12/12 | 0.0% |
| openvino | v5 | de | 7/12 | 3.6% |
| openvino | v5 | en | 12/12 | 0.0% |
| openvino | v5 | fr | 6/12 | 3.9% |
| openvino | v5 | ja | 11/12 | 1.0% |
| openvino | v5 | zh | 12/12 | 0.0% |
| openvino | v6_tiny | de | 12/12 | 0.0% |
| openvino | v6_tiny | en | 12/12 | 0.0% |
| openvino | v6_tiny | fr | 9/12 | 1.5% |
| openvino | v6_tiny | ja | 0/12 | 42.4% |
| openvino | v6_tiny | zh | 12/12 | 0.0% |
| openvino | v6_small | de | 12/12 | 0.0% |
| openvino | v6_small | en | 12/12 | 0.0% |
| openvino | v6_small | fr | 12/12 | 0.0% |
| openvino | v6_small | ja | 12/12 | 0.0% |
| openvino | v6_small | zh | 12/12 | 0.0% |
| openvino | v6_medium | de | 12/12 | 0.0% |
| openvino | v6_medium | en | 12/12 | 0.0% |
| openvino | v6_medium | fr | 12/12 | 0.0% |
| openvino | v6_medium | ja | 12/12 | 0.0% |
| openvino | v6_medium | zh | 12/12 | 0.0% |

Raw JSON files include every prediction, box, confidence, timing, labeled synthetic case, and asset hash.
