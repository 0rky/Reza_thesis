# All Experiments Comparison

| Case | Last Updated | Accuracy (%) | Loss | Model Size (MB) | Size Reduction (%) | Notes |
|---|---|---:|---:|---:|---:|---|
| baseline | 2026-03-29 20:07:48 | 85.0700 | 0.399263 | 16.7065 |  |  |
| mixed_precision | 2026-03-29 21:10:41 | 85.4500 | 0.393198 | 16.7065 |  | AMP=True, dtype=torch.float16 |
| quantization_float | 2026-03-31 00:35:31 | 80.1300 | 0.536393 | 16.7066 |  |  |
| quantization_dynamic | 2026-03-31 00:35:31 | 79.9700 | 0.536724 | 16.6378 | 0.4122 |  |
| qat_preconvert | 2026-03-31 00:39:56 | 84.4400 | 0.437665 | 16.8257 |  |  |
| qat_converted | 2026-03-31 00:39:56 | 84.3500 | 0.437515 | 4.2491 | 74.7465 |  |
| pruning_dense | 2026-03-31 00:00:11 | 79.8700 | 0.551287 | 16.7064 |  |  |
| pruning_pruned | 2026-03-31 00:00:11 | 79.6400 | 0.554264 | 16.7065 | -0.0002 | sparsity=30.00% |
| structured_dense | 2026-03-31 00:45:20 | 80.6400 | 0.531298 | 16.7065 |  |  |
| structured_pruned | 2026-03-31 00:45:20 | 82.3000 | 0.469805 | 16.7066 | -0.0002 | structured_sparsity=9.89% |
