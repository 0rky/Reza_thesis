# All Experiments Comparison Enriched

| Case | Accuracy (%) | Loss | Model Size (MB) | Size Reduction (%) | Avg Train Epoch (s) | Avg Test Epoch (s) | Max GPU Alloc (MB) | Avg RSS RAM (MB) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 85.0700 | 0.399263 |  |  | 85.08 | 10.15 | 438.56 | 1158.61 |  |
| mixed_precision | 85.4500 | 0.393198 |  |  | 84.83 | 10.49 | 279.64 | 1179.21 | AMP=True, dtype=torch.float16 |
| quantization_float | 80.1300 | 0.536393 | 16.7066 |  | 85.88 | 10.07 | 438.56 | 1129.86 |  |
| quantization_dynamic | 79.9700 | 0.536724 | 16.6378 | 0.4122 | 85.88 | 10.07 | 438.56 | 1129.86 |  |
| qat_preconvert | 84.4400 | 0.437665 | 16.8257 |  | 100.62 | 12.09 | 689.08 | 1189.72 |  |
| qat_converted | 84.3500 | 0.437515 | 4.2491 | 74.7465 | 100.62 | 12.09 | 689.08 | 1189.72 |  |
| pruning_dense | 79.8700 | 0.551287 | 16.7064 |  | 86.86 | 10.14 | 438.56 | 1127.79 |  |
| pruning_pruned | 79.6400 | 0.554264 | 16.7065 | -0.0002 | 86.86 | 10.14 | 438.56 | 1127.79 | sparsity=30.00% |
| structured_dense | 80.6400 | 0.531298 | 16.7065 |  | 86.72 | 10.29 | 484.94 | 1222.89 |  |
| structured_pruned | 82.3000 | 0.469805 | 16.7066 | -0.0002 | 86.72 | 10.29 | 484.94 | 1222.89 | structured_sparsity=9.89% |
