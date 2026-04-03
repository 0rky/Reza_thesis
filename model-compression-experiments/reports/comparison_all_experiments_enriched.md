# All Experiments Comparison Enriched

| Case | Accuracy (%) | Loss | Model Size (MB) | Size Reduction (%) | Avg Train Epoch (s) | Avg Test Epoch (s) | Max GPU Alloc (MB) | Avg RSS RAM (MB) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 88.9500 | 0.306597 |  |  | 85.56 | 10.17 | 438.56 | 1160.45 |  |
| mixed_precision | 88.7300 | 0.309739 |  |  | 85.14 | 10.14 | 279.64 | 1179.49 | AMP=True, dtype=torch.float16 |
| quantization_float | 88.9600 | 0.310175 | 16.7066 |  | 84.26 | 10.12 | 438.56 | 1166.96 |  |
| quantization_dynamic | 88.9300 | 0.310507 | 16.6378 | 0.4122 | 84.26 | 10.12 | 438.56 | 1166.96 |  |
| qat_preconvert | 90.8000 | 0.255665 | 16.8257 |  | 97.91 | 11.68 | 689.08 | 1197.61 |  |
| qat_converted | 90.7900 | 0.255665 | 4.2491 | 74.7465 | 97.91 | 11.68 | 689.08 | 1197.61 |  |
| pruning_dense | 88.6300 | 0.311569 | 16.7064 |  | 86.26 | 10.39 | 438.56 | 1145.62 |  |
| pruning_pruned | 88.9300 | 0.309282 | 16.7065 | -0.0002 | 86.26 | 10.39 | 438.56 | 1145.62 | sparsity=30.00% |
| structured_dense | 88.8200 | 0.314264 | 16.7065 |  | 84.86 | 10.09 | 484.94 | 1185.88 |  |
| structured_pruned | 89.5300 | 0.286907 | 16.7066 | -0.0002 | 84.86 | 10.09 | 484.94 | 1185.88 | structured_sparsity=9.89% |
