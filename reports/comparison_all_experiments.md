# All Experiments Comparison

| Case | Last Updated | Accuracy (%) | Loss | Model Size (MB) | Size Reduction (%) | Notes |
|---|---|---:|---:|---:|---:|---|
| baseline | 2026-03-31 01:47:18 | 88.9500 | 0.306597 | 16.7065 |  |  |
| mixed_precision | 2026-03-31 02:03:34 | 88.7300 | 0.309739 | 16.7065 |  | AMP=True, dtype=torch.float16 |
| quantization_float | 2026-03-31 02:20:17 | 88.9600 | 0.310175 | 16.7066 |  |  |
| quantization_dynamic | 2026-03-31 02:20:17 | 88.9300 | 0.310507 | 16.6378 | 0.4122 |  |
| qat_preconvert | 2026-03-31 02:39:12 | 90.8000 | 0.255665 | 16.8257 |  |  |
| qat_converted | 2026-03-31 02:39:12 | 90.7900 | 0.255665 | 4.2491 | 74.7465 |  |
| pruning_dense | 2026-03-31 02:55:53 | 88.6300 | 0.311569 | 16.7064 |  |  |
| pruning_pruned | 2026-03-31 02:55:53 | 88.9300 | 0.309282 | 16.7065 | -0.0002 | sparsity=30.00% |
| structured_dense | 2026-03-31 03:15:26 | 88.8200 | 0.314264 | 16.7065 |  |  |
| structured_pruned | 2026-03-31 03:15:26 | 89.5300 | 0.286907 | 16.7066 | -0.0002 | structured_sparsity=9.89% |
