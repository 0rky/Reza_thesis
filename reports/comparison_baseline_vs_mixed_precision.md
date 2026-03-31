# Baseline vs Mixed Precision Comparison

| Metric | Baseline | Mixed Precision |
|---|---:|---:|
| Final accuracy (%) | 85.0700 | 84.6200 |
| Average train epoch time (s) | 85.0751 | 86.8461 |
| Average test time (s) | 10.1455 | 10.3877 |
| Max GPU allocated (MB) | 438.5635 | 279.6372 |
| Max GPU reserved (MB) | 576.0000 | 344.0000 |
| Average process RSS RAM (MB) | 1158.6117 | 1168.0320 |
| Average system RAM (%) | 2.0900 | 2.0800 |
| Total parameters | 4376772 | 4376772 |
| Trainable parameters | 4376772 | 4376772 |
| Optimizer state size (MB) | 33.3922 | 33.3922 |
| State dict size (MB) | 16.7065 | 16.7065 |
| ONNX size (MB) | 16.7007 | 16.7007 |
| AMP enabled | N/A | True |
| Autocast dtype | N/A | torch.float16 |
