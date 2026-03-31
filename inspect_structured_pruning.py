import os
import sys
import torch
import torch.nn as nn

ROOT_DIR = "/users/kashkoul/model-compression-experiments"
sys.path.append(ROOT_DIR)

from shared.model import Net

CHECKPOINT_PATH = os.path.join(
    ROOT_DIR,
    "structured_pruning",
    "checkpoints",
    "structured_pruned_state_dict.pth"
)

def layer_structured_stats(name, module):
    w = module.weight.detach().cpu()

    if isinstance(module, nn.Conv2d):
        out_units = w.shape[0]
        reshaped = w.view(out_units, -1)
        norms = torch.norm(reshaped, p=2, dim=1)
        zero_units = int(torch.sum(norms == 0).item())
        total_units = int(out_units)
        zero_pct = 100.0 * zero_units / total_units if total_units > 0 else 0.0
        return {
            "layer_name": name,
            "layer_type": "Conv2d",
            "zero_units": zero_units,
            "total_units": total_units,
            "zero_pct": zero_pct,
        }

    if isinstance(module, nn.Linear):
        out_units = w.shape[0]
        norms = torch.norm(w, p=2, dim=1)
        zero_units = int(torch.sum(norms == 0).item())
        total_units = int(out_units)
        zero_pct = 100.0 * zero_units / total_units if total_units > 0 else 0.0
        return {
            "layer_name": name,
            "layer_type": "Linear",
            "zero_units": zero_units,
            "total_units": total_units,
            "zero_pct": zero_pct,
        }

    return None

model = Net()
state = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

rows = []
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        rows.append(layer_structured_stats(name, module))

print("===== STRUCTURED PRUNING PER-LAYER SUMMARY =====")
for row in rows:
    print(
        f"{row['layer_name']:10s} | "
        f"{row['layer_type']:6s} | "
        f"zero_units={row['zero_units']:4d} / {row['total_units']:4d} | "
        f"zero_pct={row['zero_pct']:.2f}%"
    )
