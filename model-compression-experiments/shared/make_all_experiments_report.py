import csv
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/users/kashkoul/model-compression-experiments")
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = REPORTS_DIR / "comparison_all_experiments.csv"
MD_OUT = REPORTS_DIR / "comparison_all_experiments.md"

CASES = [
    {
        "name": "baseline",
        "metrics": PROJECT_ROOT / "baseline" / "outputs" / "baseline_metrics.txt",
        "runtime": PROJECT_ROOT / "baseline" / "outputs" / "baseline_runtime_summary.txt",
    },
    {
        "name": "mixed_precision",
        "metrics": PROJECT_ROOT / "mixed_precision" / "outputs" / "mixed_precision_metrics.txt",
        "runtime": PROJECT_ROOT / "mixed_precision" / "outputs" / "mixed_precision_runtime_summary.txt",
    },
    {
        "name": "quantization_float",
        "metrics": PROJECT_ROOT / "quantization" / "outputs" / "quantization_float_metrics.txt",
        "runtime": PROJECT_ROOT / "quantization" / "outputs" / "quantization_runtime_summary.txt",
    },
    {
        "name": "quantization_dynamic",
        "metrics": PROJECT_ROOT / "quantization" / "outputs" / "quantization_dynamic_metrics.txt",
        "runtime": PROJECT_ROOT / "quantization" / "outputs" / "quantization_runtime_summary.txt",
    },
    {
        "name": "qat_preconvert",
        "metrics": PROJECT_ROOT / "qat" / "outputs" / "qat_preconvert_metrics.txt",
        "runtime": PROJECT_ROOT / "qat" / "outputs" / "qat_runtime_summary.txt",
    },
    {
        "name": "qat_converted",
        "metrics": PROJECT_ROOT / "qat" / "outputs" / "qat_converted_metrics.txt",
        "runtime": PROJECT_ROOT / "qat" / "outputs" / "qat_runtime_summary.txt",
    },
    {
        "name": "pruning_dense",
        "metrics": PROJECT_ROOT / "pruning" / "outputs" / "pruning_dense_metrics.txt",
        "runtime": PROJECT_ROOT / "pruning" / "outputs" / "pruning_runtime_summary.txt",
    },
    {
        "name": "pruning_pruned",
        "metrics": PROJECT_ROOT / "pruning" / "outputs" / "pruning_pruned_metrics.txt",
        "runtime": PROJECT_ROOT / "pruning" / "outputs" / "pruning_runtime_summary.txt",
    },
    {
        "name": "structured_dense",
        "metrics": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_dense_metrics.txt",
        "runtime": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_pruning_runtime_summary.txt",
    },
    {
        "name": "structured_pruned",
        "metrics": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_pruned_metrics.txt",
        "runtime": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_pruning_runtime_summary.txt",
    },
]

EXTRA_TEXT_FILES = {
    "quantization_size_comparison": PROJECT_ROOT / "quantization" / "outputs" / "quantization_size_comparison.txt",
    "qat_size_comparison": PROJECT_ROOT / "qat" / "outputs" / "qat_size_comparison.txt",
    "pruning_size_sparsity_comparison": PROJECT_ROOT / "pruning" / "outputs" / "pruning_size_sparsity_comparison.txt",
    "structured_size_sparsity_comparison": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_size_sparsity_comparison.txt",
}


def fmt_timestamp(path: Path | None):
    if path is None or not path.exists():
        return None
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def parse_accuracy_and_loss(metrics_path: Path):
    data = {
        "cross_entropy_loss": None,
        "accuracy_percent": None,
    }
    if not metrics_path or not metrics_path.exists():
        return data

    text = metrics_path.read_text()

    loss_match = re.search(r"CrossEntropyLoss:\s*([0-9.]+)", text)
    acc_match = re.search(r"Accuracy:\s*([0-9.]+)%", text)

    if loss_match:
        data["cross_entropy_loss"] = float(loss_match.group(1))
    if acc_match:
        data["accuracy_percent"] = float(acc_match.group(1))

    return data


def parse_runtime_summary(runtime_path: Path | None):
    data = {
        "state_dict_size_mb": None,
        "onnx_size_mb": None,
        "optimizer_state_size_mb": None,
        "avg_epoch_time_seconds": None,
        "amp_enabled": None,
        "autocast_dtype": None,
    }

    if runtime_path is None or not runtime_path.exists():
        return data

    text = runtime_path.read_text()

    patterns = {
        "state_dict_size_mb": r"State dict size \(MB\):\s*([0-9.]+)",
        "onnx_size_mb": r"ONNX size \(MB\):\s*([0-9.]+)",
        "optimizer_state_size_bytes": r"Optimizer state size \(bytes\):\s*(\d+)",
        "avg_epoch_time_seconds": r"AverageEpochTimeSeconds:\s*([0-9.]+)",
        "amp_enabled": r"AMP enabled:\s*(True|False)",
        "autocast_dtype": r"Autocast dtype:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if key == "optimizer_state_size_bytes":
                data["optimizer_state_size_mb"] = int(value) / (1024 * 1024)
            elif key in {"state_dict_size_mb", "onnx_size_mb", "avg_epoch_time_seconds"}:
                data[key] = float(value)
            elif key == "amp_enabled":
                data[key] = value
            else:
                data[key] = value

    return data


def parse_text_value(text_path: Path, label: str):
    if not text_path.exists():
        return None
    text = text_path.read_text()
    match = re.search(label + r":\s*(-?[0-9.]+)", text)
    if match:
        return float(match.group(1))
    return None


rows = []
for case in CASES:
    row = {"case": case["name"]}
    row.update(parse_accuracy_and_loss(case["metrics"]))
    row.update(parse_runtime_summary(case["runtime"]))

    row["metrics_file"] = str(case["metrics"]) if case["metrics"] is not None else ""
    row["runtime_file"] = str(case["runtime"]) if case["runtime"] is not None else ""

    metrics_time = fmt_timestamp(case["metrics"])
    runtime_time = fmt_timestamp(case["runtime"])

    row["metrics_last_updated"] = metrics_time
    row["runtime_last_updated"] = runtime_time
    row["last_updated"] = runtime_time if runtime_time is not None else metrics_time

    rows.append(row)

quant_float = next(r for r in rows if r["case"] == "quantization_float")
quant_dynamic = next(r for r in rows if r["case"] == "quantization_dynamic")
qat_pre = next(r for r in rows if r["case"] == "qat_preconvert")
qat_conv = next(r for r in rows if r["case"] == "qat_converted")
prune_dense = next(r for r in rows if r["case"] == "pruning_dense")
prune_pruned = next(r for r in rows if r["case"] == "pruning_pruned")
struct_dense = next(r for r in rows if r["case"] == "structured_dense")
struct_pruned = next(r for r in rows if r["case"] == "structured_pruned")

qfile = EXTRA_TEXT_FILES["quantization_size_comparison"]
quant_float["state_dict_size_mb"] = parse_text_value(qfile, "Float model size \\(MB\\)")
quant_dynamic["state_dict_size_mb"] = parse_text_value(qfile, "Quantized model size \\(MB\\)")
quant_dynamic["size_reduction_percent"] = parse_text_value(qfile, "Size reduction \\(%\\)")

qatfile = EXTRA_TEXT_FILES["qat_size_comparison"]
qat_pre["state_dict_size_mb"] = parse_text_value(qatfile, "QAT pre-convert model size \\(MB\\)")
qat_conv["state_dict_size_mb"] = parse_text_value(qatfile, "QAT converted model size \\(MB\\)")
qat_conv["size_reduction_percent"] = parse_text_value(qatfile, "Size reduction \\(%\\)")

pfile = EXTRA_TEXT_FILES["pruning_size_sparsity_comparison"]
prune_dense["state_dict_size_mb"] = parse_text_value(pfile, "Dense model size \\(MB\\)")
prune_pruned["state_dict_size_mb"] = parse_text_value(pfile, "Pruned model size \\(MB\\)")
prune_pruned["size_reduction_percent"] = parse_text_value(pfile, "Size reduction \\(%\\)")
prune_pruned["sparsity_percent"] = parse_text_value(pfile, "Global sparsity after pruning \\(%\\)")

sfile = EXTRA_TEXT_FILES["structured_size_sparsity_comparison"]
struct_dense["state_dict_size_mb"] = parse_text_value(sfile, "Dense model size \\(MB\\)")
struct_pruned["state_dict_size_mb"] = parse_text_value(sfile, "Structured pruned model size \\(MB\\)")
struct_pruned["size_reduction_percent"] = parse_text_value(sfile, "Size reduction \\(%\\)")
struct_pruned["zero_weight_sparsity_percent"] = parse_text_value(sfile, "Zero-weight sparsity after pruning \\(%\\)")
struct_pruned["structured_sparsity_percent"] = parse_text_value(sfile, "Structured sparsity after pruning \\(%\\)")

fieldnames = []
seen = set()
for row in rows:
    for key in row.keys():
        if key not in seen:
            seen.add(key)
            fieldnames.append(key)

with CSV_OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

lines = []
lines.append("# All Experiments Comparison")
lines.append("")
lines.append("| Case | Last Updated | Accuracy (%) | Loss | Model Size (MB) | Size Reduction (%) | Notes |")
lines.append("|---|---|---:|---:|---:|---:|---|")

for row in rows:
    notes = []
    if row.get("amp_enabled") is not None:
        notes.append(f"AMP={row['amp_enabled']}")
    if row.get("autocast_dtype"):
        notes.append(f"dtype={row['autocast_dtype']}")
    if row.get("sparsity_percent") is not None:
        notes.append(f"sparsity={row['sparsity_percent']:.2f}%")
    if row.get("structured_sparsity_percent") is not None:
        notes.append(f"structured_sparsity={row['structured_sparsity_percent']:.2f}%")

    acc = row.get("accuracy_percent")
    loss = row.get("cross_entropy_loss")
    size_mb = row.get("state_dict_size_mb")
    red = row.get("size_reduction_percent")
    last_updated = row.get("last_updated") or ""

    lines.append(
        f"| {row['case']} | "
        f"{last_updated} | "
        f"{'' if acc is None else f'{acc:.4f}'} | "
        f"{'' if loss is None else f'{loss:.6f}'} | "
        f"{'' if size_mb is None else f'{size_mb:.4f}'} | "
        f"{'' if red is None else f'{red:.4f}'} | "
        f"{', '.join(notes)} |"
    )

MD_OUT.write_text("\n".join(lines) + "\n")

print(f"Saved CSV report to: {CSV_OUT}")
print(f"Saved Markdown report to: {MD_OUT}")
