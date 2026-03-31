import csv
import re
from pathlib import Path
from statistics import mean
from datetime import datetime

PROJECT_ROOT = Path("/users/kashkoul/model-compression-experiments")
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = REPORTS_DIR / "comparison_all_experiments_enriched.csv"
MD_OUT = REPORTS_DIR / "comparison_all_experiments_enriched.md"

CASES = [
    {
        "name": "baseline",
        "metrics": PROJECT_ROOT / "baseline" / "outputs" / "baseline_metrics.txt",
        "runtime": PROJECT_ROOT / "baseline" / "outputs" / "baseline_runtime_summary.txt",
        "epoch": PROJECT_ROOT / "baseline" / "outputs" / "baseline_epoch_profile.csv",
    },
    {
        "name": "mixed_precision",
        "metrics": PROJECT_ROOT / "mixed_precision" / "outputs" / "mixed_precision_metrics.txt",
        "runtime": PROJECT_ROOT / "mixed_precision" / "outputs" / "mixed_precision_runtime_summary.txt",
        "epoch": PROJECT_ROOT / "mixed_precision" / "outputs" / "mixed_precision_epoch_profile.csv",
    },
    {
        "name": "quantization_float",
        "metrics": PROJECT_ROOT / "quantization" / "outputs" / "quantization_float_metrics.txt",
        "runtime": PROJECT_ROOT / "quantization" / "outputs" / "quantization_runtime_summary.txt",
        "epoch": PROJECT_ROOT / "quantization" / "outputs" / "quantization_epoch_profile.csv",
    },
    {
        "name": "quantization_dynamic",
        "metrics": PROJECT_ROOT / "quantization" / "outputs" / "quantization_dynamic_metrics.txt",
        "runtime": PROJECT_ROOT / "quantization" / "outputs" / "quantization_runtime_summary.txt",
        "epoch": PROJECT_ROOT / "quantization" / "outputs" / "quantization_epoch_profile.csv",
    },
    {
        "name": "qat_preconvert",
        "metrics": PROJECT_ROOT / "qat" / "outputs" / "qat_preconvert_metrics.txt",
        "runtime": PROJECT_ROOT / "qat" / "outputs" / "qat_runtime_summary.txt",
        "epoch": PROJECT_ROOT / "qat" / "outputs" / "qat_epoch_profile.csv",
    },
    {
        "name": "qat_converted",
        "metrics": PROJECT_ROOT / "qat" / "outputs" / "qat_converted_metrics.txt",
        "runtime": PROJECT_ROOT / "qat" / "outputs" / "qat_runtime_summary.txt",
        "epoch": PROJECT_ROOT / "qat" / "outputs" / "qat_epoch_profile.csv",
    },
    {
        "name": "pruning_dense",
        "metrics": PROJECT_ROOT / "pruning" / "outputs" / "pruning_dense_metrics.txt",
        "runtime": PROJECT_ROOT / "pruning" / "outputs" / "pruning_runtime_summary.txt",
        "epoch": PROJECT_ROOT / "pruning" / "outputs" / "pruning_epoch_profile.csv",
    },
    {
        "name": "pruning_pruned",
        "metrics": PROJECT_ROOT / "pruning" / "outputs" / "pruning_pruned_metrics.txt",
        "runtime": PROJECT_ROOT / "pruning" / "outputs" / "pruning_runtime_summary.txt",
        "epoch": PROJECT_ROOT / "pruning" / "outputs" / "pruning_epoch_profile.csv",
    },
    {
        "name": "structured_dense",
        "metrics": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_dense_metrics.txt",
        "runtime": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_pruning_runtime_summary.txt",
        "epoch": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_pruning_epoch_profile.csv",
    },
    {
        "name": "structured_pruned",
        "metrics": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_pruned_metrics.txt",
        "runtime": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_pruning_runtime_summary.txt",
        "epoch": PROJECT_ROOT / "structured_pruning" / "outputs" / "structured_pruning_epoch_profile.csv",
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
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def parse_accuracy_and_loss(metrics_path: Path):
    data = {
        "cross_entropy_loss": None,
        "accuracy_percent": None,
        "avg_epoch_time_seconds_from_metrics": None,
    }
    if not metrics_path.exists():
        return data

    text = metrics_path.read_text()

    m = re.search(r"CrossEntropyLoss:\s*([0-9.]+)", text)
    if m:
        data["cross_entropy_loss"] = float(m.group(1))

    m = re.search(r"Accuracy:\s*([0-9.]+)%", text)
    if m:
        data["accuracy_percent"] = float(m.group(1))

    m = re.search(r"AverageEpochTimeSeconds:\s*([0-9.]+)", text)
    if m:
        data["avg_epoch_time_seconds_from_metrics"] = float(m.group(1))

    return data


def parse_runtime_summary(runtime_path: Path):
    data = {
        "optimizer_state_size_mb": None,
        "state_dict_size_mb_runtime": None,
        "onnx_size_mb_runtime": None,
        "amp_enabled": None,
        "autocast_dtype": None,
    }
    if not runtime_path.exists():
        return data

    text = runtime_path.read_text()

    patterns = {
        "optimizer_state_size_bytes": r"Optimizer state size \(bytes\):\s*(\d+)",
        "state_dict_size_mb_runtime": r"State dict size \(MB\):\s*([0-9.]+)",
        "onnx_size_mb_runtime": r"ONNX size \(MB\):\s*([0-9.]+)",
        "amp_enabled": r"AMP enabled:\s*(True|False)",
        "autocast_dtype": r"Autocast dtype:\s*(.+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            continue
        value = m.group(1).strip()
        if key == "optimizer_state_size_bytes":
            data["optimizer_state_size_mb"] = int(value) / (1024 * 1024)
        elif key in {"state_dict_size_mb_runtime", "onnx_size_mb_runtime"}:
            data[key] = float(value)
        else:
            data[key] = value

    return data


def parse_epoch_profile(epoch_csv_path: Path):
    data = {
        "avg_train_epoch_seconds": None,
        "avg_test_epoch_seconds": None,
        "avg_epoch_time_seconds": None,
        "max_gpu_allocated_mb": None,
        "max_gpu_reserved_mb": None,
        "avg_process_rss_mb": None,
        "avg_system_ram_percent": None,
    }
    if not epoch_csv_path.exists():
        return data

    train_times = []
    test_times = []
    gpu_alloc = []
    gpu_reserved = []
    rss_vals = []
    ram_pct_vals = []

    with epoch_csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("split") or "").strip()
            elapsed = row.get("elapsed_seconds")
            if elapsed:
                val = float(elapsed)
                if split == "train":
                    train_times.append(val)
                elif split == "test":
                    test_times.append(val)

            x = row.get("gpu_max_allocated_bytes")
            if x not in (None, ""):
                gpu_alloc.append(float(x) / (1024 * 1024))

            x = row.get("gpu_max_reserved_bytes")
            if x not in (None, ""):
                gpu_reserved.append(float(x) / (1024 * 1024))

            x = row.get("process_rss_bytes")
            if x not in (None, ""):
                rss_vals.append(float(x) / (1024 * 1024))

            x = row.get("system_ram_percent")
            if x not in (None, ""):
                ram_pct_vals.append(float(x))

    if train_times:
        data["avg_train_epoch_seconds"] = mean(train_times)
    if test_times:
        data["avg_test_epoch_seconds"] = mean(test_times)

    combined = train_times + test_times
    if combined:
        data["avg_epoch_time_seconds"] = mean(combined)

    if gpu_alloc:
        data["max_gpu_allocated_mb"] = max(gpu_alloc)
    if gpu_reserved:
        data["max_gpu_reserved_mb"] = max(gpu_reserved)
    if rss_vals:
        data["avg_process_rss_mb"] = mean(rss_vals)
    if ram_pct_vals:
        data["avg_system_ram_percent"] = mean(ram_pct_vals)

    return data


def parse_text_value(text_path: Path, label: str):
    if not text_path.exists():
        return None
    text = text_path.read_text()
    m = re.search(label + r":\s*(-?[0-9.]+)", text)
    return float(m.group(1)) if m else None


rows = []
for case in CASES:
    row = {"case": case["name"]}
    row.update(parse_accuracy_and_loss(case["metrics"]))
    row.update(parse_runtime_summary(case["runtime"]))
    row.update(parse_epoch_profile(case["epoch"]))

    if row["avg_epoch_time_seconds"] is None and row["avg_epoch_time_seconds_from_metrics"] is not None:
        row["avg_epoch_time_seconds"] = row["avg_epoch_time_seconds_from_metrics"]

    row["metrics_last_updated"] = fmt_timestamp(case["metrics"])
    row["runtime_last_updated"] = fmt_timestamp(case["runtime"])
    row["epoch_last_updated"] = fmt_timestamp(case["epoch"])
    row["last_updated"] = row["runtime_last_updated"] or row["metrics_last_updated"] or row["epoch_last_updated"]

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
lines.append("# All Experiments Comparison Enriched")
lines.append("")
lines.append("| Case | Accuracy (%) | Loss | Model Size (MB) | Size Reduction (%) | Avg Train Epoch (s) | Avg Test Epoch (s) | Max GPU Alloc (MB) | Avg RSS RAM (MB) | Notes |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

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

    def fmt(x, digits):
        return "" if x is None else f"{x:.{digits}f}"

    lines.append(
        f"| {row['case']} | "
        f"{fmt(row.get('accuracy_percent'), 4)} | "
        f"{fmt(row.get('cross_entropy_loss'), 6)} | "
        f"{fmt(row.get('state_dict_size_mb'), 4)} | "
        f"{fmt(row.get('size_reduction_percent'), 4)} | "
        f"{fmt(row.get('avg_train_epoch_seconds'), 2)} | "
        f"{fmt(row.get('avg_test_epoch_seconds'), 2)} | "
        f"{fmt(row.get('max_gpu_allocated_mb'), 2)} | "
        f"{fmt(row.get('avg_process_rss_mb'), 2)} | "
        f"{', '.join(notes)} |"
    )

MD_OUT.write_text("\n".join(lines) + "\n")

print(f"Saved CSV report to: {CSV_OUT}")
print(f"Saved Markdown report to: {MD_OUT}")
