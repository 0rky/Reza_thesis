import csv
import json
import os
import re
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path("/users/kashkoul/model-compression-experiments")

BASELINE_DIR = PROJECT_ROOT / "baseline" / "outputs"
MIXED_DIR = PROJECT_ROOT / "mixed_precision" / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_RUNTIME = BASELINE_DIR / "baseline_runtime_summary.txt"
BASELINE_EPOCH = BASELINE_DIR / "baseline_epoch_profile.csv"
BASELINE_METRICS = BASELINE_DIR / "baseline_metrics.txt"

MIXED_RUNTIME = MIXED_DIR / "mixed_precision_runtime_summary.txt"
MIXED_EPOCH = MIXED_DIR / "mixed_precision_epoch_profile.csv"
MIXED_METRICS = MIXED_DIR / "mixed_precision_metrics.txt"

CSV_OUT = REPORTS_DIR / "comparison_baseline_vs_mixed_precision.csv"
MD_OUT = REPORTS_DIR / "comparison_baseline_vs_mixed_precision.md"


def parse_metrics_accuracy(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None

    text = metrics_path.read_text()
    match = re.search(r"Accuracy:\s*([0-9.]+)%", text)
    if match:
        return float(match.group(1))
    return None


def parse_runtime_summary(runtime_path: Path) -> dict:
    data = {
        "total_parameters": None,
        "trainable_parameters": None,
        "optimizer_state_size_bytes": None,
        "state_dict_size_bytes": None,
        "onnx_size_bytes": None,
        "state_dict_size_mb": None,
        "onnx_size_mb": None,
        "amp_enabled": None,
        "autocast_dtype": None,
    }

    if not runtime_path.exists():
        return data

    text = runtime_path.read_text().splitlines()

    patterns = {
        "total_parameters": r"Total parameters:\s*(\d+)",
        "trainable_parameters": r"Trainable parameters:\s*(\d+)",
        "optimizer_state_size_bytes": r"Optimizer state size \(bytes\):\s*(\d+)",
        "state_dict_size_bytes": r"State dict size \(bytes\):\s*(\d+)",
        "onnx_size_bytes": r"ONNX size \(bytes\):\s*(\d+)",
        "state_dict_size_mb": r"State dict size \(MB\):\s*([0-9.]+)",
        "onnx_size_mb": r"ONNX size \(MB\):\s*([0-9.]+)",
        "amp_enabled": r"AMP enabled:\s*(True|False)",
        "autocast_dtype": r"Autocast dtype:\s*(.+)",
    }

    for line in text:
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                if key in {
                    "total_parameters",
                    "trainable_parameters",
                    "optimizer_state_size_bytes",
                    "state_dict_size_bytes",
                    "onnx_size_bytes",
                }:
                    data[key] = int(value)
                elif key in {"state_dict_size_mb", "onnx_size_mb"}:
                    data[key] = float(value)
                elif key == "amp_enabled":
                    data[key] = value == "True"
                else:
                    data[key] = value.strip()

    return data


def parse_epoch_profile(epoch_csv_path: Path) -> dict:
    result = {
        "avg_train_epoch_seconds": None,
        "avg_test_epoch_seconds": None,
        "max_gpu_allocated_bytes": None,
        "max_gpu_reserved_bytes": None,
        "avg_process_rss_bytes": None,
        "avg_system_ram_percent": None,
    }

    if not epoch_csv_path.exists():
        return result

    train_times = []
    test_times = []
    gpu_alloc_peaks = []
    gpu_res_peaks = []
    rss_values = []
    ram_percent_values = []

    with epoch_csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "").strip()

            elapsed = row.get("elapsed_seconds")
            if elapsed:
                if split == "train":
                    train_times.append(float(elapsed))
                elif split == "test":
                    test_times.append(float(elapsed))

            max_alloc = row.get("gpu_max_allocated_bytes")
            if max_alloc:
                gpu_alloc_peaks.append(int(float(max_alloc)))

            max_reserved = row.get("gpu_max_reserved_bytes")
            if max_reserved:
                gpu_res_peaks.append(int(float(max_reserved)))

            rss = row.get("process_rss_bytes")
            if rss:
                rss_values.append(int(float(rss)))

            ram_percent = row.get("system_ram_percent")
            if ram_percent:
                ram_percent_values.append(float(ram_percent))

    if train_times:
        result["avg_train_epoch_seconds"] = mean(train_times)
    if test_times:
        result["avg_test_epoch_seconds"] = mean(test_times)
    if gpu_alloc_peaks:
        result["max_gpu_allocated_bytes"] = max(gpu_alloc_peaks)
    if gpu_res_peaks:
        result["max_gpu_reserved_bytes"] = max(gpu_res_peaks)
    if rss_values:
        result["avg_process_rss_bytes"] = mean(rss_values)
    if ram_percent_values:
        result["avg_system_ram_percent"] = mean(ram_percent_values)

    return result


def bytes_to_mb(value):
    if value is None:
        return None
    return value / (1024 * 1024)


def build_case_summary(name: str, runtime_path: Path, epoch_path: Path, metrics_path: Path) -> dict:
    runtime = parse_runtime_summary(runtime_path)
    epoch = parse_epoch_profile(epoch_path)
    accuracy = parse_metrics_accuracy(metrics_path)

    return {
        "case": name,
        "final_accuracy_percent": accuracy,
        "avg_train_epoch_seconds": epoch["avg_train_epoch_seconds"],
        "avg_test_epoch_seconds": epoch["avg_test_epoch_seconds"],
        "max_gpu_allocated_bytes": epoch["max_gpu_allocated_bytes"],
        "max_gpu_allocated_mb": bytes_to_mb(epoch["max_gpu_allocated_bytes"]),
        "max_gpu_reserved_bytes": epoch["max_gpu_reserved_bytes"],
        "max_gpu_reserved_mb": bytes_to_mb(epoch["max_gpu_reserved_bytes"]),
        "avg_process_rss_bytes": epoch["avg_process_rss_bytes"],
        "avg_process_rss_mb": bytes_to_mb(epoch["avg_process_rss_bytes"]),
        "avg_system_ram_percent": epoch["avg_system_ram_percent"],
        "total_parameters": runtime["total_parameters"],
        "trainable_parameters": runtime["trainable_parameters"],
        "optimizer_state_size_bytes": runtime["optimizer_state_size_bytes"],
        "optimizer_state_size_mb": bytes_to_mb(runtime["optimizer_state_size_bytes"]),
        "state_dict_size_bytes": runtime["state_dict_size_bytes"],
        "state_dict_size_mb": runtime["state_dict_size_mb"],
        "onnx_size_bytes": runtime["onnx_size_bytes"],
        "onnx_size_mb": runtime["onnx_size_mb"],
        "amp_enabled": runtime["amp_enabled"],
        "autocast_dtype": runtime["autocast_dtype"],
    }


def write_csv_report(rows: list[dict], out_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v, digits=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_markdown_report(rows: list[dict], out_path: Path):
    if len(rows) != 2:
        raise ValueError("Expected exactly two rows: baseline and mixed_precision")

    b = rows[0]
    m = rows[1]

    lines = []
    lines.append("# Baseline vs Mixed Precision Comparison")
    lines.append("")
    lines.append("| Metric | Baseline | Mixed Precision |")
    lines.append("|---|---:|---:|")

    metrics = [
        ("Final accuracy (%)", "final_accuracy_percent"),
        ("Average train epoch time (s)", "avg_train_epoch_seconds"),
        ("Average test time (s)", "avg_test_epoch_seconds"),
        ("Max GPU allocated (MB)", "max_gpu_allocated_mb"),
        ("Max GPU reserved (MB)", "max_gpu_reserved_mb"),
        ("Average process RSS RAM (MB)", "avg_process_rss_mb"),
        ("Average system RAM (%)", "avg_system_ram_percent"),
        ("Total parameters", "total_parameters"),
        ("Trainable parameters", "trainable_parameters"),
        ("Optimizer state size (MB)", "optimizer_state_size_mb"),
        ("State dict size (MB)", "state_dict_size_mb"),
        ("ONNX size (MB)", "onnx_size_mb"),
        ("AMP enabled", "amp_enabled"),
        ("Autocast dtype", "autocast_dtype"),
    ]

    for label, key in metrics:
        lines.append(f"| {label} | {fmt(b[key])} | {fmt(m[key])} |")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    baseline_summary = build_case_summary(
        "baseline",
        BASELINE_RUNTIME,
        BASELINE_EPOCH,
        BASELINE_METRICS,
    )

    mixed_summary = build_case_summary(
        "mixed_precision",
        MIXED_RUNTIME,
        MIXED_EPOCH,
        MIXED_METRICS,
    )

    rows = [baseline_summary, mixed_summary]

    write_csv_report(rows, CSV_OUT)
    write_markdown_report(rows, MD_OUT)

    print(f"Saved CSV report to: {CSV_OUT}")
    print(f"Saved Markdown report to: {MD_OUT}")


if __name__ == "__main__":
    main()
