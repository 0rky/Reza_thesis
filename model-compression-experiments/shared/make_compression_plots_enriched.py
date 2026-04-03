import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/users/kashkoul/model-compression-experiments")
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots_enriched"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = REPORTS_DIR / "comparison_all_experiments_enriched.csv"
SUMMARY_PATH = PLOTS_DIR / "plot_summary_enriched.txt"


def to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def read_rows():
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "case": row["case"],
                "accuracy_percent": to_float(row.get("accuracy_percent")),
                "cross_entropy_loss": to_float(row.get("cross_entropy_loss")),
                "state_dict_size_mb": to_float(row.get("state_dict_size_mb")) or to_float(row.get("state_dict_size_mb_runtime")),
                "size_reduction_percent": to_float(row.get("size_reduction_percent")),
                "avg_train_epoch_seconds": to_float(row.get("avg_train_epoch_seconds")),
                "avg_test_epoch_seconds": to_float(row.get("avg_test_epoch_seconds")),
                "avg_epoch_time_seconds": to_float(row.get("avg_epoch_time_seconds")),
                "max_gpu_allocated_mb": to_float(row.get("max_gpu_allocated_mb")),
                "max_gpu_reserved_mb": to_float(row.get("max_gpu_reserved_mb")),
                "avg_process_rss_mb": to_float(row.get("avg_process_rss_mb")),
            })
    return rows


def bar_plot(rows, key, title, ylabel, filename):
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return None

    labels = [r["case"] for r in valid]
    values = [r[key] for r in valid]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Case")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def scatter_plot(rows, x_key, y_key, title, xlabel, ylabel, filename):
    valid = [r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    if not valid:
        return None

    plt.figure(figsize=(9, 6))
    xs = [r[x_key] for r in valid]
    ys = [r[y_key] for r in valid]

    plt.scatter(xs, ys)

    for r in valid:
        plt.annotate(r["case"], (r[x_key], r[y_key]), fontsize=8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    rows = read_rows()
    created = []

    for spec in [
        ("accuracy_percent", "Accuracy Comparison Across All Cases", "Accuracy (%)", "accuracy_comparison.png"),
        ("cross_entropy_loss", "Loss Comparison Across All Cases", "CrossEntropyLoss", "loss_comparison.png"),
        ("state_dict_size_mb", "Model Size Comparison Across All Cases", "Model Size (MB)", "model_size_comparison.png"),
        ("size_reduction_percent", "Size Reduction Comparison Across All Cases", "Size Reduction (%)", "size_reduction_comparison.png"),
        ("avg_train_epoch_seconds", "Average Train Epoch Time Comparison", "Seconds", "avg_train_epoch_time_comparison.png"),
        ("avg_test_epoch_seconds", "Average Test Epoch Time Comparison", "Seconds", "avg_test_epoch_time_comparison.png"),
        ("max_gpu_allocated_mb", "Max GPU Allocated Memory Comparison", "MB", "max_gpu_allocated_comparison.png"),
        ("max_gpu_reserved_mb", "Max GPU Reserved Memory Comparison", "MB", "max_gpu_reserved_comparison.png"),
        ("avg_process_rss_mb", "Average Process RSS RAM Comparison", "MB", "avg_rss_ram_comparison.png"),
    ]:
        out = bar_plot(rows, *spec)
        if out is not None:
            created.append(out)

    for spec in [
        ("state_dict_size_mb", "accuracy_percent", "Accuracy vs Model Size", "Model Size (MB)", "Accuracy (%)", "accuracy_vs_model_size.png"),
        ("avg_train_epoch_seconds", "accuracy_percent", "Accuracy vs Train Time", "Average Train Epoch Time (s)", "Accuracy (%)", "accuracy_vs_train_time.png"),
        ("max_gpu_allocated_mb", "accuracy_percent", "Accuracy vs Max GPU Memory", "Max GPU Allocated (MB)", "Accuracy (%)", "accuracy_vs_gpu_memory.png"),
    ]:
        out = scatter_plot(rows, *spec)
        if out is not None:
            created.append(out)

    best_accuracy = max((r for r in rows if r["accuracy_percent"] is not None), key=lambda x: x["accuracy_percent"], default=None)
    smallest_model = min((r for r in rows if r["state_dict_size_mb"] is not None), key=lambda x: x["state_dict_size_mb"], default=None)
    fastest_train = min((r for r in rows if r["avg_train_epoch_seconds"] is not None), key=lambda x: x["avg_train_epoch_seconds"], default=None)
    lowest_gpu = min((r for r in rows if r["max_gpu_allocated_mb"] is not None), key=lambda x: x["max_gpu_allocated_mb"], default=None)

    with open(SUMMARY_PATH, "w") as f:
        f.write("===== ENRICHED PLOT SUMMARY =====\n\n")
        if best_accuracy:
            f.write(f"Best accuracy: {best_accuracy['case']} ({best_accuracy['accuracy_percent']:.4f}%)\n")
        if smallest_model:
            f.write(f"Smallest model: {smallest_model['case']} ({smallest_model['state_dict_size_mb']:.4f} MB)\n")
        if fastest_train:
            f.write(f"Fastest train epoch: {fastest_train['case']} ({fastest_train['avg_train_epoch_seconds']:.2f} s)\n")
        if lowest_gpu:
            f.write(f"Lowest max GPU allocated: {lowest_gpu['case']} ({lowest_gpu['max_gpu_allocated_mb']:.2f} MB)\n")

        f.write("\nCreated plot files:\n")
        for p in created:
            f.write(f"- {p}\n")

    print("Created plot files:")
    for p in created:
        print(f" - {p}")
    print(f"Created summary file: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
