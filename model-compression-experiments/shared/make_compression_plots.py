import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/users/kashkoul/model-compression-experiments")
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = REPORTS_DIR / "comparison_all_experiments.csv"

OUT_DIR = REPORTS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "compression_plot_summary.txt"


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


def load_rows():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV report: {CSV_PATH}")

    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        row["accuracy_percent"] = to_float(row.get("accuracy_percent"))
        row["cross_entropy_loss"] = to_float(row.get("cross_entropy_loss"))
        row["state_dict_size_mb"] = to_float(row.get("state_dict_size_mb"))
        row["size_reduction_percent"] = to_float(row.get("size_reduction_percent"))
        row["avg_epoch_time_seconds"] = to_float(row.get("avg_epoch_time_seconds"))

    return rows


def keep_rows(rows, key):
    return [r for r in rows if r.get(key) is not None]


def labels_and_values(rows, key):
    return [r["case"] for r in rows], [r[key] for r in rows]


def save_bar_chart(rows, key, title, ylabel, filename, rotate=30):
    rows = keep_rows(rows, key)
    if not rows:
        return None

    labels, values = labels_and_values(rows, key)

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Case")
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()

    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def save_baseline_relative_accuracy(rows):
    baseline_row = next((r for r in rows if r["case"] == "baseline"), None)
    if baseline_row is None or baseline_row["accuracy_percent"] is None:
        return None

    baseline_acc = baseline_row["accuracy_percent"]

    usable = []
    for row in rows:
        if row["accuracy_percent"] is None:
            continue
        delta = row["accuracy_percent"] - baseline_acc
        usable.append((row["case"], delta))

    if not usable:
        return None

    labels = [x[0] for x in usable]
    values = [x[1] for x in usable]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.axhline(0.0, linewidth=1.0)
    plt.title("Accuracy Delta vs Baseline")
    plt.xlabel("Case")
    plt.ylabel("Accuracy Difference (percentage points)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_path = OUT_DIR / "accuracy_delta_vs_baseline.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def save_tradeoff_scatter(rows):
    usable = [
        r for r in rows
        if r["accuracy_percent"] is not None and r["state_dict_size_mb"] is not None
    ]
    if not usable:
        return None

    plt.figure(figsize=(9, 7))

    for row in usable:
        x = row["state_dict_size_mb"]
        y = row["accuracy_percent"]
        label = row["case"]
        plt.scatter([x], [y])
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))

    plt.title("Accuracy vs Model Size")
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()

    out_path = OUT_DIR / "accuracy_vs_model_size.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def write_summary(rows):
    lines = []
    lines.append("===== COMPRESSION PLOTS SUMMARY =====")
    lines.append("")

    acc_rows = keep_rows(rows, "accuracy_percent")
    if acc_rows:
        best_acc = max(acc_rows, key=lambda r: r["accuracy_percent"])
        lines.append(
            f"Best accuracy case: {best_acc['case']} ({best_acc['accuracy_percent']:.4f}%)"
        )

    size_rows = keep_rows(rows, "state_dict_size_mb")
    if size_rows:
        smallest = min(size_rows, key=lambda r: r["state_dict_size_mb"])
        lines.append(
            f"Smallest model case: {smallest['case']} ({smallest['state_dict_size_mb']:.4f} MB)"
        )

    reduction_rows = keep_rows(rows, "size_reduction_percent")
    if reduction_rows:
        best_reduction = max(reduction_rows, key=lambda r: r["size_reduction_percent"])
        lines.append(
            f"Largest size reduction case: {best_reduction['case']} ({best_reduction['size_reduction_percent']:.4f}%)"
        )

    lines.append("")
    lines.append("Per-case summary:")
    for row in rows:
        lines.append(
            f"- {row['case']}: "
            f"accuracy={row['accuracy_percent']}, "
            f"loss={row['cross_entropy_loss']}, "
            f"size_mb={row['state_dict_size_mb']}, "
            f"size_reduction_percent={row['size_reduction_percent']}, "
            f"avg_epoch_time_seconds={row['avg_epoch_time_seconds']}"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main():
    rows = load_rows()

    created = []

    for spec in [
        ("accuracy_percent", "Accuracy Comparison Across All Cases", "Accuracy (%)", "accuracy_comparison.png"),
        ("cross_entropy_loss", "Loss Comparison Across All Cases", "Cross Entropy Loss", "loss_comparison.png"),
        ("state_dict_size_mb", "Model Size Comparison Across All Cases", "Model Size (MB)", "model_size_comparison.png"),
        ("size_reduction_percent", "Size Reduction Comparison Across All Cases", "Size Reduction (%)", "size_reduction_comparison.png"),
        ("avg_epoch_time_seconds", "Average Epoch Time Comparison Across All Cases", "Seconds", "avg_epoch_time_comparison.png"),
    ]:
        path = save_bar_chart(rows, *spec)
        if path is not None:
            created.append(path)

    path = save_baseline_relative_accuracy(rows)
    if path is not None:
        created.append(path)

    path = save_tradeoff_scatter(rows)
    if path is not None:
        created.append(path)

    write_summary(rows)

    print("Created plot files:")
    for p in created:
        print(f" - {p}")
    print(f"Created summary file: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
