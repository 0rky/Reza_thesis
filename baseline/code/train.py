import os
import platform
import time
import sys
import subprocess
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from shared.dataset import get_dataloaders
from shared.model import Net
from shared.profiler import (
    EpochProfiler,
    count_parameters,
    get_basic_system_info,
    get_file_size_bytes,
    get_optimizer_state_size_bytes,
    reset_gpu_peak_memory,
    write_json,
    profile_model_layers,
)

EPOCHS = int(os.getenv("EPOCHS", "5"))
SEED = 42
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SYSTEM_INFO_PATH = os.path.join(OUTPUT_DIR, "baseline_system_info.json")
EPOCH_PROFILE_PATH = os.path.join(OUTPUT_DIR, "baseline_epoch_profile.csv")
RUNTIME_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "baseline_runtime_summary.txt")
LAYER_PROFILE_PATH = os.path.join(OUTPUT_DIR, "baseline_layer_profile.csv")

print("Hostname:", platform.node())
print("User:", os.getenv("USER"))
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version (PyTorch):", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
else:
    print("No GPU detected by PyTorch.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

profiler = EpochProfiler()
write_json(SYSTEM_INFO_PATH, get_basic_system_info())

train_dataset, test_dataset, train_loader, test_loader, class_names = get_dataloaders()

model = Net().to(device)
total_params, trainable_params = count_parameters(model)
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

train_losses = []
test_losses = []
test_accuracies = []


def train(model, device, train_loader, optimizer, epoch):
    print("inside train")
    model.train()
    running_loss = 0.0
    reset_gpu_peak_memory()
    epoch_start_time = time.time()

    for _, (img, classes) in enumerate(train_loader):
        classes = classes.type(torch.LongTensor)
        img, classes = img.to(device), classes.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, classes)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    elapsed_seconds = time.time() - epoch_start_time
    profiler.record_epoch(
        epoch,
        "train",
        elapsed_seconds,
        {"train_loss": avg_train_loss}
    )

    print("Train Epoch: {} Loss: {:.6f} Time: {:.2f}s".format(epoch, avg_train_loss, elapsed_seconds))


def test(model, device, test_loader, test_dataset, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    eval_start_time = time.time()

    with torch.no_grad():
        for img, classes in test_loader:
            img, classes = img.to(device), classes.to(device)
            y_hat = model(img)

            test_loss += F.cross_entropy(y_hat, classes, reduction="sum").item()

            _, y_pred = torch.max(y_hat, 1)
            correct += (y_pred == classes).sum().item()

    test_loss /= len(test_dataset)
    accuracy = 100.0 * correct / len(test_dataset)

    test_losses.append(test_loss)
    test_accuracies.append(accuracy)

    elapsed_seconds = time.time() - eval_start_time
    profiler.record_epoch(
        epoch,
        "test",
        elapsed_seconds,
        {
            "test_loss": test_loss,
            "test_accuracy": accuracy,
        }
    )

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Eval Time: {:.2f}s\n".format(
        test_loss,
        correct,
        len(test_dataset),
        accuracy,
        elapsed_seconds
    ))
    print("=" * 30)


def evaluate_with_metrics(model, device, test_loader, class_names):
    model.eval()

    all_targets = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for img, classes in test_loader:
            img, classes = img.to(device), classes.to(device)
            outputs = model(img)

            loss = loss_fn(outputs, classes)
            total_loss += loss.item() * img.size(0)

            _, preds = torch.max(outputs, 1)

            all_targets.extend(classes.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    avg_loss = total_loss / len(test_loader.dataset)
    acc = (all_targets == all_preds).mean()

    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets,
        all_preds,
        labels=list(range(len(class_names))),
        zero_division=0
    )

    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\n===== EXTRA EVALUATION METRICS =====")
    print(f"CrossEntropyLoss: {avg_loss:.6f}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    metrics_path = os.path.join(OUTPUT_DIR, "baseline_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("===== BASELINE METRICS =====\n")
        f.write(f"CrossEntropyLoss: {avg_loss:.6f}\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")

        f.write("Per-class metrics:\n")
        for i, name in enumerate(class_names):
            f.write(
                f"{name}: "
                f"precision={precision[i]:.4f}, "
                f"recall={recall[i]:.4f}, "
                f"f1={f1[i]:.4f}, "
                f"support={support[i]}, "
                f"class_accuracy={per_class_acc[i]:.4f}\n"
            )

        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    np.save(os.path.join(OUTPUT_DIR, "baseline_confusion_matrix.npy"), cm)

    with open(os.path.join(OUTPUT_DIR, "baseline_per_class_accuracy.txt"), "w") as f:
        for i, name in enumerate(class_names):
            f.write(f"{name}: {float(per_class_acc[i]):.4f}\n")

    with open(os.path.join(OUTPUT_DIR, "baseline_history.csv"), "w") as f:
        f.write("epoch,train_loss,test_loss,test_accuracy\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1},{train_losses[i]:.6f},{test_losses[i]:.6f},{test_accuracies[i]:.4f}\n")

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix - Baseline")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Test Loss - Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy - Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved metrics to: {metrics_path}")
    print("Saved confusion matrix to: outputs/baseline_confusion_matrix.png")
    print("Saved loss curve to: outputs/baseline_loss_curve.png")
    print("Saved accuracy curve to: outputs/baseline_accuracy_curve.png")


def refresh_all_experiments_report():
    try:
        subprocess.run(
            ["python3", os.path.join(ROOT_DIR, "shared", "make_all_experiments_report.py")],
            check=True
        )
        print("Updated: comparison_all_experiments report")
    except Exception as e:
        print(f"Warning: could not update all-experiments report: {e}")

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, test_dataset, epoch)

    state_dict_path = os.path.join(CHECKPOINT_DIR, "sample_network_state_dict.pth")
    torch.save(model.state_dict(), state_dict_path)
    print(f"Saved: {state_dict_path}")

    model.eval()
    dummy_input = torch.randn(1, 1, 224, 224, device=device)

    onnx_path = os.path.join(CHECKPOINT_DIR, "sample_network.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12
    )

    print(f"Saved: {onnx_path}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX check: OK")

    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 1, 224, 224).astype(np.float32)
    outputs = session.run(None, {"input": dummy})

    print("ONNX Runtime output shape:", outputs[0].shape)
    print("ONNX Runtime inference: OK")

    evaluate_with_metrics(model, device, test_loader, class_names)
    sample_batch, _ = next(iter(test_loader))
    sample_batch = sample_batch.to(device)
    profile_model_layers(model, device, sample_batch, LAYER_PROFILE_PATH)
    print(f"Saved layer-wise profile to: {LAYER_PROFILE_PATH}")


    profiler.save_csv(EPOCH_PROFILE_PATH)

    state_dict_size_bytes = get_file_size_bytes(state_dict_path)
    onnx_size_bytes = get_file_size_bytes(onnx_path)
    optimizer_state_size_bytes = get_optimizer_state_size_bytes(optimizer)

    with open(RUNTIME_SUMMARY_PATH, "w") as f:
        f.write("===== BASELINE RUNTIME / SYSTEM SUMMARY =====\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")
        f.write(f"Optimizer state size (bytes): {optimizer_state_size_bytes}\n")
        f.write(f"State dict size (bytes): {state_dict_size_bytes}\n")
        f.write(f"ONNX size (bytes): {onnx_size_bytes}\n")
        if state_dict_size_bytes is not None:
            f.write(f"State dict size (MB): {state_dict_size_bytes / (1024 * 1024):.6f}\n")
        if onnx_size_bytes is not None:
            f.write(f"ONNX size (MB): {onnx_size_bytes / (1024 * 1024):.6f}\n")
    refresh_all_experiments_report()
