import os
import platform
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import torch.nn.utils.prune as prune

import time
import sys
import subprocess

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from shared.dataset import get_dataloaders
from shared.model import Net
from shared.profiler import (
    profile_model_layers,
    EpochProfiler,
    get_basic_system_info,
    get_file_size_bytes,
    get_optimizer_state_size_bytes,
    write_json,
)


EPOCHS = int(os.getenv("EPOCHS", "5"))
SEED = 42
PRUNING_AMOUNT = 0.30
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SYSTEM_INFO_PATH = os.path.join(OUTPUT_DIR, "pruning_system_info.json")
EPOCH_PROFILE_PATH = os.path.join(OUTPUT_DIR, "pruning_epoch_profile.csv")
RUNTIME_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "pruning_runtime_summary.txt")
LAYER_PROFILE_DENSE_PATH = os.path.join(OUTPUT_DIR, "pruning_dense_layer_profile.csv")
LAYER_PROFILE_PRUNED_PATH = os.path.join(OUTPUT_DIR, "pruning_pruned_layer_profile.csv")


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

train_dataset, test_dataset, train_loader, test_loader, class_names = get_dataloaders()

model = Net().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

train_losses = []
test_losses = []
test_accuracies = []

profiler = EpochProfiler()
write_json(SYSTEM_INFO_PATH, get_basic_system_info())


def train(model, device, train_loader, optimizer, epoch):
    print("inside train")
    model.train()
    running_loss = 0.0
    start_time = time.time()

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
    elapsed = time.time() - start_time

    print("Train Epoch: {} Loss: {:.6f} Time: {:.2f}s".format(epoch, avg_train_loss, elapsed))

    profiler.record_epoch(
        epoch,
        "train",
        elapsed,
        extra={"train_loss": avg_train_loss}
    )


def test(model, device, test_loader, test_dataset, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    start_time = time.time()

    with torch.no_grad():
        for img, classes in test_loader:
            img, classes = img.to(device), classes.to(device)
            y_hat = model(img)

            test_loss += F.cross_entropy(y_hat, classes, reduction="sum").item()

            _, y_pred = torch.max(y_hat, 1)
            correct += (y_pred == classes).sum().item()

    test_loss /= len(test_dataset)
    accuracy = 100.0 * correct / len(test_dataset)
    elapsed = time.time() - start_time

    test_losses.append(test_loss)
    test_accuracies.append(accuracy)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Eval Time: {:.2f}s\n".format(
        test_loss,
        correct,
        len(test_dataset),
        accuracy,
        elapsed
    ))
    print("=" * 30)

    profiler.record_epoch(
        epoch,
        "test",
        elapsed,
        extra={"test_loss": test_loss, "test_accuracy": accuracy}
    )


def evaluate_with_metrics(model, device, test_loader, class_names, prefix):
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

    print(f"\n===== EXTRA EVALUATION METRICS ({prefix}) =====")
    print(f"CrossEntropyLoss: {avg_loss:.6f}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    metrics_path = os.path.join(OUTPUT_DIR, f"{prefix}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"===== {prefix.upper()} METRICS =====\n")
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

    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_confusion_matrix.npy"), cm)

    with open(os.path.join(OUTPUT_DIR, f"{prefix}_per_class_accuracy.txt"), "w") as f:
        for i in range(len(class_names)):
            f.write(f"{class_names[i]}: {per_class_acc[i]:.4f}\n")

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {prefix}")
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
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to pruning/outputs/{prefix}_confusion_matrix.png")


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def compute_global_sparsity(model):
    total_zeros = 0
    total_params = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach().cpu()
            total_zeros += torch.sum(w == 0).item()
            total_params += w.numel()
    sparsity = 100.0 * total_zeros / total_params if total_params > 0 else 0.0
    return total_zeros, total_params, sparsity


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

    dense_state_path = os.path.join(CHECKPOINT_DIR, "pruning_dense_state_dict.pth")
    torch.save(model.state_dict(), dense_state_path)
    print(f"Saved: {dense_state_path}")

    model.eval()
    dummy_input = torch.randn(1, 1, 224, 224, device=device)

    onnx_path = os.path.join(CHECKPOINT_DIR, "pruning_dense.onnx")

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

    evaluate_with_metrics(model, device, test_loader, class_names, prefix="pruning_dense")
    sample_batch, _ = next(iter(test_loader))
    sample_batch = sample_batch.to(device)
    profile_model_layers(model, device, sample_batch, LAYER_PROFILE_DENSE_PATH)
    print(f"Saved layer-wise profile to: {LAYER_PROFILE_DENSE_PATH}")

    print("\nApplying global unstructured pruning...")
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=PRUNING_AMOUNT,
    )

    zeros_before_remove, total_before_remove, sparsity_before_remove = compute_global_sparsity(model)
    print(f"Pruned sparsity before remove (%): {sparsity_before_remove:.2f}")

    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")

    zeros_after_remove, total_after_remove, sparsity_after_remove = compute_global_sparsity(model)
    print(f"Pruned sparsity after remove (%): {sparsity_after_remove:.2f}")

    pruned_state_path = os.path.join(CHECKPOINT_DIR, "pruning_pruned_state_dict.pth")
    torch.save(model.state_dict(), pruned_state_path)
    print(f"Saved: {pruned_state_path}")

    dense_size_mb = get_file_size_mb(dense_state_path)
    pruned_size_mb = get_file_size_mb(pruned_state_path)
    reduction_pct = 100.0 * (dense_size_mb - pruned_size_mb) / dense_size_mb if dense_size_mb > 0 else 0.0

    print("\n===== PRUNING SIZE / SPARSITY COMPARISON =====")
    print(f"Dense model size (MB): {dense_size_mb:.4f}")
    print(f"Pruned model size (MB): {pruned_size_mb:.4f}")
    print(f"Size reduction (%): {reduction_pct:.2f}")
    print(f"Global sparsity after pruning (%): {sparsity_after_remove:.2f}")

    with open(os.path.join(OUTPUT_DIR, "pruning_size_sparsity_comparison.txt"), "w") as f:
        f.write("===== PRUNING SIZE / SPARSITY COMPARISON =====\n")
        f.write(f"Dense model size (MB): {dense_size_mb:.6f}\n")
        f.write(f"Pruned model size (MB): {pruned_size_mb:.6f}\n")
        f.write(f"Size reduction (%): {reduction_pct:.6f}\n")
        f.write(f"Global sparsity after pruning (%): {sparsity_after_remove:.6f}\n")
        f.write(f"Zero weights after pruning: {zeros_after_remove}\n")
        f.write(f"Total prunable weights: {total_after_remove}\n")

    evaluate_with_metrics(model, device, test_loader, class_names, prefix="pruning_pruned")
    sample_batch, _ = next(iter(test_loader))
    sample_batch = sample_batch.to(device)
    profile_model_layers(model, device, sample_batch, LAYER_PROFILE_PRUNED_PATH)
    print(f"Saved layer-wise profile to: {LAYER_PROFILE_PRUNED_PATH}")
    profiler.save_csv(EPOCH_PROFILE_PATH)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer_state_size_bytes = get_optimizer_state_size_bytes(optimizer)
    dense_state_size_bytes = get_file_size_bytes(dense_state_path)
    dense_onnx_size_bytes = get_file_size_bytes(onnx_path)
    pruned_state_size_bytes = get_file_size_bytes(pruned_state_path)

    with open(RUNTIME_SUMMARY_PATH, "w") as f:
        f.write("===== PRUNING RUNTIME / SYSTEM SUMMARY =====\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")
        f.write(f"Optimizer state size (bytes): {optimizer_state_size_bytes}\n")
        if dense_state_size_bytes is not None:
            f.write(f"Dense state dict size (bytes): {dense_state_size_bytes}\n")
            f.write(f"Dense state dict size (MB): {dense_state_size_bytes / (1024 * 1024):.6f}\n")
        if dense_onnx_size_bytes is not None:
            f.write(f"Dense ONNX size (bytes): {dense_onnx_size_bytes}\n")
            f.write(f"Dense ONNX size (MB): {dense_onnx_size_bytes / (1024 * 1024):.6f}\n")
        if pruned_state_size_bytes is not None:
            f.write(f"Pruned state dict size (bytes): {pruned_state_size_bytes}\n")
            f.write(f"Pruned state dict size (MB): {pruned_state_size_bytes / (1024 * 1024):.6f}\n")

    refresh_all_experiments_report()
