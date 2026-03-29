import os
import platform
import time
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from shared.dataset import get_dataloaders
from shared.model import Net


EPOCHS = 5
SEED = 42
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


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

amp_enabled = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

train_losses = []
test_losses = []
test_accuracies = []
epoch_times = []


def train(model, device, train_loader, optimizer, epoch):
    print("inside train")
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for _, (img, classes) in enumerate(train_loader):
        classes = classes.type(torch.LongTensor)
        img, classes = img.to(device), classes.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(img)
            loss = loss_fn(output, classes)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    elapsed = time.time() - start_time
    epoch_times.append(elapsed)

    print("Train Epoch: {} Loss: {:.6f} Time: {:.2f}s".format(epoch, avg_train_loss, elapsed))


def test(model, device, test_loader, test_dataset):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for img, classes in test_loader:
            img, classes = img.to(device), classes.to(device)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                y_hat = model(img)
                batch_loss = F.cross_entropy(y_hat, classes, reduction="sum")

            test_loss += batch_loss.item()

            _, y_pred = torch.max(y_hat, 1)
            correct += (y_pred == classes).sum().item()

    test_loss /= len(test_dataset)
    accuracy = 100.0 * correct / len(test_dataset)

    test_losses.append(test_loss)
    test_accuracies.append(accuracy)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss,
        correct,
        len(test_dataset),
        accuracy
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

            with torch.cuda.amp.autocast(enabled=amp_enabled):
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
    print(f"Average epoch time: {np.mean(epoch_times):.2f}s")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    metrics_path = os.path.join(OUTPUT_DIR, "mixed_precision_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("===== MIXED PRECISION METRICS =====\n")
        f.write(f"CrossEntropyLoss: {avg_loss:.6f}\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n")
        f.write(f"AverageEpochTimeSeconds: {np.mean(epoch_times):.6f}\n\n")

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

    np.save(os.path.join(OUTPUT_DIR, "mixed_precision_confusion_matrix.npy"), cm)

    with open(os.path.join(OUTPUT_DIR, "mixed_precision_per_class_accuracy.txt"), "w") as f:
        for i in range(len(class_names)):
            f.write(f"{class_names[i]}: {per_class_acc[i]:.4f}\n")

    with open(os.path.join(OUTPUT_DIR, "mixed_precision_history.csv"), "w") as f:
        f.write("epoch,train_loss,test_loss,test_accuracy,epoch_time_seconds\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1},{train_losses[i]:.6f},{test_losses[i]:.6f},{test_accuracies[i]:.4f},{epoch_times[i]:.6f}\n")

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix - Mixed Precision")
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
    plt.savefig(os.path.join(OUTPUT_DIR, "mixed_precision_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Test Loss - Mixed Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mixed_precision_loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy - Mixed Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mixed_precision_accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_times) + 1), epoch_times, label="Epoch Time (s)")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.title("Epoch Time - Mixed Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mixed_precision_epoch_time_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved metrics to: {metrics_path}")
    print("Saved confusion matrix to mixed_precision/outputs/mixed_precision_confusion_matrix.png")
    print("Saved loss curve to mixed_precision/outputs/mixed_precision_loss_curve.png")
    print("Saved accuracy curve to mixed_precision/outputs/mixed_precision_accuracy_curve.png")
    print("Saved epoch time curve to mixed_precision/outputs/mixed_precision_epoch_time_curve.png")


if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, test_dataset)

    state_dict_path = os.path.join(CHECKPOINT_DIR, "mixed_precision_state_dict.pth")
    torch.save(model.state_dict(), state_dict_path)
    print(f"Saved: {state_dict_path}")

    model.eval()
    dummy_input = torch.randn(1, 1, 224, 224, device=device)

    onnx_path = os.path.join(CHECKPOINT_DIR, "mixed_precision.onnx")

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
