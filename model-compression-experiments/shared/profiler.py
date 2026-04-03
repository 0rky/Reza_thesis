import csv
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def now_str():
    return datetime.now().isoformat()


def get_basic_system_info():
    info = {
        "timestamp": now_str(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "executable": sys.executable,
        "user": os.getenv("USER"),
        "conda_env": os.getenv("CONDA_DEFAULT_ENV"),
    }

    if torch is not None:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["current_device"] = torch.cuda.current_device()
        else:
            info["gpu_count"] = 0
            info["gpu_name"] = None
            info["current_device"] = None

    return info


def get_ram_info():
    if psutil is None:
        return {
            "process_rss_bytes": None,
            "process_vms_bytes": None,
            "system_total_ram_bytes": None,
            "system_available_ram_bytes": None,
            "system_used_ram_bytes": None,
            "system_ram_percent": None,
        }

    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    vm = psutil.virtual_memory()

    return {
        "process_rss_bytes": mem.rss,
        "process_vms_bytes": mem.vms,
        "system_total_ram_bytes": vm.total,
        "system_available_ram_bytes": vm.available,
        "system_used_ram_bytes": vm.used,
        "system_ram_percent": vm.percent,
    }


def get_gpu_memory_info():
    if torch is None or not torch.cuda.is_available():
        return {
            "gpu_allocated_bytes": None,
            "gpu_reserved_bytes": None,
            "gpu_max_allocated_bytes": None,
            "gpu_max_reserved_bytes": None,
        }

    return {
        "gpu_allocated_bytes": torch.cuda.memory_allocated(),
        "gpu_reserved_bytes": torch.cuda.memory_reserved(),
        "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def reset_gpu_peak_memory():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_file_size_bytes(path):
    if path is None or not os.path.exists(path):
        return None
    return os.path.getsize(path)


def get_optimizer_state_size_bytes(optimizer):
    total_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch is not None and isinstance(value, torch.Tensor):
                total_bytes += value.numel() * value.element_size()
    return total_bytes


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class EpochProfiler:
    def __init__(self):
        self.rows = []

    def record_epoch(self, epoch, split_name, elapsed_seconds, extra=None):
        row = {
            "epoch": epoch,
            "split": split_name,
            "elapsed_seconds": elapsed_seconds,
        }
        row.update(get_ram_info())
        row.update(get_gpu_memory_info())
        if extra:
            row.update(extra)
        self.rows.append(row)

    def save_csv(self, path):
        write_csv(path, self.rows)


def _shape_to_str(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    if isinstance(x, (list, tuple)):
        shapes = []
        for item in x:
            if torch is not None and isinstance(item, torch.Tensor):
                shapes.append(tuple(item.shape))
            else:
                shapes.append(type(item).__name__)
        return str(shapes)
    return type(x).__name__


class LayerProfiler:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.rows = []
        self.handles = []

    def _make_pre_hook(self):
        def pre_hook(module, inputs):
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            module._profile_start_time = time.perf_counter()
        return pre_hook

    def _make_hook(self, layer_name):
        def hook(module, inputs, output):
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            start_time = getattr(module, "_profile_start_time", None)
            elapsed_ms = None
            if start_time is not None:
                elapsed_ms = (end_time - start_time) * 1000.0

            row = {
                "layer_name": layer_name,
                "layer_type": module.__class__.__name__,
                "output_shape": _shape_to_str(output),
                "elapsed_ms": elapsed_ms,
            }
            row.update(get_ram_info())
            row.update(get_gpu_memory_info())
            self.rows.append(row)
        return hook

    def attach(self):
        if torch is None:
            return

        nn = torch.nn

        for name, module in self.model.named_modules():
            if name == "":
                continue

            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.Dropout, nn.ReLU)):
                self.handles.append(module.register_forward_pre_hook(self._make_pre_hook()))
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def clear(self):
        self.rows = []

    def profile_single_batch(self, batch_tensor):
        self.clear()
        self.model.eval()

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = self.model(batch_tensor)

        return self.rows

    def save_csv(self, path):
        write_csv(path, self.rows)


def profile_model_layers(model, device, sample_batch, output_path):
    profiler = LayerProfiler(model, device)
    profiler.attach()
    profiler.profile_single_batch(sample_batch)
    profiler.detach()
    profiler.save_csv(output_path)

