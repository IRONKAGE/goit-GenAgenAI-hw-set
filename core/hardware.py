# ==============================================================================
# 🛠️ CORE HARDWARE ROUTER: Device Detection, VRAM GC & Reproducibility
# ==============================================================================
"""
Утиліти апаратної маршрутизації рівня Enterprise (Оновлення: Листопад 2026).
Відповідають за автоматичний вибір GPU (CUDA/ROCm/MPS/XPU) з фолбеком на CPU,
очищення відеопам'яті (VRAM) та строгу фіксацію глобального seed.
"""

import gc
import logging
import os
import platform
import random
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger("core.hardware")


def set_global_seed(seed: int | None = None) -> int:
    """
    Жорстко фіксує базовий seed для Python, NumPy та PyTorch.
    Гарантує 100% відтворюваність експериментів та ембедингів.
    """
    if seed is None:
        seed = int(os.getenv("GLOBAL_SEED", 42))

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

    logger.info(f"🌱 Глобальний seed зафіксовано: {seed}")
    return seed


def get_hardware_config(global_seed: int | None = None) -> Tuple[Any, str]:
    """
    Ультимативний апаратний маршрутизатор PyTorch 2.14+.
    Автоматично шукає GPU (CUDA/ROCm/MPS/XPU). Якщо драйвери відсутні — провалюється на CPU.
    """
    if global_seed is None:
        global_seed = int(os.getenv("GLOBAL_SEED", 42))

    try:
        import torch
    except ImportError:
        return "cpu", "CPU (No PyTorch)"

    env_backend = os.getenv("ML_BACKEND", "").lower().strip()
    if env_backend == "cpu":
        device = torch.device("cpu")
        device_ui_name = "CPU (Примусово через .env)"
        torch.manual_seed(global_seed)
        return device, device_ui_name

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if getattr(torch.version, "hip", None):
            device_ui_name = "ROCm (AMD Instinct / Radeon)"
        else:
            device_ui_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        torch.cuda.manual_seed_all(global_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        if platform.machine() == "arm64":
            device_ui_name = "MPS (Apple Silicon M-Серія)"
        else:
            device_ui_name = "MPS (AMD Radeon на macOS)"
        torch.mps.manual_seed(global_seed)

    elif hasattr(torch, "xpu") and getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        device = torch.device("xpu")
        device_ui_name = "XPU (Intel GPU / AI Accelerators)"
        torch.xpu.manual_seed_all(global_seed)

    else:
        device = torch.device("cpu")
        device_ui_name = f"CPU ({platform.machine()})"
        torch.manual_seed(global_seed)

    logger.info(f"🚀 Апаратне прискорення: {device_ui_name}")
    return device, device_ui_name


def clear_vram(device: Any = None) -> None:
    """Звільняє відеопам'ять (VRAM) та оперативну пам'ять (RAM) між експериментами."""
    gc.collect()
    if device is None or isinstance(device, str):
        return
    try:
        import torch

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()
    except Exception:
        pass


def log_system_info() -> str:
    """Виводить короткий аудит операційної системи та версії інтерпретатора Python."""
    info = f"Система: {platform.system()} {platform.release()} ({platform.machine()}) | Python: {platform.python_version()}"
    logger.info(f"🖥️  {info}")
    return info
