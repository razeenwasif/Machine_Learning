"""GPU device helpers and memory management."""

from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceConfig:
    """Runtime device configuration."""

    device: torch.device
    deterministic: bool = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the most capable available torch device."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def configure_runtime(
    seed: Optional[int] = None,
    deterministic: bool = False,
    prefer_gpu: bool = True,
) -> DeviceConfig:
    """Configure device, seeds, and determinism flags."""
    device = get_device(prefer_gpu=prefer_gpu)

    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = not deterministic  # type: ignore[attr-defined]

    return DeviceConfig(device=device, deterministic=deterministic)


def release_gpu_cache() -> None:
    """Force release of GPU/CPU caches to avoid OOMs between trials."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
