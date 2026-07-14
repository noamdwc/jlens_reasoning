"""Compute-device selection."""

import torch


def select_device(*, require_cuda: bool = False) -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if require_cuda:
        raise RuntimeError("CUDA was required but is not available")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
