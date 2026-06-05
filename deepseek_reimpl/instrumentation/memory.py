"""Device-aware memory instrumentation utilities."""

from __future__ import annotations

import torch


def reset_peak_memory(device: torch.device) -> None:
    """Reset CUDA peak-memory stats for CUDA devices.

    CPU devices have no CUDA peak-memory stats, so this is a no-op.
    """
    if device.type != "cuda":
        return

    if not torch.cuda.is_available():
        return

    torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_bytes(device: torch.device) -> int | None:
    """Return peak allocated CUDA memory in bytes, or None when unavailable."""
    if device.type != "cuda":
        return None

    if not torch.cuda.is_available():
        return None

    return int(torch.cuda.max_memory_allocated(device))
