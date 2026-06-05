"""Throughput measurement utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ThroughputSnapshot:
    """Point-in-time throughput measurement."""

    tokens: int
    elapsed_seconds: float
    tokens_per_second: float


class ThroughputMeter:
    """Measure token throughput over elapsed wall-clock time."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset elapsed-time and token counters."""
        self._start_time = time.perf_counter()
        self._tokens = 0

    def update(self, tokens: int) -> None:
        """Add processed tokens to the meter."""
        if tokens < 0:
            raise ValueError(f"tokens must be nonnegative, got {tokens}")
        self._tokens += tokens

    def snapshot(self) -> ThroughputSnapshot:
        """Return current throughput statistics."""
        elapsed_seconds = time.perf_counter() - self._start_time
        tokens_per_second = 0.0
        if elapsed_seconds > 0:
            tokens_per_second = self._tokens / elapsed_seconds

        return ThroughputSnapshot(
            tokens=self._tokens,
            elapsed_seconds=elapsed_seconds,
            tokens_per_second=tokens_per_second,
        )
