"""Rotary positional embedding utilities."""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of features for rotary positional embeddings."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding:
    """Applies rotary positional embeddings to query/key tensors."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        if head_dim <= 0:
            msg = "head_dim must be positive"
            raise ValueError(msg)
        if head_dim % 2 != 0:
            msg = "head_dim must be even for rotary embeddings"
            raise ValueError(msg)

        self.head_dim = head_dim
        self.base = base

    def _cos_sin(
        self,
        *,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, device=device, dtype=dtype) / self.head_dim)
        )
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors shaped (batch, heads, seq, dim)."""
        if q.shape != k.shape:
            msg = "q and k must have matching shapes"
            raise ValueError(msg)
        if q.shape[-1] != self.head_dim:
            msg = "last dimension must equal head_dim"
            raise ValueError(msg)

        seq_len = q.shape[-2]
        cos, sin = self._cos_sin(seq_len=seq_len, device=q.device, dtype=q.dtype)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot
