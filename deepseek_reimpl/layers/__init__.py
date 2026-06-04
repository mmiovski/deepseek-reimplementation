"""Layer package exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CausalSelfAttention",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
]


def __getattr__(name: str) -> Any:
    """Lazily expose layer package objects without creating import cycles."""
    if name == "CausalSelfAttention":
        from deepseek_reimpl.layers.attention import CausalSelfAttention

        return CausalSelfAttention

    if name == "RMSNorm":
        from deepseek_reimpl.layers.rmsnorm import RMSNorm

        return RMSNorm

    if name == "RotaryEmbedding":
        from deepseek_reimpl.layers.rope import RotaryEmbedding

        return RotaryEmbedding

    if name == "SwiGLU":
        from deepseek_reimpl.layers.swiglu import SwiGLU

        return SwiGLU

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
