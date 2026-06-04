"""Model package exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaselineGPT",
    "DecoderBlock",
    "GPTConfig",
    "build_model_from_config",
]


def __getattr__(name: str) -> Any:
    """Lazily expose model package objects without creating import cycles."""
    if name == "BaselineGPT":
        from deepseek_reimpl.model.baseline_gpt import BaselineGPT

        return BaselineGPT

    if name == "DecoderBlock":
        from deepseek_reimpl.model.decoder_block import DecoderBlock

        return DecoderBlock

    if name == "GPTConfig":
        from deepseek_reimpl.model.config import GPTConfig

        return GPTConfig

    if name == "build_model_from_config":
        from deepseek_reimpl.model.model_factory import build_model_from_config

        return build_model_from_config

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
