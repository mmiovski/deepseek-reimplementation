"""Model factory utilities."""

from __future__ import annotations

from typing import Any

from torch import nn

from deepseek_reimpl.model.baseline_gpt import BaselineGPT
from deepseek_reimpl.model.config import GPTConfig


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    """Build a model from a loaded model configuration dictionary."""
    model_config = config.get("model", config)
    model_name = model_config.get("name")

    if model_name in {"baseline_gpt", "mla_gpt"}:
        return BaselineGPT(GPTConfig.from_dict(config))

    msg = f"Unsupported model name: {model_name}"
    raise ValueError(msg)
