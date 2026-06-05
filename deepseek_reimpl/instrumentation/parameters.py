"""Parameter-counting utilities."""

from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count all parameters in a model."""
    return int(sum(parameter.numel() for parameter in model.parameters()))


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that require gradients."""
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
