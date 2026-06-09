"""Activated-parameter accounting for dense and MoE-style models."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from deepseek_reimpl.instrumentation.parameters import count_parameters
from deepseek_reimpl.layers.moe_layer import DeepSeekMoELayer


@dataclass(frozen=True)
class ActivatedParameterSummary:
    """Estimated activated-parameter accounting for one token."""

    total_parameters: int
    always_active_parameters: int
    routed_expert_total_parameters: int
    routed_expert_active_parameters_per_token: int
    activated_parameters_per_token: int
    activated_to_total_ratio: float


def _module_parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def summarize_activated_parameters(model: nn.Module) -> ActivatedParameterSummary:
    """Estimate activated parameters per token for dense and MoE models.

    Dense parameters are treated as always active. For MoE layers, shared experts
    are always active because every token passes through them, while only top-k
    routed experts are counted as active per token.
    """
    total_parameters = count_parameters(model)
    moe_layers = [module for module in model.modules() if isinstance(module, DeepSeekMoELayer)]

    routed_expert_total = 0
    routed_expert_active = 0

    for layer in moe_layers:
        per_routed_expert_counts = [
            _module_parameter_count(expert) for expert in layer.routed_experts
        ]
        routed_expert_total += sum(per_routed_expert_counts)

        if per_routed_expert_counts:
            # Phase 5 uses homogeneous expert dimensions. If that changes later,
            # average selected-expert parameters are a transparent estimate.
            average_expert_params = int(
                round(sum(per_routed_expert_counts) / len(per_routed_expert_counts))
            )
            routed_expert_active += layer.top_k * average_expert_params

    always_active = total_parameters - routed_expert_total
    activated_per_token = always_active + routed_expert_active
    ratio = 0.0 if total_parameters == 0 else activated_per_token / total_parameters

    return ActivatedParameterSummary(
        total_parameters=total_parameters,
        always_active_parameters=always_active,
        routed_expert_total_parameters=routed_expert_total,
        routed_expert_active_parameters_per_token=routed_expert_active,
        activated_parameters_per_token=activated_per_token,
        activated_to_total_ratio=ratio,
    )
