"""Routing-stat aggregation for small-scale MoE models."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from deepseek_reimpl.layers.moe_layer import DeepSeekMoELayer


@dataclass(frozen=True)
class RoutingStatsSummary:
    """Serializable routing summary aggregated across MoE layers."""

    moe_layers: int
    tokens_per_layer: list[int]
    mean_routing_entropy: float | None
    mean_expert_load_variance: float | None
    mean_aux_loss: float | None
    routing_modes: list[str]
    expert_bias_active: list[bool]
    expert_bias: list[list[float] | None]
    expert_bias_mean: list[float | None]
    expert_bias_std: list[float | None]
    expert_bias_min: list[float | None]
    expert_bias_max: list[float | None]
    expert_bias_update_rate: list[float]
    expert_bias_update_interval: list[int]
    expert_selection_counts: list[list[float]]
    expert_selection_fraction: list[list[float]]
    mean_router_probability: list[list[float]]


def summarize_routing_stats(model: nn.Module) -> RoutingStatsSummary | None:
    """Collect latest routing diagnostics from MoE layers after a forward pass."""
    moe_layers = [module for module in model.modules() if isinstance(module, DeepSeekMoELayer)]
    stats = [
        layer.last_routing_stats for layer in moe_layers if layer.last_routing_stats is not None
    ]

    if not stats:
        return None

    entropies = [float(item.routing_entropy.item()) for item in stats]
    load_variances = [float(item.expert_load_variance.item()) for item in stats]
    aux_losses = [float(item.aux_loss.item()) for item in stats]

    expert_bias_values = [
        (
            None
            if item.expert_bias is None
            else [float(value) for value in item.expert_bias.cpu().tolist()]
        )
        for item in stats
    ]

    return RoutingStatsSummary(
        moe_layers=len(stats),
        tokens_per_layer=[item.tokens for item in stats],
        mean_routing_entropy=sum(entropies) / len(entropies),
        mean_expert_load_variance=sum(load_variances) / len(load_variances),
        mean_aux_loss=sum(aux_losses) / len(aux_losses),
        routing_modes=[item.routing_mode for item in stats],
        expert_bias_active=[item.expert_bias is not None for item in stats],
        expert_bias=expert_bias_values,
        expert_bias_mean=[
            None if item.expert_bias is None else float(item.expert_bias.mean().item())
            for item in stats
        ],
        expert_bias_std=[
            None if item.expert_bias is None else float(item.expert_bias.std(unbiased=False).item())
            for item in stats
        ],
        expert_bias_min=[
            None if item.expert_bias is None else float(item.expert_bias.min().item())
            for item in stats
        ],
        expert_bias_max=[
            None if item.expert_bias is None else float(item.expert_bias.max().item())
            for item in stats
        ],
        expert_bias_update_rate=[item.expert_bias_update_rate for item in stats],
        expert_bias_update_interval=[item.expert_bias_update_interval for item in stats],
        expert_selection_counts=[
            [float(value) for value in item.expert_selection_counts.cpu().tolist()]
            for item in stats
        ],
        expert_selection_fraction=[
            [float(value) for value in item.expert_selection_fraction.cpu().tolist()]
            for item in stats
        ],
        mean_router_probability=[
            [float(value) for value in item.mean_router_probability.cpu().tolist()]
            for item in stats
        ],
    )
