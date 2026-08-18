"""Routing-stat aggregation for small-scale MoE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
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


@dataclass
class _RoutingLayerAggregate:
    tokens: int
    top_k: int
    n_experts: int
    aux_loss_weight: float
    counts: torch.Tensor
    probability_sum: torch.Tensor
    entropy_sum: float
    routing_mode: str


class RoutingStatsAccumulator:
    """Aggregate routing diagnostics across every evaluated batch."""

    def __init__(self) -> None:
        self._layers: list[_RoutingLayerAggregate] | None = None

    def update(self, model: nn.Module) -> None:
        modules = [module for module in model.modules() if isinstance(module, DeepSeekMoELayer)]
        if not modules:
            return
        if self._layers is None:
            self._layers = [
                _RoutingLayerAggregate(
                    tokens=0,
                    top_k=module.top_k,
                    n_experts=module.n_routed_experts,
                    aux_loss_weight=module.aux_loss_weight,
                    counts=torch.zeros(module.n_routed_experts, dtype=torch.float64),
                    probability_sum=torch.zeros(module.n_routed_experts, dtype=torch.float64),
                    entropy_sum=0.0,
                    routing_mode=module.routing_mode,
                )
                for module in modules
            ]
        if len(modules) != len(self._layers):
            raise RuntimeError("MoE layer count changed during evaluation")
        for aggregate, module in zip(self._layers, modules, strict=True):
            stats = module.last_routing_stats
            if stats is None:
                raise RuntimeError("MoE layer did not expose routing statistics")
            tokens = stats.tokens
            aggregate.tokens += tokens
            aggregate.counts += stats.expert_selection_counts.cpu().double()
            aggregate.probability_sum += stats.mean_router_probability.cpu().double() * tokens
            aggregate.entropy_sum += float(stats.routing_entropy.item()) * tokens

    def summary(self) -> dict[str, object] | None:
        if self._layers is None:
            return None
        layer_records: list[dict[str, Any]] = []
        for layer_index, aggregate in enumerate(self._layers):
            tokens = aggregate.tokens
            top_k = aggregate.top_k
            counts = aggregate.counts
            probability_sum = aggregate.probability_sum
            fractions = counts / float(tokens * top_k)
            mean_probability = probability_sum / tokens
            aggregate_aux_loss = (
                aggregate.n_experts
                * float(torch.sum(fractions * mean_probability).item())
                * aggregate.aux_loss_weight
            )
            layer_records.append(
                {
                    "layer_index": layer_index,
                    "tokens": tokens,
                    "top_k": top_k,
                    "expert_selection_counts": counts.tolist(),
                    "expert_selection_fraction": fractions.tolist(),
                    "expert_load_variance": float(torch.var(fractions, unbiased=False).item()),
                    "routing_entropy": aggregate.entropy_sum / tokens,
                    "mean_aux_loss": aggregate_aux_loss,
                    "mean_router_probability": mean_probability.tolist(),
                    "routing_mode": aggregate.routing_mode,
                }
            )
        return {
            "aggregation": "complete_evaluation_sample",
            "moe_layers": len(layer_records),
            "layers": layer_records,
            "mean_routing_entropy": sum(
                float(record["routing_entropy"]) for record in layer_records
            )
            / len(layer_records),
            "mean_expert_load_variance": sum(
                float(record["expert_load_variance"]) for record in layer_records
            )
            / len(layer_records),
            "mean_aux_loss": sum(float(record["mean_aux_loss"]) for record in layer_records)
            / len(layer_records),
        }


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
