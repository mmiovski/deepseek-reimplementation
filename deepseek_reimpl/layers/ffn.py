"""Dense feedforward block factory."""

from __future__ import annotations

import torch
from torch import nn

from deepseek_reimpl.layers.moe_layer import DeepSeekMoELayer
from deepseek_reimpl.layers.swiglu import SwiGLU
from deepseek_reimpl.model.config import GPTConfig


class GELUMLP(nn.Module):
    """Standard GELU MLP feedforward block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP block."""
        output: torch.Tensor = self.net(x)
        return output


def build_ffn(config: GPTConfig) -> nn.Module:
    """Build the configured feedforward module."""
    if config.ffn_type == "swiglu":
        return SwiGLU(config.d_model, config.d_ff, config.dropout)

    if config.ffn_type == "gelu_mlp":
        return GELUMLP(config.d_model, config.d_ff, config.dropout)

    if config.ffn_type == "moe":
        if (
            config.n_routed_experts is None
            or config.moe_top_k is None
            or config.moe_expert_d_ff is None
        ):
            msg = "MoE config fields must be set when ffn_type='moe'"
            raise ValueError(msg)

        return DeepSeekMoELayer(
            d_model=config.d_model,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            top_k=config.moe_top_k,
            expert_d_ff=config.moe_expert_d_ff,
            dropout=config.dropout,
            router_score=config.moe_router_score,
            normalize_top_k_weights=config.moe_normalize_top_k_weights,
            aux_loss_weight=config.moe_aux_loss_weight,
            routing_mode=config.moe_routing_mode,
            use_expert_bias=config.moe_use_expert_bias,
            expert_bias_update_rate=config.moe_expert_bias_update_rate,
            expert_bias_update_interval=config.moe_expert_bias_update_interval,
            expert_bias_min=config.moe_expert_bias_min,
            expert_bias_max=config.moe_expert_bias_max,
        )

    msg = f"Unsupported ffn_type: {config.ffn_type}"
    raise ValueError(msg)
