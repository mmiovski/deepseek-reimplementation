"""Small-scale DeepSeekMoE-style sparse feedforward layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from deepseek_reimpl.layers.moe_expert import MoEExpert
from deepseek_reimpl.layers.moe_router import RouterOutput, TopKRouter


@dataclass(frozen=True)
class MoERoutingStats:
    """Routing diagnostics captured from one MoE forward pass."""

    tokens: int
    n_routed_experts: int
    top_k: int
    expert_selection_counts: torch.Tensor
    expert_selection_fraction: torch.Tensor
    expert_load_variance: torch.Tensor
    routing_entropy: torch.Tensor
    mean_router_probability: torch.Tensor
    aux_loss: torch.Tensor
    routing_mode: str
    expert_bias: torch.Tensor | None
    expert_bias_update_rate: float
    expert_bias_update_interval: int


class DeepSeekMoELayer(nn.Module):
    """Small-scale DeepSeekMoE-style FFN replacement.

    This layer implements:
    - routed expert pool,
    - optional shared expert path,
    - per-token top-k routing,
    - gate-weighted sparse expert aggregation,
    - auxiliary load-balancing loss,
    - routing diagnostics from the latest forward pass.

    It is not distributed expert parallelism, not DeepEP, and not a V3
    auxiliary-loss-free router. It is the ordinary measurable MoE substrate
    needed before later routing experiments.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_routed_experts: int,
        n_shared_experts: int,
        top_k: int,
        expert_d_ff: int,
        dropout: float = 0.0,
        router_score: str = "softmax",
        normalize_top_k_weights: bool = True,
        aux_loss_weight: float = 0.01,
        routing_mode: str = "aux_loss",
        use_expert_bias: bool = False,
        expert_bias_update_rate: float = 0.0,
        expert_bias_update_interval: int = 1,
        expert_bias_min: float = -1.0,
        expert_bias_max: float = 1.0,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            msg = "d_model must be positive"
            raise ValueError(msg)
        if n_routed_experts <= 0:
            msg = "n_routed_experts must be positive"
            raise ValueError(msg)
        if n_shared_experts < 0:
            msg = "n_shared_experts must be nonnegative"
            raise ValueError(msg)
        if top_k <= 0:
            msg = "top_k must be positive"
            raise ValueError(msg)
        if top_k > n_routed_experts:
            msg = "top_k must be less than or equal to n_routed_experts"
            raise ValueError(msg)
        if expert_d_ff <= 0:
            msg = "expert_d_ff must be positive"
            raise ValueError(msg)
        if aux_loss_weight < 0.0:
            msg = "aux_loss_weight must be nonnegative"
            raise ValueError(msg)

        valid_routing_modes = {"aux_loss", "aux_loss_free_bias"}
        if routing_mode not in valid_routing_modes:
            msg = f"routing_mode must be one of {sorted(valid_routing_modes)}"
            raise ValueError(msg)
        if expert_bias_update_rate < 0.0:
            msg = "expert_bias_update_rate must be nonnegative"
            raise ValueError(msg)
        if expert_bias_update_interval <= 0:
            msg = "expert_bias_update_interval must be positive"
            raise ValueError(msg)
        if expert_bias_min >= expert_bias_max:
            msg = "expert_bias_min must be less than expert_bias_max"
            raise ValueError(msg)
        if routing_mode == "aux_loss" and use_expert_bias:
            msg = "use_expert_bias must be false when routing_mode='aux_loss'"
            raise ValueError(msg)
        if routing_mode == "aux_loss" and expert_bias_update_rate != 0.0:
            msg = "expert_bias_update_rate must be 0.0 when " "routing_mode='aux_loss'"
            raise ValueError(msg)
        if routing_mode == "aux_loss_free_bias" and not use_expert_bias:
            msg = "use_expert_bias must be true when routing_mode='aux_loss_free_bias'"
            raise ValueError(msg)
        if routing_mode == "aux_loss_free_bias" and aux_loss_weight != 0.0:
            msg = "aux_loss_weight must be 0.0 when routing_mode='aux_loss_free_bias'"
            raise ValueError(msg)
        if routing_mode == "aux_loss_free_bias" and expert_bias_update_rate <= 0.0:
            msg = (
                "expert_bias_update_rate must be positive when " "routing_mode='aux_loss_free_bias'"
            )
            raise ValueError(msg)

        self.d_model = d_model
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k
        self.expert_d_ff = expert_d_ff
        self.aux_loss_weight = aux_loss_weight
        self.routing_mode = routing_mode
        self.use_expert_bias = use_expert_bias
        self.expert_bias_update_rate = expert_bias_update_rate
        self.expert_bias_update_interval = expert_bias_update_interval
        self.expert_bias_min = expert_bias_min
        self.expert_bias_max = expert_bias_max

        self.router = TopKRouter(
            d_model=d_model,
            n_experts=n_routed_experts,
            top_k=top_k,
            score_type=router_score,
            normalize_top_k_weights=normalize_top_k_weights,
            use_expert_bias=use_expert_bias,
            expert_bias_min=expert_bias_min,
            expert_bias_max=expert_bias_max,
        )
        self.routed_experts = nn.ModuleList(
            [
                MoEExpert(d_model=d_model, d_ff=expert_d_ff, dropout=dropout)
                for _ in range(n_routed_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                MoEExpert(d_model=d_model, d_ff=expert_d_ff, dropout=dropout)
                for _ in range(n_shared_experts)
            ]
        )

        self.last_aux_loss: torch.Tensor | None = None
        self.last_routing_stats: MoERoutingStats | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply shared experts and routed sparse experts to hidden states."""
        if hidden_states.ndim != 3:
            msg = (
                "DeepSeekMoELayer expects hidden states with shape "
                f"(batch, sequence, d_model), got rank {hidden_states.ndim}"
            )
            raise ValueError(msg)

        batch_size, seq_len, hidden_dim = hidden_states.shape
        if hidden_dim != self.d_model:
            msg = (
                "hidden_states final dimension must equal d_model; "
                f"got {hidden_dim} and expected {self.d_model}"
            )
            raise ValueError(msg)

        flat_hidden = hidden_states.reshape(batch_size * seq_len, hidden_dim)
        router_output = self.router(flat_hidden)

        routed_output = self._dispatch_to_routed_experts(flat_hidden, router_output)
        shared_output = self._apply_shared_experts(flat_hidden)

        output = routed_output + shared_output
        aux_loss = self._compute_aux_loss(router_output)
        stats = self._compute_routing_stats(router_output, aux_loss)

        self.last_aux_loss = aux_loss
        self.last_routing_stats = stats

        if self.training and self.routing_mode == "aux_loss_free_bias":
            self._update_expert_bias(stats)

        return output.reshape(batch_size, seq_len, hidden_dim)

    def _dispatch_to_routed_experts(
        self,
        flat_hidden: torch.Tensor,
        router_output: RouterOutput,
    ) -> torch.Tensor:
        routed_output = torch.zeros_like(flat_hidden)

        for expert_idx, expert in enumerate(self.routed_experts):
            selected = router_output.top_k_indices == expert_idx
            if not selected.any():
                continue

            token_indices, selected_slots = torch.where(selected)
            expert_input = flat_hidden[token_indices]
            expert_output = expert(expert_input)
            expert_weights = router_output.top_k_weights[token_indices, selected_slots].unsqueeze(
                -1
            )
            routed_output.index_add_(0, token_indices, expert_output * expert_weights)

        return routed_output

    def _apply_shared_experts(self, flat_hidden: torch.Tensor) -> torch.Tensor:
        if self.n_shared_experts == 0:
            return torch.zeros_like(flat_hidden)

        shared_output = torch.zeros_like(flat_hidden)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(flat_hidden)

        return shared_output

    def _update_expert_bias(self, stats: MoERoutingStats) -> None:
        """Update non-gradient expert bias from latest observed routing load."""
        if not self.use_expert_bias:
            return

        if stats.tokens <= 0:
            return

        target_fraction = 1.0 / float(self.n_routed_experts)
        expert_bias = cast(torch.Tensor, self.router.expert_bias)
        load_error = target_fraction - stats.expert_selection_fraction.to(
            device=expert_bias.device,
            dtype=expert_bias.dtype,
        )

        with torch.no_grad():
            expert_bias.add_(self.expert_bias_update_rate * load_error)
            expert_bias.clamp_(
                min=self.expert_bias_min,
                max=self.expert_bias_max,
            )

    def _compute_aux_loss(self, router_output: RouterOutput) -> torch.Tensor:
        if self.aux_loss_weight == 0.0:
            return router_output.scores.new_zeros(())

        tokens = router_output.scores.shape[0]
        expert_mask = torch.nn.functional.one_hot(
            router_output.top_k_indices,
            num_classes=self.n_routed_experts,
        ).to(router_output.scores.dtype)
        expert_fraction = expert_mask.sum(dim=(0, 1)) / float(tokens * self.top_k)
        mean_router_probability = router_output.scores.mean(dim=0)

        load_balance = self.n_routed_experts * torch.sum(expert_fraction * mean_router_probability)
        return load_balance * self.aux_loss_weight

    def _compute_routing_stats(
        self,
        router_output: RouterOutput,
        aux_loss: torch.Tensor,
    ) -> MoERoutingStats:
        tokens = router_output.scores.shape[0]
        expert_mask = torch.nn.functional.one_hot(
            router_output.top_k_indices,
            num_classes=self.n_routed_experts,
        ).to(router_output.scores.dtype)
        expert_selection_counts = expert_mask.sum(dim=(0, 1))
        expert_selection_fraction = expert_selection_counts / float(tokens * self.top_k)
        expert_load_variance = torch.var(expert_selection_fraction, unbiased=False)

        entropy_per_token = -torch.sum(
            router_output.scores * torch.log(router_output.scores.clamp_min(1e-12)),
            dim=-1,
        )
        routing_entropy = entropy_per_token.mean()
        mean_router_probability = router_output.scores.mean(dim=0)

        expert_bias = (
            cast(torch.Tensor, self.router.expert_bias).detach().clone()
            if self.use_expert_bias
            else None
        )

        return MoERoutingStats(
            tokens=tokens,
            n_routed_experts=self.n_routed_experts,
            top_k=self.top_k,
            expert_selection_counts=expert_selection_counts.detach(),
            expert_selection_fraction=expert_selection_fraction.detach(),
            expert_load_variance=expert_load_variance.detach(),
            routing_entropy=routing_entropy.detach(),
            mean_router_probability=mean_router_probability.detach(),
            aux_loss=aux_loss.detach(),
            routing_mode=self.routing_mode,
            expert_bias=expert_bias,
            expert_bias_update_rate=self.expert_bias_update_rate,
            expert_bias_update_interval=self.expert_bias_update_interval,
        )
