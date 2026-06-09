"""Top-k router for small-scale DeepSeekMoE-style feedforward layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RouterOutput:
    """Outputs produced by a top-k expert router.

    Attributes:
        scores: Full router probabilities/logits after score transform with shape
            (tokens, n_experts).
        top_k_indices: Selected expert indices with shape (tokens, top_k).
        top_k_weights: Gate weights for selected experts with shape (tokens, top_k).
    """

    scores: torch.Tensor
    top_k_indices: torch.Tensor
    top_k_weights: torch.Tensor


class TopKRouter(nn.Module):
    """Per-token top-k router over routed experts.

    This router is intentionally small-scale and PyTorch-native. It provides the
    ordinary auxiliary-loss-based MoE routing substrate needed before later V3
    auxiliary-loss-free routing experiments.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_experts: int,
        top_k: int,
        score_type: str = "softmax",
        normalize_top_k_weights: bool = True,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            msg = "d_model must be positive"
            raise ValueError(msg)
        if n_experts <= 0:
            msg = "n_experts must be positive"
            raise ValueError(msg)
        if top_k <= 0:
            msg = "top_k must be positive"
            raise ValueError(msg)
        if top_k > n_experts:
            msg = "top_k must be less than or equal to n_experts"
            raise ValueError(msg)
        if score_type != "softmax":
            msg = "TopKRouter currently supports only score_type='softmax'"
            raise ValueError(msg)

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.score_type = score_type
        self.normalize_top_k_weights = normalize_top_k_weights
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> RouterOutput:
        """Route flattened hidden states to top-k experts.

        Args:
            hidden_states: Tensor with shape (tokens, d_model).

        Returns:
            RouterOutput with full router scores and selected expert assignments.
        """
        if hidden_states.ndim != 2:
            msg = (
                "TopKRouter expects flattened hidden states with shape "
                f"(tokens, d_model), got rank {hidden_states.ndim}"
            )
            raise ValueError(msg)

        if hidden_states.shape[-1] != self.d_model:
            msg = (
                "hidden_states final dimension must equal d_model; "
                f"got {hidden_states.shape[-1]} and expected {self.d_model}"
            )
            raise ValueError(msg)

        logits = self.gate(hidden_states)
        scores = torch.softmax(logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(scores, k=self.top_k, dim=-1)

        if self.normalize_top_k_weights:
            denominator = top_k_weights.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(scores.dtype).eps
            )
            top_k_weights = top_k_weights / denominator

        return RouterOutput(
            scores=scores,
            top_k_indices=top_k_indices,
            top_k_weights=top_k_weights,
        )
