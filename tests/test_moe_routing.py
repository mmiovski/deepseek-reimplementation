from __future__ import annotations

from typing import Any

import pytest
import torch

from deepseek_reimpl.layers.ffn import build_ffn
from deepseek_reimpl.layers.moe_expert import MoEExpert
from deepseek_reimpl.layers.moe_layer import DeepSeekMoELayer
from deepseek_reimpl.layers.moe_router import TopKRouter
from deepseek_reimpl.model.config import GPTConfig


def test_top_k_router_preserves_expected_shapes() -> None:
    router = TopKRouter(d_model=16, n_experts=4, top_k=2)
    hidden_states = torch.randn(6, 16)

    output = router(hidden_states)

    assert output.scores.shape == (6, 4)
    assert output.top_k_indices.shape == (6, 2)
    assert output.top_k_weights.shape == (6, 2)


def test_top_k_router_scores_are_probabilities() -> None:
    router = TopKRouter(d_model=16, n_experts=4, top_k=2)
    hidden_states = torch.randn(6, 16)

    output = router(hidden_states)

    assert torch.all(output.scores >= 0)
    assert torch.allclose(output.scores.sum(dim=-1), torch.ones(6), atol=1e-6)


def test_top_k_router_normalizes_selected_weights() -> None:
    router = TopKRouter(
        d_model=16,
        n_experts=4,
        top_k=2,
        normalize_top_k_weights=True,
    )
    hidden_states = torch.randn(6, 16)

    output = router(hidden_states)

    assert torch.all(output.top_k_weights >= 0)
    assert torch.allclose(output.top_k_weights.sum(dim=-1), torch.ones(6), atol=1e-6)


def test_top_k_router_can_leave_selected_weights_unnormalized() -> None:
    router = TopKRouter(
        d_model=16,
        n_experts=4,
        top_k=2,
        normalize_top_k_weights=False,
    )
    hidden_states = torch.randn(6, 16)

    output = router(hidden_states)

    assert torch.all(output.top_k_weights >= 0)
    assert torch.all(output.top_k_weights.sum(dim=-1) <= 1.0)


def test_top_k_router_outputs_are_finite() -> None:
    router = TopKRouter(d_model=16, n_experts=4, top_k=2)
    hidden_states = torch.randn(6, 16)

    output = router(hidden_states)

    assert torch.isfinite(output.scores).all()
    assert torch.isfinite(output.top_k_weights).all()


def test_top_k_router_backward_updates_gate_gradients() -> None:
    router = TopKRouter(d_model=16, n_experts=4, top_k=2)
    hidden_states = torch.randn(6, 16, requires_grad=True)

    output = router(hidden_states)
    loss = output.top_k_weights.sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert router.gate.weight.grad is not None
    assert torch.isfinite(router.gate.weight.grad).all()


@pytest.mark.parametrize(
    ("n_experts", "top_k", "match"),
    [
        (0, 1, "n_experts must be positive"),
        (4, 0, "top_k must be positive"),
        (2, 3, "top_k must be less than or equal to n_experts"),
    ],
)
def test_top_k_router_rejects_invalid_expert_configuration(
    n_experts: int,
    top_k: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        TopKRouter(d_model=16, n_experts=n_experts, top_k=top_k)


def test_top_k_router_rejects_invalid_hidden_rank() -> None:
    router = TopKRouter(d_model=16, n_experts=4, top_k=2)

    with pytest.raises(ValueError, match="expects flattened hidden states"):
        router(torch.randn(2, 3, 16))


def test_top_k_router_rejects_invalid_hidden_dimension() -> None:
    router = TopKRouter(d_model=16, n_experts=4, top_k=2)

    with pytest.raises(ValueError, match="final dimension must equal d_model"):
        router(torch.randn(6, 12))


def test_moe_expert_preserves_flattened_token_shape() -> None:
    expert = MoEExpert(d_model=16, d_ff=32)
    hidden_states = torch.randn(6, 16)

    output = expert(hidden_states)

    assert output.shape == hidden_states.shape


def test_moe_expert_outputs_are_finite() -> None:
    expert = MoEExpert(d_model=16, d_ff=32)
    hidden_states = torch.randn(6, 16)

    output = expert(hidden_states)

    assert torch.isfinite(output).all()


def test_moe_expert_backward_creates_gradients() -> None:
    expert = MoEExpert(d_model=16, d_ff=32)
    hidden_states = torch.randn(6, 16, requires_grad=True)

    output = expert(hidden_states)
    loss = output.square().mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert expert.gate_proj.weight.grad is not None
    assert expert.up_proj.weight.grad is not None
    assert expert.down_proj.weight.grad is not None


@pytest.mark.parametrize(
    ("d_model", "d_ff", "dropout", "match"),
    [
        (0, 32, 0.0, "d_model must be positive"),
        (16, 0, 0.0, "d_ff must be positive"),
        (16, 32, 1.0, "dropout must be in the interval"),
    ],
)
def test_moe_expert_rejects_invalid_configuration(
    d_model: int,
    d_ff: int,
    dropout: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MoEExpert(d_model=d_model, d_ff=d_ff, dropout=dropout)


def test_moe_expert_rejects_invalid_hidden_rank() -> None:
    expert = MoEExpert(d_model=16, d_ff=32)

    with pytest.raises(ValueError, match="expects flattened hidden states"):
        expert(torch.randn(2, 3, 16))


def test_moe_expert_rejects_invalid_hidden_dimension() -> None:
    expert = MoEExpert(d_model=16, d_ff=32)

    with pytest.raises(ValueError, match="final dimension must equal d_model"):
        expert(torch.randn(6, 12))


def test_deepseek_moe_layer_preserves_batch_sequence_shape() -> None:
    layer = DeepSeekMoELayer(
        d_model=16,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        expert_d_ff=32,
    )
    hidden_states = torch.randn(2, 3, 16)

    output = layer(hidden_states)

    assert output.shape == hidden_states.shape


def test_deepseek_moe_layer_outputs_are_finite() -> None:
    layer = DeepSeekMoELayer(
        d_model=16,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        expert_d_ff=32,
    )
    hidden_states = torch.randn(2, 3, 16)

    output = layer(hidden_states)

    assert torch.isfinite(output).all()


def test_deepseek_moe_layer_sets_aux_loss_and_routing_stats() -> None:
    layer = DeepSeekMoELayer(
        d_model=16,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        expert_d_ff=32,
        aux_loss_weight=0.01,
    )
    hidden_states = torch.randn(2, 3, 16)

    layer(hidden_states)

    assert layer.last_aux_loss is not None
    assert layer.last_routing_stats is not None
    assert layer.last_routing_stats.tokens == 6
    assert layer.last_routing_stats.n_routed_experts == 4
    assert layer.last_routing_stats.top_k == 2
    assert layer.last_routing_stats.expert_selection_counts.shape == (4,)
    assert torch.isfinite(layer.last_aux_loss)


def test_deepseek_moe_layer_backward_reaches_router_and_selected_experts() -> None:
    layer = DeepSeekMoELayer(
        d_model=16,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        expert_d_ff=32,
    )
    hidden_states = torch.randn(2, 3, 16, requires_grad=True)

    output = layer(hidden_states)
    assert layer.last_aux_loss is not None
    loss = output.square().mean() + layer.last_aux_loss
    loss.backward()

    assert hidden_states.grad is not None
    assert layer.router.gate.weight.grad is not None
    assert torch.isfinite(layer.router.gate.weight.grad).all()

    routed_grads = [
        expert.down_proj.weight.grad
        for expert in layer.routed_experts
        if expert.down_proj.weight.grad is not None
    ]
    assert routed_grads
    assert all(torch.isfinite(grad).all() for grad in routed_grads)


def test_deepseek_moe_layer_sparse_activation_uses_only_selected_routed_experts() -> None:
    layer = DeepSeekMoELayer(
        d_model=4,
        n_routed_experts=3,
        n_shared_experts=0,
        top_k=1,
        expert_d_ff=4,
    )

    with torch.no_grad():
        layer.router.gate.weight.zero_()
        layer.router.gate.weight[0, 0] = 10.0
        layer.router.gate.weight[1, 0] = -10.0
        layer.router.gate.weight[2, 0] = -20.0

        for expert_idx, expert in enumerate(layer.routed_experts):
            expert.gate_proj.weight.fill_(1.0)
            expert.up_proj.weight.fill_(1.0)
            expert.down_proj.weight.fill_(float(expert_idx + 1))

    hidden_states = torch.ones(1, 2, 4)

    layer(hidden_states)

    assert layer.last_routing_stats is not None
    assert torch.equal(
        layer.last_routing_stats.expert_selection_counts.cpu(),
        torch.tensor([2.0, 0.0, 0.0]),
    )


def test_deepseek_moe_layer_without_shared_experts_still_runs() -> None:
    layer = DeepSeekMoELayer(
        d_model=16,
        n_routed_experts=4,
        n_shared_experts=0,
        top_k=2,
        expert_d_ff=32,
    )
    hidden_states = torch.randn(2, 3, 16)

    output = layer(hidden_states)

    assert output.shape == hidden_states.shape


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"d_model": 0}, "d_model must be positive"),
        ({"n_routed_experts": 0}, "n_routed_experts must be positive"),
        ({"n_shared_experts": -1}, "n_shared_experts must be nonnegative"),
        ({"top_k": 0}, "top_k must be positive"),
        ({"top_k": 5}, "top_k must be less than or equal to n_routed_experts"),
        ({"expert_d_ff": 0}, "expert_d_ff must be positive"),
        ({"aux_loss_weight": -0.1}, "aux_loss_weight must be nonnegative"),
    ],
)
def test_deepseek_moe_layer_rejects_invalid_configuration(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    base_kwargs: dict[str, Any] = {
        "d_model": 16,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "top_k": 2,
        "expert_d_ff": 32,
        "aux_loss_weight": 0.01,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=match):
        DeepSeekMoELayer(**base_kwargs)


def test_deepseek_moe_layer_rejects_invalid_hidden_rank() -> None:
    layer = DeepSeekMoELayer(
        d_model=16,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        expert_d_ff=32,
    )

    with pytest.raises(ValueError, match="expects hidden states"):
        layer(torch.randn(6, 16))


def test_deepseek_moe_layer_rejects_invalid_hidden_dimension() -> None:
    layer = DeepSeekMoELayer(
        d_model=16,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        expert_d_ff=32,
    )

    with pytest.raises(ValueError, match="final dimension must equal d_model"):
        layer(torch.randn(2, 3, 12))


def tiny_moe_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=64,
        dropout=0.0,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_top_k=2,
        moe_expert_d_ff=32,
        moe_router_score="softmax",
        moe_normalize_top_k_weights=True,
        moe_aux_loss_weight=0.01,
        moe_drop_tokens=False,
    )


def test_gpt_config_accepts_valid_moe_fields() -> None:
    config = tiny_moe_config()

    assert config.ffn_type == "moe"
    assert config.n_routed_experts == 4
    assert config.n_shared_experts == 1
    assert config.moe_top_k == 2
    assert config.moe_expert_d_ff == 32


def test_build_ffn_creates_deepseek_moe_layer_from_config() -> None:
    config = tiny_moe_config()

    ffn = build_ffn(config)

    assert isinstance(ffn, DeepSeekMoELayer)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_routed_experts": None}, "n_routed_experts must be set"),
        ({"moe_top_k": None}, "moe_top_k must be set"),
        ({"moe_expert_d_ff": None}, "moe_expert_d_ff must be set"),
        ({"n_routed_experts": 0}, "n_routed_experts must be positive"),
        ({"n_shared_experts": -1}, "n_shared_experts must be nonnegative"),
        ({"moe_top_k": 0}, "moe_top_k must be positive"),
        ({"moe_top_k": 5}, "moe_top_k must be less than or equal"),
        ({"moe_expert_d_ff": 0}, "moe_expert_d_ff must be positive"),
        ({"moe_router_score": "sigmoid"}, "moe_router_score currently supports"),
        ({"moe_aux_loss_weight": -0.1}, "moe_aux_loss_weight must be nonnegative"),
        ({"moe_drop_tokens": True}, "moe_drop_tokens is reserved"),
    ],
)
def test_gpt_config_rejects_invalid_moe_fields(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    base_kwargs: dict[str, Any] = {
        "vocab_size": 128,
        "block_size": 16,
        "n_layers": 1,
        "n_heads": 2,
        "d_model": 16,
        "d_ff": 64,
        "dropout": 0.0,
        "ffn_type": "moe",
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "moe_top_k": 2,
        "moe_expert_d_ff": 32,
        "moe_router_score": "softmax",
        "moe_normalize_top_k_weights": True,
        "moe_aux_loss_weight": 0.01,
        "moe_drop_tokens": False,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=match):
        GPTConfig(**base_kwargs)
