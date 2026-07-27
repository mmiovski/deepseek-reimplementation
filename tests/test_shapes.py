from __future__ import annotations

from typing import Any

import pytest
import torch

from deepseek_reimpl.layers.rmsnorm import RMSNorm
from deepseek_reimpl.layers.rope import RotaryEmbedding
from deepseek_reimpl.layers.swiglu import SwiGLU
from deepseek_reimpl.model.baseline_gpt import BaselineGPT
from deepseek_reimpl.model.config import GPTConfig
from deepseek_reimpl.model.decoder_block import DecoderBlock


def tiny_gpt_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=64,
        block_size=8,
        n_layers=2,
        n_heads=2,
        d_model=16,
        d_ff=64,
        dropout=0.0,
        norm_type="rmsnorm",
        positional_encoding="rope",
        ffn_type="swiglu",
        tie_embeddings=True,
    )


def test_gpt_config_rejects_invalid_head_dimension() -> None:
    with pytest.raises(ValueError, match="d_model must be divisible by n_heads"):
        GPTConfig(
            vocab_size=100,
            block_size=8,
            n_layers=2,
            n_heads=3,
            d_model=16,
            d_ff=64,
        )


@pytest.mark.parametrize(
    "field_name",
    ["vocab_size", "block_size", "n_layers", "n_heads", "d_model", "d_ff"],
)
def test_gpt_config_rejects_nonpositive_integer_fields(field_name: str) -> None:
    kwargs: dict[str, Any] = {
        "vocab_size": 100,
        "block_size": 8,
        "n_layers": 2,
        "n_heads": 4,
        "d_model": 16,
        "d_ff": 64,
    }
    kwargs[field_name] = 0

    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        GPTConfig(**kwargs)


def test_gpt_config_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError, match="dropout must be in the interval"):
        GPTConfig(
            vocab_size=100,
            block_size=8,
            n_layers=2,
            n_heads=4,
            d_model=16,
            d_ff=64,
            dropout=1.0,
        )


def test_rmsnorm_preserves_shape() -> None:
    norm = RMSNorm(d_model=16)
    x = torch.randn(2, 5, 16)

    y = norm(x)

    assert y.shape == x.shape


def test_rotary_embedding_preserves_query_key_shapes() -> None:
    rope = RotaryEmbedding(head_dim=8)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 5, 8)

    q_rot, k_rot = rope.apply(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rotary_embedding_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError, match="head_dim must be even"):
        RotaryEmbedding(head_dim=7)


def test_swiglu_preserves_sequence_shape() -> None:
    ffn = SwiGLU(d_model=16, d_ff=64)
    x = torch.randn(2, 5, 16)

    y = ffn(x)

    assert y.shape == x.shape


def test_decoder_block_preserves_shape() -> None:
    config = tiny_gpt_config()
    block = DecoderBlock(config)
    x = torch.randn(2, 5, config.d_model)

    y = block(x)

    assert y.shape == x.shape


def test_baseline_gpt_outputs_logits_shape() -> None:
    config = tiny_gpt_config()
    model = BaselineGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits = model(input_ids)

    assert logits.shape == (2, 5, config.vocab_size)


def test_baseline_gpt_ties_embeddings_when_configured() -> None:
    config = tiny_gpt_config()
    model = BaselineGPT(config)

    assert model.lm_head.weight is model.token_embedding.weight


def test_baseline_gpt_rejects_sequence_longer_than_block_size() -> None:
    config = tiny_gpt_config()
    model = BaselineGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size + 1))

    with pytest.raises(ValueError, match="sequence length exceeds configured block_size"):
        model(input_ids)


def test_model_package_exports() -> None:
    from deepseek_reimpl.model import BaselineGPT as ExportedBaselineGPT
    from deepseek_reimpl.model import GPTConfig as ExportedGPTConfig

    assert ExportedBaselineGPT is BaselineGPT
    assert ExportedGPTConfig is GPTConfig


def test_mla_moe_gpt_backward_reaches_router_experts_and_embeddings() -> None:
    config = GPTConfig(
        vocab_size=64,
        block_size=8,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=64,
        dropout=0.0,
        positional_encoding="rope",
        attention_type="mla",
        mla_kv_latent_dim=8,
        mla_q_rope_dim=4,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_top_k=2,
        moe_expert_d_ff=32,
        moe_aux_loss_weight=0.01,
    )
    model = BaselineGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits = model(input_ids)
    aux_loss = model.auxiliary_loss()
    assert aux_loss is not None

    loss = logits.square().mean() + aux_loss
    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()

    from deepseek_reimpl.layers.moe_layer import DeepSeekMoELayer

    first_block = model.blocks[0]
    assert isinstance(first_block.ffn, DeepSeekMoELayer)
    assert first_block.ffn.router.gate.weight.grad is not None
    assert torch.isfinite(first_block.ffn.router.gate.weight.grad).all()

    routed_grads = [
        expert.down_proj.weight.grad
        for expert in first_block.ffn.routed_experts
        if expert.down_proj.weight.grad is not None
    ]
    assert routed_grads
    assert all(torch.isfinite(grad).all() for grad in routed_grads)


def test_v3_routing_gpt_backward_reaches_router_experts_and_embeddings() -> None:
    from deepseek_reimpl.layers.moe_layer import DeepSeekMoELayer

    config = GPTConfig(
        vocab_size=64,
        block_size=8,
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
        moe_aux_loss_weight=0.0,
        moe_routing_mode="aux_loss_free_bias",
        moe_use_expert_bias=True,
        moe_expert_bias_update_rate=0.001,
    )
    model = BaselineGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits = model(input_ids)
    aux_loss = model.auxiliary_loss()
    assert aux_loss is not None

    loss = logits.square().mean() + aux_loss
    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()

    first_block = model.blocks[0]
    assert isinstance(first_block.ffn, DeepSeekMoELayer)
    assert first_block.ffn.router.gate.weight.grad is not None
    assert torch.isfinite(first_block.ffn.router.gate.weight.grad).all()

    routed_grads = [
        expert.down_proj.weight.grad
        for expert in first_block.ffn.routed_experts
        if expert.down_proj.weight.grad is not None
    ]
    assert routed_grads
    assert all(torch.isfinite(grad).all() for grad in routed_grads)
