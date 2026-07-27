from __future__ import annotations

import pytest
import torch

from deepseek_reimpl.layers.mla import MLAAttention
from deepseek_reimpl.model.baseline_gpt import BaselineGPT
from deepseek_reimpl.model.config import GPTConfig


def tiny_mla_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=64,
        block_size=8,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=64,
        dropout=0.0,
        norm_type="rmsnorm",
        positional_encoding="rope",
        ffn_type="swiglu",
        attention_type="mla",
        mla_kv_latent_dim=8,
        mla_q_rope_dim=4,
        tie_embeddings=True,
    )


def test_mla_attention_preserves_shape() -> None:
    config = tiny_mla_config()
    attention = MLAAttention(config)
    x = torch.randn(2, 5, config.d_model)

    y = attention(x)

    assert y.shape == x.shape


def test_mla_attention_outputs_finite_values() -> None:
    config = tiny_mla_config()
    attention = MLAAttention(config)
    x = torch.randn(2, 5, config.d_model)

    y = attention(x)

    assert torch.isfinite(y).all()


def test_mla_attention_backward_pass_produces_gradients() -> None:
    config = tiny_mla_config()
    attention = MLAAttention(config)
    x = torch.randn(2, 5, config.d_model, requires_grad=True)

    loss = attention(x).pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mla_attention_rejects_sequence_longer_than_block_size() -> None:
    config = tiny_mla_config()
    attention = MLAAttention(config)
    x = torch.randn(2, config.block_size + 1, config.d_model)

    with pytest.raises(ValueError, match="sequence length exceeds configured block_size"):
        attention(x)


def test_mla_config_rejects_missing_latent_dim() -> None:
    with pytest.raises(ValueError, match="mla_kv_latent_dim must be set"):
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layers=1,
            n_heads=2,
            d_model=16,
            d_ff=64,
            attention_type="mla",
            mla_q_rope_dim=4,
        )


def test_mla_config_rejects_latent_dim_not_smaller_than_d_model() -> None:
    with pytest.raises(ValueError, match="mla_kv_latent_dim must be smaller than d_model"):
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layers=1,
            n_heads=2,
            d_model=16,
            d_ff=64,
            attention_type="mla",
            mla_kv_latent_dim=16,
            mla_q_rope_dim=4,
        )


def test_mla_config_rejects_odd_rope_dim() -> None:
    with pytest.raises(ValueError, match="mla_q_rope_dim must be even"):
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layers=1,
            n_heads=2,
            d_model=16,
            d_ff=64,
            attention_type="mla",
            mla_kv_latent_dim=8,
            mla_q_rope_dim=3,
        )


def test_mla_gpt_outputs_logits_shape() -> None:
    config = tiny_mla_config()
    model = BaselineGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits = model(input_ids)

    assert logits.shape == (2, 5, config.vocab_size)
