from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from deepseek_reimpl.layers.attention import CausalSelfAttention, build_attention
from deepseek_reimpl.layers.mla import MLAAttention
from deepseek_reimpl.model.config import GPTConfig


def tiny_attention_config(attention_type: str = "dense") -> GPTConfig:
    kwargs: dict[str, Any] = {
        "vocab_size": 32,
        "block_size": 8,
        "n_layers": 1,
        "n_heads": 2,
        "d_model": 8,
        "d_ff": 32,
        "dropout": 0.0,
        "positional_encoding": "rope",
        "attention_type": attention_type,
    }
    if attention_type == "mla":
        kwargs["mla_kv_latent_dim"] = 4
        kwargs["mla_q_rope_dim"] = 2
    return GPTConfig(**kwargs)


def _assert_no_future_token_dependence(attention: nn.Module, d_model: int) -> None:
    torch.manual_seed(0)
    attention.eval()

    x = torch.randn(1, 5, d_model)
    changed_future = x.clone()
    changed_future[:, 3:, :] = torch.randn_like(changed_future[:, 3:, :]) * 100.0

    with torch.no_grad():
        original_output = attention(x)
        changed_output = attention(changed_future)

    assert torch.allclose(
        original_output[:, :3, :],
        changed_output[:, :3, :],
        atol=1e-5,
    )


def test_causal_self_attention_preserves_shape() -> None:
    config = tiny_attention_config()
    attention = CausalSelfAttention(config)
    x = torch.randn(2, 5, config.d_model)

    y = attention(x)

    assert y.shape == x.shape


def test_causal_self_attention_rejects_sequence_longer_than_block_size() -> None:
    config = tiny_attention_config()
    attention = CausalSelfAttention(config)
    x = torch.randn(2, config.block_size + 1, config.d_model)

    with pytest.raises(ValueError, match="sequence length exceeds configured block_size"):
        attention(x)


def test_dense_causal_mask_prevents_future_token_dependence() -> None:
    config = tiny_attention_config()
    attention = CausalSelfAttention(config)

    _assert_no_future_token_dependence(attention, config.d_model)


def test_mla_causal_mask_prevents_future_token_dependence() -> None:
    config = tiny_attention_config(attention_type="mla")
    attention = MLAAttention(config)

    _assert_no_future_token_dependence(attention, config.d_model)


def test_build_attention_returns_dense_attention() -> None:
    config = tiny_attention_config()

    attention = build_attention(config)

    assert isinstance(attention, CausalSelfAttention)


def test_build_attention_returns_mla_attention() -> None:
    config = tiny_attention_config(attention_type="mla")

    attention = build_attention(config)

    assert isinstance(attention, MLAAttention)
