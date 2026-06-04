from __future__ import annotations

import pytest
import torch

from deepseek_reimpl.layers.attention import CausalSelfAttention
from deepseek_reimpl.model.config import GPTConfig


def tiny_attention_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layers=1,
        n_heads=2,
        d_model=8,
        d_ff=32,
        dropout=0.0,
        positional_encoding="rope",
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


def test_causal_mask_prevents_future_token_dependence() -> None:
    torch.manual_seed(0)

    config = tiny_attention_config()
    attention = CausalSelfAttention(config)
    attention.eval()

    x = torch.randn(1, 5, config.d_model)
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
