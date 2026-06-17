from __future__ import annotations

import torch

from deepseek_reimpl.layers.mla import MLAAttention
from deepseek_reimpl.model.config import GPTConfig


def test_decoupled_mla_allows_deepseek_style_head_geometry() -> None:
    config = GPTConfig(
        vocab_size=10000,
        block_size=256,
        n_layers=12,
        n_heads=12,
        d_model=768,
        d_ff=3072,
        attention_type="mla",
        ffn_type="swiglu",
        mla_kv_latent_dim=192,
        mla_qk_nope_head_dim=128,
        mla_q_rope_dim=64,
        mla_v_head_dim=128,
    )

    assert config.head_dim == 64
    assert config.mla_qk_nope_dim == 128
    assert config.mla_q_rope_dim == 64
    assert config.mla_qk_head_dim == 192
    assert config.mla_v_dim == 128

    layer = MLAAttention(config)
    inputs = torch.randn(2, 8, 768)
    outputs = layer(inputs)

    assert outputs.shape == (2, 8, 768)


def test_legacy_small_mla_config_remains_supported() -> None:
    config = GPTConfig(
        vocab_size=10000,
        block_size=128,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        attention_type="mla",
        ffn_type="swiglu",
        mla_kv_latent_dim=64,
        mla_q_rope_dim=32,
    )

    assert config.head_dim == 64
    assert config.mla_qk_nope_dim == 32
    assert config.mla_qk_head_dim == 64
    assert config.mla_v_dim == 64

    layer = MLAAttention(config)
    inputs = torch.randn(2, 8, 256)
    outputs = layer(inputs)

    assert outputs.shape == (2, 8, 256)
