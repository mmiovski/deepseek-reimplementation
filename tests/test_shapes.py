from __future__ import annotations

import pytest
import torch

from deepseek_reimpl.layers.rmsnorm import RMSNorm
from deepseek_reimpl.layers.rope import RotaryEmbedding
from deepseek_reimpl.layers.swiglu import SwiGLU
from deepseek_reimpl.model.baseline_gpt import BaselineGPT
from deepseek_reimpl.model.config import GPTConfig
from deepseek_reimpl.model.decoder_block import DecoderBlock
from deepseek_reimpl.utils.config import load_yaml_config


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


def test_baseline_gpt_config_loads_from_yaml() -> None:
    config_dict = load_yaml_config("configs/model/baseline_gpt.yaml")

    config = GPTConfig.from_dict(config_dict)

    assert config.vocab_size == 10000
    assert config.block_size == 128
    assert config.n_layers == 4
    assert config.n_heads == 4
    assert config.d_model == 256
    assert config.d_ff == 1024
    assert config.head_dim == 64
    assert config.norm_type == "rmsnorm"
    assert config.positional_encoding == "rope"
    assert config.ffn_type == "swiglu"
    assert config.tie_embeddings


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
    kwargs = {
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


def test_model_factory_builds_baseline_gpt_from_yaml() -> None:
    from deepseek_reimpl.model.model_factory import build_model_from_config

    config_dict = load_yaml_config("configs/model/baseline_gpt.yaml")

    model = build_model_from_config(config_dict)

    assert isinstance(model, BaselineGPT)


def test_model_package_exports() -> None:
    from deepseek_reimpl.model import BaselineGPT as ExportedBaselineGPT
    from deepseek_reimpl.model import GPTConfig as ExportedGPTConfig

    assert ExportedBaselineGPT is BaselineGPT
    assert ExportedGPTConfig is GPTConfig


def test_baseline_gpt_yaml_config_forward_smoke() -> None:
    config_dict = load_yaml_config("configs/model/baseline_gpt.yaml")
    config = GPTConfig.from_dict(config_dict)
    model = BaselineGPT(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, 8, config.vocab_size)
