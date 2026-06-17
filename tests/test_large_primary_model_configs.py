from __future__ import annotations

from typing import Any, cast

from deepseek_reimpl.utils.config import load_yaml_config

LARGE_CONFIGS = {
    "dense_121m": ("configs/model/dense_121m.yaml", "baseline_gpt"),
    "mla_121m": ("configs/model/mla_121m.yaml", "mla_gpt"),
    "mtp_121m": ("configs/model/mtp_121m.yaml", "mtp_gpt"),
    "moe_220m": ("configs/model/moe_220m.yaml", "moe_gpt"),
    "mla_moe_220m": ("configs/model/mla_moe_220m.yaml", "mla_moe_gpt"),
    "v3_routing_220m": ("configs/model/v3_routing_220m.yaml", "v3_routing_gpt"),
}


def _model(path: str) -> dict[str, Any]:
    wrapper = load_yaml_config(path)
    assert set(wrapper.keys()) == {"model"}
    return cast(dict[str, Any], wrapper["model"])


def test_large_primary_configs_exist_and_are_not_small_smoke_scale() -> None:
    for _, (path, expected_factory_name) in LARGE_CONFIGS.items():
        model = _model(path)
        assert model["name"] == expected_factory_name
        assert model["vocab_size"] == 10000
        assert model["block_size"] == 256
        assert model["n_layers"] == 12
        assert model["n_heads"] == 12
        assert model["d_model"] == 768
        assert model["d_ff"] == 3072


def test_large_primary_dense_style_configs_are_matched() -> None:
    dense = _model("configs/model/dense_121m.yaml")
    mla = _model("configs/model/mla_121m.yaml")
    mtp = _model("configs/model/mtp_121m.yaml")

    for model in [dense, mla, mtp]:
        assert model["ffn_type"] == "swiglu"

    assert dense["name"] == "baseline_gpt"
    assert dense["attention_type"] == "dense"
    assert "mtp_enabled" not in dense or dense["mtp_enabled"] is False

    assert mla["name"] == "mla_gpt"
    assert mla["attention_type"] == "mla"
    assert mla["mla_kv_latent_dim"] == 192
    assert mla["mla_q_rope_dim"] == 96
    assert "mtp_enabled" not in mla or mla["mtp_enabled"] is False

    assert mtp["name"] == "mtp_gpt"
    assert mtp["attention_type"] == "dense"
    assert mtp["mtp_enabled"] is True
    assert mtp["mtp_num_future_tokens"] == 2
    assert mtp["mtp_loss_weight"] == 0.3
    assert mtp["mtp_share_lm_head"] is False


def test_large_primary_sparse_configs_are_matched_to_220m_anchor() -> None:
    moe = _model("configs/model/moe_220m.yaml")
    mla_moe = _model("configs/model/mla_moe_220m.yaml")
    v3 = _model("configs/model/v3_routing_220m.yaml")

    for model in [moe, mla_moe, v3]:
        assert model["ffn_type"] == "moe"
        assert model["n_routed_experts"] == 12
        assert model["n_shared_experts"] == 1
        assert model["moe_top_k"] == 2
        assert model["moe_expert_d_ff"] == 512

    assert moe["name"] == "moe_gpt"
    assert moe["attention_type"] == "dense"
    assert moe["moe_routing_mode"] == "aux_loss"
    assert moe["moe_use_expert_bias"] is False

    assert mla_moe["name"] == "mla_moe_gpt"
    assert mla_moe["attention_type"] == "mla"
    assert mla_moe["mla_kv_latent_dim"] == 192
    assert mla_moe["mla_q_rope_dim"] == 96
    assert mla_moe["moe_routing_mode"] == "aux_loss"
    assert mla_moe["moe_use_expert_bias"] is False

    assert v3["name"] == "v3_routing_gpt"
    assert v3["attention_type"] == "dense"
    assert v3["moe_aux_loss_weight"] == 0.0
    assert v3["moe_routing_mode"] == "aux_loss_free_bias"
    assert v3["moe_use_expert_bias"] is True
    assert v3["moe_expert_bias_update_rate"] == 0.001
