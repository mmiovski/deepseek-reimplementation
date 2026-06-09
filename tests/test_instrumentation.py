from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from deepseek_reimpl.instrumentation.logging_utils import append_jsonl, write_json
from deepseek_reimpl.instrumentation.memory import (
    get_peak_memory_bytes,
    reset_peak_memory,
)
from deepseek_reimpl.instrumentation.parameters import (
    count_parameters,
    count_trainable_parameters,
)
from deepseek_reimpl.instrumentation.throughput import ThroughputMeter
from deepseek_reimpl.model.baseline_gpt import BaselineGPT
from deepseek_reimpl.model.config import GPTConfig


def test_count_parameters_linear_exact() -> None:
    model = nn.Linear(3, 4)

    # weight: 4 * 3 = 12, bias: 4
    assert count_parameters(model) == 16


def test_count_trainable_parameters_excludes_frozen_params() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))

    for parameter in model[0].parameters():
        parameter.requires_grad = False

    # second layer weight: 2 * 4 = 8, bias: 2
    assert count_trainable_parameters(model) == 10
    assert count_parameters(model) > count_trainable_parameters(model)


def test_count_parameters_baseline_gpt_positive() -> None:
    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layers=1,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )
    model = BaselineGPT(config)

    assert count_parameters(model) > 0
    assert count_trainable_parameters(model) > 0
    assert count_trainable_parameters(model) <= count_parameters(model)


def test_throughput_meter_update_increments_token_count() -> None:
    meter = ThroughputMeter()

    meter.update(10)
    meter.update(5)

    snapshot = meter.snapshot()
    assert snapshot.tokens == 15
    assert snapshot.elapsed_seconds >= 0.0
    assert snapshot.tokens_per_second >= 0.0


def test_throughput_meter_reset_clears_token_count() -> None:
    meter = ThroughputMeter()

    meter.update(10)
    meter.reset()

    snapshot = meter.snapshot()
    assert snapshot.tokens == 0
    assert snapshot.tokens_per_second >= 0.0


def test_throughput_meter_rejects_negative_tokens() -> None:
    meter = ThroughputMeter()

    try:
        meter.update(-1)
    except ValueError as error:
        assert "tokens must be nonnegative" in str(error)
    else:
        raise AssertionError("Expected ValueError for negative token update")


def test_cpu_peak_memory_returns_none() -> None:
    assert get_peak_memory_bytes(torch.device("cpu")) is None


def test_cpu_reset_peak_memory_does_not_crash() -> None:
    reset_peak_memory(torch.device("cpu"))


def test_memory_helpers_import_without_cuda_requirement() -> None:
    device = torch.device("cpu")

    reset_peak_memory(device)
    assert get_peak_memory_bytes(device) is None


def test_write_json_creates_parent_and_writes_loadable_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "summary.json"

    returned_path = write_json(output_path, {"loss": 1.25, "step": 3})

    assert returned_path == output_path
    assert returned_path.exists()
    assert json.loads(returned_path.read_text(encoding="utf-8")) == {"loss": 1.25, "step": 3}


def test_append_jsonl_creates_parent_and_appends_records(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "train_log.jsonl"

    returned_path = append_jsonl(output_path, {"step": 1, "loss": 2.0})
    append_jsonl(output_path, {"step": 2, "loss": 1.5})

    assert returned_path == output_path
    assert returned_path.exists()

    lines = returned_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"loss": 2.0, "step": 1}
    assert json.loads(lines[1]) == {"loss": 1.5, "step": 2}


def test_activated_parameter_summary_dense_model_counts_all_parameters_active() -> None:
    from deepseek_reimpl.instrumentation.activated_params import summarize_activated_parameters

    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layers=1,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )
    model = BaselineGPT(config)

    summary = summarize_activated_parameters(model)

    assert summary.total_parameters == count_parameters(model)
    assert summary.routed_expert_total_parameters == 0
    assert summary.routed_expert_active_parameters_per_token == 0
    assert summary.activated_parameters_per_token == summary.total_parameters
    assert summary.activated_to_total_ratio == 1.0


def test_activated_parameter_summary_moe_model_excludes_unselected_routed_experts() -> None:
    from deepseek_reimpl.instrumentation.activated_params import summarize_activated_parameters

    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layers=1,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_top_k=2,
        moe_expert_d_ff=16,
        moe_aux_loss_weight=0.01,
    )
    model = BaselineGPT(config)

    summary = summarize_activated_parameters(model)

    assert summary.total_parameters == count_parameters(model)
    assert summary.routed_expert_total_parameters > 0
    assert summary.routed_expert_active_parameters_per_token > 0
    assert summary.activated_parameters_per_token < summary.total_parameters
    assert 0.0 < summary.activated_to_total_ratio < 1.0


def test_routing_stats_summary_returns_none_before_moe_forward() -> None:
    from deepseek_reimpl.instrumentation.routing_stats import summarize_routing_stats

    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layers=1,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_top_k=2,
        moe_expert_d_ff=16,
        moe_aux_loss_weight=0.01,
    )
    model = BaselineGPT(config)

    assert summarize_routing_stats(model) is None


def test_routing_stats_summary_collects_moe_layer_stats_after_forward() -> None:
    from deepseek_reimpl.instrumentation.routing_stats import summarize_routing_stats

    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_top_k=2,
        moe_expert_d_ff=16,
        moe_aux_loss_weight=0.01,
    )
    model = BaselineGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    model(input_ids)
    summary = summarize_routing_stats(model)

    assert summary is not None
    assert summary.moe_layers == 2
    assert summary.tokens_per_layer == [10, 10]
    assert summary.mean_routing_entropy is not None
    assert summary.mean_expert_load_variance is not None
    assert summary.mean_aux_loss is not None
    assert len(summary.expert_selection_counts) == 2
    assert len(summary.expert_selection_counts[0]) == 4


def test_activated_parameter_summary_mla_moe_model_excludes_unselected_routed_experts() -> None:
    from deepseek_reimpl.instrumentation.activated_params import summarize_activated_parameters

    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layers=1,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
        positional_encoding="rope",
        attention_type="mla",
        mla_kv_latent_dim=16,
        mla_q_rope_dim=8,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_top_k=2,
        moe_expert_d_ff=16,
        moe_aux_loss_weight=0.01,
    )
    model = BaselineGPT(config)

    summary = summarize_activated_parameters(model)

    assert summary.total_parameters == count_parameters(model)
    assert summary.always_active_parameters > 0
    assert summary.routed_expert_total_parameters > 0
    assert summary.routed_expert_active_parameters_per_token > 0
    assert summary.activated_parameters_per_token < summary.total_parameters
    assert 0.0 < summary.activated_to_total_ratio < 1.0


def test_routing_stats_summary_returns_none_before_mla_moe_forward() -> None:
    from deepseek_reimpl.instrumentation.routing_stats import summarize_routing_stats

    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layers=1,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
        positional_encoding="rope",
        attention_type="mla",
        mla_kv_latent_dim=16,
        mla_q_rope_dim=8,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_top_k=2,
        moe_expert_d_ff=16,
        moe_aux_loss_weight=0.01,
    )
    model = BaselineGPT(config)

    assert summarize_routing_stats(model) is None


def test_routing_stats_summary_collects_mla_moe_layer_stats_after_forward() -> None:
    from deepseek_reimpl.instrumentation.routing_stats import summarize_routing_stats

    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
        positional_encoding="rope",
        attention_type="mla",
        mla_kv_latent_dim=16,
        mla_q_rope_dim=8,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_top_k=2,
        moe_expert_d_ff=16,
        moe_aux_loss_weight=0.01,
    )
    model = BaselineGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    model(input_ids)
    summary = summarize_routing_stats(model)

    assert summary is not None
    assert summary.moe_layers == 2
    assert summary.tokens_per_layer == [10, 10]
    assert summary.mean_routing_entropy is not None
    assert summary.mean_expert_load_variance is not None
    assert summary.mean_aux_loss is not None
    assert len(summary.expert_selection_counts) == 2
    assert len(summary.expert_selection_counts[0]) == 4
