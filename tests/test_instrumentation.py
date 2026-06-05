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
