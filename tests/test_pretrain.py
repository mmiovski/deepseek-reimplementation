from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from deepseek_reimpl.train.pretrain import _build_lm_dataloader, _require_file
from tokenizers import Tokenizer


def _write_tiny_wordlevel_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def test_require_file_returns_existing_absolute_path(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.txt"
    file_path.write_text("content", encoding="utf-8")

    assert _require_file(file_path, purpose="test artifact", remediation="create it") == file_path


def test_require_file_raises_clear_error_for_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Missing test artifact"):
        _require_file(missing_path, purpose="test artifact", remediation="create it")


def test_build_lm_dataloader_from_text_and_tokenizer_artifacts(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    text_path = tmp_path / "train.txt"

    _write_tiny_wordlevel_tokenizer(tokenizer_path)
    text_path.write_text("one two three four five six seven eight", encoding="utf-8")

    dataloader = _build_lm_dataloader(
        text_path=text_path,
        tokenizer_path=tokenizer_path,
        block_size=3,
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )

    input_ids, targets = next(iter(dataloader))

    assert input_ids.shape == (2, 3)
    assert targets.shape == (2, 3)
    assert input_ids.dtype == torch.long
    assert targets.dtype == torch.long


def test_build_lm_dataloader_uses_fixed_corpus_spanning_evaluation_windows(
    tmp_path: Path,
) -> None:
    token_ids_path = tmp_path / "tokens.int32.bin"
    np.arange(41, dtype=np.int32).tofile(token_ids_path)

    dataloader = _build_lm_dataloader(
        text_path=tmp_path / "unused.txt",
        tokenizer_path=tmp_path / "unused.json",
        token_ids_path=token_ids_path,
        token_count=41,
        split_name="validation",
        block_size=4,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        fixed_eval_samples=4,
    )

    starts = [int(input_ids[0].item()) for batch, _ in dataloader for input_ids in batch]
    assert starts == [0, 12, 24, 36]


def test_pretraining_refuses_to_overwrite_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepseek_reimpl.train import pretrain

    root = tmp_path
    (root / "configs" / "model").mkdir(parents=True)
    (root / "configs" / "data").mkdir(parents=True)
    (root / "configs" / "tokenizer").mkdir(parents=True)
    (root / "configs" / "train").mkdir(parents=True)
    (root / "configs" / "experiment").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    (root / "tokenizers").mkdir(parents=True)

    tokenizer_path = root / "tokenizers" / "tiny.json"
    train_text = root / "data" / "train.txt"
    validation_text = root / "data" / "validation.txt"
    test_text = root / "data" / "test.txt"

    _write_tiny_wordlevel_tokenizer(tokenizer_path)
    text = "one two three four five six seven eight " * 4
    train_text.write_text(text, encoding="utf-8")
    validation_text.write_text(text, encoding="utf-8")
    test_text.write_text(text, encoding="utf-8")

    (root / "configs" / "model" / "tiny.yaml").write_text(
        """
model:
  name: baseline_gpt
  vocab_size: 9
  block_size: 3
  n_layers: 1
  n_heads: 1
  d_model: 4
  d_ff: 16
  dropout: 0.0
  norm_type: rmsnorm
  positional_encoding: rope
  ffn_type: swiglu
  attention_type: dense
  tie_embeddings: true
""",
        encoding="utf-8",
    )
    (root / "configs" / "data" / "tiny.yaml").write_text(
        """
artifacts:
  train_text: data/train.txt
  validation_text: data/validation.txt
  test_text: data/test.txt
""",
        encoding="utf-8",
    )
    (root / "configs" / "tokenizer" / "tiny.yaml").write_text(
        """
artifacts:
  tokenizer_json: tokenizers/tiny.json
""",
        encoding="utf-8",
    )
    (root / "configs" / "train" / "tiny.yaml").write_text(
        """
train:
  seed: 1337
  device: cpu
  batch_size: 2
  block_size: 3
  max_steps: 2
  max_tokens: 12
  eval_interval: 2
  eval_batches: 1
  learning_rate: 0.0003
  weight_decay: 0.0
  betas: [0.9, 0.95]
  grad_clip: null
  num_workers: 0
  checkpoint_interval: null
  log_interval: 1
  precision: fp32
""",
        encoding="utf-8",
    )
    experiment_path = root / "configs" / "experiment" / "tiny.yaml"
    experiment_path.write_text(
        f"""
experiment:
  name: tiny
  description: Tiny regression experiment.
  model_config: {root / "configs" / "model" / "tiny.yaml"}
  data_config: {root / "configs" / "data" / "tiny.yaml"}
  tokenizer_config: {root / "configs" / "tokenizer" / "tiny.yaml"}
  train_config: {root / "configs" / "train" / "tiny.yaml"}
  output_dir: results/raw_logs/tiny
  metrics_dir: results/metrics/tiny
  checkpoint_dir: checkpoints/tiny
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(pretrain, "project_path", lambda *parts: root.joinpath(*map(str, parts)))

    train_log_path = root / "results" / "raw_logs" / "tiny" / "train_log.jsonl"
    summary_path = root / "results" / "metrics" / "tiny" / "summary.json"
    first_summary = pretrain.run_pretraining_from_experiment_config(experiment_path)
    original_log = train_log_path.read_bytes()
    original_summary = summary_path.read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        pretrain.run_pretraining_from_experiment_config(experiment_path)

    assert train_log_path.read_bytes() == original_log
    assert summary_path.read_bytes() == original_summary

    lines = train_log_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert len(records) == 3
    assert [record["record_type"] for record in records] == ["train", "train", "eval"]
    assert records[-1]["split"] == "validation"

    assert first_summary["experiment_config_path"] == str(experiment_path)
    assert first_summary["config_paths"]["model_config"] == str(
        root / "configs" / "model" / "tiny.yaml"
    )
    assert first_summary["model_config"]["name"] == "baseline_gpt"
    assert first_summary["train_config"]["max_steps"] == 2
    assert first_summary["tokenizer_artifact"] == "tokenizers/tiny.json"
    assert first_summary["runtime"]["device"] == "cpu"
    assert first_summary["runtime"]["torch_version"]
    assert first_summary["elapsed_seconds"] >= 0.0


def test_incomplete_primary_run_is_archived_before_clean_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepseek_reimpl.train import pretrain

    root = tmp_path
    run_root = root / "results" / "runs" / "interrupted_dense"
    output_dir = run_root / "logs"
    metrics_dir = run_root / "metrics"
    checkpoint_dir = run_root / "checkpoints"
    output_dir.mkdir(parents=True)
    metrics_dir.mkdir()
    checkpoint_dir.mkdir()
    (output_dir / "train_log.jsonl").write_text('{"step": 1}\n', encoding="utf-8")

    monkeypatch.setattr(pretrain, "project_path", lambda *parts: root.joinpath(*map(str, parts)))
    archive_path = pretrain._archive_incomplete_run_artifacts(
        experiment_name="interrupted_dense",
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        checkpoint_dir=checkpoint_dir,
    )

    assert not run_root.exists()
    assert archive_path.parent == root / "tmp" / "interrupted_runs"
    assert (archive_path / "logs" / "train_log.jsonl").is_file()
    recovery = json.loads((archive_path / "recovery.json").read_text(encoding="utf-8"))
    assert recovery == {
        "artifact_type": "incomplete_training_recovery",
        "experiment_name": "interrupted_dense",
        "reason": "run artifacts existed without a completed summary or resumable checkpoint",
        "schema_version": 1,
        "source_run_root": "results/runs/interrupted_dense",
    }


def test_pretraining_summary_helpers_include_dense_activated_metrics() -> None:
    from deepseek_reimpl.model.baseline_gpt import BaselineGPT
    from deepseek_reimpl.model.config import GPTConfig
    from deepseek_reimpl.train.pretrain import (
        _activated_parameter_summary_to_dict,
        _routing_stats_summary_to_dict,
    )

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

    activated = _activated_parameter_summary_to_dict(model)
    routing_stats = _routing_stats_summary_to_dict(model)

    assert activated["total_parameters"] > 0
    assert activated["activated_parameters_per_token"] == activated["total_parameters"]
    assert activated["activated_to_total_ratio"] == 1.0
    assert routing_stats is None


def test_pretraining_summary_helpers_include_moe_routing_metrics_after_forward() -> None:
    from deepseek_reimpl.model.baseline_gpt import BaselineGPT
    from deepseek_reimpl.model.config import GPTConfig
    from deepseek_reimpl.train.pretrain import (
        _activated_parameter_summary_to_dict,
        _routing_stats_summary_to_dict,
    )

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
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    model(input_ids)
    activated = _activated_parameter_summary_to_dict(model)
    routing_stats = _routing_stats_summary_to_dict(model)

    assert activated["routed_expert_total_parameters"] > 0
    assert activated["activated_parameters_per_token"] < activated["total_parameters"]
    assert routing_stats is not None
    assert routing_stats["moe_layers"] == 1
    assert routing_stats["mean_aux_loss"] is not None


def test_mtp_summary_metadata_uses_independent_head_defaults() -> None:
    from deepseek_reimpl.train.pretrain import _mtp_summary_metadata

    assert _mtp_summary_metadata({}) == {
        "mtp_enabled": False,
        "mtp_horizons": [],
        "mtp_auxiliary_head_count": 0,
        "mtp_loss_weight": 0.0,
        "mtp_share_lm_head": False,
    }


def test_mtp_summary_metadata_preserves_explicit_values() -> None:
    from deepseek_reimpl.train.pretrain import _mtp_summary_metadata

    assert _mtp_summary_metadata(
        {
            "mtp_enabled": True,
            "mtp_horizons": [2, 3],
            "mtp_loss_weight": 0.3,
            "mtp_share_lm_head": False,
        }
    ) == {
        "mtp_enabled": True,
        "mtp_horizons": [2, 3],
        "mtp_auxiliary_head_count": 2,
        "mtp_loss_weight": 0.3,
        "mtp_share_lm_head": False,
    }


def test_validate_precision_accepts_fp32() -> None:
    from deepseek_reimpl.train.pretrain import _validate_precision

    assert _validate_precision({"precision": "fp32"}) == "fp32"


def test_validate_precision_rejects_unsupported_precision() -> None:
    import pytest

    from deepseek_reimpl.train.pretrain import _validate_precision

    with pytest.raises(
        ValueError,
        match="Only precision='fp32' is implemented",
    ):
        _validate_precision({"precision": "bf16"})
