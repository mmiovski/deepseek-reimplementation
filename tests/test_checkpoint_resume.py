from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from deepseek_reimpl.data.collators import causal_lm_collate
from deepseek_reimpl.data.datasets import RandomMemmapLanguageModelingDataset
from deepseek_reimpl.model.baseline_gpt import BaselineGPT
from deepseek_reimpl.model.config import GPTConfig
from deepseek_reimpl.train.checkpointing import (
    atomic_save_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
)
from deepseek_reimpl.train.optim import build_optimizer
from deepseek_reimpl.train.train_utils import set_seed
from deepseek_reimpl.train.trainer import TrainingLoopConfig, train_loop


class PlannedInterruption(RuntimeError):
    pass


def _objects(token_path: Path):
    set_seed(1234)
    model = BaselineGPT(
        GPTConfig(
            vocab_size=32,
            block_size=4,
            n_layers=1,
            n_heads=2,
            d_model=16,
            d_ff=32,
            dropout=0.2,
        )
    )
    optimizer = build_optimizer(
        model,
        {"learning_rate": 0.001, "weight_decay": 0.01, "betas": [0.9, 0.95]},
    )
    dataset = RandomMemmapLanguageModelingDataset(
        token_path,
        num_tokens=256,
        block_size=4,
        seed=77,
    )
    loader_generator = torch.Generator().manual_seed(88)
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=causal_lm_collate,
        generator=loader_generator,
    )
    return model, optimizer, dataset, loader


def test_interrupted_resume_matches_uninterrupted_state_bitwise(tmp_path: Path) -> None:
    token_path = tmp_path / "tokens.bin"
    (np.arange(256, dtype=np.int32) % 32).tofile(token_path)
    config = TrainingLoopConfig(
        max_steps=4,
        max_tokens=None,
        eval_interval=None,
        log_interval=1,
        eval_batches=1,
        checkpoint_interval=2,
    )

    reference_model, reference_optimizer, _, reference_loader = _objects(token_path)
    reference_summary = train_loop(
        reference_model,
        reference_loader,
        reference_optimizer,
        device=torch.device("cpu"),
        config=config,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    interrupted_model, interrupted_optimizer, interrupted_dataset, interrupted_loader = _objects(
        token_path
    )

    def checkpoint_and_interrupt(training_state):
        atomic_save_checkpoint(
            checkpoint_path,
            {
                "schema_version": 1,
                "model_state": interrupted_model.state_dict(),
                "optimizer_state": interrupted_optimizer.state_dict(),
                "dataset_state": interrupted_dataset.state_dict(),
                "rng_state": capture_rng_state(),
                "training_state": training_state,
            },
        )
        raise PlannedInterruption

    with pytest.raises(PlannedInterruption):
        train_loop(
            interrupted_model,
            interrupted_loader,
            interrupted_optimizer,
            device=torch.device("cpu"),
            config=config,
            checkpoint_callback=checkpoint_and_interrupt,
        )

    resumed_model, resumed_optimizer, resumed_dataset, resumed_loader = _objects(token_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
    resumed_model.load_state_dict(checkpoint["model_state"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
    resumed_dataset.load_state_dict(checkpoint["dataset_state"])
    restore_rng_state(checkpoint["rng_state"])
    resumed_summary = train_loop(
        resumed_model,
        resumed_loader,
        resumed_optimizer,
        device=torch.device("cpu"),
        config=config,
        initial_state=checkpoint["training_state"],
    )

    for name, tensor in reference_model.state_dict().items():
        assert torch.equal(tensor, resumed_model.state_dict()[name]), name
    assert reference_summary.steps == resumed_summary.steps
    assert reference_summary.train_tokens == resumed_summary.train_tokens
    assert reference_summary.final_train_loss == resumed_summary.final_train_loss
    assert reference_summary.final_lm_loss == resumed_summary.final_lm_loss


def test_malformed_checkpoint_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": 999}, path)
    with pytest.raises(ValueError, match="Unsupported or malformed"):
        load_checkpoint(path, map_location=torch.device("cpu"))
