"""Tests for model initialization and initial logit scale."""

from __future__ import annotations

import math

import torch

from deepseek_reimpl.model.baseline_gpt import BaselineGPT
from deepseek_reimpl.model.config import GPTConfig
from deepseek_reimpl.train.losses import next_token_cross_entropy
from deepseek_reimpl.train.train_utils import set_seed


def test_baseline_gpt_untrained_loss_is_near_uniform_vocab_scale() -> None:
    set_seed(1337)
    config = GPTConfig(
        vocab_size=128,
        block_size=64,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=64,
        dropout=0.0,
    )
    model = BaselineGPT(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (4, 64))
    targets = torch.randint(0, config.vocab_size, (4, 64))

    with torch.no_grad():
        logits = model(input_ids)
        loss = next_token_cross_entropy(logits, targets)

    assert logits.std().item() < 1.0
    assert abs(loss.item() - math.log(config.vocab_size)) < 1.0
