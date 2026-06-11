"""Tests for model initialization and initial logit scale."""

from __future__ import annotations

import math

import torch

from deepseek_reimpl.model.model_factory import build_model_from_config
from deepseek_reimpl.train.losses import next_token_cross_entropy
from deepseek_reimpl.train.train_utils import set_seed
from deepseek_reimpl.utils.config import load_yaml_config


def test_baseline_gpt_untrained_loss_is_near_uniform_vocab_scale() -> None:
    set_seed(1337)
    model_config = load_yaml_config("configs/model/baseline_gpt.yaml")
    model = build_model_from_config(model_config)
    model.eval()

    vocab_size = int(model_config["model"]["vocab_size"])
    input_ids = torch.randint(0, vocab_size, (4, 64))
    targets = torch.randint(0, vocab_size, (4, 64))

    with torch.no_grad():
        logits = model(input_ids)
        loss = next_token_cross_entropy(logits, targets)

    assert logits.std().item() < 1.0
    assert abs(loss.item() - math.log(vocab_size)) < 1.0
