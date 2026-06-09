"""Tests for multi-token prediction layers and losses."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from deepseek_reimpl.layers.mtp import MultiTokenPredictionHead
from deepseek_reimpl.model.baseline_gpt import BaselineGPT
from deepseek_reimpl.model.config import GPTConfig
from deepseek_reimpl.train.losses import multi_token_cross_entropy


def test_multi_token_prediction_head_returns_stacked_future_logits() -> None:
    head = MultiTokenPredictionHead(d_model=8, vocab_size=13, num_future_tokens=3)
    hidden_states = torch.randn(2, 5, 8)

    logits = head(hidden_states)

    assert logits.shape == (3, 2, 5, 13)
    assert torch.isfinite(logits).all()


def test_multi_token_prediction_head_backpropagates_to_hidden_states() -> None:
    head = MultiTokenPredictionHead(d_model=8, vocab_size=13, num_future_tokens=2)
    hidden_states = torch.randn(2, 5, 8, requires_grad=True)

    loss = head(hidden_states).sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


@pytest.mark.parametrize(
    ("d_model", "vocab_size", "num_future_tokens"),
    [
        (0, 13, 2),
        (8, 0, 2),
        (8, 13, 0),
    ],
)
def test_multi_token_prediction_head_rejects_invalid_dimensions(
    d_model: int,
    vocab_size: int,
    num_future_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        MultiTokenPredictionHead(
            d_model=d_model,
            vocab_size=vocab_size,
            num_future_tokens=num_future_tokens,
        )


def test_multi_token_prediction_head_rejects_non_rank_three_hidden_states() -> None:
    head = MultiTokenPredictionHead(d_model=8, vocab_size=13, num_future_tokens=2)

    with pytest.raises(ValueError, match="hidden_states must have shape"):
        head(torch.randn(2, 8))


def test_multi_token_cross_entropy_matches_manual_shifted_targets() -> None:
    token_ids = torch.tensor([[0, 1, 2, 3, 4]])
    logits = torch.randn(2, 1, 5, 7)

    loss, per_horizon = multi_token_cross_entropy(logits, token_ids)

    horizon_1 = F.cross_entropy(logits[0, :, :-1, :].reshape(-1, 7), token_ids[:, 1:].reshape(-1))
    horizon_2 = F.cross_entropy(logits[1, :, :-2, :].reshape(-1, 7), token_ids[:, 2:].reshape(-1))
    expected = torch.stack([horizon_1, horizon_2]).mean()

    assert torch.allclose(loss, expected)
    assert per_horizon == pytest.approx((float(horizon_1.item()), float(horizon_2.item())))


def test_multi_token_cross_entropy_backpropagates_to_future_logits() -> None:
    token_ids = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    logits = torch.randn(2, 2, 5, 7, requires_grad=True)

    loss, _ = multi_token_cross_entropy(logits, token_ids)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_multi_token_cross_entropy_rejects_invalid_future_logits_rank() -> None:
    with pytest.raises(ValueError, match="future_token_logits must have shape"):
        multi_token_cross_entropy(torch.randn(2, 5, 7), torch.ones(2, 5, dtype=torch.long))


def test_multi_token_cross_entropy_rejects_invalid_token_rank() -> None:
    with pytest.raises(ValueError, match="token_ids must have shape"):
        multi_token_cross_entropy(torch.randn(2, 2, 5, 7), torch.ones(2, 5, 1, dtype=torch.long))


def test_multi_token_cross_entropy_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="batch/sequence dimensions must match"):
        multi_token_cross_entropy(torch.randn(2, 2, 5, 7), torch.ones(2, 6, dtype=torch.long))


def test_multi_token_cross_entropy_rejects_horizon_at_or_above_sequence_length() -> None:
    with pytest.raises(ValueError, match="num_future_tokens must be smaller than sequence length"):
        multi_token_cross_entropy(torch.randn(5, 2, 5, 7), torch.ones(2, 5, dtype=torch.long))


def _tiny_mtp_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=31,
        block_size=8,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        mtp_enabled=True,
        mtp_num_future_tokens=2,
        mtp_loss_weight=0.5,
        mtp_share_lm_head=False,
    )


def test_baseline_gpt_forward_mtp_returns_main_and_future_logits() -> None:
    config = _tiny_mtp_config()
    model = BaselineGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    output = model.forward_mtp(input_ids)

    assert output.next_token_logits.shape == (2, 5, config.vocab_size)
    assert output.future_token_logits.shape == (
        config.mtp_num_future_tokens,
        2,
        5,
        config.vocab_size,
    )
    assert torch.isfinite(output.next_token_logits).all()
    assert torch.isfinite(output.future_token_logits).all()


def test_baseline_gpt_forward_still_returns_plain_logits_when_mtp_enabled() -> None:
    config = _tiny_mtp_config()
    model = BaselineGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits = model(input_ids)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 5, config.vocab_size)


def test_baseline_gpt_forward_mtp_requires_enabled_config() -> None:
    config = GPTConfig(
        vocab_size=31,
        block_size=8,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
    )
    model = BaselineGPT(config)

    with pytest.raises(RuntimeError, match="forward_mtp requires mtp_enabled=True"):
        model.forward_mtp(torch.randint(0, config.vocab_size, (2, 5)))


def test_baseline_gpt_forward_mtp_backpropagates_to_trunk_and_mtp_head() -> None:
    config = _tiny_mtp_config()
    model = BaselineGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    output = model.forward_mtp(input_ids)
    loss = output.next_token_logits.mean() + output.future_token_logits.mean()
    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert model.mtp_head is not None
    first_mtp_head = model.mtp_head.heads[0]
    assert first_mtp_head.weight.grad is not None
