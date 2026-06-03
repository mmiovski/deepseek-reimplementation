from __future__ import annotations

import pytest
import torch

from deepseek_reimpl.data.collators import causal_lm_collate
from deepseek_reimpl.data.datasets import LanguageModelingDataset
from deepseek_reimpl.data.tokenization import read_token_ids, write_token_ids


def test_language_modeling_dataset_returns_shifted_input_target() -> None:
    dataset = LanguageModelingDataset([10, 11, 12, 13, 14], block_size=3)

    x, y = dataset[0]

    assert torch.equal(x, torch.tensor([10, 11, 12]))
    assert torch.equal(y, torch.tensor([11, 12, 13]))


def test_language_modeling_dataset_length() -> None:
    dataset = LanguageModelingDataset([1, 2, 3, 4, 5, 6], block_size=4)

    assert len(dataset) == 2


def test_language_modeling_dataset_rejects_too_short_sequence() -> None:
    with pytest.raises(ValueError, match="greater than block_size"):
        LanguageModelingDataset([1, 2, 3], block_size=3)


def test_language_modeling_dataset_rejects_invalid_block_size() -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        LanguageModelingDataset([1, 2, 3], block_size=0)


def test_causal_lm_collate_returns_expected_batch_shapes() -> None:
    dataset = LanguageModelingDataset([1, 2, 3, 4, 5, 6], block_size=3)

    batch = causal_lm_collate([dataset[0], dataset[1]])

    x, y = batch

    assert x.shape == (2, 3)
    assert y.shape == (2, 3)
    assert torch.equal(x[0], torch.tensor([1, 2, 3]))
    assert torch.equal(y[0], torch.tensor([2, 3, 4]))
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_causal_lm_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="batch must contain"):
        causal_lm_collate([])


def test_write_and_read_token_ids_roundtrip(tmp_path) -> None:
    output_path = tmp_path / "tokens.txt"

    written_path = write_token_ids([1, 2, 3, 4], output_path)
    token_ids = read_token_ids(written_path)

    assert written_path == output_path
    assert token_ids == [1, 2, 3, 4]
