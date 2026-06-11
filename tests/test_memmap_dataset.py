"""Tests for scalable memory-mapped LM datasets."""

from __future__ import annotations

import numpy as np

from deepseek_reimpl.data.datasets import (
    MemmapLanguageModelingDataset,
    RandomMemmapLanguageModelingDataset,
)


def test_memmap_language_modeling_dataset_reads_shifted_windows(tmp_path):
    token_ids_path = tmp_path / "tokens.int32.bin"
    np.arange(20, dtype=np.int32).tofile(token_ids_path)

    dataset = MemmapLanguageModelingDataset(token_ids_path, num_tokens=20, block_size=4)

    x, y = dataset[3]

    assert len(dataset) == 16
    assert x.tolist() == [3, 4, 5, 6]
    assert y.tolist() == [4, 5, 6, 7]


def test_random_memmap_language_modeling_dataset_yields_valid_windows(tmp_path):
    token_ids_path = tmp_path / "tokens.int32.bin"
    np.arange(50, dtype=np.int32).tofile(token_ids_path)

    dataset = RandomMemmapLanguageModelingDataset(
        token_ids_path,
        num_tokens=50,
        block_size=8,
        seed=123,
    )

    x, y = next(iter(dataset))

    assert x.shape[0] == 8
    assert y.shape[0] == 8
    assert y[:-1].tolist() == x[1:].tolist()
