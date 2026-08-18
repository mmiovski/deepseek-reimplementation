"""Tests for scalable memory-mapped LM datasets."""

from __future__ import annotations

import numpy as np
import pytest

from deepseek_reimpl.data.datasets import (
    FixedEvaluationWindowDataset,
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


def test_memmap_dataset_rejects_metadata_file_size_mismatch(tmp_path):
    token_ids_path = tmp_path / "tokens.int32.bin"
    np.arange(21, dtype=np.int32).tofile(token_ids_path)

    with pytest.raises(ValueError, match="Token-ID artifact size mismatch"):
        MemmapLanguageModelingDataset(token_ids_path, num_tokens=20, block_size=4)


def test_fixed_evaluation_windows_are_non_overlapping_and_corpus_spanning(tmp_path):
    token_ids_path = tmp_path / "tokens.int32.bin"
    np.arange(41, dtype=np.int32).tofile(token_ids_path)
    source = MemmapLanguageModelingDataset(token_ids_path, num_tokens=41, block_size=4)

    dataset = FixedEvaluationWindowDataset(source, block_size=4, num_samples=4)

    assert dataset.start_indices == [0, 12, 24, 36]
    assert [dataset[index][0][0].item() for index in range(len(dataset))] == [0, 12, 24, 36]
    assert all(
        next_start - start >= dataset.block_size
        for start, next_start in zip(
            dataset.start_indices,
            dataset.start_indices[1:],
            strict=False,
        )
    )


def test_fixed_evaluation_windows_reject_overlapping_request(tmp_path):
    token_ids_path = tmp_path / "tokens.int32.bin"
    np.arange(20, dtype=np.int32).tofile(token_ids_path)
    source = MemmapLanguageModelingDataset(token_ids_path, num_tokens=20, block_size=4)

    with pytest.raises(ValueError, match="available non-overlapping windows"):
        FixedEvaluationWindowDataset(source, block_size=4, num_samples=5)
