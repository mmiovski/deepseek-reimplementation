"""Dataset classes for language modeling."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Fixed-length next-token-prediction dataset.

    Each sample returns:

    x = token_ids[i : i + block_size]
    y = token_ids[i + 1 : i + block_size + 1]
    """

    def __init__(self, token_ids: list[int], *, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        if len(token_ids) <= block_size:
            raise ValueError(
                "token_ids length must be greater than block_size "
                "so at least one shifted sample can be created."
            )

        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        x = self.token_ids[index : index + self.block_size]
        y = self.token_ids[index + 1 : index + self.block_size + 1]

        return x, y
