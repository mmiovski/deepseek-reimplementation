"""Dataset classes for language modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from deepseek_reimpl.utils.paths import project_path


class LanguageModelingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Fixed-length next-token-prediction dataset backed by in-memory token IDs.

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


def _resolve_token_ids_path(token_ids_path: str | Path) -> Path:
    path = Path(token_ids_path)
    if not path.is_absolute():
        path = project_path(path)
    return path


def _validate_memmap_args(*, num_tokens: int, block_size: int) -> None:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if num_tokens <= block_size:
        raise ValueError(
            "num_tokens must be greater than block_size "
            "so at least one shifted sample can be created."
        )


def _window_to_tensors(token_ids: Any, *, index: int, block_size: int) -> tuple[Tensor, Tensor]:
    x_np = np.asarray(token_ids[index : index + block_size], dtype=np.int64)
    y_np = np.asarray(token_ids[index + 1 : index + block_size + 1], dtype=np.int64)

    return torch.from_numpy(x_np), torch.from_numpy(y_np)


class MemmapLanguageModelingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Fixed-length next-token dataset backed by an int32 binary token-ID file."""

    def __init__(self, token_ids_path: str | Path, *, num_tokens: int, block_size: int) -> None:
        _validate_memmap_args(num_tokens=num_tokens, block_size=block_size)

        resolved_path = _resolve_token_ids_path(token_ids_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Missing token-ID artifact: {resolved_path}")

        self.token_ids_path = resolved_path
        self.num_tokens = num_tokens
        self.block_size = block_size
        self.token_ids = np.memmap(
            resolved_path,
            dtype=np.int32,
            mode="r",
            shape=(num_tokens,),
        )

    def __len__(self) -> int:
        return self.num_tokens - self.block_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        return _window_to_tensors(self.token_ids, index=index, block_size=self.block_size)


class RandomMemmapLanguageModelingDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """Infinite random-window LM dataset backed by an int32 binary token-ID file.

    This avoids PyTorch RandomSampler constructing a huge permutation for very large
    corpora while still drawing randomized training windows.
    """

    def __init__(
        self,
        token_ids_path: str | Path,
        *,
        num_tokens: int,
        block_size: int,
        seed: int,
    ) -> None:
        _validate_memmap_args(num_tokens=num_tokens, block_size=block_size)

        resolved_path = _resolve_token_ids_path(token_ids_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Missing token-ID artifact: {resolved_path}")

        self.token_ids_path = resolved_path
        self.num_tokens = num_tokens
        self.block_size = block_size
        self.seed = seed

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        seed = self.seed + worker_id

        generator = torch.Generator()
        generator.manual_seed(seed)

        token_ids = np.memmap(
            self.token_ids_path,
            dtype=np.int32,
            mode="r",
            shape=(self.num_tokens,),
        )

        max_start = self.num_tokens - self.block_size - 1
        while True:
            index = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
            yield _window_to_tensors(token_ids, index=index, block_size=self.block_size)
