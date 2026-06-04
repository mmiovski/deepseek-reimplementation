"""Deterministic dataset splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class IndexedSplit(Generic[T]):
    """Container for deterministic split outputs and source indices."""

    validation_items: list[T]
    test_items: list[T]
    validation_indices: list[int]
    test_indices: list[int]


def split_sequence_by_fraction(
    items: list[T],
    *,
    validation_fraction: float,
) -> IndexedSplit[T]:
    """Split a sequence into deterministic validation and test portions.

    The first ``validation_fraction`` portion of ``items`` is assigned to
    validation. The remaining portion is assigned to test. No shuffling or
    randomness is used.

    Args:
        items: Ordered source items to split.
        validation_fraction: Fraction of items assigned to validation.
            Must be strictly between 0 and 1.

    Returns:
        IndexedSplit containing validation/test items and their original
        zero-based source indices.
    """
    if not 0.0 < validation_fraction < 1.0:
        msg = "validation_fraction must be strictly between 0 and 1"
        raise ValueError(msg)

    split_index = int(len(items) * validation_fraction)

    if items and split_index == 0:
        split_index = 1

    if items and split_index == len(items):
        split_index = len(items) - 1

    validation_items = items[:split_index]
    test_items = items[split_index:]

    validation_indices = list(range(0, split_index))
    test_indices = list(range(split_index, len(items)))

    return IndexedSplit(
        validation_items=validation_items,
        test_items=test_items,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )
