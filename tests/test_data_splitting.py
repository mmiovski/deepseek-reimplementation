from __future__ import annotations

import pytest

from deepseek_reimpl.data.splitting import split_sequence_by_fraction


def test_split_sequence_by_fraction_is_deterministic() -> None:
    items = list(range(10))

    first = split_sequence_by_fraction(items, validation_fraction=0.5)
    second = split_sequence_by_fraction(items, validation_fraction=0.5)

    assert first == second


def test_split_sequence_by_fraction_uses_first_portion_for_validation() -> None:
    items = ["a", "b", "c", "d", "e", "f"]

    split = split_sequence_by_fraction(items, validation_fraction=0.5)

    assert split.validation_items == ["a", "b", "c"]
    assert split.test_items == ["d", "e", "f"]


def test_split_sequence_by_fraction_returns_expected_indices() -> None:
    items = ["a", "b", "c", "d", "e"]

    split = split_sequence_by_fraction(items, validation_fraction=0.4)

    assert split.validation_indices == [0, 1]
    assert split.test_indices == [2, 3, 4]


def test_split_sequence_by_fraction_indices_do_not_overlap() -> None:
    items = list(range(11))

    split = split_sequence_by_fraction(items, validation_fraction=0.6)

    validation_indices = set(split.validation_indices)
    test_indices = set(split.test_indices)

    assert validation_indices.isdisjoint(test_indices)


def test_split_sequence_by_fraction_indices_cover_all_items() -> None:
    items = list(range(11))

    split = split_sequence_by_fraction(items, validation_fraction=0.6)

    all_indices = split.validation_indices + split.test_indices

    assert sorted(all_indices) == list(range(len(items)))


def test_split_sequence_by_fraction_keeps_nonempty_parts_for_small_inputs() -> None:
    items = ["only_validation_candidate", "only_test_candidate"]

    split = split_sequence_by_fraction(items, validation_fraction=0.9)

    assert split.validation_items == ["only_validation_candidate"]
    assert split.test_items == ["only_test_candidate"]


@pytest.mark.parametrize("validation_fraction", [0.0, 1.0, -0.1, 1.1])
def test_split_sequence_by_fraction_rejects_invalid_fraction(
    validation_fraction: float,
) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_sequence_by_fraction([1, 2, 3], validation_fraction=validation_fraction)
