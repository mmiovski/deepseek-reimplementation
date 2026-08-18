from __future__ import annotations

import pytest

from scripts.analysis.analyze_balanced_10seed_matrix_budget_trends import (
    exact_sign_flip_p_value as trend_sign_flip_p_value,
)
from scripts.analysis.analyze_balanced_10seed_matrix_global_tests import friedman_q
from scripts.analysis.analyze_balanced_10seed_matrix_paired_contrasts import (
    exact_sign_flip_p_value as contrast_sign_flip_p_value,
)


@pytest.mark.parametrize(
    "function",
    [contrast_sign_flip_p_value, trend_sign_flip_p_value],
)
def test_exact_sign_flip_keeps_zero_pairs_in_a_consistent_statistic(function) -> None:
    assert function([5.0, 4.0, 1.0, 0.0]) == pytest.approx(0.25)
    assert function([0.0, 0.0]) == 1.0


def test_friedman_q_matches_known_untied_and_tied_cases() -> None:
    assert friedman_q([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]) == pytest.approx(4.0)
    assert friedman_q([[1.0, 1.0, 2.0], [1.0, 2.0, 2.0]]) == pytest.approx(3.0)
    assert friedman_q([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]) == 0.0
