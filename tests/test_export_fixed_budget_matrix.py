from __future__ import annotations

import pytest

from scripts.analysis.export_fixed_budget_matrix import _budget_from_name, _suffix_from_name


def test_budget_from_name_parses_local_experiment_budget() -> None:
    assert _budget_from_name("local_10m_00_baseline") == "10m"
    assert _budget_from_name("local_25m_04_v3_routing") == "25m"
    assert _budget_from_name("local_50m_05_mtp") == "50m"


def test_suffix_from_name_parses_architecture_suffix() -> None:
    assert _suffix_from_name("local_10m_00_baseline") == "00_baseline"
    assert _suffix_from_name("local_25m_03_mla_moe") == "03_mla_moe"
    assert _suffix_from_name("local_50m_04_v3_routing") == "04_v3_routing"


def test_name_parsers_reject_unexpected_experiment_names() -> None:
    with pytest.raises(ValueError):
        _budget_from_name("fineweb_pilot_00_baseline")
    with pytest.raises(ValueError):
        _suffix_from_name("baseline")
