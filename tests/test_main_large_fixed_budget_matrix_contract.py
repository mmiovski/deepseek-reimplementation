from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from deepseek_reimpl.utils.config import load_yaml_config

BUDGETS = {
    "10m": 10_000_000,
    "25m": 25_000_000,
    "50m": 50_000_000,
}

ARCHITECTURES = {
    "00_dense_121m": "configs/model/dense_121m.yaml",
    "01_mla_121m": "configs/model/mla_121m.yaml",
    "02_mtp_121m": "configs/model/mtp_121m.yaml",
    "03_moe_220m": "configs/model/moe_220m.yaml",
    "04_mla_moe_220m": "configs/model/mla_moe_220m.yaml",
    "05_v3_routing_220m": "configs/model/v3_routing_220m.yaml",
}


def _load(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], load_yaml_config(path))


def test_main_large_matrix_must_cover_all_budgets_and_architectures_when_created() -> None:
    missing: list[str] = []

    for budget in BUDGETS:
        for suffix in ARCHITECTURES:
            path = Path("configs/experiment") / f"main_large_{budget}_{suffix}.yaml"
            if not path.exists():
                missing.append(str(path))

    assert missing == [], (
        "Main large experiment matrix must cover all 18 runs: "
        "6 large architectures x 10M/25M/50M budgets. Missing: " + ", ".join(missing)
    )


def test_main_large_train_configs_use_fixed_requested_token_budgets_when_created() -> None:
    missing: list[str] = []

    for budget, requested_tokens in BUDGETS.items():
        path = Path("configs/train") / f"main_large_{budget}.yaml"
        if not path.exists():
            missing.append(str(path))
            continue

        wrapper = _load(path)
        assert set(wrapper.keys()) == {"train"}
        train = cast(dict[str, Any], wrapper["train"])
        assert train["max_tokens"] == requested_tokens
        assert train["max_steps"] is None
        assert train["device"] == "cuda"

    assert missing == [], "Missing train configs: " + ", ".join(missing)


def test_main_large_experiment_configs_use_exact_large_model_paths_when_created() -> None:
    for budget in BUDGETS:
        for suffix, expected_model_path in ARCHITECTURES.items():
            path = Path("configs/experiment") / f"main_large_{budget}_{suffix}.yaml"
            if not path.exists():
                continue

            wrapper = _load(path)
            assert set(wrapper.keys()) == {"experiment"}
            experiment = cast(dict[str, Any], wrapper["experiment"])
            assert experiment["model_config"] == expected_model_path
            assert experiment["data_config"] == "configs/data/fineweb_edu_10bt.yaml"
            assert (
                experiment["tokenizer_config"]
                == "configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml"
            )
            assert experiment["train_config"] == f"configs/train/main_large_{budget}.yaml"
