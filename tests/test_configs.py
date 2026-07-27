from __future__ import annotations

from pathlib import Path

import pytest

from deepseek_reimpl.model.config import GPTConfig
from deepseek_reimpl.utils.paths import ensure_project_dirs, find_project_root


def test_find_project_root_contains_pyproject() -> None:
    root = find_project_root()
    assert (root / "pyproject.toml").exists()


def test_ensure_project_dirs_creates_expected_directories(tmp_path: Path) -> None:
    dirs = ensure_project_dirs(root=tmp_path)

    expected_keys = {
        "data_raw",
        "data_interim",
        "data_processed",
        "data_tokenized",
        "tokenizers_trained",
        "tokenizers_metadata",
        "results",
        "checkpoints",
    }

    assert set(dirs) == expected_keys
    assert all(path.exists() and path.is_dir() for path in dirs.values())


def test_gpt_config_accepts_valid_mtp_config() -> None:
    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        mtp_enabled=True,
        mtp_num_future_tokens=2,
        mtp_loss_weight=0.5,
        mtp_share_lm_head=False,
    )

    assert config.mtp_enabled is True
    assert config.mtp_num_future_tokens == 2
    assert config.mtp_loss_weight == 0.5
    assert config.mtp_share_lm_head is False


def test_gpt_config_rejects_disabled_mtp_with_future_tokens() -> None:
    with pytest.raises(ValueError, match="mtp_num_future_tokens must be 0"):
        GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layers=2,
            n_heads=4,
            d_model=32,
            d_ff=64,
            mtp_num_future_tokens=2,
        )


def test_gpt_config_rejects_enabled_mtp_without_positive_horizons() -> None:
    with pytest.raises(ValueError, match="mtp_num_future_tokens must be positive"):
        GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layers=2,
            n_heads=4,
            d_model=32,
            d_ff=64,
            mtp_enabled=True,
            mtp_num_future_tokens=0,
            mtp_loss_weight=0.5,
        )


def test_gpt_config_rejects_enabled_mtp_with_invalid_loss_weight() -> None:
    with pytest.raises(ValueError, match="mtp_loss_weight must be positive"):
        GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layers=2,
            n_heads=4,
            d_model=32,
            d_ff=64,
            mtp_enabled=True,
            mtp_num_future_tokens=2,
            mtp_loss_weight=0.0,
        )


def test_gpt_config_rejects_mtp_horizon_at_or_above_block_size() -> None:
    with pytest.raises(ValueError, match="mtp_num_future_tokens must be smaller than block_size"):
        GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layers=2,
            n_heads=4,
            d_model=32,
            d_ff=64,
            mtp_enabled=True,
            mtp_num_future_tokens=16,
            mtp_loss_weight=0.5,
        )


def test_gpt_config_defaults_to_independent_mtp_heads() -> None:
    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        mtp_enabled=True,
        mtp_num_future_tokens=2,
        mtp_loss_weight=0.5,
    )

    assert config.mtp_share_lm_head is False


def test_gpt_config_rejects_unimplemented_shared_mtp_head() -> None:
    with pytest.raises(ValueError, match="mtp_share_lm_head=True is not implemented"):
        GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layers=2,
            n_heads=4,
            d_model=32,
            d_ff=64,
            mtp_share_lm_head=True,
        )
