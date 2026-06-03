"""Config loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from deepseek_reimpl.utils.paths import assert_relative_path, project_path

ConfigDict = dict[str, Any]


def load_yaml_config(path: str | Path) -> ConfigDict:
    """Load a YAML config file.

    Relative paths are interpreted from the project root.
    """
    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = project_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        raise TypeError(f"Expected YAML mapping in {config_path}, got {type(loaded)}")

    return loaded


def require_keys(config: ConfigDict, required_keys: set[str], *, name: str) -> None:
    """Validate required top-level config keys."""
    missing = required_keys.difference(config)

    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"{name} config missing required keys: {missing_str}")


def validate_relative_paths(config: ConfigDict, path_keys: tuple[str, ...]) -> None:
    """Validate that selected config entries are repo-relative paths."""
    for key in path_keys:
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"Expected path config value for {key} to be str, got {type(value)}")
        assert_relative_path(value)
