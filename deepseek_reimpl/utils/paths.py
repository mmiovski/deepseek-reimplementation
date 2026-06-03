"""Path utilities for repo-relative, cross-platform artifact handling."""

from __future__ import annotations

from pathlib import Path

_PROJECT_MARKERS = ("pyproject.toml", ".git")


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by walking upward from start."""
    current = (start or Path.cwd()).resolve()

    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _PROJECT_MARKERS):
            return candidate

    raise FileNotFoundError(
        f"Could not find project root from {current}. "
        f"Expected one of these markers: {_PROJECT_MARKERS}"
    )


def project_path(*parts: str | Path, root: Path | None = None) -> Path:
    """Return a path inside the project root."""
    base = root or find_project_root()
    return base.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_project_dirs(root: Path | None = None) -> dict[str, Path]:
    """Ensure standard local artifact directories exist."""
    base = root or find_project_root()

    dirs = {
        "data_raw": base / "data" / "raw",
        "data_interim": base / "data" / "interim",
        "data_processed": base / "data" / "processed",
        "data_tokenized": base / "data" / "tokenized",
        "tokenizers_trained": base / "tokenizers" / "trained",
        "tokenizers_metadata": base / "tokenizers" / "metadata",
        "results": base / "results",
        "checkpoints": base / "checkpoints",
    }

    for directory in dirs.values():
        ensure_dir(directory)

    return dirs


def assert_relative_path(path_value: str) -> None:
    """Raise if a config path is absolute."""
    if Path(path_value).is_absolute():
        raise ValueError(f"Config paths must be repo-relative, got: {path_value}")
