from __future__ import annotations

import json

import pytest

from deepseek_reimpl.instrumentation.logging_utils import write_json


def test_write_json_atomically_replaces_complete_artifact(tmp_path) -> None:
    output_path = tmp_path / "summary.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")

    write_json(output_path, {"run_completed": True, "value": 3})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "run_completed": True,
        "value": 3,
    }
    assert not list(tmp_path.glob(".summary.json.*.tmp"))


def test_write_json_preserves_prior_artifact_when_serialization_fails(tmp_path) -> None:
    output_path = tmp_path / "summary.json"
    original = '{"old": true}\n'
    output_path.write_text(original, encoding="utf-8")

    with pytest.raises(TypeError):
        write_json(output_path, {"not_json": {1, 2, 3}})

    assert output_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".summary.json.*.tmp"))
