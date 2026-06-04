from __future__ import annotations

from pathlib import Path

from deepseek_reimpl.data.download import iter_texts_from_records, write_text_stream
from deepseek_reimpl.data.preprocess import format_lm_text, keep_text, normalize_text


def test_normalize_text_strips_and_normalizes_newlines() -> None:
    text = "  hello\r\nworld\r  "
    assert normalize_text(text) == "hello\nworld"


def test_keep_text_respects_min_chars() -> None:
    assert keep_text("abc", min_chars=3)
    assert not keep_text("ab", min_chars=3)


def test_format_lm_text_joins_examples() -> None:
    assert format_lm_text(["one", "two"], eos_text="\n\n") == "one\n\ntwo"


def test_iter_texts_from_records_filters_invalid_and_short_records() -> None:
    records = [
        {"text": "  first story  "},
        {"text": ""},
        {"text": "x"},
        {"text": None},
        {"other": "missing text field"},
        {"text": "second\r\nstory"},
    ]

    texts = iter_texts_from_records(
        records,
        text_field="text",
        min_chars=2,
    )

    assert texts == ["first story", "second\nstory"]


def test_write_text_stream_writes_joined_text(tmp_path: Path) -> None:
    output_path = tmp_path / "stories.txt"

    written_path = write_text_stream(["first", "second"], output_path)

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8") == "first\n\nsecond"


def test_write_train_validation_test_text_artifacts(tmp_path: Path) -> None:
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    test_path = tmp_path / "test.txt"

    written_train_path = write_text_stream(["train story"], train_path)
    written_validation_path = write_text_stream(["validation story"], validation_path)
    written_test_path = write_text_stream(["test story"], test_path)

    assert written_train_path == train_path
    assert written_validation_path == validation_path
    assert written_test_path == test_path
    assert train_path.read_text(encoding="utf-8") == "train story"
    assert validation_path.read_text(encoding="utf-8") == "validation story"
    assert test_path.read_text(encoding="utf-8") == "test story"
