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
