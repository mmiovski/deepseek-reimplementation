from __future__ import annotations

import json

from deepseek_reimpl.data.text_quality import (
    compute_text_quality_report,
    count_mojibake_markers,
    split_lm_documents,
)


def test_split_lm_documents_drops_empty_documents() -> None:
    text = "first doc\n\n\n\nsecond doc\n\n  \n\nthird doc"
    assert split_lm_documents(text) == ["first doc", "second doc", "third doc"]


def test_count_mojibake_markers_counts_common_artifacts() -> None:
    counts = count_mojibake_markers("Janeâ€™s book â€œquotedâ€ badly Ã Â �")

    assert counts["â€™"] == 1
    assert counts["â€œ"] == 1
    assert counts["â€"] >= 1
    assert counts["Ã"] == 1
    assert counts["Â"] == 1
    assert counts["�"] == 1


def test_compute_text_quality_report_summarizes_local_file(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "short doc\n\n" "Janeâ€™s longer document with mojibake\n" "and another line\n\n" "final",
        encoding="utf-8",
    )

    report = compute_text_quality_report(corpus_path)

    assert report["bytes"] == corpus_path.stat().st_size
    assert report["documents"] == 3
    assert report["chars"] == len(corpus_path.read_text(encoding="utf-8"))
    assert report["document_length_chars"]["max"] > report["document_length_chars"]["min"]
    assert report["mojibake_markers"]["â€™"] == 1
    assert report["total_mojibake_markers"] >= 1


def test_audit_text_corpus_cli_writes_json(tmp_path) -> None:
    import subprocess
    import sys
    from pathlib import Path

    corpus_path = tmp_path / "corpus.txt"
    output_path = tmp_path / "audit.json"
    corpus_path.write_text("alpha\n\nbeta â€™", encoding="utf-8")

    script_path = Path("scripts/data/audit_text_corpus.py").resolve()
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--inputs",
            str(corpus_path),
            "--output-json",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["reports"]) == 1
    assert payload["reports"][0]["documents"] == 2
