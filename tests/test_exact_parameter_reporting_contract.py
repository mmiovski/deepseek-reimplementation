from __future__ import annotations

from pathlib import Path


def test_pretrain_metrics_include_exact_parameter_accounting_fields() -> None:
    source = Path("deepseek_reimpl/train/pretrain.py").read_text(encoding="utf-8")

    required_fragments = [
        "total_parameters = count_parameters(model)",
        "trainable_parameters = count_trainable_parameters(model)",
        "activated_parameter_summary = _activated_parameter_summary_to_dict(model)",
        '"total_parameters": total_parameters',
        '"trainable_parameters": trainable_parameters',
        '"activated_parameters": activated_parameter_summary',
        '"tokens_per_total_parameter"',
        '"tokens_per_trainable_parameter"',
        '"tokens_per_activated_parameter"',
        '"requested_tokens_per_total_parameter"',
        '"requested_tokens_per_trainable_parameter"',
        '"requested_tokens_per_activated_parameter"',
    ]

    missing = [fragment for fragment in required_fragments if fragment not in source]
    assert missing == []
