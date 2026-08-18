"""Regenerate the complete statistical and figure evidence pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "scripts" / "analysis"

ANALYSIS_STAGES = (
    ("build_balanced_10seed_matrix_manifest.py", ()),
    ("../validation/preflight_primary_matrix.py", ("--require-complete",)),
    ("extract_balanced_10seed_matrix_artifacts.py", ()),
    ("summarize_balanced_10seed_matrix_descriptives.py", ()),
    ("analyze_balanced_10seed_matrix_global_tests.py", ()),
    ("analyze_balanced_10seed_matrix_paired_contrasts.py", ()),
    ("analyze_balanced_10seed_matrix_budget_trends.py", ()),
    ("build_balanced_10seed_mechanism_profiles.py", ()),
)

PLOT_STAGES = (
    "plot_mean_aux_loss_by_budget.py",
    "plot_mean_expert_load_variance_by_budget.py",
    "plot_mean_routing_entropy_by_budget.py",
    "plot_mtp_loss_by_budget.py",
    "plot_peak_memory_gib_by_budget.py",
    "plot_report_mtp_optimization_comparison.py",
    "plot_report_paired_test_loss_contrasts_50m.py",
    "plot_report_planned_test_loss_contrasts_by_budget.py",
    "plot_report_quality_throughput_tradeoff_50m.py",
    "plot_report_routing_behavior_comparison.py",
    "plot_report_test_loss_by_budget.py",
    "plot_report_test_loss_vs_tokens_per_activated_parameter.py",
    "plot_report_total_vs_activated_parameter_exposure_50m.py",
    "plot_test_loss_by_budget.py",
    "plot_test_perplexity_by_budget.py",
    "plot_tokens_per_activated_parameter_by_budget.py",
    "plot_tokens_per_total_parameter_by_budget.py",
    "plot_tokens_per_trainable_parameter_by_budget.py",
    "plot_train_loss_by_budget.py",
    "plot_train_tokens_per_second_by_budget.py",
    "plot_validation_loss_by_budget.py",
    "plot_validation_perplexity_by_budget.py",
)


def _validate_stage_contract() -> None:
    observed_plots = {path.name for path in ANALYSIS_DIR.glob("plot_*.py")}
    expected_plots = set(PLOT_STAGES)
    if observed_plots != expected_plots:
        raise RuntimeError(
            "Plot-stage contract mismatch; "
            f"missing={sorted(expected_plots - observed_plots)}, "
            f"unexpected={sorted(observed_plots - expected_plots)}"
        )


def main() -> None:
    _validate_stage_contract()
    stages = (
        *ANALYSIS_STAGES,
        *((plot_stage, ()) for plot_stage in PLOT_STAGES),
        ("build_balanced_10seed_evidence_index.py", ()),
    )
    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONHASHSEED"] = "0"

    for index, (relative_stage, arguments) in enumerate(stages, start=1):
        stage = (ANALYSIS_DIR / relative_stage).resolve()
        if not stage.is_file():
            raise FileNotFoundError(f"Missing analysis stage: {stage}")
        print(f"[{index}/{len(stages)}] {stage.name}", flush=True)
        subprocess.run(
            [sys.executable, str(stage), *arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
