# DeepSeek-Inspired Efficiency Study

This repository contains a controlled, single-GPU comparison of six decoder-only
language-model variants: Dense, Multi-head Latent Attention (MLA), multi-token
prediction (MTP), mixture-of-experts (MoE), MLA+MoE, and a V3-style
auxiliary-loss-free expert-bias routing analogue.

The corrected primary matrix is complete: three fixed token budgets, ten aligned
seeds, and six model families, for 180 runs in total.

## Report

The [project report](reports/DS_proj_report.pdf) presents the complete
180-run study, including its methodology, results, systems measurements,
limitations, and conclusions.

## Study design

| Dimension | Setting |
|---|---|
| Experiment matrix | 6 models x 3 token budgets x 10 aligned seeds |
| Completed runs | 180 |
| Token budgets | 10M, 25M, and 50M |
| Context length | 256 tokens |
| Corpus | FineWeb-Edu `sample-10BT` at a pinned revision |
| Training records | 50,000 |
| Validation/test protocol | 400 fixed, non-overlapping windows per split |
| Tokenizer | Byte-level BPE, vocabulary 10,000 |
| Model scale | Approximately 121M-237M total parameters |
| Hardware | NVIDIA RTX 4050 Laptop GPU |
| Precision | FP32 |
| Primary outcome | Test next-token cross-entropy |

The implementations are local PyTorch analogues. They are not claims of exact
production DeepSeek systems, distributed expert parallelism, FlashMLA, or the
sequential MTP architecture used by DeepSeek-V3.

Statistical claims must remain proportional to this design: three budgets, ten
paired seeds, one dataset sample, one tokenizer, and one hardware/software setup.

## Current figures

### Test loss across token budgets

![Test loss across token budgets](results/figures/balanced_10seed_matrix_report/report_test_loss_by_budget.png)

### Planned test-loss contrasts

![Planned test-loss contrasts](results/figures/balanced_10seed_matrix_report/report_planned_test_loss_contrasts_by_budget.png)

### Quality-throughput tradeoff at 50M tokens

![Quality-throughput tradeoff](results/figures/balanced_10seed_matrix_report/report_quality_throughput_tradeoff_50m.png)

### Total versus activated parameter exposure

![Total versus activated parameter exposure](results/figures/balanced_10seed_matrix_report/report_total_vs_activated_parameter_exposure_50m.png)

All report-ready and diagnostic figures are under `results/figures/`.

## Repository structure

```text
deepseek-reimplementation/
|-- configs/                 # Data, experiment, model, tokenizer, and train configs
|-- deepseek_reimpl/         # Model, training, data, evaluation, and utilities
|-- scripts/
|   |-- analysis/            # Audited statistics, evidence indexing, and figures
|   |-- data/                # Corpus preparation and tokenization
|   |-- tokenizer/           # Tokenizer training
|   |-- train/               # Experiment entry point
|   `-- validation/          # Matrix preflight and model smoke tests
|-- results/
|   |-- analysis/            # Canonical matrix and statistical artifacts
|   |-- figures/             # Regenerated report and diagnostic figures
|   `-- runs/                # Self-authenticating summaries; local logs/checkpoints ignored
|-- tests/
|-- reports/
|-- requirements.txt
|-- requirements-cuda.txt
|-- requirements-dev.txt
`-- pyproject.toml
```

## Reproducible environment

The tested environment is Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, and an
NVIDIA RTX 4050 Laptop GPU. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -m pytest
```

`requirements.txt`, `requirements-cuda.txt`, and `requirements-dev.txt` pin all
direct dependencies. Every completed run records dependency-lock hashes,
Python/PyTorch/CUDA versions, deterministic backend flags, GPU identity, the code
tree and commit, resolved configurations, and input-artifact hashes.

## Data and tokenizer regeneration

The dataset revision and streaming shuffle are pinned in
`configs/data/fineweb_edu_10bt.yaml`. The split writer produces disjoint source
records and explicit record manifests. The tokenizer is trained on the training
split only, seeds the complete ByteLevel alphabet, and disables BOS/EOS
post-processing for raw language-model streams.

```powershell
.\.venv\Scripts\python.exe scripts\data\prepare_hf_streaming_text.py --config configs\data\fineweb_edu_10bt.yaml
.\.venv\Scripts\python.exe scripts\tokenizer\train_tokenizer.py --config configs\tokenizer\bpe_fineweb_edu_10bt_local_experiment.yaml
.\.venv\Scripts\python.exe scripts\data\tokenize_lm_corpus.py --data-config configs\data\fineweb_edu_10bt.yaml --tokenizer-config configs\tokenizer\bpe_fineweb_edu_10bt_local_experiment.yaml
.\.venv\Scripts\python.exe scripts\analysis\build_data_tokenizer_provenance.py
.\.venv\Scripts\python.exe scripts\analysis\verify_final_data_tokenizer_provenance.py --require-local-artifacts
```

Prepared corpora, token binaries, and tokenizer files are intentionally ignored
because of their size. `results/analysis/final_data_tokenizer_provenance.json`
retains their exact paths, byte sizes, and SHA-256 hashes.

## Verification before primary training

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\analysis\build_balanced_10seed_matrix_manifest.py
.\.venv\Scripts\python.exe scripts\validation\preflight_primary_matrix.py
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe scripts\validation\smoke_model_variants.py
.\.venv\Scripts\python.exe scripts\validation\smoke_model_variants.py --full-model-step --primary-shape
```

The preflight verifies all 180 identities, exact generated configurations,
counterbalanced queue positions, local artifact hashes, tokenizer/model agreement,
and fixed held-out samples. Validation and test each use 400 deterministic,
non-overlapping, corpus-spanning windows: 102,400 scored tokens per split for every
model, seed, and budget.

## Primary matrix runner and recovery

Run the queue from an elevated PowerShell session only after committing the exact
source/configuration state and closing interactive applications:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_balanced_10seed_matrix_queue.ps1
```

The runner refuses a dirty source tree or invalid artifact, counterbalances model
order, prevents system sleep, temporarily suppresses Windows automatic updates,
checks for competing interactive/CUDA processes, and restores the update policy on
exit. It checkpoints each run atomically and resumes only when code, configuration,
data, tokenizer, environment, and experiment identity still match. Completed runs
are immutable and semantically validated immediately and before they are skipped.
After the final run, the command regenerates every statistical artifact and all 22
figures, then rebuilds the evidence index.

## Evidence and analysis

- `results/runs/<experiment>/metrics/summary.json` is the self-authenticating run
  summary committed for each matrix cell.
- `results/runs/<experiment>/logs/` contains local durable trajectories and is
  intentionally ignored because of its size.
- `results/analysis/balanced_10seed_matrix_manifest.json` is the canonical 180-cell
  design and counterbalanced queue order.
- `results/analysis/balanced_10seed_matrix_evidence_index.json` hashes the analysis
  artifacts, figures, and all 180 run summaries.
- `scripts/analysis/run_balanced_10seed_pipeline.py` regenerates the manifest,
  validation, extraction, statistics, profiles, figures, and evidence index.

The analysis includes descriptives, repeated-measures global tests, paired exact
tests, bootstrap intervals, Holm-adjusted planned contrasts, budget trends, and
mechanism profiles. Training-step throughput excludes evaluation and artifact I/O;
active end-to-end throughput includes them. Training and evaluation peak GPU
allocation are recorded separately.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The completed corrected matrix passes the full 291-test suite. Before the evidence
commit was created, the complete preflight validated all 180 summaries against the
exact recorded training commit, environment, configurations, data, tokenizer, and
training logs. All extraction, descriptive, global-test, paired-contrast,
budget-trend, and mechanism-profile audits pass with no missing primary metrics.

## License

This project is released under the [MIT License](LICENSE).
