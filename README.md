# Controlled DeepSeek-Inspired Architecture Study

This repository implements a controlled, single-GPU comparison of six decoder-only
language-model variants: dense attention, Multi-head Latent Attention (MLA),
multi-token prediction (MTP), mixture-of-experts (MoE), MLA+MoE, and an
auxiliary-loss-free expert-bias routing analogue. The primary design uses three
fixed token budgets, ten aligned seeds, one pinned FineWeb-Edu sample, and paired
statistical inference across 180 runs.

The implementations are local PyTorch analogues. They are not claims of exact
production DeepSeek systems, distributed expert parallelism, FlashMLA, or the
sequential MTP architecture used by DeepSeek-V3.

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
direct dependencies. Every completed run records the dependency-lock hashes,
Python/PyTorch/CUDA versions, deterministic backend flags, GPU identity, code-tree
hash, resolved configurations, and input-artifact hashes.

## Data and tokenizer regeneration

The dataset revision and streaming shuffle are pinned in
`configs/data/fineweb_edu_10bt.yaml`. The split writer produces disjoint source
records and explicit record manifests. The tokenizer is trained on the training
split only, seeds the complete ByteLevel alphabet, and raw language-model streams
disable BOS/EOS post-processing.

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
and the fixed held-out samples. Validation and test each use 400 deterministic,
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
After the final run, the same command regenerates every statistical artifact and all
22 figures, then rebuilds the evidence index.

The complete matrix is expected to require roughly two to four weeks of continuous
single-GPU execution on the tested laptop. Training-step throughput excludes
evaluation and artifact I/O; active end-to-end throughput includes them. Both are
recorded separately, as are training and evaluation peak GPU allocation.

## Evidence and analysis

- `results/runs/<experiment>/logs/` contains durable structured trajectories.
- `results/runs/<experiment>/metrics/summary.json` is the self-authenticating run
  summary.
- `results/analysis/balanced_10seed_matrix_manifest.json` is the canonical 180-cell
  design and queue order.
- `scripts/analysis/` produces the flattened evidence, descriptives, paired exact
  tests, Holm-adjusted contrasts, budget trends, mechanism profiles, and evidence
  index.
- Plot scripts preserve the existing report color, typography, and layout themes;
  corrected results change values, not the visual design contract.

Statistical claims must remain proportional to this design: three budgets, ten
paired seeds, one dataset sample, one tokenizer, and one hardware/software setup.
