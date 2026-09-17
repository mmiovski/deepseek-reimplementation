# Experiment Design

The primary evidence comes from a balanced matrix of six model families, three
training-token budgets, and ten aligned seeds, producing 180 completed runs.

## Model Families

The evaluated model configurations are:

- `configs/model/dense_121m.yaml`
- `configs/model/mla_121m.yaml`
- `configs/model/mtp_121m.yaml`
- `configs/model/moe_220m.yaml`
- `configs/model/mla_moe_220m.yaml`
- `configs/model/v3_routing_220m.yaml`

These implementations are local DeepSeek-inspired analogues rather than exact
reproductions of the production systems.

## MLA Geometry

The MLA configurations use separate non-rotary query/key, rotary query/key,
value-head, and compressed key/value dimensions:

- `d_model = 768`
- `n_heads = 12`
- `mla_qk_nope_head_dim = 128`
- `mla_q_rope_dim = 64`
- `mla_v_head_dim = 128`
- `mla_kv_latent_dim = 192`

## Training and Evaluation Policy

The experiment uses:

- Nominal training budgets of 10M, 25M, and 50M tokens
- Ten aligned seeds
- Batch size 4
- Context length 256
- FP32 precision
- One fixed FineWeb-Edu-derived corpus and tokenizer
- Identical deterministic held-out windows across models, seeds, and budgets
- Counterbalanced model order within each budget and seed

The variants are compared through predictive quality, throughput, peak GPU
allocation, total and activated parameters, routing diagnostics, and MTP
diagnostics.

## Canonical Evidence

The authoritative experiment artifacts are:

- `results/analysis/balanced_10seed_matrix_manifest.json`
- `results/runs/<experiment>/metrics/summary.json`
- `results/analysis/balanced_10seed_matrix_summary_flat.csv`
- `results/analysis/balanced_10seed_matrix_evidence_index.json`
- `scripts/analysis/run_balanced_10seed_pipeline.py`

The manifest defines all 180 matrix cells. Each run summary records its code,
configuration, environment, data, tokenizer, evaluation-window, and metric
provenance. The evidence index hashes the canonical summaries, analysis outputs,
and generated figures.
