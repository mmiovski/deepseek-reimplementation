# Experiment Scope Correction

Tiny/small models are only for smoke testing, code-path validation, schema validation, GPU sanity checks, and instrumentation/logging validation.

The primary experimental evidence must come from larger feasible local models. The main large local matrix is centered on:

- `configs/model/dense_121m.yaml`
- `configs/model/mla_121m.yaml`
- `configs/model/mtp_121m.yaml`
- `configs/model/moe_220m.yaml`
- `configs/model/mla_moe_220m.yaml`
- `configs/model/v3_routing_220m.yaml`

The completed 6M-12M 10M/25M/50M matrix is calibration only and should not be presented as the main experimental result.

The actual experimental interpretation must emphasize efficiency tradeoffs: validation/test loss, perplexity, tokens/sec, peak GPU memory, total parameters, trainable parameters, activated parameters per token, tokens per total/trainable/activated parameter, routing/expert diagnostics, and MTP next-token loss separately from auxiliary MTP loss.

The large config filenames identify the experimental variants. The internal `model.name` values remain factory-compatible implementation names.

## Large MLA Scaling Rule

The large MLA configs use a DeepSeek-style decoupled query/key/value head geometry.

The implementation should not force `mla_q_rope_dim < d_model / n_heads`. DeepSeek-style MLA separates:

- `mla_qk_nope_head_dim`: non-rotary query/key head dimension.
- `mla_q_rope_dim`: rotary query/key head dimension.
- `mla_v_head_dim`: value head dimension.
- `mla_kv_latent_dim`: compressed key/value latent rank.

For the local large configs, the selected MLA geometry is:

- `d_model = 768`
- `n_heads = 12`
- ordinary dense `head_dim = 64`
- `mla_qk_nope_head_dim = 128`
- `mla_q_rope_dim = 64`
- `mla_v_head_dim = 128`
- `mla_kv_latent_dim = 192`

This is a DeepSeek-inspired local analogue, not an exact DeepSeek-V2/V3 reproduction.

## Main Large Fixed-Budget Matrix

The main experimental matrix must train every large primary architecture at every fixed token budget.

Required runs:

- `main_large_10m_00_dense_121m`
- `main_large_10m_01_mla_121m`
- `main_large_10m_02_mtp_121m`
- `main_large_10m_03_moe_220m`
- `main_large_10m_04_mla_moe_220m`
- `main_large_10m_05_v3_routing_220m`
- `main_large_25m_00_dense_121m`
- `main_large_25m_01_mla_121m`
- `main_large_25m_02_mtp_121m`
- `main_large_25m_03_moe_220m`
- `main_large_25m_04_mla_moe_220m`
- `main_large_25m_05_v3_routing_220m`
- `main_large_50m_00_dense_121m`
- `main_large_50m_01_mla_121m`
- `main_large_50m_02_mtp_121m`
- `main_large_50m_03_moe_220m`
- `main_large_50m_04_mla_moe_220m`
- `main_large_50m_05_v3_routing_220m`

The final analysis must report exact total parameters, trainable parameters, activated parameters per token, and tokens per total/trainable/activated parameter for each model. The variants must not be described as identical-size models.

## Matrix Manifest

The machine-readable main matrix manifest is:

- `configs/experiment/main_large_matrix_manifest.json`

It records every main large fixed-budget run and the exact model-specific total, trainable, and activated-parameter accounting used for tokens-per-parameter calculations.

## Feasibility Probe Reporting Policy

Large feasibility probes are tracked separately from main experimental results.

Tracked feasibility artifacts:

- `configs/train/main_large_feasibility_probe.yaml`
- `configs/experiment/main_large_feasibility_*.yaml`
- `results/metrics/main_large_feasibility_*/summary.json`
- `results/raw_logs/main_large_feasibility_*/train_log.jsonl` when small enough to audit
- `results/analysis/main_large_feasibility_summary.json`
- `results/analysis/main_large_feasibility_summary.csv`
- `scripts/analysis/export_main_large_feasibility_summary.py`

The reporting exporter must use the actual pretraining summary schema. In particular, it must use `steps`, `train_tokens_per_second`, and `peak_memory_bytes`, not stale aliases such as `train_steps`, `tokens_per_second`, or `peak_memory_mb`.

Feasibility probes validate code paths, CUDA memory, exact parameter accounting, MTP diagnostics, and routing diagnostics. They must not be treated as main 10M/25M/50M experimental results.

Checkpoints and bulky transient runtime artifacts should not be committed unless specifically selected as part of a separate archival policy.

## Main Large Batch-Size Policy

The main large fixed-token matrix uses one standardized batch size across all six architecture variants.

Selected policy:

- `batch_size: 4`
- `block_size: 256`
- `precision: fp32`
- `device: cuda`

Selection artifact:

- `results/analysis/main_large_batch_sweep_summary.json`
- `results/analysis/main_large_batch_sweep_summary.csv`

Rationale:

Batch size 4 completed all six large variants in the batch-size sweep. The maximum observed peak memory in that sweep was below approximately 5 GB, and the slowest observed throughput was higher than the slowest batch-size-2 run. Using one common batch size avoids architecture-specific optimizer-step counts and keeps the comparison cleaner under fixed token budgets.
