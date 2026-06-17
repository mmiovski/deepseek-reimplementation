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
