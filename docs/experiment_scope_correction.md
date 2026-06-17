# Experiment Scope Correction

Tiny/small models are only for smoke testing, code-path validation, schema validation, GPU sanity checks, and instrumentation/logging validation.

The primary experimental evidence must come from larger feasible local models. The main large local matrix is centered on:

- `dense_121m`
- `mla_121m`
- `mtp_121m`
- `moe_220m`
- `mla_moe_220m`
- `v3_routing_220m`

The completed 6M?12M 10M/25M/50M matrix is calibration only and should not be presented as the main experimental result.

The actual experimental interpretation must emphasize efficiency tradeoffs:
validation/test loss, perplexity, tokens/sec, peak GPU memory, total parameters, trainable parameters, activated parameters per token, tokens per total/trainable/activated parameter, routing/expert diagnostics, and MTP next-token loss separately from auxiliary MTP loss.
