# 25-Seed 50M Statistical Closure

## Scope

This closure applies only to the confirmatory 25-seed 50M-token subset:

- dense_121m
- mtp_121m
- moe_220m
- v3_routing_220m

MLA and MLA+MoE remain part of the broader six-model exploratory matrix, but they are not part of this 25-seed confirmatory statistical block.

## Statistical Design

The design is paired/blocking by seed. Each seed block contains the same four architecture variants. The primary endpoints are validation loss and test loss. Secondary endpoints are validation perplexity and test perplexity.

Global tests:
- repeated-measures ANOVA with seed as subject/block
- Friedman nonparametric robustness test

Planned paired contrasts:
- paired t-tests
- Holm correction within each metric
- Wilcoxon signed-rank robustness checks
- sign-flip permutation p-values
- bootstrap 95% confidence intervals

## Global Results

| Metric | ANOVA p | Friedman p | Interpretation |
|---|---:|---:|---|
| validation_loss | 0.0022 | 0.0043 | reliable global model effect |
| test_loss | 0.0215 | 0.0876 | mixed/moderate global evidence |
| validation_perplexity | 0.0027 | 0.0043 | reliable global model effect |
| test_perplexity | 0.0179 | 0.0876 | mixed/moderate global evidence |

## Key Planned Contrasts

| Contrast | Metric | Mean diff | Relative diff vs baseline | 95% CI | Holm p | Sign count |
|---|---|---:|---:|---|---:|---:|
| MTP - Dense | validation_loss | -0.02949 | -0.686% | [-0.04552, -0.01345] | 0.0053 | 20/25 |
| MoE - Dense | validation_loss | -0.02874 | -0.668% | [-0.04775, -0.00974] | 0.0232 | 18/25 |
| V3 - Dense | validation_loss | -0.02127 | -0.495% | [-0.03931, -0.00322] | 0.0913 | 19/25 |
| MoE - Dense | test_loss | -0.03005 | -0.670% | [-0.05041, -0.00970] | 0.0333 | 18/25 |

## Supported Claims

1. Validation loss differs reliably across the four replicated models.
2. MTP improves validation loss over dense after Holm correction.
3. MoE improves validation loss over dense after Holm correction.
4. MoE improves test loss over dense after Holm correction.
5. Dense remains the fastest and lowest-memory model among the four replicated models.

## Suggestive Claims

1. V3-style routing is directionally better than dense on validation loss, but it does not survive Holm correction.
2. Test-loss/perplexity global evidence is mixed: repeated-measures ANOVA is significant, but Friedman is not.

## Unsupported Claims / Guardrails

- Do not claim MTP clearly beats MoE.
- Do not claim V3-style routing clearly beats standard MoE.
- Do not claim all sparse/routing variants dominate dense on test loss.
- Do not describe the quality gaps as large.
- Do not include MLA or MLA+MoE in the 25-seed confirmatory statistical test unless those variants are later replicated.

## Recommended Report Framing

The 25-seed paired 50M-token analysis shows small but replicated quality improvements for MTP and MoE over the dense baseline, while the dense model remains the lowest-cost model in throughput and memory, with especially large efficiency advantages over the MoE-based variants. The replicated results do not identify a single uniformly dominant architecture. Instead, they show a tradeoff: MTP has the strongest validation-loss profile, MoE has the strongest test-loss evidence versus dense, dense remains the most efficient baseline, and V3-style routing is directionally useful but not clearly superior to standard MoE.
