# Stage32-D2 Coverage Owner v2 Threshold Sweep Report

## Purpose
Stage32-D2 adds offline Coverage Owner v2 threshold sweeping to `scripts/evaluate_stage32_shadow_composer.py`. It evaluates confidence and margin thresholds from existing prediction exports without rerunning training or changing model outputs.

## Diagnostics Fix
The evaluator now emits global Coverage Owner v2 diagnostics whenever v2 fields exist:

- `coverage_v2_pred_label_counts`
- `coverage_v2_route_counts`
- `coverage_v2_reason_counts`
- `coverage_v2_abstain_count`
- `coverage_v2_abstain_rate`
- `coverage_v2_margin_summary`
- `coverage_v2_top_prob_summary`

The margin and top-probability summaries include mean, min, max, p10, p25, p50, p75, and p90.

## Offline Sweep Mode
New CLI options:

| Flag | Default |
|---|---|
| `--coverage-v2-offline-sweep` | disabled |
| `--coverage-v2-sweep-confidences` | `0.30,0.35,0.40,0.45,0.50` |
| `--coverage-v2-sweep-margins` | `0.00,0.02,0.05,0.08,0.10` |

For each confidence/margin pair, the evaluator recomputes v2 routing from exported coverage probabilities:

- `stage32_coverage_entails_support_prob`
- `stage32_coverage_overclaim_ne_prob`
- `stage32_coverage_contradicts_refute_prob`

It then recomputes the shadow label using Stage32-D rules and reports current-vs-offline-shadow metrics.

## Sweep Output
JSON includes:

- `coverage_v2_offline_sweep`
- `coverage_v2_offline_sweep_best`

Each sweep row includes thresholds, shadow metrics, route/pred/reason counts, unresolved rate, Stage31 support/overclaim/refute diagnostics, safety counters, and a D2 decision label.

Markdown includes a compact sweep table sorted by safety first, then support-entailment SUPPORT recovery, then macro-F1.

## Decision Labels
| Label | Meaning |
|---|---|
| `STAGE32_D2_NO_SAFE_SUPPORT_RECOVERY` | No threshold safely recovers SUPPORT. |
| `STAGE32_D2_DIAGNOSTIC_THRESHOLD_FOUND_NOT_APPLY` | A safe threshold recovers SUPPORT but macro-F1 remains weak. |
| `STAGE32_D2_PROMISING_BUT_STILL_SHADOW_ONLY` | A safe threshold recovers SUPPORT and improves macro-F1 materially, but still must remain shadow-only. |

## Guardrail
This evaluator is diagnostic-only. It must not be used for training, calibration, threshold selection, checkpoint selection, or final prediction replacement.

## Remaining Risk
Offline threshold sweeps can expose useful routing behavior, but they do not validate a deployable composer. Any promising setting still requires separate shadow validation and leakage-safe evaluation.
