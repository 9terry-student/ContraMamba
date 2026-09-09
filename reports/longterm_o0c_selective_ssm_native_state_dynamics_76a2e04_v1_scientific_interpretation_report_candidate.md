# O0c Validated Scientific Interpretation Report

## Scope and validated identity

This is a report-only scientific interpretation of the frozen validated O0c run.  No code, tests, data, checkpoints, imported artifacts, or execution identities were modified; no training or evaluation was run.

- Scientific/design authority: `242ad9ed70fc995ebda560911a7d0dfd2f18f9b3`
- Validated execution / run name: `longterm-o0c-selective-ssm-native-state-dynamics-76a2e04-v1`
- Execution commit: `76a2e04b3f78b76bb81d6828260fa8a0b13c2258`
- Manifest schema: `longterm_o0c_selective_ssm_native_state_dynamics_v1`
- Manifest experiment: `longterm_o0c_selective_ssm_native_state_dynamics`
- Implementation authority commit: `6eca52722aaffa214e8546c6b616e1f670aecf77`
- Observer implementation commit: `60a53b6f5d1db8d7cbecded5b94d5231adcfc520`
- Model/tokenizer revision: `state-spaces/mamba-130m-hf` at `5708daa364c50b880e7bd92eab456e0d34492ee9`
- Dataset: `data/longterm_o0b_matched_controls_v1.jsonl` (`75a675bee49cb26eb0935d364f0f5d090922dd01576dfc23294961b28394aec2`)

Code correctness and noninterference are PASS (`PASS_EXACT_EQUIVALENCE_NONINTERFERENCE`). Execution is PASS (`PASS_EXECUTION_COMPLETE`), complete native-state capture is PASS (`PASS_COMPLETE_NATIVE_STATE_CAPTURE`), and source/provenance validation is PASS (`PASS_RECONCILED_UNIQUE_TRANSFORMERS_SOURCE`; `PASS_PROVENANCE_VALIDATED`). Collection/import provenance is PASS: the supplied import result is `IMPORT PASS, VALIDATED=7, COPIED=7`, and all seven manifest-listed imported artifacts reproduce their `SHA256SUMS.txt` hashes.

## Independent measurement audit

The quoted results below were recomputed directly from `paired_measurements.jsonl`, grouping each pair, anchor, and layer and comparing comparison-A against comparison-B and comparison-C. There are 3 frozen pairs, 24 layers, and 5 post-divergence anchors, hence 360 post-divergence pair-layer-anchor comparisons; each individual anchor has 72 and each pair has 120.

`anchor_pre_minus_1` is a negative-control/integrity anchor, not a positive-signal opportunity. Its pre-divergence integrity field is PASS for all 216 rows, and normalized-L2 distance is zero in all 216 rows. It is therefore excluded from the post-divergence signal totals.

### Primary metric: `normalized_l2_state_distance`

Post-divergence comparison-A separation exceeds comparison-B in 202/360 = 56.1% and comparison-C in 199/360 = 55.3%; it exceeds both in 146/360 = 40.6%.

| Anchor | Comparison-A exceeds both B and C |
| --- | ---: |
| divergence | 33/72 = 45.8% |
| post_plus_1 | 17/72 = 23.6% |
| post_plus_2 | 9/72 = 12.5% |
| post_plus_4 | 17/72 = 23.6% |
| terminal | 70/72 = 97.2% |

The pair-specific post-divergence BOTH values are: `o0b_pair_001` 33/120 = 27.5%, `o0b_pair_002` 40/120 = 33.3%, and `o0b_pair_003` 73/120 = 60.8%. These heterogeneous values do not meet the frozen broad-consistency requirement.

### Tier-2 metrics

For `paired_transition_delta`, post-divergence BOTH is 172/360 = 47.8%. This is mixed and not consistently A-specific.

For `transition_direction_cosine`, lower cosine is the directional-separation orientation used for the frozen comparison. Post-divergence A-versus-B is 247/360 = 68.6%, A-versus-C is 285/360 = 79.2%, and BOTH is 222/360 = 61.7%. This is a secondary directional tendency. It does not override failure of the broad primary recurrent-state consistency criterion.

## Frozen falsification interpretation

A strong terminal-localized observation is present within the frozen schedule: normalized-L2 BOTH is 70/72 and transition-direction BOTH is 71/72. Terminal is not promoted as a selected best anchor. This is an observation within the frozen schedule, not a post-hoc decision rule.

Under the frozen falsification matrix, the evidence does **not** satisfy: “comparison-A consistently > comparison-B / comparison-C across multiple pairs, layers, and anchors.” The primary normalized-L2 result is broadly inconsistent across anchors and frozen pairs despite the terminal-localized separation; the Tier-2 transition result is mixed, and the directional result remains secondary.

Frozen O0c evidence does not support a broad, consistently expressed native sufficiency-sensitive recurrent-state precursor clue. A strong terminal-anchor-localized recurrent-state separation and a weaker transition-direction tendency are observed, but the effect is not consistent across the frozen pairs, layers, and post-divergence anchor schedule.

This report preserves O0b: it neither rewrites nor invalidates the frozen O0b hidden-state-proxy conclusion.

## Limitations and authorization boundary

- `n=3` frozen pairs.
- This is descriptive observational evidence only; it provides no statistical-significance result or population estimate.
- It makes no causal claim.
- It makes no predictive or calibrated-detector claim.
- It performs no best-layer or best-anchor selection.
- It provides no O1 promotion authorization.

SCIENTIFIC_CONCLUSION:
BROAD_NATIVE_PRECURSOR_NOT_SUPPORTED;
TERMINAL_LOCALIZED_RECURRENT_STATE_SEPARATION_OBSERVED
