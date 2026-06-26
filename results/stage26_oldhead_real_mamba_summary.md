# Stage26 Old-Head Real-Mamba Validation

Date: 2026-06-26

## Final decision

- **Final v6B old-head config:** `frame+preserve_w03`
- **Boundary:** valid secondary signal
- **Pair contrastive support-safe:** component signal **KEEP**, final config **DROP / NOT ROBUST**
- **Next stage:** Stage27-H2A, v7-H1 explicit entitlement decision signal

## frame+preserve_w03, 3 seeds

- Macro-F1: **0.9785 ± 0.0159**
- Accuracy: **0.9856 ± 0.0112**
- SUPPORT precision: **0.9293 ± 0.0850**
- SUPPORT recall: **0.9667 ± 0.0484**
- NOT_ENTITLED recall: **0.9864 ± 0.0172**
- REFUTE recall: **1.0000 ± 0.0000**
- Location false SUPPORT total: **8**
- Role false SUPPORT total: **6**
- Bad SUPPORT total: **22**
- Missing-evidence false SUPPORT total: **0**

Decision: **final v6B old-head winner**.

## frame+preserve+pair_w01_supportsafe_sample300, 3 seeds

- Macro-F1: **0.9778 ± 0.0142**
- Accuracy: **0.9852 ± 0.0095**
- SUPPORT precision: **0.9116 ± 0.0490**
- SUPPORT recall: **0.9778 ± 0.0294**
- NOT_ENTITLED recall: **0.9840 ± 0.0091**
- REFUTE recall: **1.0000 ± 0.0000**
- Location false SUPPORT total: **16**
- Role false SUPPORT total: **5**
- Bad SUPPORT total: **26**
- Missing-evidence false SUPPORT total: **0**

Decision: **component signal KEEP; final config DROP / NOT ROBUST**.

## Head-to-head delta: pair minus baseline

- Macro-F1: **-0.0007**
- Accuracy: **-0.0005**
- SUPPORT precision: **-0.0177**
- SUPPORT recall: **+0.0111**
- NOT_ENTITLED recall: **-0.0025**
- Location false SUPPORT: **+8**
- Bad SUPPORT: **+4**
- Missing-evidence false SUPPORT: **0**

## Interpretation

Old Stage22 diagnostic heads are real-Mamba-valid in `v6b_minimal`.

The most robust old-head configuration is `frame+preserve_w03`. Pair-contrastive frame supervision learns a nonzero ranking signal and improves some seeds, but direct mixing into final v6B training is not robust because it increases location-swap false SUPPORT. Keep pair contrastive as a component-level H2/H3 hypothesis, not as the current final v6B configuration.

## Next step

Stage27-H2A: add explicit compositional entitlement decision signal selection to v7-H1.
