# Stage117-A vNext External Collapse Synthesis

Decision: `STAGE117A_VNEXT_EXTERNAL_COLLAPSE_SYNTHESIS_LOCKED_RECONSTRUCTED`

Status: reconstructed from Stage112-116 pasted Kaggle outputs after Kaggle runtime loss.

## Locked interpretation

ContraMamba-vNext minimal learns a strong clean-controlled epistemic geometry, but that geometry is tied to a lexical-overlap/template-heavy clean distribution. Under open-domain fact-verification format shift, SUPPORT/REFUTE examples often fail to enter the model's judgment-enabled frame/predicate/polarity manifold and collapse to NOT_ENTITLED.

## Stage chain

- Stage111: clean-controlled dev hard pass.
- Stage112: external VitaminC diagnostic hard fail by NOT_ENTITLED collapse.
- Stage113: product-gate mechanism confirmed for `learned_x_product`.
- Stage114: router ablations failed to recover external performance.
- Stage115: clean/external scalar geometry shift confirmed.
- Stage116: surface/format shift root cause confirmed.

## Stage114 Wave 1 router ablation summary

| mode | accuracy | macro_f1 | SUPPORT recall | REFUTE recall | NOT_ENTITLED pred count | false_NE_total |
|---|---:|---:|---:|---:|---:|---:|
| learned_x_sufficiency | 0.148 | 0.091720 | 0.006 | 0.005634 | 989 | 846 |
| sufficiency_only | 0.147 | 0.095885 | 0.004 | 0.016901 | 977 | 838 |
| learned_only | 0.148 | 0.090261 | 0.006 | 0.002817 | 992 | 848 |

Conclusion: removing product/frame/predicate gates did not recover external performance. Collapse is not solved by router ablation alone.

## Stage115 clean-dev vs external metrics

| domain | n | accuracy | macro_f1 | false_NE_total | false_entitlement_total | pred_counts |
|---|---:|---:|---:|---:|---:|---|
| clean_dev | 720 | 0.984722 | 0.977373 | 0 | 11 | NOT_ENTITLED=529, REFUTE=90, SUPPORT=101 |
| external_vitaminc | 1000 | 0.149000 | 0.093106 | 845 | 2 | NOT_ENTITLED=988, REFUTE=5, SUPPORT=7 |

## Stage115 clean-dev scalar geometry

| gold_label | frame median | predicate median | sufficiency median | entitlement median | polarity median |
|---|---:|---:|---:|---:|---:|
| SUPPORT | 0.964179 | 0.996356 | 0.997218 | 0.861664 | 5.660862 |
| REFUTE | 0.997581 | 0.996033 | 0.994712 | 0.887199 | -5.689252 |
| NOT_ENTITLED | 0.006241 | 0.036081 | 0.996981 | 0.000377 | 4.444390 |

Clean-dev shows strong channel separation for SUPPORT/REFUTE, with high frame, predicate, sufficiency, entitlement, and signed polarity.

## Stage115 external VitaminC scalar geometry

| gold_label | frame median | predicate median | sufficiency median | entitlement median | polarity median |
|---|---:|---:|---:|---:|---:|
| SUPPORT | 0.014200 | 0.006733 | 0.695391 | 0.000004 | 0.250387 |
| REFUTE | 0.012924 | 0.006965 | 0.596711 | 0.000004 | 0.239785 |
| NOT_ENTITLED | 0.017272 | 0.007964 | 0.799662 | 0.000007 | 0.206705 |

External SUPPORT/REFUTE fall into the same low-entitlement geometry as external NOT_ENTITLED. Polarity also collapses toward near-zero instead of clean signed margins.

## Stage116 compact surface-shift table

| gold_label | clean token_jaccard median | external token_jaccard median | clean entity coverage median | external entity coverage median | clean evidence token len median | external evidence token len median |
|---|---:|---:|---:|---:|---:|---:|
| SUPPORT | 1.000000 | 0.250000 | 1.000000 | 0.500000 | 12 | 14 |
| REFUTE | 1.000000 | 0.176471 | 1.000000 | 0.500000 | 12 | 14 |

Clean SUPPORT/REFUTE are near-copy lexical/entity-overlap examples. External SUPPORT/REFUTE have shorter claims, longer evidence, lower token overlap, lower entity coverage, paraphrase, numeric comparison, alias/coreference, and implicit relation shifts.

## Root cause

The failure is not primarily:
- threshold tuning,
- router mode choice,
- product gate alone,
- or polarity-only error.

The root cause is:

`clean-controlled lexical/template overfit -> external open-domain format shift -> frame/predicate/polarity manifold miss -> entitlement collapse -> NOT_ENTITLED collapse`

## Research value

This is a useful failure, not a dead end. vNext demonstrates that ContraMamba can learn a clean epistemic geometry, and the scalar probes expose exactly how that geometry fails under open-domain format shift.

## Recommendation

Next stage: `Stage117-B controlled hard-clean diagnostic design`

Do:
- Build a controlled-derived hard-clean split.
- Lower lexical overlap.
- Lower entity overlap.
- Increase evidence length relative to claim.
- Add paraphrase/alias/numeric/date comparison variants.
- Keep VitaminC external-only.
- Test whether hard-clean reproduces the Stage112 external NOT_ENTITLED collapse.

Do not:
- Do not continue router ablation as the main path.
- Do not tune thresholds on VitaminC.
- Do not train on VitaminC.
- Do not add bridge/synthetic external append yet.
- Do not claim external generalization yet.

## Final locked statement

ContraMamba-vNext minimal learns clean-controlled epistemic separation, but the current clean data distribution is too lexical-overlap/template-heavy. Under open-domain fact-verification format shift, SUPPORT/REFUTE examples fail to activate the model's judgment-enabled frame/predicate/polarity geometry and collapse into NOT_ENTITLED.
