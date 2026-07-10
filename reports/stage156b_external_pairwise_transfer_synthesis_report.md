# Stage156-B External Pairwise Transfer Synthesis

## 1. Summary decision

Decision: `STAGE156B_EXTERNAL_BRIDGE_IMPROVES_COLLAPSE_BUT_OVER_RELEASES_ENTITLEMENT`

Stage156-B synthesizes the Stage156-A pairwise transfer inventory into a core diagnosis: Stage63 substantially improves Stage53's NOT_ENTITLED collapse, but it also over-releases into SUPPORT/REFUTE. The remaining bottleneck is coarse external entitlement/routing, not a need for more route/location rules.

## 2. Why Stage156-A was needed

Stage155-A identified external/generalization transfer as the dominant current bottleneck. On external or FactVer rows, the model heavily underpredicted SUPPORT and REFUTE into NOT_ENTITLED, with top confusions of SUPPORT->NOT_ENTITLED and REFUTE->NOT_ENTITLED.

Stage156-A was needed to compare the frozen Stage53 external behavior against the Stage63 bridge-enabled behavior on paired external examples. This made it possible to separate genuine recovery from new regressions, rather than treating the external accuracy gain as a single undifferentiated improvement.

## 3. Stage53 frozen external behavior

Stage53 frozen external metrics show severe NOT_ENTITLED collapse:

- Accuracy: `0.156`
- Macro-F1: `0.11671748688294066`
- Prediction counts: REFUTE `33`, NOT_ENTITLED `944`, SUPPORT `23`
- False SUPPORT: `6`
- False NOT_ENTITLED: `814`
- False REFUTE: `24`
- SUPPORT recall: `0.034`
- NOT_ENTITLED recall: `0.896551724137931`
- REFUTE recall: `0.02535211267605634`

The Stage53 problem is not excessive external assertion. It is an overly conservative routing pattern that keeps external SUPPORT and REFUTE examples trapped in NOT_ENTITLED.

## 4. Stage63 bridge external behavior

Stage63 bridge-enabled external metrics show partial collapse recovery with over-release:

- Accuracy: `0.322`
- Macro-F1: `0.3153191098829531`
- Prediction counts: REFUTE `217`, NOT_ENTITLED `492`, SUPPORT `291`
- False SUPPORT: `135`
- False NOT_ENTITLED: `411`
- False REFUTE: `132`
- SUPPORT recall: `0.312`
- NOT_ENTITLED recall: `0.5586206896551724`
- REFUTE recall: `0.23943661971830985`

Stage63 opens SUPPORT and REFUTE paths that Stage53 barely used. This is directionally beneficial, but it reduces NOT_ENTITLED safety and creates many new false SUPPORT and false REFUTE cases.

## 5. Stage63 vs Stage53 deltas

Stage63 minus Stage53:

- Accuracy: `+0.166`
- Macro-F1: `+0.19860162300001244`
- False SUPPORT: `+129`
- False NOT_ENTITLED: `-403`
- False REFUTE: `+108`
- SUPPORT recall: `+0.278`
- NOT_ENTITLED recall: `-0.33793103448275863`
- REFUTE recall: `+0.2140845070422535`

The bridge therefore improves the external collapse problem, but it does so by releasing too many examples from NOT_ENTITLED into SUPPORT/REFUTE. The benefit and the regression are coupled.

## 6. Pairwise transition interpretation

Stage156-A pairwise transitions:

- Both wrong: `626`
- Recovered by Stage63: `218`
- Both correct: `104`
- Regressed by Stage63: `52`

Prediction transitions:

- NOT_ENTITLED->NOT_ENTITLED: `489`
- NOT_ENTITLED->SUPPORT: `264`
- NOT_ENTITLED->REFUTE: `191`
- REFUTE->REFUTE: `24`
- SUPPORT->SUPPORT: `18`
- REFUTE->SUPPORT: `9`
- SUPPORT->NOT_ENTITLED: `3`
- SUPPORT->REFUTE: `2`

The strongest movement is out of Stage53's NOT_ENTITLED prediction. Some of that movement is justified and recovers previously wrong examples. Some of it is unsafe and creates regressions. This confirms that the problem is not simply whether to be more conservative or more permissive; the model needs finer entitlement/routing calibration.

## 7. Core diagnosis

External transfer status: `improved_but_still_weak`

Main failure mode: `coarse_entitlement_release`

Stage53 is overly conservative and collapses external examples into NOT_ENTITLED. Stage63 partially fixes that collapse, but over-releases into SUPPORT/REFUTE and weakens NOT_ENTITLED safety. The dominant remaining issue is that external examples need finer internal entitlement/routing calibration.

## 8. Why this is not a shadow-rule problem

This is a model-internal entitlement/routing problem. The Stage63 bridge improves transfer by changing the model's ability to route external examples into SUPPORT and REFUTE, but the over-release shows that the release decision is too coarse.

More shadow diagnostics, route/location rules, or post-hoc patches would not answer the core question: when should an external example be released from NOT_ENTITLED into SUPPORT/REFUTE, and when should it remain safely NOT_ENTITLED?

## 9. Stage157 recommendation

Recommended next stage: `Stage157-A`

Goal: design an external entitlement/routing diagnostic or architecture response that separates justified release from NOT_ENTITLED from unsafe over-release.

Recommended direction: `external_bridge_rerouting_or_entitlement_calibration`

Candidate designs:

- Post-freeze external scalar export for Stage63/current-best predictions
- Compare recovered_by_stage63 vs regressed_by_stage63 examples
- Learn diagnostic criteria for safe NE release without using external labels for training
- Inspect whether frame/predicate/sufficiency/learned entitlement disagree on recovered vs regressed rows
- Design an internal calibration head or router constraint on clean-controlled data only

Stage157 should focus on safe release from NOT_ENTITLED, not more route/location rules.

## 10. Safety constraints

This report is analysis-only. No training code, model code, export behavior, analyzer scripts, logits, final predictions, checkpoint selection, or source predictions are modified.

No external labels may be used for training, threshold tuning, or checkpoint selection.

Required safety policy:

- Analysis only: `true`
- Shadow diagnostics integrated: `false`
- Source predictions mutated: `false`
- Final logits modified: `false`
- Final predictions modified: `false`
- Training modified: `false`
- Checkpoint selection modified: `false`
- External data used for training: `false`
- Threshold used for model selection: `false`
