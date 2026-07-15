# Stage186-A compatible-positive margin fixed-spec report

## Decision

`STAGE186A_FIXED_NO_SWEEP_COMPATIBLE_POSITIVE_MARGIN_SPEC_READY`

Authorized next: `STAGE187_COMPATIBLE_POSITIVE_MARGIN_DEFAULT_OFF_IMPLEMENTATION`. Loss implementation, checkpoint execution, and training remain unauthorized.

## Stage185 closure and sidecar identity

The authoritative 3,600-row sidecar exact-joins the source by unique row ID. Dataset SHA-256: `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`. Semantic sidecar SHA-256: `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`. Stage182 overlap regression and 22/22 deterministic contamination recovery passed with zero blocked invariants.

## Native frame BCE detection

The trainer does not directly recompute native frame BCE as `F.binary_cross_entropy_with_logits(output["frame_logit"], ...)`. Detection mode is `delegated_model_forward_stage183_crosscheck`: the model-native auxiliary loss is consumed through the trainer's delegated loss path and wired into `losses["total"]`. Stage183-A static facts establish row-mean BCEWithLogits, no `pos_weight`, native weight 1.0, the pre-sigmoid frame logit, and a trainable frame-classifier bias. Generic BCE substrings—including Stage22 frame-violation, entitlement, temporal, and location-boundary losses—are not used as native frame-BCE evidence.

## Eligible cohort

Exactly 605 of 1440 train-compatible rows are eligible (0.4201388888888889). They cover 121 pairs and 5 families; each eligible pair contributes five rows and each family contributes 121. Families: evidence_deletion, evidence_truncation, none, paraphrase, predicate_swap.

Only 121 of 240 train pairs contribute, so this is pair-concentrated integrity-filtered coverage, not population-wide cleanliness. Exact five-family balance does not remove that limitation.

## Directional evidence and fixed specification

Stage182-B reports 13 compatible false negatives versus 1 incompatible false positives. Its candidate-minus-control direction and bootstrap interval are negative. This justifies intervention direction only; no Stage182-B statistic is fitted into a hyperparameter.

The fixed target is native pre-sigmoid `output["frame_logit"]` with margin `0.0`:

```text
L_margin = mean(relu(-frame_logit[eligible_mask]))
L_total = L_existing + 0.05 * L_margin
```

Zero is the sigmoid boundary, not a confidence-inflation target. The weight is fixed at 0.05 while native frame BCE remains weight 1.0. Near zero on the active side, the hinge gradient is 10% of positive BCE's magnitude. No sweep, calibration, schedule, family weight, or automatic rescaling is permitted.

## Normalization and zero-eligible batches

Normalization is eligible-row mean only. Current 5-per-pair topology makes row mean equal to equal-pair mean, but later topology changes require a new gate. An empty eligible mask returns graph-compatible scalar zero `frame_logit.sum() * 0.0`; division by zero, NaN, batch skipping, and optimizer-step changes are forbidden.

## Consumption and objective integration

Activation requires the authoritative eligibility boolean plus train/compatible/integrity/time/source checks, exact dataset and semantic SHA values, and an exact one-to-one row-ID join. Family names, reason-code text, Stage182 membership, predictions, probabilities, final labels, heuristics, and row order may not infer eligibility.

Static trainer inspection identified native `frame_logit`, row-mean frame BCE, final CE from `output["logits"]`, existing objective assembly, and clean-dev `final_macro_f1` selection. Stage187 may append the default-off term only; final CE, existing auxiliary weights, optimizer/scheduler, architecture, and checkpoint selection remain unchanged.

Stage175 targets a final-classifier SUPPORT anchor against a detached reference. Stage177 targets within-pair ordering. Stage186 targets an absolute native frame-head boundary on independently eligible rows, so it is nonredundant with both.

## Default-off contract and safety

Implementation defaults: `--compatible-positive-margin-weight 0.0` and `--compatible-positive-margin-logit 0.0`. The fixed intervention setting is weight 0.05 and logit 0.0. Sidecar path and expected semantic SHA are separate activation inputs. No trainer/model/loss modification, Torch import, checkpoint load, forward, training, smoke run, evaluation, fitting, sweep, or annotation occurred in this audit.
