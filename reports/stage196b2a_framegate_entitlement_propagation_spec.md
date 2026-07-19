# Stage196-B2-A FrameGate-to-entitlement propagation mediation audit

## Scope and scientific boundary

Stage196-B2-A is an artifact-only audit of observed paired changes in FrameGate, downstream local channels, entitlement aggregation, and the final SUPPORT-versus-NOT_ENTITLED decision under frozen Mamba. It does not establish formal causal mediation from observational correlation. It does not authorize promotion of `frame_local_only`, training, replay, checkpoint loading, model loading, checkpoint mutation, a new loss, trainer modification, threshold search, calibration, external/OOD evaluation, or full retraining.

The analyzer is `scripts/analyze_stage196b2a_framegate_entitlement_propagation.py`. Every CLI argument is required:

```text
python scripts/analyze_stage196b2a_framegate_entitlement_propagation.py \
  --repo-root <repository> \
  --run-root <repository>/reports/stage196b2p0_epoch_channel_observability_runs \
  --stage196a-report-json <stage196a_persistent_support_boundary_report.json> \
  --stage196b1c-analysis-json <stage196b1c_analysis.json> \
  --current-git-commit <repository-HEAD-containing-this-analyzer> \
  --stage196b1-runtime-git-commit <historical-six-run-runtime-commit> \
  --stage196b2p0-runtime-git-commit <observability-rich-rerun-runtime-commit> \
  --output-dir <new-empty-directory-below-reports>
```

No path is silently hard-coded. The analyzer commit must be lowercase full 40-hex and equal repository HEAD. The historical Stage196-B1 runtime is independently supplied as the source role for B1-C; the Stage196-B2-P0 runtime is separately checked against all six observability-rich run artifacts. The FrameGate gradient-ownership implementation origin remains `5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8`.

## Exact input and provenance closure

The run root must contain exactly these six directories:

1. `seed183_joint`
2. `seed183_frame_local_only`
3. `seed184_joint`
4. `seed184_frame_local_only`
5. `seed185_joint`
6. `seed185_frame_local_only`

The analyzer reuses the Stage196-B1-C implementation's explicit provenance-source ownership validators rather than recursive generic field search. This closes split seed 174; Mamba `state-spaces/mamba-130m-hf`; `v6b_minimal`; CUDA; 20 epochs; frozen encoder and A_log; no shared-encoder gradient path; unchanged direct FrameGate BCE at weight 1.0; no external/OOD evaluation; no bridge training; no calibration or threshold search; no compatible-positive margin; no Stage195 SWA; no state capsules; no new loss; and `data/controlled_v5_v3_without_time_swap.jsonl` as main data.

The supplied B1-C analysis directory must contain exactly the required named companions for closure:

```text
stage196b1c_analysis.json
stage196b1c_report.md
stage196b1c_run_summary.csv
stage196b1c_paired_seed_deltas.csv
stage196b1c_tail3_persistent_rows.csv
stage196b1c_recurrent_position_effects.csv
stage196b1c_epoch_trajectory.csv
stage196b1c_contract.csv
```

The decision must be `STAGE196B1C_MIXED_GRADIENT_OWNERSHIP_EFFECT`, the recommended next stage must be `STAGE196B2_NO_PROMOTION_TARGETED_CAUSAL_FOLLOWUP`, blockers must be empty, all contract rows must pass, the run summary must be the exact six-run matrix, the paired table must expose seeds 183–185, the trajectory must contain 120 rows, the tail table must cover all six runs, recurrent sets must cover all four named sets, and selected epochs must match the artifacts.

## Alignment and fixed views

The certified join is `id`, cross-checked with trajectory `source_row_id`; `dev_position` is the certified position key. Every run and epoch requires 720 unique IDs, identical paired populations, identical gold labels and intervention metadata, no duplicate or missing paired row, no epoch drift, and one consistent position map across all six runs. Incidental file order is never a join key.

The views remain distinct:

- selected: the actual selected-checkpoint output for each run, never epoch 20 by substitution;
- tail-three: exactly epochs 18, 19, and 20, with continuous scalar means and ordered label patterns;
- epoch: all 20 epochs, paired into exactly 60 seed-by-epoch rows.

Stage196-A recurrence rows are loaded from the supplied report's companions. They are not replaced by embedded lists. Required counts are baseline recurrent 22, intervention recurrent 19, common recurrent 19, and universal all-six 10.

## Resolved semantic schema and the fail-closed prerequisite

Selected-checkpoint fields are explicit: `gold_final_label`, `pred_final_label`, `frame_prob`, `predicate_coverage_prob`, `sufficiency_prob`, `polarity_margin`, `entitlement_prob`, canonical `final_probs[2]` for SUPPORT, canonical `final_probs[1]` for NOT_ENTITLED, `intervention_type`, and `id`. Prediction/scalar duplicates are cross-checked within serialization tolerance.

Epoch rows are explicit: `gold_final_label`, `predicted_final_label`, `sigmoid(frame_logit)`, canonical `final_logits[2]` for SUPPORT, canonical `final_logits[1]` for NOT_ENTITLED, `source_row_id`, and `dev_position`.

Stage196-B2-P0 supplies 20 explicit `stage196b2p0_epoch_channels_NNN.jsonl` sidecars per run. The analyzer requires their exact stable semantic schema, 720 rows, identity and position alignment, native downstream probabilities/margin, and equality with the existing trajectory predictions and canonical final logits. Missing or contradictory sidecars still fail closed. Selected-checkpoint scalars are never substituted for any epoch.

This is an input-contract result, not scientific evidence that any channel is or is not a bottleneck.

The uniform SUPPORT-versus-NOT_ENTITLED margin source is `support_logit - not_entitled_logit` from the P0 sidecars. Both logits are mandatory in every view; there is no probability fallback or mixed source.

## Fixed classifications

Tail-three status is exactly one of `STABLE_SUPPORT`, `PERSISTENT_NOT_ENTITLED`, `PERSISTENT_REFUTE`, or `UNSTABLE`. The ordered paired transition precedence is:

1. `RESCUE_NE_TO_STABLE_SUPPORT`
2. `HARM_STABLE_SUPPORT_TO_PERSISTENT_NE`
3. `HARM_STABLE_SUPPORT_TO_PERSISTENT_REFUTE`
4. `HARM_STABLE_SUPPORT_TO_UNSTABLE`
5. `PERSISTENT_NE_BOTH`
6. `STABLE_SUPPORT_BOTH`
7. `NEW_PERSISTENT_NE_FROM_UNSTABLE`
8. `PERSISTENT_REFUTE_BOTH`
9. `OTHER_TRANSITION`

Threshold crossings are exactly `FAIL_TO_PASS`, `PASS_TO_FAIL`, `PASS_TO_PASS`, or `FAIL_TO_FAIL`. Native passes are frame, predicate, sufficiency, and entitlement probability at least 0.5 and SUPPORT-facing polarity margin at least zero. There is no threshold search or calibration.

For gold SUPPORT with exact-positive FrameGate delta, the fixed first-blocker order is frame remains subthreshold, predicate, sufficiency, polarity, entitlement aggregation, then final SUPPORT-vs-NOT_ENTITLED composition. `POLARITY_BLOCKED` remains explicit. `MULTI_CHANNEL_DEGRADATION` is used when more than one downstream channel crosses pass-to-fail and one first blocker is inadequate. `PROPAGATED_TO_SUPPORT` requires selected SUPPORT or tail-three `STABLE_SUPPORT`, according to view. `UNRESOLVED_PROPAGATION` is used only with complete required fields. Missing fields always fail the analysis instead of creating an unresolved scientific row.

The final-composition condition is parenthesized so positive frame delta and every upstream pass are prerequisites. A final non-SUPPORT label or nonpositive SUPPORT-vs-NE margin can then identify final composition blocking.

## Analysis populations and paired estimands

Gold SUPPORT rows are isolated. The audit separately precommits the four Stage196-A sets, joint persistent SUPPORT-to-NOT_ENTITLED rows by seed, intervention persistent SUPPORT-to-NOT_ENTITLED rows by seed, joint-defined stable-correct SUPPORT controls, intervention-induced harm inside that fixed control population, and rescue rows that are joint persistent NOT_ENTITLED and intervention stable SUPPORT. Controls and rescues are never inferred from aggregates or frame increase alone.

Selected and tail-three per-row deltas are `frame_local_only - joint` for frame, predicate, sufficiency, polarity, entitlement, SUPPORT-vs-NE margin, SUPPORT probability, and NOT_ENTITLED probability. Both arm values, deltas, pass states, crossings, and final class/pattern are retained.

Positive-frame-shift seeds are derived only from the completed B1-C paired row `mean_frame_probability_stage196a_common_recurrent > 0`. The recorded deltas `-0.136243`, `+0.140103`, and `+0.147282` are reproduced within serialization tolerance before deriving positive seeds 184/185 and negative contrast seed 183. Seed numbers are not the derivation rule.

Per seed and population, the complete path reports direction counts, frame crossings, downstream positive fractions among frame-up rows, selected and stable-tail propagation rates, every first-blocker count/rate, rescue, harm, persistent failure, descriptive Spearman correlations, and sign concordance. Constant vectors yield unavailable correlation, not zero. No correlation or p-value enters a causal decision.

## Harm, rescue, and epoch contracts

Every harm and rescue row would include both tail statuses, selected transition, all values/deltas/crossings, first blocker, all recurrent memberships, intervention type, and the three requested despite-harm direction flags. Summary retains seed and intervention type. Baseline controls remain baseline-defined.

Epoch propagation requires exactly 60 rows: three seeds by 20 epochs. Each row contains paired gold-SUPPORT channel means, crossings, SUPPORT rescue/harm counts, false-NOT_ENTITLED and false-entitlement deltas, common-recurrent frame/entitlement/final-margin deltas, tail markers, and both selected-epoch markers. Tail epochs and selected epochs never substitute for each other. On incomplete input, zero scientific epoch rows are emitted and the failed schema gate explains why; fabricated 60-row placeholders are prohibited.

## Precommitted decision rule

The primary evidence population is positive-frame-shift seeds × Stage196-A common recurrent × gold SUPPORT × tail-three × positive frame delta × intervention not stable SUPPORT. Its size is retained per seed.

A single bottleneck requires at least two positive seeds; at least five eligible rows per seed; the same largest first blocker in every positive seed; at least 50% of eligible rows in every positive seed; and no different largest harm blocker in two or more seeds. `FRAME_REMAINS_SUBTHRESHOLD` maps to the frame-primary decision only under this same rule. Otherwise valid but conflicting evidence maps to seed-specific mixed propagation, including different largest blockers, lack of a common 50% blocker, recurrent/harm conflict, material selected/tail conflict, or one-seed aggregate dominance.

Missing or contradictory artifacts, provenance or B1-C closure failure, alignment failure, unresolved required schema, insufficient eligible rows, or output-contract violation maps only to `STAGE196B2A_ANALYSIS_INCOMPLETE`. Runtime/schema failure is never converted into mechanistic evidence.

## Exact output closure

The analyzer creates exactly nine files in a new output directory:

```text
stage196b2a_analysis.json
stage196b2a_report.md
stage196b2a_seed_summary.csv
stage196b2a_support_transition_rows.csv
stage196b2a_channel_transition_summary.csv
stage196b2a_recurrent_position_propagation.csv
stage196b2a_harm_rescue_rows.csv
stage196b2a_epoch_propagation.csv
stage196b2a_contract.csv
```

The analysis JSON always records the decision, next stage, blockers, three commit roles, B1-C source decision, derived seed roles when closure reaches that gate, schema, uniform margin source, primary population counts, blocker distributions, rule evaluation, authorized/prohibited interpretations, file count, and the activity flags `training_performed=false`, `checkpoint_loaded=false`, `model_loaded=false`, `external_evaluation_performed=false`, and `artifact_only_analysis=true`.

Incomplete output preserves all nine schemas: scientific CSVs contain headers only, the contract contains every gate reached plus the terminal failure, and the Markdown contains all nineteen required sections. This exact closure prevents partial scientific tables from being mistaken for a complete analysis.

## Static review checklist

The implementation statically establishes:

1. the analyzer and specification remain the Stage196-B2-A definition while Stage196-B2-P0 adds the separately specified observability support;
2. the exact six-run matrix is required;
3. completed B1-C and all eight companions are required;
4. Stage196-A recurrent sets are loaded, with 22/19/19/10 closure;
5. selected, tail-three, and epoch views remain distinct;
6. a complete epoch table requires exactly 60 rows;
7. controls remain joint/baseline-defined;
8. first-blocker order is fixed before results;
9. positive-shift seeds are derived from B1-C deltas, not hard-coded as the rule;
10. seed directions remain explicit;
11. correlations alone never enter the decision;
12. no metric or threshold search exists;
13. output closure is exactly nine files;
14. no analyzer, Python, compile, smoke, training, checkpoint, model, Kaggle, commit, or push execution is part of this implementation stage.

## Remaining risk

The requested complete scientific audit is blocked by information that was not exported at epochs 18–20 or the other trajectory epochs. Repair requires a separately authorized artifact source that supplies native aligned per-epoch downstream scalars without replay, or a separately scoped decision about whether new artifact generation is permitted. Neither is authorized here.
