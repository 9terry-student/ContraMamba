# Stage196-B2-B6P2 Action-Conditional Composer Margin Export Specification

## Objective

Stage196-B2-B6P2 materializes exact action-conditional final composer state over
the frozen full candidate-row population. It is an export and exactness stage
only. It does not evaluate a safety gate, fit a threshold, train a model,
enumerate feature subsets, or use P0 row safety targets.

B2-B6P1 identified an export boundary, not an instrumentation requirement.
The existing boundary is the `ContraMambaV6BMinimal.forward` return where
`output["logits"]` is `final_logits` and `output["predictions"]` is
`final_logits.argmax(dim=-1)`. Candidate actions reuse the frozen
`reconstruct` and `apply_mask` functions from
`scripts/analyze_stage196b2b6_minimal_selector_intervention.py`; the exporter
does not implement a second composer formula or reinterpret primitive-mask
bits.

## Required invocation

```text
--repo-root
--stage196b2b6p1-analysis-json
--stage196b2b6p0-analysis-json
--stage196b2b6-analysis-json
--stage196b2b5-analysis-json
--stage196b2b4-analysis-json
--stage196b2b5-recipient-signature-rows-csv
--current-git-commit
--output-dir
```

The B2-B5 recipient-signature CSV is an explicit external authority. The
exporter uses only its exact CLI path and never discovers it by timestamp glob.
The output directory must not already exist.

## Frozen authorities

The exporter requires:

- B2-B6P1 decision
  `STAGE196B2B6P1_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_REQUIRED`,
  recommendation
  `STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT`, and no blockers.
- B2-B6P0 decision
  `STAGE196B2B6P0_CURRENT_SAFETY_OBSERVABILITY_INSUFFICIENT` and no blockers.
  Only the frozen 6,480 candidate-row, 2,160 native-row, and three-candidate
  population facts are validated. The P0 row-safety target CSV is never opened.
- B2-B6 decision `STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE` and the exact
  candidate masks `00100000000000`, `01000000000000`, and
  `10000000000000`.
- B2-B5 decision `STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED`,
  its analysis, feature dictionary, row action sets, recipient selector
  summary, and explicit recipient-signature CSV.
- B2-B4 decision `STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION`, primitive
  action semantics, and controlled score coordinates.

The B2-B6 clean-dev signature audit supplies exactly one frozen primitive
action per candidate mask and recipient data identity. No donor action is
selected from an outcome label.

## Population closure

The native key is `(seed, data_identity)`. The candidate key is
`(seed, data_identity, candidate_mask)`. A data identity is the canonical
structured value of `id`, `source_row_id`, and `dev_position`.

Required closure is:

```text
seeds = 183, 184, 185
720 unique recipient identities per seed
2,160 native rows
3 exact candidate masks per recipient
6,480 candidate-action rows
```

The same 720-identity set must occur in every seed. Missing, extra, duplicate,
unknown-mask, and cross-seed-confused rows fail closed. `seed`,
`stable_row_id`, and `data_identity` are provenance keys only.

## Class-coordinate mapping

The source decision head constructs coordinates in the validated internal
order:

```text
0 = REFUTE
1 = NOT_ENTITLED
2 = SUPPORT
```

The exported semantic fields are named explicitly:

```text
score_support
score_not_entitled
score_refute
```

The semantic presentation order `SUPPORT, NOT_ENTITLED, REFUTE` does not change
the internal authority. The exporter maps names explicitly and never assumes
or silently changes positional order.

## Native export

There is one native row per seed and recipient data identity. It includes the
authoritative stored prediction, exact class scores, top-1 and runner-up
classes, the exact top1-minus-runner-up margin, all three named pairwise
margins, source dtype, and source-boundary identity.

## Candidate-action export

There is one row per seed, recipient identity, and candidate mask. It records
the frozen primitive `candidate_action_key`, native and counterfactual
predictions, exact counterfactual class scores, top-1 and runner-up state,
named margins, dtype, and source-boundary identity.

## Margin and delta definitions

For class \(c\):

```text
delta_score_c = counterfactual_score_c - native_score_c
```

Named margins are:

```text
SUPPORT-minus-NOT_ENTITLED = score_support - score_not_entitled
SUPPORT-minus-REFUTE = score_support - score_refute
REFUTE-minus-NOT_ENTITLED = score_refute - score_not_entitled
```

Each named margin delta is its counterfactual value minus its native value.
`delta_top1_runner_up_margin` is the counterfactual top1-minus-runner-up margin
minus the native top1-minus-runner-up margin.

The response exports all three coordinate deltas. L1 is their sum of absolute
values, L2 is their Euclidean norm, and L-infinity is their maximum absolute
value. These norms are supplemental diagnostics, never substitutes for the
coordinate deltas.

## Branch-transition definitions

The entitled branch is the set `{SUPPORT, REFUTE}` and the non-entitled branch
is `{NOT_ENTITLED}`.

- `entitlement_branch_changed` is true exactly when native and
  counterfactual predictions occupy different entitlement branches.
- `polarity_branch_changed` is true exactly when both predictions are entitled
  and one is SUPPORT while the other is REFUTE.
- `polarity_direction_preserved` is true exactly when both predictions are
  entitled and the SUPPORT/REFUTE class is unchanged.

All flags use native and counterfactual model outputs only. Gold labels are not
read or used.

## Numerical fidelity

Native coordinates preserve the serialized `torch.float32` `final_logits`
values. The frozen B2-B6 action composer uses Python binary64 arithmetic over
serialized composer inputs, so counterfactual rows record
`python_float_binary64`. P2 serializes numeric CSV fields with `%.17g`,
sufficient for exact binary64 round-trip and more than sufficient to preserve
the serialized float32 values. Values are not rounded before serialization.
Float64 values are not cast to float32, and no cross-dtype bitwise-exactness
claim is made.

Every numeric score, margin, delta, and norm must be finite. NaN and infinity
block the run.

## Prediction reproduction

For all 2,160 native rows, argmax of the exported native score coordinates must
equal the authoritative native prediction. For all 6,480 candidate-action
rows, argmax of the exported counterfactual coordinates must equal the
prediction returned by the frozen B2-B6 action application. The exported
`prediction_changed` flag must equal inequality of the two exported
predictions.

All three disagreement counts must be zero. Any disagreement produces
`STAGE196B2B6P2_BLOCKED_SCORE_REPRODUCTION_FAILURE`.

## Controlled-row cross-check

Where epoch-20 forward B2-B4 primitive coalition rows overlap candidate
actions, P2 compares all three score coordinates, the SUPPORT-minus-
NOT_ENTITLED margin, and prediction. It reports matched controlled rows,
coordinate disagreements, margin disagreements, prediction disagreements,
maximum absolute difference, and the `1e-15` text-serialization tolerance.

Controlled coverage need not span the full population, and zero overlap alone
does not block. Any disagreement in an existing overlap blocks.

## Leakage boundary

The following fields are `PROVENANCE_ONLY_NOT_FEATURE_AUTHORIZED`:

```text
seed
stable_row_id
data_identity
```

Integration-authorized candidate quantities are limited to:

```text
native class scores and margins
counterfactual class scores and margins
exact score and margin deltas
model-output transition flags
explicit action-response norms
```

The following are prohibited feature inputs:

```text
seed
stable_row_id
data_identity
raw text
gold label
correctness
recovery
harm
MUST_ALLOW
MUST_BLOCK
OPTIONAL
discovery membership
primary-case membership
candidate outcome frequency
```

P2 never opens the P0 row-safety target CSV and never computes precision,
recall, F1, AUROC, thresholds, feasible gates, candidate gates, or
target-conditioned separation.

## Decisions

Success:

```text
decision = STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_COMPLETE
recommended_next_stage = STAGE196B2B6P3_ACTION_RESPONSE_SAFETY_GATE_DIAGNOSTIC
```

Missing or invalid authority/source boundary:

```text
decision = STAGE196B2B6P2_BLOCKED_SOURCE_OR_ARTIFACT_FAILURE
recommended_next_stage = STAGE196B2B6P2_REPAIR_SOURCE_OR_ARTIFACT
```

Score reproduction failure:

```text
decision = STAGE196B2B6P2_BLOCKED_SCORE_REPRODUCTION_FAILURE
recommended_next_stage = STAGE196B2B6P2_REPAIR_COMPOSER_SCORE_EXPORT
```

Exit status is zero only when `blocking_reasons` is empty; otherwise it is two.
Unhandled exceptions produce the exact blocked output set where the
non-overwrite contract permits it.

## Exact outputs

The output directory contains exactly:

```text
stage196b2b6p2_analysis.json
stage196b2b6p2_report.md
stage196b2b6p2_source_closure.csv
stage196b2b6p2_native_composer_scores.csv
stage196b2b6p2_candidate_action_composer_scores.csv
stage196b2b6p2_action_response_margin_rows.csv
stage196b2b6p2_coverage_and_reproduction_summary.csv
stage196b2b6p2_decision_gate.csv
stage196b2b6p2_contract.csv
```

Writes are staged and atomically published. Existing output directories are
never overwritten.

## Contract

The structured contract columns are:

```text
scope
run
gate
required
observed
passed
blocking_reason
```

The contract covers current commit identity; P1 and P0 closure; B2-B6 masks;
B2-B5 source, schema, provenance, seed, and large-file identity; B2-B4 action
semantics; class order; source boundary; native and candidate key closure;
2,160/6,480 counts; three masks per recipient; score completeness and
finiteness; margin and delta arithmetic; native, counterfactual, and categorical
response reproduction; controlled-row comparison; leakage boundaries; and the
exact nine-file declaration.
