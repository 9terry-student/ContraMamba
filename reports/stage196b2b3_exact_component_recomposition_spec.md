# Stage196-B2-B3 Exact Inference-Only Component Recomposition Probe

## Purpose

Stage196-B2-B3 asks whether actual causal inputs to the frozen `v6b_minimal` final decision graph can be exchanged between `joint` and `frame_local_only` at the same seed, stable row, and epoch. It is an artifact-only, inference-only diagnostic. It introduces no learned or fitted decision mechanism.

The analyzer is `scripts/analyze_stage196b2b3_exact_component_recomposition.py`. Its required CLI is exactly:

```text
--repo-root
--stage196b2b2-analysis-json
--stage196b2b2-analyzer-git-commit
--stage196b2b1-analysis-json
--stage196b2a-analysis-json
--stage196b2p0-run-root
--trainer-path
--current-git-commit
--output-dir
```

All arguments are required and have no optional defaults.

## Frozen provenance roles

The analyzer keeps these roles distinct:

```text
Stage196-B2-B2 analyzer          85b571610c00a4a1658229051bd6d9fcfabcf408
Stage196-B2-B1 analyzer          85f1de8f9e0393ccdca5da4bc0725d88d8f427c9
Stage196-B2-A analyzer           833752ec25890e76ee3c2dc3f3bc3c3c1b7428a6
Stage196-B2-P0 training runtime  e9aaff24054f1d409119b70df13b94159a34a8e4
Original Stage196-B1 runtime     9835cbbf86d83aca0964821669e63f7f6deb1c59
FrameGate implementation origin 5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8
```

The new analyzer commit is supplied separately through `--current-git-commit` and must equal repository `HEAD` when the analyzer is eventually run.

## Exact source and artifact closure

The parent directory of `--stage196b2b2-analysis-json` must contain exactly the nine frozen B2-B2 outputs, with no missing or extra entries. The analyzer requires the completed B2-B2 decision and recommended stage, empty blocking reasons, exactly 155 passed contract gates with empty gate blockers, 16 row-path rows, and 320 paired-epoch rows.

The predecessor paths are explicit CLI inputs. No timestamp glob, sibling guessing, repository search, or hard-coded hosted path is permitted.

The parent of `--stage196b2b1-analysis-json` must contain exactly the eight frozen B2-B1 files: analysis JSON, report, group summary, row profiles, signature summary, cross-seed transfer, intervention-type summary, and contract. The decision must be `STAGE196B2B1_SEED_SPECIFIC_NO_STABLE_BIFURCATION`, the next stage must be `STAGE196B2B2_NO_PROMOTION_ROW_LEVEL_CAUSAL_PROBE`, blockers must be empty, and exactly 23 contract gates must pass with empty blockers.

The parent of `--stage196b2a-analysis-json` must contain exactly the nine frozen B2-A files: analysis JSON, report, seed summary, support-transition rows, channel-transition summary, recurrent-position propagation, harm-rescue rows, epoch propagation, and contract. The decision must be `STAGE196B2A_SEED_SPECIFIC_MIXED_PROPAGATION`, the next stage must be `STAGE196B2B_NO_PROMOTION_MINIMAL_SEED_SPECIFIC_FOLLOWUP`, blockers must be empty, and every contract gate must pass with an empty blocker.

The frozen primary population must be exactly:

```text
seed184 recovery  5 MULTI_CHANNEL_CONFLICT
seed184 harm      3 POLARITY_OVERRIDE_DESPITE_FRAME_GAIN
                  2 FRAME_ENTITLEMENT_LOSS
                  1 COMPOSITION_DIVERGENCE_WITHOUT_LOCAL_BOUNDARY
seed185 recovery  2 FRAME_ENTITLEMENT_GAIN
seed185 harm      3 MULTI_CHANNEL_CONFLICT
```

The row CSV, analysis JSON, and Markdown report must agree. Seeds 184 and 185 are positive primary seeds; seed183 is contrast-only.

The P0 root must contain exactly the six expected run directories. Each run may contain unrelated standard artifacts, but the `stage196b2p0_epoch_channels_NNN.jsonl` namespace must contain exactly basenames `001` through `020`. Every sidecar must have exactly 720 rows and the exact frozen 18-field P0 schema. IDs, source-row IDs, dense dev positions, seed, epoch, mode, labels, probabilities, and finite logits are validated across epochs and arms.

## Margin-source normalization

The authoritative margin source is:

```text
support_logit - not_entitled_logit
```

Authority uses this exact precedence:

1. The mandatory primary authority is exactly one passed B2-B1 contract row with `scope=source` and `gate=contract_native_logit_margin_authority`. The formula is read from `required`, never from `observed`.
2. B2-B1 `provenance/normalized_source_roles` is parsed from its `observed` JSON when present. A non-null `support_vs_not_entitled_margin_source` must agree. An absent row or absent/null field passes with a schema warning because the native B2-B1 authority remains controlling. A null B2-B1 top-level normalized alias is not a blocker.
3. B2-A remains an explicit input with exact nine-file closure and provenance-chain validation. Its contract is not required to contain either `contract_native_logit_margin_authority` or `normalized_source_roles`. An absent optional row or absent/null margin field passes with a schema warning; a non-null agreement passes without a warning; malformed present evidence or a non-null conflict fails closed. Historical commit roles remain validated from all available contract-role and analysis top-level evidence, with disagreement rejected.
4. B2-B2 top-level `normalized_margin_source` is optional. Null passes with a schema warning; a non-null value must agree. Every other already-supported non-null B2-B2 margin alias is also a consistency check.

The analyzer never fabricates a missing B2-A value and never silently selects between conflicting non-null values. Every available non-null source must equal `support_logit - not_entitled_logit`; otherwise the input is malformed and the decision is `STAGE196B2B3_ANALYSIS_INCOMPLETE`.

The output analysis retains `normalized_margin_source`, the complete ordered `margin_source_authorities` evidence, `source_schema_warnings`, both explicit predecessor analysis paths, and `normalized_historical_provenance_roles`. The first authority entry identifies the B2-B1 contract `required` column as primary.

The contract records diagnostic evidence under these corrected gates:

```text
b2b1_native_margin_authority
b2b1_normalized_margin_consistency
b2a_optional_margin_consistency
b2b2_optional_margin_consistency
cross_generation_margin_source_agreement
```

`b2b1_native_margin_authority` records analysis path, contract path, scope, source gate, `required` column, match count, passed state, and value. Each optional contract-mapping gate records analysis path, contract path, scope, gate, field, column, match count, row/field presence, value, parse error, status, and passed state. `optional_row_absent` and `optional_field_absent_or_null` pass with warnings; `optional_value_agrees` passes; malformed present evidence and `optional_value_conflicts` fail closed. Cross-generation agreement compares only valid non-null values and still requires the B2-B1 native authority. The provenance-chain gate continues to cross-check B2-B1 and B2-A analyzer commits and normalized runtime/origin roles against B2-B2 source provenance. The B2-B3 commit remains a separate current-analyzer role.

## Static composer audit

The trainer path must resolve exactly to `scripts/train_controlled_v6b_minimal.py`. The analyzer parses source with the Python AST; it does not import the trainer, import Torch, instantiate the model, or load a checkpoint.

The audit follows the trainer's explicit import to `ContraMambaV6BMinimal`, verifies the `build_model` frozen settings, identifies `ContraMambaV6BMinimal.forward`, and resolves `FinalEntitlementDecisionHead.forward`. It records callable spans, source files, trainer SHA-256, primitive inputs, derived entitlement, native logit equations, final comparator modulation, and FrameGate origin.

The frozen explicit-product graph is source-defined as:

```text
entitlement = Frame * PredicateCoverage * Sufficiency
SUPPORT = entitlement * positive_energy
REFUTE = entitlement * negative_energy
NOT_ENTITLED = learned_bias + softplus(learned_raw_alpha) * (1 - entitlement)
native logit order = [REFUTE, NOT_ENTITLED, SUPPORT]
```

The v6b wrapper then applies learned temporal and predicate comparator modulation to the three logits when the corresponding actual flags are active.

`polarity_support_margin` is explicitly classified as an exported diagnostic equal to `positive_energy - negative_energy`. It is not consumed by the final decision head and cannot stand in for the two actual energies. The exported `entitlement_probability` is a derived intermediate; primitive swaps must recompute it.

## Native reconstruction precondition

Before any swap, exact native reconstruction is mandatory for:

```text
6 runs x 20 epochs x 720 rows = 86,400 rows
```

For every row, reconstruction must compare SUPPORT, REFUTE, NOT_ENTITLED, SUPPORT-vs-NOT_ENTITLED margin, and final prediction. Maximum absolute logit and margin errors must be at most `1e-6`, and prediction equality must be exactly `1.0`.

The current exact P0 schema omits actual inputs required by the source graph: positive and negative energies, the REFUTE logit, learned decision-head parameters, and actual epoch-specific final-comparator flags/parameters. Therefore the current artifacts cannot represent the full-composer positive control or satisfy native reconstruction. The analyzer must not solve for these values from correlations or exported logits.

This is a completed diagnostic outcome:

```text
STAGE196B2B3_ADDITIONAL_COMPOSER_OBSERVABILITY_REQUIRED
```

with recommended next stage:

```text
STAGE196B2B3P0_EPOCH_COMPOSER_INPUT_OBSERVABILITY_DESIGN
```

It is not `ANALYSIS_INCOMPLETE` when source closure, artifact closure, alignment, and graph identification pass.

## Precommitted swaps and directions

If a future artifact closure exports every required actual input and exact reconstruction is implemented and passes, the only authorized variants, in order, remain:

```text
FRAME_ONLY
PREDICATE_ONLY
SUFFICIENCY_ONLY
ENTITLEMENT_PRIMITIVES
POLARITY_ONLY
ENTITLEMENT_PLUS_POLARITY
FULL_COMPOSER_INPUT_POSITIVE_CONTROL
```

Both directions remain precommitted:

```text
recipient joint, donor frame_local_only
recipient frame_local_only, donor joint
```

All unswapped values must remain from the recipient at the same seed, stable row, and epoch. Derived entitlement must be recomputed after primitive swaps. No component-subset search is permitted. Epochs 18, 19, and 20 remain the primary scientific summary; selected-checkpoint behavior cannot replace them.

Because current reconstruction is unavailable, the analyzer emits no scientific swap rows. This avoids implementing a dormant approximate composer under the guise of a future path.

## Subtype audit

Frozen B2-B2 path classes are never relabeled. The additional audit groups are:

- frame/entitlement-loss-like: the two `FRAME_ENTITLEMENT_LOSS` rows plus the three seed185 harm `MULTI_CHANNEL_CONFLICT` rows whose terminal Frame, predicate, entitlement, and final-margin deltas are all negative;
- polarity override: exactly `POLARITY_OVERRIDE_DESPITE_FRAME_GAIN`;
- composition residual: exactly `COMPOSITION_DIVERGENCE_WITHOUT_LOCAL_BOUNDARY`;
- seed184 recovery `MULTI_CHANNEL_CONFLICT`;
- seed185 recovery `FRAME_ENTITLEMENT_GAIN`.

The subtype table records these frozen populations and marks recomposition results unavailable; it does not fabricate reproduction rates.

## Decision discipline

The fixed seven-decision set is encoded. Under the current fixed P0 schema, only the completed additional-observability decision can follow valid closure. All mechanistic decision rules remain unevaluated because their denominators require exact bidirectional counterfactuals.

`STAGE196B2B3_ANALYSIS_INCOMPLETE` is reserved for malformed inputs, provenance disagreement, closure failure, alignment failure, duplicate identities, unexpected counts, source-graph changes, or output-contract failure.

No training authorization is granted by any decision.

## Exact output closure

The analyzer creates exactly nine files:

```text
stage196b2b3_analysis.json
stage196b2b3_report.md
stage196b2b3_composer_graph.csv
stage196b2b3_native_reconstruction.csv
stage196b2b3_component_swap_rows.csv
stage196b2b3_row_swap_summary.csv
stage196b2b3_group_swap_summary.csv
stage196b2b3_subtype_summary.csv
stage196b2b3_contract.csv
```

For the additional-observability outcome, native reconstruction and all counterfactual swap tables are header-only. `composer_graph.csv` contains the static causal/diagnostic audit, and `subtype_summary.csv` contains population-only audit rows marked unavailable. No counterfactual logits are emitted.

The analysis JSON records all required negative activity flags, including no training, optimizer, backward pass, checkpoint-selection change, decision-rule change, external evaluation, threshold search, or classifier fit. It also records `artifact_component_recomposition = true`, `model_loaded = false`, and `checkpoint_loaded = false`.

## Prohibited operations

The implementation performs no trainer or model modification, training, backward pass, optimizer creation, checkpoint loading or selection, threshold search, calibration, classifier fitting, regression approximation, component-subset search, external evaluation, data modification, or historical artifact overwrite.

Static review of this change must not execute Python, `py_compile`, a smoke test, the analyzer, Kaggle, commit, or push.
