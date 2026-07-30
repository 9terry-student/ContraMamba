# ContraMamba Reason Router P3-W0 Reason-Loss Weight Resolution Specification

## Scope

This revision responds to `P3W0_FINAL_REVIEW_V2_BLOCKED`. It modifies only:

```text
reports/reason_router_p2_p3w0_reason_weight_resolution_spec.md
reports/reason_router_p2_p3w0_reason_weight_resolution_manifest.json
```

No trainer, model/head, tests, data, existing P3 reports/manifests, other reports, or patch files are modified. No calibration, training, evaluation, model load, Python, pytest, py_compile, Kaggle, or git operation is executed.

Both files must be UTF-8 without BOM. The Markdown first character is `#`; the JSON first character is `{`.

## Authority

`b118a984c749ca70bad63919f95aad918881a29d` is the A0 execution checkout commit supplied by the P3 A0 audit. It must not be renamed into the P2 implementation tested commit.

Current repository/report evidence records that the P2 test result was user-reported as `26 passed`, `P2_TEST_RETURN_CODE=0`, but the actual full SHA of the P2 implementation tested commit is not present in the current source/report authority. Therefore:

```text
p2_implementation_tested_commit = UNRESOLVED_P2_IMPLEMENTATION_TESTED_COMMIT
p3_a0_execution_commit = b118a984c749ca70bad63919f95aad918881a29d
p2_code_tree_unchanged_at_a0_execution_commit = true
p2_code_tree_identity_verification.status = RECORDED_BY_PRIOR_P3_EXECUTION_GATE
```

`p2_code_tree_unchanged_at_a0_execution_commit = true` is a prior identity-gate record only. It does not replace the unresolved tested commit SHA.

Code-tree identity files recorded by the prior P3 gate:

```text
scripts/train_controlled_v6b_minimal.py
src/contramamba/heads/entitlement_decision.py
src/contramamba/modeling_v6b_minimal.py
tests/test_reason_router_p2_contract.py
```

Authoritative A0 identity retained:

```text
authoritative_dev_row_count = 720
authoritative_dev_row_identity_hash = 3b807eeecbff097efdae5dfef73786ea1bd3414236237673c3ae42ecc4f52f4e
```

## Static Loss Algebra

The accurate loss algebra from prior P3-W0 inspection is retained.

For A1/A3, `scripts/train_controlled_v6b_minimal.py::_p2_reason_router_losses(...)` is the authority path:

```text
total =
  final_3way_ce
  + frame_bce
  + predicate_bce
  + sufficiency_bce
  + authorized_polarity_ce
  + reason_loss_weight * primary_reason_ce
```

### Final 3-Way CE

```text
source_file = scripts/train_controlled_v6b_minimal.py
function_or_block = _p2_reason_router_losses
relevant_argument = F.cross_entropy(output["logits"].index_select(0, indices), selected_labels, weight=weights if weighted_label_loss else None)
reduction = PyTorch default mean over selected rows; optional class-weighted mean if weighted_label_loss is active
eligibility = selected indices from v5.sample_indices
gradient_owner = final decision router path; in A3 explicit_local the local inputs to final router are detached
```

```text
logits shape = [N_selected, 3]
target shape = [N_selected]
class order = REFUTE, NOT_ENTITLED, SUPPORT
denominator = N_selected
ignored rows = rows absent from selected indices
class weight = weights if weighted_label_loss else None
zero selected rows = not expected for non-empty train labels
```

`output["logits"]` is final logits only; `base_logits` is diagnostic.

### Primary Reason CE

```text
source_file = scripts/train_controlled_v6b_minimal.py
function_or_block = _p2_reason_router_losses
relevant_argument = F.cross_entropy(output["reason_logits_4"][reason_active], reason_targets[reason_active])
reduction = PyTorch default mean over reason_active rows
eligibility = p2_reason_supervision_eligible & p2_primary_reason_targets_4 != -100
gradient_owner = conditional_first_blocker reason router; A1 reaches F/P/S paths, A3 does not because local-to-router inputs are detached
```

```text
logits shape = [N_reason_eligible, 4]
target shape = [N_reason_eligible]
class order = FRAME, PREDICATE, SUFFICIENCY, AUTHORIZED
denominator = N_reason_eligible
ignored rows = boolean-index excluded; ignore_index is not used
class weight = none
zero eligible batch = _p2_differentiable_zero(output["reason_logits_4"])
```

### Local Losses

F local:

```text
source_file = scripts/train_controlled_v6b_minimal.py
function_or_block = _p2_reason_router_losses -> _p2_masked_bce_loss
relevant_argument = F.binary_cross_entropy_with_logits(output["frame_logit"][active], frame_compatible_labels[active].float())
reduction = default mean
eligibility = p2_frame_applicability_mask = reason_supervision_eligible
gradient_owner = FrameGate local owner
```

P local:

```text
source_file = scripts/train_controlled_v6b_minimal.py
function_or_block = _p2_reason_router_losses -> _p2_masked_bce_loss
relevant_argument = F.binary_cross_entropy_with_logits(output["predicate_coverage_logit"][active], predicate_covered_labels[active].float())
reduction = default mean
eligibility = p2_predicate_applicability_mask = reason_supervision_eligible and frame_compatible_label == 1
gradient_owner = PredicateCoverageHead local owner
```

S local:

```text
source_file = scripts/train_controlled_v6b_minimal.py
function_or_block = _p2_reason_router_losses -> _p2_masked_bce_loss
relevant_argument = F.binary_cross_entropy_with_logits(output["sufficiency_logit"][active], sufficiency_labels[active].float())
reduction = default mean
eligibility = p2_sufficiency_applicability_mask = reason_supervision_eligible and frame_compatible_label == 1 and predicate_covered_label == 1
gradient_owner = SufficiencyGate local owner
```

Polarity local:

```text
source_file = scripts/train_controlled_v6b_minimal.py
function_or_block = _p2_reason_router_losses
relevant_argument = F.cross_entropy(torch.stack([negative_energy, positive_energy], dim=-1)[polarity_active], polarity_targets[polarity_active])
reduction = default mean
eligibility = p2_polarity_targets_2 != -100 and p2_polarity_applicability_mask
gradient_owner = PolarityEnergyHead local owner
```

Polarity class order is `REFUTE, SUPPORT`; denominator is `N_polarity_applicable`; no class weight is used; zero eligible batch returns differentiable zero.

## Reason Supervision Eligibility And Denominators

Primary reason is first-blocker ordered:

```text
if frame_compatible_label == 0 -> FRAME
elif predicate_covered_label == 0 -> PREDICATE
elif sufficiency_label == 0 -> SUFFICIENCY
else -> AUTHORIZED
```

A row is reason-supervision eligible only if no P2 exclusion code is produced. Ineligible rows receive `p2_primary_reason_targets_4 = -100` and are excluded by boolean indexing.

Denominators:

```text
final_3way_ce denominator = selected final-label row count
primary_reason_ce denominator = reason-eligible row count
F denominator = reason-eligible row count
P denominator = reason-eligible and F == 1 row count
S denominator = reason-eligible and F == 1 and P == 1 row count
polarity denominator = reason-eligible and F == 1 and P == 1 and S == 1 and final in {REFUTE,SUPPORT}
```

These denominators are not generally equal.

## Measurement Ownership

The calibration measurement path is now fixed:

```text
measurement_arm = conditional_first_blocker
measurement_gradient_ownership = explicit_local
detach_value_preserving_for_scalar_forward = true
```

Rationale:

```text
calibration is no-backward scalar measurement, so detach does not change scalar values
explicit_local prevents the false interpretation that reason/final calibration grants optimization authority to local heads
explicit_local aligns with the A3 ownership contract
one fixed measurement path is provided
one fixed measurement configuration supports one common A1/A3 weight
```

In `src/contramamba/modeling_v6b_minimal.py::ContraMambaV6BMinimal.forward`, explicit-local mode routes detached aliases into downstream consumers:

```text
frame_prob.detach()
predicate_coverage_prob.detach()
sufficiency_prob.detach()
positive_energy.detach()
negative_energy.detach()
```

Detach is value-preserving. Therefore, for the same initialized parameters, same model mode, same inputs, and same RNG/dropout realization, A1 joint and A3 explicit-local scalar forward values are numerically the same for final and reason losses; only gradient reachability differs. This statement depends on the same RNG realization because the model contains dropout and training forward runs under `model.train()`.

## `reason_loss_weight = 1.0`

`reason_loss_weight = 1.0` means only:

```text
total = final_ce_mean + F_mean + P_mean + S_mean + polarity_mean + primary_reason_ce_mean
```

It does not establish:

```text
per-example contribution equivalence = not established
per-eligible-example contribution equivalence = not established
gradient-norm equivalence = not established
optimization authority equivalence = not established
local-owner balance = not established
total-objective balance = not established
A1/A3 optimization equivalence = not established
causal superiority = not established
```

Chance-level CE normalization such as `log(3)` versus `log(4)` remains rejected because it does not account for eligibility denominators, binary local summands, sampled final CE denominator, or A1/A3 detach differences.

## Candidate Selection Rejected

The prior protocol is rejected:

```text
base_weight = L_final / L_reason
candidate_weights = base_weight x [2^-0.5, 1.0, 2^0.5]
selection = minimize abs(log((candidate_weight x L_reason) / L_final))
```

Mathematical reason:

```text
When base_weight = L_final / L_reason:

(base_weight x L_reason) / L_final = 1

Therefore:
abs(log(1)) = 0

The central candidate is selected by construction.
```

Therefore the candidate comparison is tautological and provides no independent calibration authority. The multipliers `[2^-0.5, 1.0, 2^0.5]` may only be used later as:

```text
diagnostic sensitivity multipliers
not weight-selection candidates
not execution-authorizing evidence
```

## Narrow Calibration Authority

The only defensible target authority type is:

```text
TRAIN_ONLY_FINAL_VS_REASON_SCALAR_CONTRIBUTION_MATCHING
```

This means the reason weight is an engineering calibration that matches final CE scalar contribution and reason CE scalar contribution on an initial train-only measurement surface. It does not establish:

```text
total-objective balance = not established
gradient-norm balance = not established
local-owner balance = not established
per-example equivalence = not established
per-eligible-example equivalence = not established
A1/A3 optimization equivalence = not established
causal superiority = not established
```

## Calibration Unit Contract

The actual P3-W0 calibration execution contract is fixed:

```text
train_batch_size = null
calibration_units_per_seed = 1
unit_index = 0
unit_scope = COMPLETE_AUTHORITATIVE_TRAIN_SPLIT
model_mode = train
balanced_sampler = false
calibration_seeds = [180, 181, 182]
measurement_arm = conditional_first_blocker
measurement_gradient_ownership = explicit_local
```

Each seed uses a fresh initialization, then measures the complete authoritative train split as one forward surface.

The unit identity format is fixed to ordered hashes, not per-row expansion:

```text
ordered_train_row_count = exact authoritative train row count
ordered_train_row_identity_hash = SHA256 of ordered (row_id, pair_id, normalized_gold_label)
```

## Pooled Denominator-Weighted Estimator

The general `s,b` estimator remains the theoretical form, but actual P3-W0 execution has one complete-train unit per seed.

For each seed `s`:

```text
n_final[s] = final applicable count from seed s complete-train unit
ell_final[s] = mean final CE from seed s complete-train unit
n_reason[s] = reason eligible count from seed s complete-train unit
ell_reason[s] = mean reason CE from seed s complete-train unit
```

Global pooled means:

```text
mu_final =
  sum_s n_final[s] * ell_final[s]
  /
  sum_s n_final[s]

mu_reason =
  sum_s n_reason[s] * ell_reason[s]
  /
  sum_s n_reason[s]
```

Scalar-matching weight:

```text
resolved_reason_loss_weight = mu_final / mu_reason
```

Forbidden:

```text
mean of seed loss means without count weighting
mean of seed-level ratios
seed-specific weights
seed180-only authority
dev-selected weight
A0-performance-selected weight
A0 prediction/error/logit/probability based selection
A0 trained checkpoint
A0 selected epoch state
```

The resulting weight, if ever computed, must be one global value shared by seeds `180,181,182` and by A1/A3.

## Calibration-Only P2 Gate Exception

Current A1/A3 arm contracts require a positive `reason_loss_weight`, but calibration exists to determine that weight. Therefore minimal instrumentation must provide a separate calibration-only mode activated by an equivalent of:

```text
--reason-router-weight-calibration-export PATH
```

Calibration-only activation contract:

```text
reason_router_weight_calibration_only = true
reason_router_mode = conditional_first_blocker
gradient_ownership_mode = explicit_local
reason_loss_weight = 0.0
```

Only in this mode, the positive A1/A3 `reason_loss_weight` requirement is bypassed. The exception applies only to that one requirement. All other P2 fail-fast and objective-neutralization contracts remain active.

Calibration-only mode must:

```text
compute unweighted final 3-way CE
compute unweighted primary reason CE
compute applicable/eligible counts
persist calibration artifact
do not construct an execution-authorizing A1/A3 training result
do not backward
do not optimizer.step
do not scheduler.step
do not evaluate dev
do not evaluate external/OOD
do not save causal-run checkpoint
do not write normal A1/A3 training report
```

`reason_loss_weight = 0.0` is a measurement placeholder only. It is not the resolved A1/A3 execution weight, and it does not multiply the exported unweighted reason CE. Without this calibration-only exception, any measurement would require an arbitrary positive placeholder and become circular.

## Calibration Artifact Namespace

Calibration artifacts are separate from normal training artifacts:

```text
reports/reason_router_p2_p3w0_calibration/
  seed180/calibration_unit.json
  seed181/calibration_unit.json
  seed182/calibration_unit.json
  calibration_aggregate.json
```

Each seed unit artifact must include:

```text
schema_version
seed
unit_index = 0
unit_scope = COMPLETE_AUTHORITATIVE_TRAIN_SPLIT
ordered_train_row_count
ordered_train_row_identity_hash
model_mode = train
measurement_arm = conditional_first_blocker
measurement_gradient_ownership = explicit_local
fresh_initialization = true
checkpoint_loaded = false
before_backward = true
before_optimizer_step = true
before_scheduler_step = true
parameter_update_count = 0
dev_forward_executed = false
external_eval_executed = false
final_loss_mean
final_applicable_count
final_loss_sum_reconstructed
reason_loss_mean
reason_eligible_count
reason_loss_sum_reconstructed
final_loss_finite
reason_loss_finite
dataset_sha256
sidecar_semantic_sha256
split_seed
execution_commit
```

Aggregate artifact must include:

```text
schema_version
calibration_seeds
seed_unit_artifact_sha256
total_final_count
total_reason_count
total_final_loss_sum
total_reason_loss_sum
mu_final
mu_reason
resolved_reason_loss_weight
all_three_seeds_present
all_unit_gates_pass
nonfinite_count
decision
```

Loss sums must satisfy:

```text
final_loss_sum_reconstructed = final_loss_mean * final_applicable_count
reason_loss_sum_reconstructed = reason_loss_mean * reason_eligible_count
```

## `p2_loss_export` Aggregation Semantics

Source evidence:

```text
scripts/train_controlled_v6b_minimal.py::_p2_record_epoch_loss_snapshot appends {"epoch", "loss_summary"} to _p2_epoch_loss_history.
The append occurs inside the epoch loop after _p2_reason_router_losses/_p2_reason_arm_loss_export and before loss_for_backward.backward() and optimizer.step().
The final report writes reason_router_p2.epoch_loss_history, reason_router_p2.final_epoch_loss_summary, and reason_router_p2.loss_summaries.
```

Classification:

```text
p2_loss_export = PER_EPOCH_PRE_STEP_FULL_TRAIN_FORWARD_SNAPSHOT
```

It is not:

```text
PER_BATCH_PRE_STEP_EXPORT
LAST_BATCH_ONLY
a dedicated calibration artifact
a no-step execution mode
an all-three-seed pooled authority artifact
sufficient to release A1/A2/A3
```

If `train_batch_size is None`, the train forward is a single full-train forward. If `train_batch_size` is set, `_vnext_forward_maybe_batched` slices the train tensors, forwards deterministic chunks, concatenates outputs, and loss export is still one merged full-train epoch snapshot, not persisted per-mini-batch units.

## Existing Trainer Path Audit

Path A, dedicated no-step calibration, is rejected because no source evidence was found for an opt-in reason-router weight calibration export mode that is forward-only, no-backward, no-optimizer-step, train-only, and persists all scalar/count/unit identity records.

Path B, throwaway training invocation, is rejected because it lacks a calibration-only mode, no-step artifact proof, explicit calibration artifact namespace, unit identity hashes, nonfinite counts, all-three-seed pooled aggregate, and proceeds to optimizer updates and dev evaluation.

## Observability Reclassification

```text
calibration-only mode = REQUIRES_MINIMAL_INSTRUMENTATION
no-step artifact persistence = REQUIRES_MINIMAL_INSTRUMENTATION
complete-train unit identity proof = REQUIRES_MINIMAL_INSTRUMENTATION
parameter_update_count proof = REQUIRES_MINIMAL_INSTRUMENTATION
pooled three-seed aggregate = REQUIRES_MINIMAL_INSTRUMENTATION
pre-update final CE scalar = REQUIRES_MINIMAL_INSTRUMENTATION
pre-update reason CE scalar = REQUIRES_MINIMAL_INSTRUMENTATION
final applicable count = REQUIRES_MINIMAL_INSTRUMENTATION
reason eligible count = REQUIRES_MINIMAL_INSTRUMENTATION
seed identity = ALREADY_AVAILABLE
nonfinite count = REQUIRES_MINIMAL_INSTRUMENTATION
```

Existing `p2_loss_export` is insufficient for P3-W0 release.

## Validity Gates

All gates are required:

```text
calibration seeds exactly [180,181,182]
all three seed records present
dataset SHA exact match
sidecar semantic SHA exact match
split seed exact match
train split only
no dev access
no A0 prediction access
measurement_arm == conditional_first_blocker
measurement_gradient_ownership == explicit_local
train_batch_size is null
units_per_seed == 1
unit_index == 0
unit_scope == COMPLETE_AUTHORITATIVE_TRAIN_SPLIT
model_mode == train
balanced_sampler == false
fresh_initialization == true
checkpoint_loaded == false
parameter_update_count == 0
before_backward == true
before_optimizer_step == true
before_scheduler_step == true
dev_forward_executed == false
external_eval_executed == false
ordered train row count exact match
ordered train row identity hash exact match across all three seeds
total_final_count > 0
total_reason_count > 0
mu_final finite
mu_reason finite
mu_final > 0
mu_reason > 0
resolved weight finite
resolved weight > 0
no duplicate calibration unit identity
no missing calibration unit identity
```

Any failed gate yields:

```text
P3W0_BLOCKED_BY_INSUFFICIENT_SCALE_AUTHORITY
```

An existing calibration artifact does not authorize a weight unless every gate passes.

## A0 Information Policy

Allowed:

```text
execution commit identity
dataset identity
sidecar identity
split identity
train/dev universe identity
artifact existence
artifact schema identity
```

Forbidden:

```text
A0 accuracy
A0 macro-F1
A0 prediction counts
false-entitlement count
stable true-SUPPORT count
seed-specific A0 error pattern
A0 prediction probabilities/logits
A0 selected epoch state or checkpoint as weight-selection signal
```

A0 error populations remain reserved only for later recovery/harm denominators.

## A1/A2/A3 Status

```text
A0 = EXECUTION_AND_REFERENCE_AUDIT_PASS
A1 = BLOCKED_BY_UNRESOLVED_REASON_LOSS_WEIGHT
A2 = BLOCKED_BY_MATCHED_FACTORIAL_RELEASE
A3 = BLOCKED_BY_UNRESOLVED_REASON_LOSS_WEIGHT
```

A2 has `reason_loss_weight = 0.0`, but must not be released alone before the matched A1/A2/A3 factorial batch is released.

## Decision

```text
P3W0_BLOCKED_BY_INSUFFICIENT_SCALE_AUTHORITY
```

Required fields:

```text
resolved_reason_loss_weight = null
calibration_executed = false
calibration_protocol_complete = false
candidate_selection_rejected_as_tautological = true
scalar_matching_formula_defined = true
execution_observability_sufficient = false
minimal_instrumentation_required = true
measurement_gradient_ownership = explicit_local
calibration_unit_contract_complete = true
A1_A3_common_weight = true
A1_A3_released = false
blockers = [INSUFFICIENT_SCALE_AUTHORITY, MISSING_CALIBRATION_ONLY_INSTRUMENTATION, UNRESOLVED_P2_IMPLEMENTATION_TESTED_COMMIT]
```

The following are not claimed:

```text
P3W0_TRAIN_ONLY_CALIBRATION_PROTOCOL_READY
P3W0_REASON_LOSS_WEIGHT_RESOLVED
A1_READY
A2_READY
A3_READY
```

## Non-Executed Verification List

```text
trainer modification: not executed
model/head modification: not executed
test modification: not executed
data modification: not executed
calibration: not executed
training: not executed
evaluation: not executed
model load: not executed
Python: not executed
pytest: not executed
py_compile: not executed
Kaggle: not executed
git add/commit/push/pull: not executed
```