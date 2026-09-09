# Seed8192 Revised-Split Reason-Loss Calibration Authority Specification Candidate

## Status and boundary

This document is a **report-only calibration-authority candidate** derived from the frozen Seed8192 A0 N=3 validated-evidence analysis.

Governing analysis freeze commit:

```text
dd183f59f4040405c178da193fe99c7c7f3ef57f
```

Governing validated-evidence report:

```text
reports/reason_router_p3w7_seed8192_a0_n3_validated_evidence_analysis_report_candidate.md
```

Seed8192 calibration split-rebind implementation authority:

```text
authority_commit = 6fd1b5e8e89295aeefe94ea8bec53a66de7cf082
authority_report = reports/reason_router_p3w7_seed8192_reason_loss_calibration_split_rebind_implementation_authority_spec_candidate.md
```

Verified split-rebind implementation commit:

```text
implementation_commit = 47ff8d16a28a17cb3dca2104c51b4d63c67d6109
implementation_verification = PASS_READY_FOR_COMMIT_REVIEW
focused_tests = 107 passed
training_or_evaluation_executed = false
calibration_execution_executed = false
```

The implementation changed only the frozen calibration split binding from historical `174` to exact `8192` in the trainer, pure-JSON aggregator, and focused calibration tests. The trainer and aggregator retain fail-closed exact-equality split gates.

This candidate authorizes no training, evaluation, Kaggle execution, A1/A2/A3 execution, commit, or push by itself.

Its sole purpose is to define a reproducible **calibration-only measurement authority** for resolving one common nonzero `reason_loss_weight` for later A1/A3 execution.

## Scientific motivation

The frozen A0 N=3 evidence establishes:

- stable aggregate A0 behavior across seeds 180/181/182;
- stable SUPPORT under-entitlement;
- stable AUTHORIZED-to-FRAME overblocking;
- stable PREDICATE-to-FRAME reason-ownership leakage;
- saturated REFUTE behavior;
- saturated sufficiency behavior;
- saturated entitled polarity behavior.

The analysis explicitly does not establish that reason-specific supervision improves these failures.

Therefore the next bounded question is not whether A1/A3 is superior. It is:

> What single engineering weight should scale primary-reason CE for later A1/A3 runs without using dev performance, A0 predictions, A0 checkpoints, or post-training model selection?

## Reused loss semantics

The current trainer's reason-router semantics preserve the prior P3 calibration algebra.

For A1/A3:

```text
total =
  final_3way_ce
  + frame_bce
  + predicate_bce
  + sufficiency_bce
  + authorized_polarity_ce
  + reason_loss_weight * primary_reason_ce
```

A1/A3 use `_p2_reason_router_losses(...)`.

A0/A2 do not use the primary-reason CE training path; they retain the product-arm loss/export path.

The calibration is therefore an A1/A3 weight-resolution operation only.

## Arm and gradient-ownership interpretation

The existing factorial contract is retained:

- reason-supervision factor is active in A1/A3;
- product-arm/no-primary-reason-loss factor is retained in A0/A2;
- A1/A3 share one resolved `reason_loss_weight`;
- calibration measurement uses `conditional_first_blocker`;
- calibration measurement uses `explicit_local`;
- detach under calibration is value-preserving because the calibration path performs no backward pass.

Calibration does not select between A1 and A3 and does not authorize either arm.

## Rejected approaches

The following are explicitly forbidden:

```text
reuse old split174 resolved weight as current authority
seed180-only calibration
seed-specific weights
dev-selected weight
A0-performance-selected weight
A0 prediction/error/logit/probability based weight selection
A0 trained checkpoint based calibration
selected-epoch checkpoint calibration
manual weight grid selected by downstream performance
mean of seed-level ratios
unweighted mean of seed loss means when denominators differ
```

The prior diagnostic multiplier family:

```text
[2^-0.5, 1.0, 2^0.5]
```

remains diagnostic-only and is not a candidate-selection rule.

## Why the historical value cannot be reused

The historical calibration aggregate resolved:

```text
resolved_reason_loss_weight = 0.6518018402446165
```

under a different split/data authority, including split seed 174.

The current primary research line uses split seed 8192 and a revised P4-L integrity sidecar.

Therefore the old numerical value is context only and is not execution-authorizing evidence for the current lineage.

## Current fixed inputs

### Execution source authority

The verified calibration implementation source is frozen at:

```text
implementation_commit = 47ff8d16a28a17cb3dca2104c51b4d63c67d6109
```

Exact implementation identities at that commit:

```text
scripts/train_controlled_v6b_minimal.py
git_blob = bb1639525916d99cbc4d458ba4771c277cd4d46b

scripts/aggregate_reason_router_p3w1_calibration.py
git_blob = 418f747df0e4224bc25124d0053e235fa8bd95b9

tests/test_reason_router_p3w1_calibration.py
git_blob = 723c63297338aa4d412b39cfcafb19bd7d7e798a
```

The active production calibration gates now require exact split seed `8192`; historical `174` is rejected. The focused calibration suite passed `107` tests after the rebind.

This calibration-authority candidate still does not authorize execution. A separate calibration execution authority/activation commit must be frozen after this candidate is independently verified and committed. That later execution authority must bind the exact implementation identities above and the exact authority-report identity produced by freezing this candidate.

### Dataset

```text
reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl
physical_sha256 = eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3
semantic_sha256 = 3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b
```

### Revised P4-L integrity sidecar

```text
reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl
physical_sha256 = 9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d
semantic_sha256 = 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9
```

### Split

```text
split_seed = 8192
dev_ratio = 0.2
train_rows = 2880
dev_rows = 720
calibration_seeds = [180, 181, 182]
```

Calibration consumes train only.

## Calibration-only execution contract

Exactly one logical calibration unit is measured per seed.

```text
calibration_units_per_seed = 1
unit_index = 0
unit_scope = COMPLETE_AUTHORITATIVE_TRAIN_SPLIT
model_mode = train
balanced_sampler = false
measurement_arm = conditional_first_blocker
measurement_gradient_ownership = explicit_local
fresh_initialization = true
checkpoint_loaded = false
reason_loss_weight_placeholder = 0.0
before_backward = true
before_optimizer_step = true
before_scheduler_step = true
parameter_update_count = 0
dev_forward_executed = false
external_eval_executed = false
```

The placeholder zero weight exists only because the measurement exports unweighted `final_3way_ce` and unweighted `primary_reason_ce`.

It is not a candidate execution weight.

## Required calibration measurements

For each seed `s`:

```text
n_final[s]  = final CE applicable count
L_final[s]  = mean unweighted final 3-way CE

n_reason[s] = primary-reason eligible count
L_reason[s] = mean unweighted primary-reason CE
```

Reconstruct:

```text
sum_final[s]  = n_final[s]  * L_final[s]
sum_reason[s] = n_reason[s] * L_reason[s]
```

All losses and reconstructed sums must be finite.

## Global estimator

Across seeds 180/181/182:

```text
mu_final =
  sum_s sum_final[s]
  / sum_s n_final[s]

mu_reason =
  sum_s sum_reason[s]
  / sum_s n_reason[s]

resolved_reason_loss_weight =
  mu_final / mu_reason
```

The resulting value is one global engineering calibration shared by A1 and A3.

No per-seed weights are permitted.

## Required reason-supervision readiness

Before weight release, each calibration unit must validate:

```text
primary reason classes:
FRAME
PREDICATE
SUFFICIENCY
AUTHORIZED
```

Each class must satisfy the trainer's frozen minimum-count gates for the current train split.

Calibration weight resolution and later A1/A3 training readiness remain distinct concepts.

A finite scalar weight does not by itself prove every downstream training contract is ready.

## No-dev / no-A0 leakage gate

Calibration must fail closed if any of the following occurs:

```text
dev input accessed
dev label accessed
dev metric computed
A0 checkpoint loaded
A0 selected checkpoint accessed
A0 prediction accessed
A0 logit/probability accessed
A0 reference predictions required
external/OOD evaluation executed
backward executed
optimizer step executed
scheduler step executed
parameter update count > 0
```

The Seed8192 A0 N=3 report motivates why calibration is needed but may not numerically select the weight.

## Required artifacts

Use a dedicated current-lineage namespace, separate from normal training outputs.

Proposed namespace:

```text
reports/reason_router_p3w7_seed8192_reason_loss_calibration/
  seed180/calibration_unit.json
  seed181/calibration_unit.json
  seed182/calibration_unit.json
  calibration_aggregate.json
```

Each unit must record at minimum:

```text
schema_version
seed
unit_index
unit_scope
ordered_train_row_count
ordered_train_row_identity_hash
model_mode
measurement_arm
measurement_gradient_ownership
fresh_initialization
checkpoint_loaded
before_backward
before_optimizer_step
before_scheduler_step
parameter_update_count
dev_forward_executed
external_eval_executed
final_loss_mean
final_applicable_count
final_loss_sum_reconstructed
reason_loss_mean
reason_eligible_count
reason_loss_sum_reconstructed
final_loss_finite
reason_loss_finite
primary_reason_class_counts
dataset_sha256
dataset_semantic_sha256
sidecar_physical_sha256
sidecar_semantic_sha256
split_seed
execution_commit
```

The aggregate must record at minimum:

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
all_primary_reason_min_count_gates_pass
all_dev_inputs_unaccessed
all_dev_labels_unused
all_dev_metrics_unused
all_a0_checkpoints_unused
all_a0_predictions_unused
all_a0_logits_unused
nonfinite_count
decision
```

## Acceptance gate for weight resolution

Calibration weight resolution passes only if all are true:

```text
all_three_seeds_present = true
all_unit_gates_pass = true
all_primary_reason_min_count_gates_pass = true
all_dev_inputs_unaccessed = true
all_dev_labels_unused = true
all_dev_metrics_unused = true
all_a0_checkpoints_unused = true
all_a0_predictions_unused = true
all_a0_logits_unused = true
nonfinite_count = 0
mu_final > 0
mu_reason > 0
resolved_reason_loss_weight is finite
resolved_reason_loss_weight > 0
```

No target numerical interval is imposed in advance.

An unexpected but valid positive finite result is evidence to review, not grounds to alter the estimator after seeing the value.

## Post-calibration decision boundary

A successful calibration resolves only:

```text
one common A1/A3 reason_loss_weight
```

It does not establish:

```text
A1/A3 performance
A1/A3 superiority to A0
A3 superiority to A1
reason-loss causal benefit
gradient-ownership causal benefit
factorial promotion
```

After validated calibration import, a separate result-review/freeze step must decide whether the resolved value is provenance-valid and training-ready.

Only after that may a separate A1/A2/A3 execution authority be authored or rebound.

## Execution sequencing

If this authority is later frozen and activated:

1. run CPU/static preflight with GPU OFF;
2. execute one calibration-only unit for seed180;
3. validate/import it before trusting it;
4. execute seed181 under the same frozen execution commit;
5. validate/import it;
6. execute seed182 under the same frozen execution commit;
7. validate/import it;
8. construct and independently verify the pooled aggregate;
9. freeze the resolved common weight in a separate result artifact/report.

No normal A1/A3 training may occur during this sequence.

## Candidate verification requirements

Independent static verification must confirm:

- governing analysis freeze commit and report identity;
- split-rebind implementation-authority commit identity;
- verified implementation commit `47ff8d16a28a17cb3dca2104c51b4d63c67d6109`;
- trainer, aggregator, and focused-test blob identities;
- exact trainer/aggregator split8192 gates and historical split174 rejection;
- focused calibration test result `107 passed`;
- A1/A3 reason-loss wiring;
- A0/A2 product-arm loss path;
- calibration-only path performs no backward/optimizer/scheduler step;
- train-only scope;
- current dataset and sidecar identities;
- split seed 8192;
- seeds 180/181/182;
- pooled denominator-weighted estimator;
- old split174 weight is not reused;
- no dev/A0-performance leakage;
- no A1/A2/A3 execution authorization;
- no repository mutation other than this candidate file.

## Candidate verdict

```text
PASS_READY_FOR_INDEPENDENT_SEED8192_REASON_LOSS_CALIBRATION_AUTHORITY_VERIFICATION
```
