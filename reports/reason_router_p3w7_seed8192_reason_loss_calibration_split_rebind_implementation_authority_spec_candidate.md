# Seed8192 Reason-Loss Calibration Split-Rebind Implementation Authority Specification Candidate

## Status

This is a **report-only implementation-authority candidate**.

It does not itself modify code, execute calibration, train, evaluate, run Kaggle, commit, or push.

Candidate verdict:

```text
CURRENT_CALIBRATION_SPLIT8192_EXECUTABLE = NO
BLOCKER = P3W1_CALIBRATION_SPLIT_SEED_HARDCODED_TO_174
IMPLEMENTATION_REBIND_REQUIRED = YES
TRAINING_EVALUATION_AUTHORIZED = NO
CALIBRATION_EXECUTION_AUTHORIZED = NO
```

## Governing authority

Frozen Seed8192 A0 N=3 validated-evidence analysis:

```text
commit = dd183f59f4040405c178da193fe99c7c7f3ef57f
report = reports/reason_router_p3w7_seed8192_a0_n3_validated_evidence_analysis_report_candidate.md
```

The validated-evidence analysis permits authoring a revised reason-loss calibration authority, but does not authorize calibration execution or A1/A2/A3 training.

A provisional calibration-authority candidate has been authored locally:

```text
reports/reason_router_p3w7_seed8192_reason_loss_calibration_authority_spec_candidate.md
```

That provisional candidate must not be staged/frozen as execution-ready until this split-rebind blocker is resolved and independently verified.

## Static blocker evidence at governing HEAD

At exact governing HEAD:

```text
dd183f59f4040405c178da193fe99c7c7f3ef57f
```

the calibration-only trainer path contains:

```text
P3W1_CALIBRATION_SPLIT_SEED = 174
```

and `_p3w1_validate_calibration_only_args(...)` requires:

```text
args.resolved_split_seed == P3W1_CALIBRATION_SPLIT_SEED
```

with the error semantics:

```text
resolved split seed must be 174
```

Therefore a scientifically intended Seed8192 calibration invocation is rejected before calibration forward execution.

The existing pure-JSON aggregator likewise contains:

```text
EXPECTED_SPLIT_SEED = 174
```

and `_validate_expected_identity(...)` requires:

```text
expected_split_seed == EXPECTED_SPLIT_SEED
```

with the error semantics:

```text
expected split seed must be exactly 174
```

The existing calibration tests also construct the normal calibration fixture with:

```text
resolved_split_seed = 174
```

and encode the old split-gate behavior.

This is a code/authority compatibility blocker, not scientific evidence and not a failed calibration run.

## Current exact source identities

Trainer:

```text
path = scripts/train_controlled_v6b_minimal.py
git_blob = 8252f5778944974e20c21acfe203e8f7bb5f3218
```

Aggregator:

```text
path = scripts/aggregate_reason_router_p3w1_calibration.py
git_blob = 506ac1937294e9d0267493de3793a5df11d88db5
```

The implementation must begin from governing HEAD and preserve all unrelated code.

## Goal

Make the existing P3-W1 calibration-only instrumentation and aggregate validator accept exactly the current revised split authority:

```text
split_seed = 8192
```

instead of the historical:

```text
split_seed = 174
```

This is a narrow lineage rebind only.

## Allowed implementation files

Exactly these files are authorized for modification:

```text
scripts/train_controlled_v6b_minimal.py
scripts/aggregate_reason_router_p3w1_calibration.py
tests/test_reason_router_p3w1_calibration.py
```

No other file may be modified under this authority.

If implementation requires any additional production or test file, stop and report the need for expanded authority.

## Required semantic delta

### Trainer

Rebind the calibration-only split constant/gate from historical split174 to current split8192.

Expected semantic change:

```text
P3W1_CALIBRATION_SPLIT_SEED:
174 -> 8192
```

All calibration-only validation must then require exact resolved split seed 8192.

Error/help text that explicitly says `174` must be updated to say `8192`.

### Aggregator

Rebind the independent aggregate expected-split authority from 174 to 8192.

Expected semantic change:

```text
EXPECTED_SPLIT_SEED:
174 -> 8192
```

The independent aggregate validator must fail closed for any expected split other than 8192.

Error text that explicitly says `174` must be updated to say `8192`.

### Tests

Update the P3-W1 calibration test authority so the valid/default calibration fixture uses:

```text
resolved_split_seed = 8192
```

and tests explicitly establish:

```text
8192 accepted
historical 174 rejected
another non-authoritative split rejected
aggregate expected split 8192 accepted
aggregate expected split 174 rejected
```

Existing unrelated calibration contracts must remain covered.

## Invariants that must not change

The following are frozen and must not be modified:

```text
calibration seeds = [180, 181, 182]
calibration arm = A3
resolved router = conditional_first_blocker
resolved gradient ownership = explicit_local
reason_loss_weight placeholder = 0.0
calibration forward batch size = 8
logical units per seed = 1
logical unit scope = COMPLETE_AUTHORITATIVE_TRAIN_SPLIT
model mode = train
torch.no_grad calibration measurement
fresh initialization
no checkpoint load
no backward
no optimizer step
no scheduler step
no dev calibration
no A0 reference/prediction/logit/checkpoint use
primary reason minimum train count = 50
pooled estimator = mu_final / mu_reason
one common A1/A3 resolved weight
dev ratio = 0.2
architecture = v6b_minimal
backbone = mamba
model = state-spaces/mamba-130m-hf
max length = 128
device contract = cuda
flag source = controlled_heuristic
frozen encoder
class weighting = none
```

## Data and sidecar boundaries

This implementation authority does not change dataset bytes, sidecar bytes, split construction, labels, or row identities.

Current scientific lineage remains:

```text
dataset physical SHA256 =
eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3

dataset semantic SHA256 =
3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b

revised P4-L sidecar physical SHA256 =
9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d

revised P4-L sidecar semantic SHA256 =
2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9

split seed = 8192
train rows = 2880
dev rows = 720
```

The implementation must not manufacture or alter these identities.

## Historical calibration result boundary

The historical resolved weight:

```text
0.6518018402446165
```

belongs to the prior split174 calibration lineage.

This implementation must not:

- copy that value into current authority;
- use it as a default;
- compare current calibration to it as a pass/fail target;
- change the pooled estimator to reproduce it.

Current Seed8192 weight remains unresolved until authorized calibration-only execution is later completed and validated.

## Validation authority

Training/evaluation is not allowed.

Allowed validation is static/unit-test validation only.

Minimum required validation:

```text
python -m pytest tests/test_reason_router_p3w1_calibration.py -q
git diff --check
```

If the focused test imports or exercises production calibration helpers, that is allowed as unit/static validation only.

No model loading, CUDA forward, calibration execution, training, dev evaluation, or artifact production is authorized.

The implementer must also report an exact grep/static audit showing no active calibration split gate in the three allowed files still requires historical split174.

Historical comments/tests may mention `174` only when explicitly asserting that old split174 is rejected.

## High-risk verification requirement

Because this change touches split/provenance authority, an independent verifier is required after implementation.

The verifier must confirm independently:

1. exact three-file scope;
2. trainer calibration gate accepts 8192 and rejects 174;
3. aggregator expected identity accepts 8192 and rejects 174;
4. tests encode the same authority;
5. no change to loss algebra, arm, ownership, estimator, seeds, batch size, data, labels, or dev/A0 isolation;
6. no hidden widening from exact split8192 to arbitrary caller-selected split;
7. no training/evaluation was executed;
8. no commit/push occurred.

## Commit boundary

Implementation and verification must complete before any implementation commit.

This authority candidate itself may be frozen first as a report-only commit.

After implementation verification PASS, use `cm ship` and explicitly stage only the verified implementation files authorized by the then-active frozen authority.

## Stop conditions

Stop immediately if:

- governing HEAD differs before implementation begins;
- any required change falls outside the three-file whitelist;
- current trainer/aggregator semantics differ materially from this static audit;
- implementing split8192 would require altering dataset/split generation rather than only calibration binding;
- a test failure indicates a broader scientific-contract conflict;
- calibration execution or model forward would be required to validate the patch;
- unrelated local tracked changes conflict with the implementation.

## Candidate verdict

```text
PASS_READY_FOR_INDEPENDENT_SEED8192_CALIBRATION_SPLIT_REBIND_AUTHORITY_VERIFICATION
```
