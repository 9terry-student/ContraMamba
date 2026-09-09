# Seed8192 Reason-Loss Calibration Execution-Authority Specification Candidate

## Verdict and boundary

`PASS_READY_FOR_INDEPENDENT_EXECUTION_AUTHORITY_VERIFICATION`

This is the sole calibration-only execution-authority candidate for Seed8192
reason-loss calibration seeds 180/181/182. It authorizes neither execution,
normal A1/A2/A3 training, evaluation, Kaggle, commit, nor push. It becomes
execution authority only after independent verification and manual clean freeze.

Governing calibration authority:
`reports/reason_router_p3w7_seed8192_reason_loss_calibration_authority_spec_candidate.md`
at commit `4a5494df4f5e049c1673cf337d3a763064a37751`, blob
`05ef59aa6a7f94c92cc1e795d26eb3c5a83cade9`. Verified split-rebind
implementation: `47ff8d16a28a17cb3dca2104c51b4d63c67d6109`.

The Artifact Index and PRE_URP handoff are historical naming/provenance context
only. The frozen calibration authority and revised Seed8192 P4-L provenance
below supersede their older P4-L path for this narrow authority.

## Fixed identities

```text
trainer blob    = bb1639525916d99cbc4d458ba4771c277cd4d46b
aggregator blob = 418f747df0e4224bc25124d0053e235fa8bd95b9
focused test    = 723c63297338aa4d412b39cfcafb19bd7d7e798a
dataset Git blob = 2b6829bf04a1333446aac6f7c603d9178b339f36
dataset canonical/execution SHA256 = eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3
dataset semantic SHA256 = 3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b
sidecar physical SHA256 = 9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d
sidecar semantic SHA256 = 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9
split_seed = 8192; dev_ratio = 0.2; train_rows = 2880; dev_rows = 720
ordered_train_row_identity_hash = 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8
calibration_seeds = [180, 181, 182]
```

Dataset:
`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Revised sidecar:
`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl`

## Dataset representation distinction

`eb1e0614…c0d3` is the authoritative canonical repository/execution SHA256:
the exact bytes of committed Git blob `2b6829…f36`. The Windows working-copy
raw SHA256 `eedbf93cf7fc3e141c4a49511750cbe4d8b0443e7de3463ea7e77696aca2c572`
is non-authoritative representation-only evidence. It contains exactly 3600
CRLF sequences; CRLF-to-LF normalization yields `eb1e0614…c0d3` and the exact
committed blob bytes. It is not a data-lineage change and must neither replace
the canonical identity nor block this report.

At execution, raw bytes consumed must hash exactly to `eb1e0614…c0d3`.
Different execution bytes fail closed; execution must not normalize bytes or
change authority.

## Frozen calibration contract

```text
architecture=v6b_minimal; backbone=mamba; model=state-spaces/mamba-130m-hf
max_length=128; device=cuda; flag_source=controlled_heuristic
freeze_encoder=true; balanced_sampler=false; weighted_label_loss=false; class_weighting=none
measurement_arm=conditional_first_blocker; measurement_gradient_ownership=explicit_local
reason_loss_weight_placeholder=0.0; calibration_forward_batch_size=8
logical_units_per_seed=1; logical_unit_scope=COMPLETE_AUTHORITATIVE_TRAIN_SPLIT
model_mode=train; fresh_initialization=true; checkpoint_loaded=false
calibration_data_scope=TRAIN_ONLY; backward=false; optimizer_step=false
scheduler_step=false; parameter_update_count=0; dev_forward=false; external_eval=false
A0 checkpoint/prediction/logits/probabilities/metrics access=false
primary_reason_min_train_count=50
```

The trainer's `_p3w1_ordered_train_identity(train_records)` hashes every
ordered record as canonical row identity, reference pair id, and normalized
final/gold label, serialized as `row_id\\tpair_id\\tgold_label\\n` UTF-8.
That semantic construction is compatible with the frozen Seed8192 identity
`478013…c2a8`.

Output namespace (no overwrite):

```text
reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed180/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed181/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed182/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration/calibration_aggregate.json
```

## Execution-commit self-binding

After this candidate independently passes verification and is manually
committed/frozen, that resulting clean freeze commit is the sole valid
calibration execution HEAD:

```text
<CALIBRATION_EXECUTION_COMMIT>
```

No existing or predicted SHA may substitute. At execution the declared
`--reason-router-weight-calibration-execution-commit` equals observed
`git rev-parse HEAD`, and that same value is required for all units and the
aggregator's expected-execution-commit gate.

## CPU/static preflight (GPU OFF)

Run before model loading. It validates raw execution bytes without normalizing.

```bash
set -euo pipefail
E='<CALIBRATION_EXECUTION_COMMIT>'; D='reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl'; S='reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl'; P='reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json'; O='reports/reason_router_p3w7_seed8192_reason_loss_calibration'
fail(){ echo "CALIBRATION_PREFLIGHT_REJECTED:$1" >&2; exit 64; }
[[ "$E" =~ ^[0-9a-f]{40}$ && "$(git rev-parse HEAD)" == "$E" ]] || fail HEAD_MISMATCH
[[ -z "$(git status --short --untracked-files=no)" && -z "$(git diff --cached --name-status)" ]] || fail DIRTY_TRACKED_OR_INDEX
[[ "$(git rev-parse HEAD:reports/reason_router_p3w7_seed8192_reason_loss_calibration_authority_spec_candidate.md)" == 05ef59aa6a7f94c92cc1e795d26eb3c5a83cade9 ]] || fail GOVERNING_AUTHORITY_BLOB
[[ "$(git rev-parse HEAD:scripts/train_controlled_v6b_minimal.py)" == bb1639525916d99cbc4d458ba4771c277cd4d46b && "$(git rev-parse HEAD:scripts/aggregate_reason_router_p3w1_calibration.py)" == 418f747df0e4224bc25124d0053e235fa8bd95b9 && "$(git rev-parse HEAD:tests/test_reason_router_p3w1_calibration.py)" == 723c63297338aa4d412b39cfcafb19bd7d7e798a ]] || fail SOURCE_BLOB
[[ "$(git rev-parse HEAD:"$D")" == 2b6829bf04a1333446aac6f7c603d9178b339f36 && "$(sha256sum "$D" | awk '{print $1}')" == eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3 ]] || fail DATASET_IDENTITY
[[ "$(sha256sum "$S" | awk '{print $1}')" == 9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d && -f "$P" ]] || fail SIDECAR_PHYSICAL
grep -Fq '"source_dataset_semantic_sha256": "3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"' "$P" && grep -Fq '"sidecar_semantic_sha256": "2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9"' "$P" || fail SEMANTIC_PROVENANCE
for p in "$O" "$O/seed180/calibration_unit.json" "$O/seed181/calibration_unit.json" "$O/seed182/calibration_unit.json" "$O/calibration_aggregate.json"; do [[ ! -e "$p" ]] || fail OUTPUT_EXISTS; done
```

## Exact trainer CLI templates (GPU ON only for these forwards)

Run in this order only: seed180 then validate its artifact; seed181 then
validate; seed182 then validate. These are existing CLI options; normal
training/checkpoint/prediction/dev/A0 options are absent.

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 180 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed180/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 181 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed181/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 182 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed182/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

## Artifact validation, aggregation, and scientific boundary

Artifact/provenance validation is GPU OFF and rejects a unit unless its fixed
identities, self-bound execution commit, train-only/no-dev/no-A0 booleans,
no-step counters, ordered identity/count, minimum primary-class counts, and
finite positive losses pass. It accesses no model, checkpoint, predictions,
logits, probabilities, metrics, or dev data.

The pure-JSON aggregator imports no model code or torch. Only after all three
valid units exist, run it with GPU OFF:

```bash
python scripts/aggregate_reason_router_p3w1_calibration.py --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed180/calibration_unit.json --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed181/calibration_unit.json --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed182/calibration_unit.json --output-json reports/reason_router_p3w7_seed8192_reason_loss_calibration/calibration_aggregate.json --expected-execution-commit <CALIBRATION_EXECUTION_COMMIT> --expected-dataset-sha256 eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3 --expected-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --expected-split-seed 8192 --expected-ordered-train-row-count 2880 --expected-ordered-train-row-identity-hash 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8 --expected-dev-ratio 0.2
```

```text
mu_final = sum_s(n_final[s] * L_final[s]) / sum_s(n_final[s])
mu_reason = sum_s(n_reason[s] * L_reason[s]) / sum_s(n_reason[s])
resolved_reason_loss_weight = mu_final / mu_reason
```

One common A1/A3 weight only: no seed-specific weights, mean seed ratios,
unweighted seed means, dev selection, A0 performance selection, or reuse of
historical split174 value `0.6518018402446165`.

Fail closed on every specified identity/semantic/split/count/seed/destination
mismatch; differing unit commits; dev/A0/checkpoint access; backward,
optimizer/scheduler step, parameter update; nonfinite/nonpositive loss;
incomplete seeds; and historical-weight reuse.

Successful calibration establishes only one provenance-valid common A1/A3
`reason_loss_weight`. It establishes no A1/A3 performance or superiority,
reason-supervision or gradient-ownership causal benefit, factorial promotion,
or normal A1/A2/A3 execution authority.

## Non-execution declaration

```text
training_executed = false
evaluation_executed = false
calibration_executed = false
model_loaded = false
staged = false
commit = false
push = false
```
