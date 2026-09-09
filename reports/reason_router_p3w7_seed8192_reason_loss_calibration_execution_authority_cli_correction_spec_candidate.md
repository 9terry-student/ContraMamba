# Seed8192 Reason-Loss Calibration Execution-Authority CLI Correction Candidate

## Verdict and narrow supersession

`PASS_READY_FOR_INDEPENDENT_CLI_CORRECTION_AUTHORITY_VERIFICATION`

This report supersedes **only** the exact trainer CLI templates and their
activation validity in the frozen defective execution authority.  It does not
alter any other contract, identity, dataset/sidecar rule, split semantic,
output namespace, estimator, aggregation semantic, scientific boundary,
fail-closed condition, GPU boundary, or execution ordering.  Those unaffected
terms are inherited unchanged from the parent defective authority.

Parent defective authority: commit
`1cfc41ddf70a716954fbb2bdb824e1af5ac8139c`; frozen report blob
`dae1a848d709f8cf113a212fde6fdf79f95058bb`.

Governing calibration authority: commit
`4a5494df4f5e049c1673cf337d3a763064a37751`, blob
`05ef59aa6a7f94c92cc1e795d26eb3c5a83cade9`.  Verified implementation commit:
`47ff8d16a28a17cb3dca2104c51b4d63c67d6109`.

## Preserved fixed identities and scope

```text
trainer blob = bb1639525916d99cbc4d458ba4771c277cd4d46b
aggregator blob = 418f747df0e4224bc25124d0053e235fa8bd95b9
focused test blob = 723c63297338aa4d412b39cfcafb19bd7d7e798a
dataset Git blob = 2b6829bf04a1333446aac6f7c603d9178b339f36
dataset canonical/execution SHA256 = eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3
dataset semantic SHA256 = 3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b
sidecar physical SHA256 = 9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d
sidecar semantic SHA256 = 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9
split_seed = 8192; dev_ratio = 0.2; train_rows = 2880; dev_rows = 720
ordered_train_row_identity_hash = 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8
calibration_seeds = [180, 181, 182]
```

The inherited no-overwrite namespace remains exactly:

```text
reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed180/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed181/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed182/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration/calibration_aggregate.json
```

## Historical failed-run provenance and defect

```text
run_name = p3w7-seed8192-reason-calibration-seed180-retry1
execution commit = 1cfc41ddf70a716954fbb2bdb824e1af5ac8139c
command SHA256 = 34cf440d4a3c07c7bbcf478f659e9290dd58300639fbd3e8cb921b6d9154ab24
started UTC = 2026-09-09T15:00:00Z
finished UTC = 2026-09-09T15:00:06Z
exit code = 2
observed error = P2_INCOMPATIBLE_OPTION: option=stage174c_clean_polarity_preservation_weight value=1.0
```

The failed frozen command omitted
`--stage174c-clean-polarity-preservation-weight`.  In trainer blob
`bb1639525916d99cbc4d458ba4771c277cd4d46b`, this is a v6b parser option with
default `1.0`; P2/A3 objective fail-closed validation requires `0.0`.  Static
failure-recovery audit found this to be the sole latent non-neutral default
conflicting with the frozen calibration CLI.  No trainer implementation change
is required.  The exact and only CLI correction is:

```text
--stage174c-clean-polarity-preservation-weight 0.0
```

The retry1 failure occurred before tokenizer/model construction and before a
CUDA calibration forward.  No valid calibration measurement occurred; no
seed180 calibration unit is accepted as scientific evidence; there is no
execution-success, artifact/provenance-validity, or scientific-conclusion
claim.  The run must not be reused as a successful calibration run.  Its
command identity is historical failure provenance only.

## Corrected exact trainer CLI templates

Each command below is byte-for-token identical in arguments to the
corresponding frozen template except for exactly the explicit added
`--stage174c-clean-polarity-preservation-weight 0.0` pair.  The execution
commit placeholder is intentionally not the old commit.

### Seed 180

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 180 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed180/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

### Seed 181

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 181 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed181/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

### Seed 182

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 182 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed182/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

## Execution-commit self-binding

1. `1cfc41ddf70a716954fbb2bdb824e1af5ac8139c` remains provenance only for the
   defective authority and failed retry1.
2. It is not authorized as the execution commit for a corrected CLI.
3. After this candidate independently passes verification and is manually
   frozen in a clean commit, that resulting full 40-character freeze commit is
   the sole valid `<CALIBRATION_EXECUTION_COMMIT>`.
4. All three units must declare exactly that same commit with
   `--reason-router-weight-calibration-execution-commit`.
5. The unchanged aggregator expected-execution-commit gate must use that same
   correction-authority freeze commit.
6. A command from the old commit must not be reused merely by manually
   appending the corrected flag.

## Mandatory CPU/static and parser/helper preflight (GPU OFF)

Preserve the inherited CPU/static preflight unchanged.  Before any GPU
execution, additionally parse each corrected seed command through production
`build_parser()` and exercise `_p2_resolve_arm_contract` plus
`_p3w1_validate_calibration_only_args`.  This pre-model check must run with
GPU off and must not construct a tokenizer or model, load CUDA, or perform a
CUDA forward.  It must require resolved
`stage174c_clean_polarity_preservation_weight == 0.0`, A3,
`conditional_first_blocker`, `explicit_local`, calibration exception
`reason_loss_weight=0.0`, `split_seed=8192`, and `dev_ratio=0.2`; it fails
closed on any P2 incompatible option or any additional incompatible default.

The static failure-recovery audit performed that parser/helper-only validation
for all three corrected token sets, substituting a syntactically valid neutral
40-hex value only for the future commit placeholder and nonexisting temporary
export destinations.  It passed parser acceptance, P2 incompatible-option
validation, A3 resolution, conditional-first-blocker and explicit-local
resolution, the calibration-only zero-weight exception, split/dev checks, and
calibration-only argument validation.  It used no tokenizer, model, or CUDA.

At the future preflight, the same safe production-source procedure is:

```text
for each template: replace <CALIBRATION_EXECUTION_COMMIT> with the frozen 40-hex commit;
parse argv with scripts.train_controlled_v6b_minimal.build_parser(); set resolved_split_seed
from split_seed as main does; call _p2_resolve_arm_contract(args, argv, parser); then call
_p3w1_validate_calibration_only_args(args).  Do not invoke trainer main().
```

No helper is invented and no source modification is authorized for this
preflight.

## Preserved order, aggregation, names, and boundaries

The sole permitted future order remains: CPU/static preflight (GPU OFF) ->
seed180 corrected unit (GPU ON) -> validate seed180 (GPU OFF) -> seed181
corrected unit (GPU ON) -> validate seed181 (GPU OFF) -> seed182 corrected
unit (GPU ON) -> validate seed182 (GPU OFF) -> pure-JSON aggregation (GPU
OFF).  No normal A1/A2/A3 training is authorized.

The failed name `p3w7-seed8192-reason-calibration-seed180-retry1` must not be
reused.  After correction-authority freeze, the first corrected seed180 run
should use `p3w7-seed8192-reason-calibration-seed180-retry2`.  Its exact pinned
command and command SHA are generated only after the freeze commit exists;
this report authorizes no pinned Kaggle wrapper.

The aggregator semantics and command remain unchanged, except that its
`--expected-execution-commit` must bind to the future correction-authority
freeze commit.  The historical split174 calibration weight
`0.6518018402446165` remains forbidden: this is CLI activation repair only,
not an estimator change or authorization to reuse a historical weight.

Successful corrected calibration can establish only a provenance-valid common
reason-loss weight for A1/A3 under the frozen Seed8192 calibration contract.
It does not establish improved performance, A1/A3 superiority, causal
mechanism, promotion, factorial authority, or normal training/evaluation
authority.

## Fail-closed additions and non-execution declaration

In addition to inherited fail-closed conditions, reject omission of the zero
flag; any resolved nonzero
`stage174c_clean_polarity_preservation_weight`; reuse of retry1 command SHA;
use of `1cfc41ddf70a716954fbb2bdb824e1af5ac8139c` as corrected execution
commit; a corrected runtime command not exactly authorized by the newly frozen
correction authority; or a parser/helper preflight that finds any additional
P2 incompatible option.

```text
files_created = 1
existing_files_modified = 0
training_executed = false
evaluation_executed = false
calibration_executed = false
model_loaded = false
tokenizer_loaded = false
cuda_forward = false
kaggle_used = false
staged = false
commit = false
push = false
```
