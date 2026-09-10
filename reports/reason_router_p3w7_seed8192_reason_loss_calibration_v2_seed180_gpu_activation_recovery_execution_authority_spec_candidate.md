# Seed8192 Calibration-V2 Seed180 GPU-Activation Recovery Execution Authority Specification Candidate

## Candidate status, frozen retry3 provenance, and sole permitted delta

```text
CANDIDATE_ONLY — NO EXECUTION AUTHORITY YET
authority HEAD = 510d29a9267832dcae521e0c419c2ae0a95c575f
frozen calibration-v2 execution authority = 510d29a9267832dcae521e0c419c2ae0a95c575f
```

This report authorizes no execution until it is independently verified, manually committed, and pushed. Its only semantic deltas from the frozen calibration-v2 authority are: immutable disposition of the consumed Seed180 retry3 environment failure; the fresh Seed180 retry4 run name; and re-self-binding every future accepted calibration-v2 unit and aggregate to the resulting recovery freeze commit. No scientific or calibration semantic changes are authorized.

The following retry3 provenance is immutable and historical:

```text
run_name = p3w7-seed8192-reason-calibration-v2-seed180-retry3
run wrapper expected commit = 510d29a9267832dcae521e0c419c2ae0a95c575f
command SHA256 = 5ddb7efb69824e9d0960339e292e3d64839237e0152348e8ae9958ee609918cc
command bytes = 1415
STARTED_UTC = 2026-09-10T01:56:30Z
FINISHED_UTC = 2026-09-10T01:57:12Z
EXIT_CODE = 1
command file SHA256 independently observed = 5ddb7efb69824e9d0960339e292e3d64839237e0152348e8ae9958ee609918cc
classification = SEED180_RETRY3_GPU_ACTIVATION_ENVIRONMENT_FAILURE
```

The authorized `--device cuda` command ran while Kaggle GPU was OFF. The terminal exception was `AssertionError: Torch not compiled with CUDA enabled`, at `scripts/train_controlled_v6b_minimal.py main()` → `train_controlled_v5.move_inputs(...)` → `value.to(device)` → `torch.cuda._lazy_init()`. This is neither a trainer implementation defect nor a command-authority defect.

```text
retry3 command integrity = PASS
retry3 wrapper commit identity = PASS
retry3 process execution = FAIL
retry3 exit code = 1
retry3 calibration artifact = ABSENT
retry3 calibration measurement = NONE / NOT ACCEPTED
retry3 scientific evidence = NONE
retry3 provenance namespace = CONSUMED
retry3 namespace reuse = FORBIDDEN
retry3 log/meta/command/start-marker mutation = FORBIDDEN
retry3 deletion/recreation = FORBIDDEN
retry3 collection/import = NOT AUTHORIZED
```

Do not alter collector recovery or retry3 provenance.

## Fresh state and run namespace preflight

At report creation, the following target is absent, including no tracked file at that location:

```text
reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json
SEED180_V2_OUTPUT_ABSENT=true
```

Because retry3 created no unit artifact, retry4 may use that same frozen Seed180 v2 output path. It may not create another scientific output namespace. Seed181, Seed182, and aggregate output paths remain unchanged.

The corrected controller registry was read only. These exact names were unused at report creation:

```text
RUN_NAME_RETRY4_UNUSED=true
p3w7-seed8192-reason-calibration-v2-seed181 UNUSED=true
p3w7-seed8192-reason-calibration-v2-seed182 UNUSED=true
p3w7-seed8192-reason-calibration-v2-aggregate UNUSED=true
```

The sole fresh Seed180 name is `p3w7-seed8192-reason-calibration-v2-seed180-retry4`. Retry3 remains consumed and forbidden to reuse. The unchanged remaining names are `p3w7-seed8192-reason-calibration-v2-seed181`, `p3w7-seed8192-reason-calibration-v2-seed182`, and `p3w7-seed8192-reason-calibration-v2-aggregate`.

## Recovery self-binding and unchanged scientific contract

After independent verification, manual commit, and push, the resulting full 40-character recovery freeze commit becomes the new and sole `<CALIBRATION_EXECUTION_COMMIT>`. It supersedes `510d29a9267832dcae521e0c419c2ae0a95c575f` only as the execution-commit binding for future accepted units.

Seed180 retry4, Seed181, Seed182, and the aggregate must all declare and validate the same resulting recovery freeze commit. Binding Seed180 to the new commit while Seed181/Seed182 remain on `510d29a9267832dcae521e0c419c2ae0a95c575f` is forbidden. Historical retry3 remains bound to `510d29a9267832dcae521e0c419c2ae0a95c575f` and is never promoted.

The governing controller remains `C:\Users\Home1\.contramamba\cm.ps1`, with bytes `91954` and SHA256 `d619329478197bee866b91ca95bf52d26dcb8500f350449e3f27e60f6f40800e`. Verify both before every run and collect; mismatch is `BLOCKED`.

The unchanged calibration contract is:

```text
unit schema = reason_router_p3w1_calibration_unit_v2
aggregate schema = reason_router_p3w1_calibration_aggregate_v2
P4-X = 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8
P3-W1 = 4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
split_seed=8192; dev_ratio=0.2; train_rows=2880; dev_rows=720
seeds=[180,181,182]
A3; conditional_first_blocker; explicit_local
reason_loss_weight=0.0 calibration placeholder
stage174c_clean_polarity_preservation_weight=0.0
forward batch size=8
```

No backward, optimizer, scheduler, dev/external/OOD evaluation, checkpoint load, or normal factorial training is authorized. Historical split174 weight `0.6518018402446165` remains forbidden.

## Exact future CUDA unit command templates

Replace `<CALIBRATION_EXECUTION_COMMIT>` only after the recovery candidate is independently verified, manually committed, and pushed. The three commands differ only by seed and export path after that common substitution.

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 180 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 181 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed181/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 182 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed182/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

Using neutral valid substitute `0000000000000000000000000000000000000000` only for static checking, current production `build_parser()`, `_p2_resolve_arm_contract(...)`, and `_p3w1_validate_calibration_only_args(...)` passed for every listed unit command. No `main()`, CUDA, model, tokenizer, output creation, or calibration ran.

```text
SEED180_RETRY4_PARSER_HELPER_PREFLIGHT=PASS
SEED181_PARSER_HELPER_PREFLIGHT=PASS
SEED182_PARSER_HELPER_PREFLIGHT=PASS
```

## Exact aggregate command and parser-only preflight

Run only after all three accepted units have been imported and validated, with GPU OFF:

```bash
python scripts/aggregate_reason_router_p3w1_calibration.py --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed181/calibration_unit.json --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed182/calibration_unit.json --output-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/calibration_aggregate.json --expected-execution-commit <CALIBRATION_EXECUTION_COMMIT> --expected-dataset-sha256 eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3 --expected-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --expected-split-seed 8192 --expected-ordered-train-row-count 2880 --expected-p4x-ordered-train-row-sha256 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8 --expected-p3w1-ordered-train-row-label-sha256 4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b --expected-dev-ratio 0.2
```

With the same neutral 40-hex substitute, parser-only validation passed without `main()`, aggregate construction, unit reads, or output creation:

```text
AGGREGATE_PARSER_PREFLIGHT=PASS
```

## GPU activation, collection, and sequence gates

Before every CUDA unit invocation, Kaggle GPU must be ON and this non-scientific environment check (or a semantic equivalent) must pass:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Require `torch.cuda.is_available() == true`. If false, STOP before the cm-generated pinned run cell; do not consume the registered run namespace. This check loads no ContraMamba model and writes no scientific artifact. Immediately after each actual unit process ends, GPU must be OFF before collection, import, and validation.

The frozen recovery order is:

```text
GPU OFF: final local/controller/registry/output preflight
Kaggle GPU ON → CUDA availability PASS → Seed180 retry4 pinned cell → GPU OFF → collect/import/unit-v2 validation
only if accepted: GPU ON → CUDA availability PASS → Seed181 pinned cell → GPU OFF → collect/import/unit-v2 validation
only if accepted: GPU ON → CUDA availability PASS → Seed182 pinned cell → GPU OFF → collect/import/unit-v2 validation
only if accepted: GPU OFF → aggregate → aggregate validation/import
```

For each accepted unit, require process exit `0`; `FILE_COUNT >= 1`; manifest contains the exact expected unit path; v2 schema; correct seed; common recovery execution commit; exact P4-X and P3-W1; and exact split/train/dev identities. Re-verify controller bytes and SHA before every run and collect. No retry3 collection is authorized. `FILES=0` is rejected.

## Scientific boundary and candidate-task boundary

Successful recovery establishes only the same claim as the frozen authority: a provenance-valid common Seed8192 reason-loss calibration weight for A1/A3. It establishes no performance, superiority, causality, promotion, or factorial-execution authority.

While authoring this candidate: no existing file was edited; no staging, commit, push, Kaggle, CUDA, model/tokenizer, training, evaluation, calibration, aggregation, retry3 mutation, controller mutation, or registry mutation occurred. The two pre-existing untracked A0 roots remain untouched.

```text
PASS_READY_FOR_INDEPENDENT_SEED180_GPU_ACTIVATION_RECOVERY_AUTHORITY_VERIFICATION
```
