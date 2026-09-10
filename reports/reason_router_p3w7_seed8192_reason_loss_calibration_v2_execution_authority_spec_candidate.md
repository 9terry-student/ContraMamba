# Seed8192 Reason-Loss Calibration V2 Execution Authority Specification Candidate

## Candidate status and binding rule

```text
CANDIDATE_ONLY — NO EXECUTION AUTHORITY YET
repository HEAD inspected: 0360a830012e03e25b2dda097ef2190fc7d6056e
frozen dual-identity implementation: ee0ddd3154f1dceba79683ded152a1c23610ea3d
frozen collector correction: 4d0b55e5258e7b6adf6f39c82aeb3c7db3c72ed7
frozen collector implementation: 0360a830012e03e25b2dda097ef2190fc7d6056e
```

This candidate becomes an execution authority only after independent verification, a manual freeze commit, and push.  The resulting full 40-character commit is the sole `<CALIBRATION_EXECUTION_COMMIT>`.  It is self-binding: every unit command declares it, the Kaggle repository `HEAD` must equal it, the controller registry entry must be saved at it, and the aggregate command must expect it.  Do not bind execution to `ee0ddd3154f1dceba79683ded152a1c23610ea3d`, `0360a830012e03e25b2dda097ef2190fc7d6056e`, or `a44c6394323da14b423654a88a11a9d0ed3507f6`.

The governing controller is `C:\Users\Home1\.contramamba\cm.ps1`, frozen at 91,954 bytes and SHA256 `d619329478197bee866b91ca95bf52d26dcb8500f350449e3f27e60f6f40800e`.  Immediately before **each** `cm run <name>` and **each** `cm collect <name>`, locally recompute both values.  Any mismatch is `BLOCKED`; do not save, run, collect, import, recreate a marker, or alter provenance.

## Frozen implementation and scientific contract

Current production contracts inspected at `HEAD` are:

```text
trainer:    scripts/train_controlled_v6b_minimal.py
unit schema: reason_router_p3w1_calibration_unit_v2
aggregator: scripts/aggregate_reason_router_p3w1_calibration.py
aggregate:   reason_router_p3w1_calibration_aggregate_v2
```

The trainer's current calibration flags are `--reason-router-weight-calibration-export`, `--reason-router-weight-calibration-execution-commit`, and `--reason-router-weight-calibration-forward-batch-size`.  The aggregate parser's current input flag is repeatable `--unit-json`; its output flag is `--output-json`.

All three units freeze: `architecture=v6b_minimal`; `backbone=mamba`; `model_name=state-spaces/mamba-130m-hf`; `freeze_encoder=true`; `frame_downstream_gradient_mode=joint`; `max_length=128`; A3; `conditional_first_blocker`; `explicit_local`; `reason_router_epsilon=1e-8`; `reason_min_train_count=50`; `ranking_weight=0.0`; `class_weighting=none`; placeholder `reason_loss_weight=0.0`; `stage174c_clean_polarity_preservation_weight=0.0`; `flag_source=controlled_heuristic`; and calibration forward batch size 8.

The frozen data binding is:

```text
data = reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl
data Git blob = 2b6829bf04a1333446aac6f7c603d9178b339f36
data canonical SHA256 = eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3
sidecar = reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl
sidecar semantic SHA256 = 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9
split_seed = 8192; dev_ratio = 0.2; train_rows = 2880; dev_rows = 720
P4-X = 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8
P3-W1 = 4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
P4-X != P3-W1 (required)
```

Calibration is complete authoritative train split, forward measurement only: no backward, optimizer, scheduler, dev/external/OOD evaluation, checkpoint loading, or normal A1/A2/A3 outputs.  The historical split174 weight `0.6518018402446165` is forbidden.

## Fresh namespace and single-use names

The following no-overwrite namespace is fresh evidence only:

```text
reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed181/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed182/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/calibration_aggregate.json
```

No historical calibration artifact may be copied, moved, promoted, rewritten, or accepted as a unit there.  Each of these run names is single-use under the corrected controller semantics, and its provenance namespace may never be reused after creation:

```text
p3w7-seed8192-reason-calibration-v2-seed180-retry3
p3w7-seed8192-reason-calibration-v2-seed181
p3w7-seed8192-reason-calibration-v2-seed182
p3w7-seed8192-reason-calibration-v2-aggregate
```

Historical retry2 (`p3w7-seed8192-reason-calibration-seed180-retry2`, execution commit `a44c6394323da14b423654a88a11a9d0ed3507f6`, command SHA256 `5cbba8dba815fc50aef822099d0a678f37ebe1f456cca7dfb6b7b4b676c2fe06`) remains historical only: process PASS, v1 schema, corrected provenance INVALID, measurement NOT ACCEPTED.  Mutation, marker recreation, v1-to-v2 promotion, and `FILES=0` ZIP acceptance are forbidden.

## Exact unit commands

After self-binding, save each exact one-line command using the corrected controller's byte-exact `cm run save <name>` flow, with the controller hash checked first.  The only permitted differences among these commands are `--seed` and the export path.

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 180 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 181 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed181/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --max-length 128 --dev-ratio 0.2 --seed 182 --split-seed 8192 --device cuda --flag-source controlled_heuristic --reason-router-epsilon 1e-8 --reason-min-train-count 50 --ranking-weight 0.0 --class-weighting none --reason-router-arm A3 --reason-router-mode conditional_first_blocker --gradient-ownership-mode explicit_local --reason-loss-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --reason-router-weight-calibration-export reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed182/calibration_unit.json --reason-router-weight-calibration-execution-commit <CALIBRATION_EXECUTION_COMMIT> --reason-router-weight-calibration-forward-batch-size 8
```

## CPU/static preflight performed for this candidate

Parser and helper-only preflight was performed without trainer `main()`, model/tokenizer construction, CUDA use, calibration, or artifact writes.  For each seed's exact argv template, `<CALIBRATION_EXECUTION_COMMIT>` was replaced only for this static check by `0000000000000000000000000000000000000000`; production `build_parser()`, `_p2_resolve_arm_contract(...)`, and `_p3w1_validate_calibration_only_args(...)` returned PASS.  Each resolved A3, `conditional_first_blocker`, `explicit_local`, `reason_loss_weight=0.0` via the calibration exception, stage174c weight 0.0, split seed 8192, dev ratio 0.2, and forward batch size 8.  No latent incompatible default reached either helper.

## Exact pure-JSON aggregate command

Run this only after all three imported units pass, with GPU OFF.  It imports no model or torch code and writes no output during parser validation.

```bash
python scripts/aggregate_reason_router_p3w1_calibration.py --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed181/calibration_unit.json --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed182/calibration_unit.json --output-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/calibration_aggregate.json --expected-execution-commit <CALIBRATION_EXECUTION_COMMIT> --expected-dataset-sha256 eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3 --expected-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --expected-split-seed 8192 --expected-ordered-train-row-count 2880 --expected-p4x-ordered-train-row-sha256 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8 --expected-p3w1-ordered-train-row-label-sha256 4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b --expected-dev-ratio 0.2
```

The aggregate parser-only validation of that exact argv, again using the neutral 40-hex placeholder, passed.  It did not call `build_aggregate`, read units, or create the output.

## Mandatory sequential gates

1. CPU/static final preflight with GPU OFF.  Verify clean tracked/index state, this candidate's eventual self-binding commit, the controller bytes/SHA, all command bytes/SHA values after substituting the resulting commit, and all four fresh output paths absent.
2. GPU ON: execute only Seed180 retry3.  GPU OFF immediately after exit.  Advance only on process exit 0.  Re-verify controller SHA, `cm collect p3w7-seed8192-reason-calibration-v2-seed180-retry3`, then import the produced ZIP without modifying it.  Require `FILE_COUNT >= 1` and a manifest entry for exactly `reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json`; reject zero files, absent path, wrong run name, execution commit, command SHA, or collector/controller provenance.
3. Validate the imported Seed180 unit-v2 before any Seed181 action.  Require its v2 schema; seed 180; split seed 8192; dev ratio 0.2; ordered train rows 2880; the exact distinct P4-X and P3-W1 identities; execution commit `<CALIBRATION_EXECUTION_COMMIT>`; and every current frozen validator field, including `model_mode`, measurement-contract, no-grad, no-backward/no-optimizer/no-scheduler, train-only, no-dev/no-external/no-checkpoint, and finite-loss fields.
4. Only if Seed180 passes, repeat the same GPU-on/run then GPU-off/collect/import/unit-validation sequence for Seed181.  Its manifest must contain exactly the Seed181 fresh path and its unit must have seed 181.
5. Only if Seed181 passes, repeat the same sequence for Seed182.  Its manifest must contain exactly the Seed182 fresh path and its unit must have seed 182.
6. Only if all units pass, keep GPU OFF, run the pure-JSON aggregate, then validate/import it.  Any failure is STOP; never proceed to a later seed from an invalid unit.

For every collection, the corrected controller's manifest must agree with the local registry authorization on run name, expected/actual commit, command SHA, exit code 0, and the artifact list.  `FILES=0` is rejected even if a ZIP exists.  Import is allowed only after those checks and must preserve the fresh no-overwrite namespace.

Aggregate acceptance requires schema `reason_router_p3w1_calibration_aggregate_v2`; seeds exactly `[180, 181, 182]`; one common execution commit, dataset identity, sidecar identity, split, row count, P4-X identity, and P3-W1 identity; and finite positive `resolved_reason_loss_weight`.  Its estimator is exclusively count weighted:

```text
mu_final  = sum_s(n_final[s]  * ell_final[s])  / sum_s(n_final[s])
mu_reason = sum_s(n_reason[s] * ell_reason[s]) / sum_s(n_reason[s])
resolved_reason_loss_weight = mu_final / mu_reason
```

No mean-of-means substitution is allowed.

## Scientific boundary

A successful chain establishes only a provenance-valid Seed8192 common reason-loss calibration weight for A1/A3 under this frozen measurement contract.  It establishes no improved model quality, A1/A3 superiority, causal mechanism, promotion, normal factorial execution authority, or test/dev/OOD performance claim.  This authority does not authorize normal A1/A2/A3 training.

## Candidate-task boundary

No calibration, training, evaluation, CUDA, model/tokenizer loading, Kaggle activity, stage execution, commit, push, staging, or controller mutation was performed while authoring this candidate.  The two pre-existing untracked A0 roots remain untouched.

Expected final verification verdict after this candidate is independently checked:

```text
PASS_READY_FOR_INDEPENDENT_SEED8192_CALIBRATION_V2_EXECUTION_AUTHORITY_VERIFICATION
```
