# Seed8192 Revised-Split A0 N=3 Clean-Replacement Execution-Authority Candidate

## Verdict, scope, and authoring boundary

This is the single report-only candidate for the frozen Seed8192 revised-split A0 N=3 clean-replacement execution.  It creates no collector, recovery framework, validator family, or `cm.ps1` change.  It preserves all existing recovery artifacts and reports as historical records.

Candidate path: `reports/reason_router_p3w7_seed8192_a0_n3_clean_replacement_execution_authority_spec_candidate.md`.

At authoring time: `CANDIDATE_ONLY`.

This report authorizes no training, evaluation, Kaggle work, packaging execution, staging, commit, push, or modification of an existing file.  It does not authorize calibration or A1/A2/A3.  Training/Evaluation allowed: `NO`.  Kaggle allowed: `NO`.  Commit/Push: `NO`.

## Authority reconciliation and historical disposition

Commit `abd85a088c274678004432160625d42208112848` has subject `Activate Seed8192 revised-split A0 execution authority`, but its report body explicitly says it is report-only and that execution occurs only after a separate activation.  Therefore:

```text
ABD85_BODY_LEVEL_AUTHORITY = CANDIDATE_ONLY
COMMIT_MESSAGE_ALONE_DOES_NOT_AUTHORIZE_EXECUTION = TRUE
```

The completed r2 run is classified only as controller-directed historical evidence:

```text
SEED180_R2_EXECUTION_SUCCESS = ESTABLISHED
SEED180_R2_EXECUTION_DISPOSITION = VALID_BY_SEPARATE_CONTROLLER_ACTIVATION
SEED180_R2_STANDARD_CM_WRAPPER_PROVENANCE = INCOMPLETE
SEED180_R2_PRIMARY_SEED8192_A0_ADMISSION = EXCLUDED
SEED180_R2_DISPOSITION = HISTORICAL_PROVENANCE_INCOMPLETE_EVIDENCE_ONLY
SCIENTIFIC_CONCLUSION_FROM_R2 = NONE
```

This is not a training-failure finding.  Its artifacts are neither deleted nor mutated.

The recovery bridge at `4d26d5601b10714d0158049357d600901274a054` remains in history and is not reverted or deleted.  Its proposed collector/package path is not implemented:

```text
WRAPPER_LOSS_RECOVERY_BRIDGE = PRESERVED_HISTORICALLY
FURTHER_RECOVERY_IMPLEMENTATION = CANCELLED_BY_CURRENT_CONTROLLER_DIRECTION
```

## Body-level execution-authority rule

This candidate avoids the `abd85` subject/body ambiguity.  If and only if, for these exact candidate bytes, all of the following complete: independent high-risk verification `PASS`; exact byte/blob identity freeze; dedicated commit; push; and remote commit/blob authentication, then **the frozen commit itself is the execution authority**.

No additional activation commit is required, and no commit-subject inference is permitted.  The authorization derives from this report body.  Execution remains conditional on every runtime gate in this report.

## Exact primary matrix, sequence, namespaces, and run names

The complete and exclusive primary Seed8192 revised A0 N=3 membership is:

1. `seed180 REPLACEMENT_R1`
2. `seed181`
3. `seed182`

Execution is strictly sequential: seed180 replacement R1, successful handoff/import validation, seed181, successful handoff/import validation, seed182, successful handoff/import validation.  Stop after any failure.  No seed is silently retried; a later retry requires controller authorization and must not pre-create another recovery-authority family.

| Member | Exact output namespace | Frozen run name |
| --- | --- | --- |
| seed180 replacement R1 | `reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0` | `p3w7-seed8192-a0-seed180-replacement-r1` |
| seed181 | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed181/A0` | `p3w7-seed8192-a0-seed181-r1` |
| seed182 | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed182/A0` | `p3w7-seed8192-a0-seed182-r1` |

The original r2 namespace, `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0`, remains untouched.  The name `p3w7-seed8192-a0-seed180-r2` is not reused.

## Frozen scientific inputs, A0 semantics, and split contract

| Binding | Exact value |
| --- | --- |
| Dataset path | `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` |
| Dataset Git blob | `2b6829bf04a1333446aac6f7c603d9178b339f36` |
| Dataset canonical SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| Dataset semantic SHA256 | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| Sidecar path | `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl` |
| Sidecar semantic SHA256 | `2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9` |
| Sidecar physical SHA256 | `9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d` |
| Sidecar provenance physical SHA256 | `170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8` |

```text
architecture = v6b_minimal
backbone = mamba
model = state-spaces/mamba-130m-hf
freeze_encoder = true
frame_downstream_gradient_mode = joint
reason_router_arm = A0
reason_router_mode = explicit_product
gradient_ownership_mode = joint
reason_loss_weight = 0.0
A0_REFERENCE_PREDICTIONS_CONSUMED = NONE
```

No historical split174 A0 prediction is consumed.

| Split field | Exact value |
| --- | --- |
| Split seed / dev ratio | `8192` / `0.2` |
| Pairs: total / train / dev | `300` / `240` / `60` |
| Rows: train / dev | `2880` / `720` |
| Pair universe SHA256 | `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2` |
| Shuffled pair SHA256 | `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55` |
| Train pair SHA256 | `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049` |
| Dev pair SHA256 | `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4` |
| Ordered train rows SHA256 | `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8` |
| Ordered dev rows SHA256 | `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4` |

## Exact training command contract

The following is exact, with only `SEED` and `OUTDIR` substituted:

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --epochs 20 --max-length 128 --dev-ratio 0.2 --seed SEED --split-seed 8192 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --ranking-weight 0.0 --class-weighting none --stage174c-clean-pairwise-mode off --stage174c-clean-pairwise-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --stage175b-support-anchor-mode off --stage175b-support-anchor-weight 0.0 --stage177c-frame-pairwise-mode off --stage177c-frame-pairwise-weight 0.0 --compatible-positive-margin-logit 0.0 --compatible-positive-margin-weight 0.0 --lr 0.001 --reason-router-arm A0 --reason-router-mode explicit_product --gradient-ownership-mode joint --reason-loss-weight 0.0 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --save-selected-checkpoint --selected-checkpoint-filename selected_checkpoint.pt --output-json OUTDIR/training_report.json --output-predictions-json OUTDIR/clean_dev_predictions.json
```

```text
seed180 replacement: SEED=180
OUTDIR=reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0

seed181: SEED=181
OUTDIR=reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed181/A0

seed182: SEED=182
OUTDIR=reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed182/A0
```

## Five-output and no-collision contract

Before each run, its output namespace must be absent.  No overwrite, resume, or collision is permitted.  Exactly these scientific artifacts are required:

```text
training_report.json
training_report_predictions.jsonl
clean_dev_predictions.json
run_provenance.json
selected_checkpoint.pt
```

## Fixed-path v3 handoff contract

Standard `cm run` remains the execution wrapper.  `cm.ps1` remains unchanged, and no generic `cm collect` patch is authorized.  After each successful run, do not use standard `cm collect` recursive filesystem-start-marker/find-newer/mtime discovery.

Instead, use one reusable, parameterized, CPU-only fixed-path v3 packaging protocol for all three runs.  It reads only the authentic wrapper files generated by that exact `cm run` (`command.sh`, `run.log`, `run.meta`) and the five exact authorized scientific paths in the current run namespace.  It creates a truthful package with `schema = contramamba-handoff-v3`, compatible with the existing standard `cm import` contract, and with `artifact_discovery = fixed_authorized_paths_sha256`.  It must not claim filesystem-start-marker discovery.

The exact package membership is only:

```text
manifest.json
run.log
run.meta
command.sh
files/<OUTDIR>/training_report.json
files/<OUTDIR>/training_report_predictions.jsonl
files/<OUTDIR>/clean_dev_predictions.json
files/<OUTDIR>/run_provenance.json
files/<OUTDIR>/selected_checkpoint.pt
```

Here `OUTDIR` is exactly the current member's namespace in the matrix above; no other member is selected, inferred, or traversed.

The later parameterized protocol must fail closed unless it verifies all of the following: `run.meta` semantics; the frozen `RUN_NAME`; `EXPECTED_COMMIT == ACTUAL_COMMIT`; command SHA against authentic `command.sh`; `EXIT_CODE=0`; SHA256 hashes of `run.log`, `run.meta`, and `command.sh`; exactly the five required regular output files; and each output's path, size, and SHA256.  It must reject extra ZIP members, reopen and fully revalidate the ZIP before `PASS`, and mutate no scientific output.

It uses no glob artifact selection, no recursive artifact discovery, no mtime comparison, and no `find -newer`.  It is CPU-only.  This report freezes the protocol, not a repository collector implementation and not a recovery package.

Before each run, the controller must derive and independently review the one exact parameterized packaging command from this fixed protocol for that member's `RUN_NAME`, authority commit, and `OUTDIR`; deriving that command does not authorize a repository implementation or any change to `cm.ps1`.

## Standard cm import compatibility

The locally downloaded fixed-path v3 ZIP must use the existing standard `cm import` without weakening it, without a recovery-specific schema, and without wrapper fabrication.  Before each import, local validation must require: the run registry entry matches the run name; registered HEAD equals handoff HEAD; registered command hash equals the manifest and authentic wrapper command hash; and the local checkout is pinned to the execution-authority commit required by standard import.

Only after successful standard import and exact imported-artifact/provenance audit may the next sequential seed begin.

## Sequential runtime gates and stop conditions

For every run, stop immediately on a HEAD mismatch; dirty tracked worktree/index; remote identity mismatch; trainer/blob mismatch; dataset/sidecar/provenance mismatch; split mismatch; output collision; nonzero exit; NaN/nonfinite training failure; missing output; wrapper inconsistency; fixed-path handoff failure; ZIP validation failure; or import failure/provenance mismatch.

Do not proceed until the current seed has successful training execution, a valid authentic wrapper, successful fixed-path handoff, and successful standard import/audit.  If a run fails before valid evidence, stop and diagnose.  If it succeeds scientifically but handoff/import fails, preserve the run and require the controller to decide the smallest bounded remedy.  No automatic escalation into another multi-stage recovery authority stack is authorized.

## Downstream boundary

Successful completion and import of all three runs establishes no scientific conclusion.  The next phase is `Seed8192 revised A0 N=3 validated-evidence analysis`; only after that is revised reason-loss calibration allowed; only after calibration acceptance may a new A1/A2/A3 factorial authority be considered.  This report authorizes none of those downstream stages.

## Authoring record

Required authoring validation is: required HEAD `4d26d5601b10714d0158049357d600901274a054`; zero staged files; zero tracked modifications; exactly this one new candidate; `git diff --check`; candidate byte count; SHA256; Git blob OID; CR count; UTF-8 BOM absence; and exactly one final LF.  These facts must be recorded after the report bytes are written and before independent verification.

Explicit confirmation: no code, no `cm.ps1` change, no recovery implementation, no execution, no Kaggle, no training, no evaluation, no staging, no commit, and no push.

PASS_READY_FOR_INDEPENDENT_SEED8192_A0_N3_CLEAN_REPLACEMENT_EXECUTION_AUTHORITY_VERIFICATION
