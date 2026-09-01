# P3-W7-A0 Seed181 Runtime-Loss Replacement Execution Authority Specification Candidate

Authority/version:

`P3W7_A0_SEED181_RUNTIME_LOSS_REPLACEMENT_EXECUTION_AUTHORITY_V1`

## Status

CANDIDATE ONLY.

This downstream candidate is the exact-command materialization required by the
frozen seed181 runtime-loss recovery authority:

`74defa2c679ca2244d69b6ee950dd4a6a7a643b4`

This file does not authorize execution by existence alone. It authorizes no
training, evaluation, Kaggle execution, `cm run` registration, `cm run`
execution, collection, import, trainer modification, data modification,
tooling modification, A1, A2, A3, result promotion, or scientific
interpretation while it remains an unfrozen candidate.

Future replacement execution is authorized at most once only after this exact
candidate is independently verified, frozen in an immutable Git commit, and
then separately authorized for execution.

## Authority Basis

Recovery authority freeze:

`74defa2c679ca2244d69b6ee950dd4a6a7a643b4`

Recovery authority path:

`reports/reason_router_p3w7_a0_seed181_runtime_loss_recovery_execution_authority_spec_candidate.md`

Formal P3-W7-A0 A0 execution authority freeze:

`2737c3c6116ae3766b469801f990e2c45ba9a55e`

Formal authority path:

`reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`

Original consumed seed181 command SHA256, computed by the formal authority's
final-LF convention:

`3794fbdcb9e347a13aef02a258bab2a7a597d49acee12686d363cb178e5ae1ea`

Original seed181 attempt disposition:

`CONSUMED`

Original execution success status:

`OBSERVED`

Original artifact/provenance status:

`DESTROYED_OR_UNAVAILABLE`

Scientific conclusion:

`NOT_ESTABLISHED`

## Preliminary Verification Results

Current HEAD at materialization time:

`74defa2c679ca2244d69b6ee950dd4a6a7a643b4`

Initial tracked/staged delta before drafting:

`NONE`

Recovery short commit prefix:

`74defa2c679c`

Prefix ambiguity check:

`git rev-parse --disambiguate=74defa2c679c` returned exactly
`74defa2c679ca2244d69b6ee950dd4a6a7a643b4`; therefore the preferred prefix is
unambiguous in the available repository history.

Formal-to-recovery full diff includes added downstream research/report
artifacts, but the execution-relevant material named below is byte-identical
between `2737c3c6116ae3766b469801f990e2c45ba9a55e` and
`74defa2c679ca2244d69b6ee950dd4a6a7a643b4`.

| Path | Formal blob | Recovery blob | Result |
| --- | --- | --- | --- |
| `scripts/train_controlled_v6b_minimal.py` | `3dcc0864b85bde5fb8090c3b7bdbd04de02025e0` | `3dcc0864b85bde5fb8090c3b7bdbd04de02025e0` | BYTE_IDENTICAL |
| `reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md` | `73e9c1c7cbbe2528279687827c6ddb96614fb9d2` | `73e9c1c7cbbe2528279687827c6ddb96614fb9d2` | BYTE_IDENTICAL |
| `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` | `2b6829bf04a1333446aac6f7c603d9178b339f36` | `2b6829bf04a1333446aac6f7c603d9178b339f36` | BYTE_IDENTICAL |
| `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl` | `867ece69f4da680b4bb036530a96f586467d8421` | `867ece69f4da680b4bb036530a96f586467d8421` | BYTE_IDENTICAL |
| `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json` | `8082b3fe67532a8df4b92b95abd25fe4e16eb823` | `8082b3fe67532a8df4b92b95abd25fe4e16eb823` | BYTE_IDENTICAL |

No additional non-output file directly consumed by the formal seed181 wrapper
command changed between the formal A0 freeze and recovery freeze.

## Execution HEAD Decision

The frozen recovery authority is read literally. Replacement execution code
checkout must be the exact clean recovery authority freeze:

`74defa2c679ca2244d69b6ee950dd4a6a7a643b4`

The future downstream documentation commit containing this candidate must not
be substituted as the trainer execution HEAD.

## Replacement Identity

Replacement attempt label:

`REPLACEMENT_R1`

Exact replacement run name:

`p3w7-a0-seed181-runtime-loss-replacement-r1-74defa2c679c`

Exact replacement output root:

`/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0`

The replacement output root is distinct from the consumed original output root:

`/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed181/A0`

The replacement does not use any seed180 or seed182 namespace.

Expected scientific files remain:

- `training_report.json`
- `clean_dev_predictions.json`
- `training_report_predictions.jsonl`
- `selected_checkpoint.pt`
- `run_provenance.json`

## Exact Replacement Command

The command below was derived by extracting the frozen formal seed181 command
directly from the formal A0 authority and changing only the three original
seed181 output-destination occurrences to the replacement output namespace:
`OUTDIR`, `--output-json`, and `--output-predictions-json`.

It preserves the formal seed181 command's flag ordering, architecture,
backbone, model, frozen encoder setting, frame downstream gradient mode,
epochs, maximum length, dev ratio, seed `181`, split seed `174`, device
`cuda`, flag source `controlled_heuristic`, selection metric
`final_macro_f1`, learning rate `0.001`, class weighting `none`, ranking
weight `0.0`, reason-router arm `A0`, router mode `explicit_product`,
gradient ownership `joint`, Stage174C/Stage175B/Stage177C neutralizations,
compatible-positive-margin neutralizations, P4-L sidecar path, P4-L semantic
SHA, and checkpoint-save behavior.

It continues to omit `--reason-loss-weight`.

```bash
P3W7_A0_AUTHORITY_FREEZE="$(git rev-parse HEAD)" bash -lc 'set -euo pipefail; fail(){ printf "%s\n" "P3W7_A0_WRAPPER_REJECTED:$1" >&2; exit 64; }; AUTHORITY="reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md"; DATA="reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"; SIDECAR="reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"; PROVENANCE="reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"; OUTDIR="/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0"; REPORT="$OUTDIR/training_report.json"; PREDJSON="$OUTDIR/clean_dev_predictions.json"; PREDJSONL="$OUTDIR/training_report_predictions.jsonl"; CKPT="$OUTDIR/selected_checkpoint.pt"; [[ "${P3W7_A0_AUTHORITY_FREEZE:-}" =~ ^[0-9a-f]{40}$ ]] || fail FREEZE_NOT_LOWERCASE_40_HEX; [[ "$(git cat-file -t "$P3W7_A0_AUTHORITY_FREEZE" 2>/dev/null)" == "commit" ]] || fail FREEZE_NOT_COMMIT; [[ "$(git rev-parse HEAD)" == "$P3W7_A0_AUTHORITY_FREEZE" ]] || fail HEAD_MISMATCH; [[ -z "$(git status --short --untracked-files=no)" ]] || fail TRACKED_WORKTREE_DIRTY; [[ -z "$(git diff --cached --name-status)" ]] || fail INDEX_DIRTY; for p in "$AUTHORITY" "$DATA" "$SIDECAR" "$PROVENANCE"; do git cat-file -e "$P3W7_A0_AUTHORITY_FREEZE:$p" || fail "FREEZE_TREE_MISSING:$p"; [[ -e "$p" ]] || fail "WORKTREE_PATH_MISSING:$p"; done; for p in "$OUTDIR" "$REPORT" "$PREDJSON" "$PREDJSONL" "$CKPT"; do [[ ! -e "$p" ]] || fail "OUTPUT_COLLISION:$p"; done; python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --epochs 20 --max-length 128 --dev-ratio 0.2 --seed 181 --split-seed 174 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --ranking-weight 0.0 --class-weighting none --stage174c-clean-pairwise-mode off --stage174c-clean-pairwise-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --stage175b-support-anchor-mode off --stage175b-support-anchor-weight 0.0 --stage177c-frame-pairwise-mode off --stage177c-frame-pairwise-weight 0.0 --compatible-positive-margin-logit 0.0 --lr 0.001 --reason-router-arm A0 --reason-router-mode explicit_product --gradient-ownership-mode joint --controlled-integrity-sidecar-path reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08 --compatible-positive-margin-weight 0.0 --save-selected-checkpoint --selected-checkpoint-filename selected_checkpoint.pt --output-json /kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/training_report.json --output-predictions-json /kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/clean_dev_predictions.json'
```

Command byte checks:

- command contains no CR byte;
- command contains no embedded newline;
- command ends logically after the intended trainer invocation;
- command contains no Markdown fence bytes;
- command contains no shell-cell wrapper bytes;
- authoritative hash convention is `UTF-8(exact command) + exactly one LF byte`.

Authoritative final-LF replacement command SHA256:

`2b4722e3442580eae21b676d5a4a82f1b5aebbb776f159ace68ebe1571a42d0d`

No-final-LF contrast SHA256, non-authoritative:

`dd4d0b1c0d1d09bc1bb11d5cf13c2f3f4f080ed8c30b15bde990ffd9d18c30bc`

## Semantic Diff Versus Original Seed181 Command

Exactly three output-destination byte changes were made:

1. `OUTDIR` changed from
   `/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed181/A0`
   to
   `/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0`.
2. `--output-json` changed from the original consumed seed181 A0
   `training_report.json` path to the replacement namespace's
   `training_report.json` path.
3. `--output-predictions-json` changed from the original consumed seed181 A0
   `clean_dev_predictions.json` path to the replacement namespace's
   `clean_dev_predictions.json` path.

No other wrapper, guard, trainer path, data path, sidecar path, provenance path,
flag, flag order, seed, split seed, hyperparameter, objective weight, model,
backbone, device, reason-router arm, router mode, gradient ownership mode,
neutralization flag, P4-L semantic SHA, or checkpoint-save behavior was changed.

## Prohibited Scope Audit

The exact replacement command:

- does not contain `--reason-loss-weight`;
- does not contain A0 reference predictions;
- does not contain A1, A2, or A3;
- does not contain seed180 or seed182 output namespaces;
- does not contain alternate seed or alternate split values;
- does not introduce class weighting beyond `--class-weighting none`;
- does not introduce extra objective terms;
- does not use the consumed original run output path.

## Provenance Sufficiency Verdict

Verdict:

`SUFFICIENT_WITHOUT_TOOLING_OR_TRAINER_CHANGE`

The following layers together satisfy the frozen recovery authority's required
replacement provenance links without weakening `cm` validation and without
fabricating historical seed181 provenance:

1. frozen recovery authority
   `74defa2c679ca2244d69b6ee950dd4a6a7a643b4`;
2. exact replacement run name
   `p3w7-a0-seed181-runtime-loss-replacement-r1-74defa2c679c`;
3. exact replacement output namespace
   `/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0`;
4. standard `cm run save` / `cm run` wrapper metadata, including registered
   HEAD, command, command SHA, wrapper command file, start marker, run log, run
   meta, expected commit, actual commit, timestamps, and exit code;
5. trainer `run_provenance.json` for the replacement artifact set;
6. standard `cm collect` manifest with per-file SHA256 and size records;
7. standard `cm import` audit with registry, command, commit, manifest, file
   hash, and size validation.

The frozen recovery authority and this downstream candidate record the original
formal A0 authority SHA, original seed181 command SHA, original `CONSUMED`
disposition, original execution success `OBSERVED`, original artifact status
`DESTROYED_OR_UNAVAILABLE`, recovery authority freeze SHA, replacement command
SHA, replacement run name, replacement output namespace, and scientific
interpretation boundary. Standard `cm` collection/import and trainer provenance
then record replacement artifact SHA256/size and handoff/import identity.

No additional trainer flag is required or authorized merely to encode recovery
metadata. No historical seed181 wrapper or trainer artifact is reconstructed.

## Pre-Run Validation Gates

Before any future replacement trainer launch, a separate verifier must confirm:

1. this candidate has been independently verified and frozen in an immutable
   Git commit;
2. explicit subsequent execution authorization exists;
3. execution checkout is exactly clean
   `74defa2c679ca2244d69b6ee950dd4a6a7a643b4`, not the downstream
   documentation commit;
4. `git status --porcelain=v1 --untracked-files=no` is empty;
5. `git diff --cached --name-status` is empty;
6. recovery short prefix `74defa2c679c` remains unambiguous or any longer
   replacement prefix is explicitly re-frozen in a superseding authority;
7. the formal seed181 command is still uniquely extractable from
   `reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`;
8. the formal seed181 command final-LF SHA remains
   `3794fbdcb9e347a13aef02a258bab2a7a597d49acee12686d363cb178e5ae1ea`;
9. execution-relevant blob identity between formal and recovery freeze remains
   byte-identical for the trainer, formal authority, canonical dataset, P4-L
   sidecar, P4-L provenance, and any other non-output path directly consumed by
   the command;
10. the replacement command final-LF SHA is exactly
    `2b4722e3442580eae21b676d5a4a82f1b5aebbb776f159ace68ebe1571a42d0d`;
11. the no-final-LF contrast SHA is exactly
    `dd4d0b1c0d1d09bc1bb11d5cf13c2f3f4f080ed8c30b15bde990ffd9d18c30bc`;
12. the replacement output namespace does not collide with any existing output;
13. the consumed original namespace is not reused;
14. targeted prohibited-flag/path checks remain clean for
    `--reason-loss-weight`, A0 reference predictions, A1/A2/A3, seed180,
    seed182, alternate seed, alternate split, class weighting changes, and
    extra objective terms;
15. standard `cm` validation remains unmodified and unweakened.

## Attempt Semantics

The original formal seed181 attempt remains permanently:

`CONSUMED`

The replacement attempt label is:

`REPLACEMENT_R1`

This candidate authorizes at most one future replacement trainer launch after
independent verification, freeze, and subsequent explicit execution
authorization. If the replacement trainer process launches, `REPLACEMENT_R1` is
consumed regardless of later PASS or FAIL. It must not be silently retried,
resumed, overwritten, cleaned, or represented as the original consumed seed181
attempt.

## Collection And Import Boundary

If future execution is separately authorized and launched, collection, local ZIP
acquisition, and `cm import` must complete before accelerator changes, session
changes, notebook resets, runtime resets, filesystem resets, or other runtime
changes that could destroy `/kaggle/working`.

Scientific interpretation remains:

`NOT_ESTABLISHED`

Execution success, artifact validity, wrapper provenance validity, collection,
import, and scientific interpretation remain separate workflow states.
