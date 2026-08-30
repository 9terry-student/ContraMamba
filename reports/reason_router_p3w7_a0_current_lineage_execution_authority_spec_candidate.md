# P3-W7-A0 Current-Lineage Execution Authority Specification Candidate

## Status

CANDIDATE ONLY: formal Week 1 P3-W7-A0 execution-authority candidate.

This file does not authorize training, evaluation, Kaggle execution, GPU use, artifact promotion, A1, A2, or A3 while it is merely an uncommitted or unverified candidate.

It becomes the formal Week 1 P3-W7-A0 execution authority only after:
1. this exact candidate is independently verified;
2. the verified file is frozen in an immutable Git commit;
3. local post-freeze gates and per-seed preflight pass.

After freeze, the authority authorizes exactly the bounded A0 executions defined below and nothing else.

## Authority creation basis

Formal Week 1 candidate materialization basis HEAD:

`dd40d7a9514aaaed8ee7c24c06fed80598c2b0f1`

The immutable Week 0 pre-start candidate freeze is:

`ecda9707cc054ec26428b3f0937be8829f754f1b`

That Week 0 freeze is retained only as immutable pre-start provenance and as
the independently verified technical/V3 draft basis for this formal Week 1
candidate. It must not be consumed as the formal Week 1 A0 execution
authority.

The pre-start V3 technical lineage was:

`9f12fd3a65c94006a76d6c20ccbfaeb6728c44ba`

The intervening pre-start documentation basis was:

`bca6db6de2e1bb5d1b81188b61b2023be20eadd3`

The delta from the independently verified Week 0 pre-start freeze
`ecda9707cc054ec26428b3f0937be8829f754f1b` to the formal Week 1
materialization basis `dd40d7a9514aaaed8ee7c24c06fed80598c2b0f1` is limited
to Week 0 closure documentation/state updates. No trainer, parser, dataset,
P4-L sidecar/provenance artifact, P2 arm contract, loss semantics,
gradient-ownership semantics, execution parameter, seed, split, model,
hyperparameter, output contract, attempt semantics, recovery semantics, or
A1/A2/A3 boundary changed between the verified technical basis and this formal
Week 1 candidate materialization basis.

The materialization-basis HEAD is not the future execution freeze.

The new future formal Week 1 authority freeze is the commit that contains this
newly materialized exact file. Each authorized wrapper captures that commit
dynamically with:

`P3W7_A0_AUTHORITY_FREEZE="$(git rev-parse HEAD)"`

No future literal freeze SHA is embedded in the command text.

## Canonical current-lineage binding

Dataset:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Dataset physical SHA256:

`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

Dataset semantic SHA256:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

P4-L sidecar:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`

P4-L sidecar physical SHA256:

`2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`

P4-L sidecar semantic SHA256:

`0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

P4-L provenance:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`

P4-L provenance physical SHA256:

`9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`

Expected sidecar row count:

`3600`

P4-L exact-byte provisioning and provisioning-result validation are already closed upstream. This authority consumes those frozen artifacts; it does not reopen, regenerate, rewrite, or reinterpret them.

## Active trainer and model envelope

Trainer:

`scripts/train_controlled_v6b_minimal.py`

Architecture:

`v6b_minimal`

Backbone:

`mamba`

Model:

`state-spaces/mamba-130m-hf`

Encoder:

frozen, via `--freeze-encoder true`

Legacy frame downstream mode:

`joint`

The P2 A0 arm itself resolves:

- router: `explicit_product`
- gradient ownership: `joint`
- reason loss: effective `0.0`

No A0 reference predictions input is passed or consumed.

## Exact A0 execution scope

Authorized arm:

`A0`

Authorized training seeds:

`180`, `181`, `182`

Split seed:

`174`

Epochs:

`20`

Learning rate:

`0.001`

Maximum sequence length:

`128`

Dev ratio:

`0.2`

Device:

`cuda`

Flag source:

`controlled_heuristic`

Selection metric:

`final_macro_f1`

Class weighting:

`none`

Train/eval batch-size flags are intentionally omitted, preserving the established P3 A0 behavior.

`--reason-loss-weight` is intentionally omitted. Under A0 resolution, the effective reason-loss weight is `0.0`.

No ad-hoc hyperparameter search, alternate seed, A1, A2, A3, recovery attempt, retry, resume, cleanup, overwrite, or scope change is authorized.

## Required objective neutralization

The exact commands preserve the established P3 A0 scientific envelope:

- `--ranking-weight 0.0`
- `--class-weighting none`
- `--stage174c-clean-pairwise-mode off`
- `--stage174c-clean-pairwise-weight 0.0`
- `--stage174c-clean-polarity-preservation-weight 0.0`
- `--stage175b-support-anchor-mode off`
- `--stage175b-support-anchor-weight 0.0`
- `--stage177c-frame-pairwise-mode off`
- `--stage177c-frame-pairwise-weight 0.0`
- `--compatible-positive-margin-logit 0.0`
- `--compatible-positive-margin-weight 0.0`

No historical Stage175 distributional-separation alias or Stage177 counterfactual-consistency alias is used.

## Output contract

Run root:

`/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs`

Seed output directories:

- `/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0`
- `/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed181/A0`
- `/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0`

Required artifacts per seed:

- `training_report.json`
- `clean_dev_predictions.json`
- `training_report_predictions.jsonl`
- `selected_checkpoint.pt`

Before trainer launch, the exact seed output directory and every required target artifact must be absent.

Existing outputs must not be deleted, renamed, cleaned, overwritten, or reused.

## Future immutable A0 reference contract

After valid A0 collection/import, each seed's future same-seed A1/A2/A3 reference is the collected/imported `training_report_predictions.jsonl` from that seed's A0 run together with:

- its recorded SHA256;
- selected-checkpoint identity;
- training-report metadata;
- A0 authority freeze SHA;
- exact A0 command SHA256.

Those references remain unusable by A1/A2/A3 until a separate authority explicitly authorizes their consumption.

## Attempt-consumption contract

Exactly one authorized A0 trainer attempt exists for each of seeds 180, 181, and 182.

Wrapper rejection before the trainer process launches does not consume the seed attempt.

Trainer process launch consumes the seed attempt regardless of later PASS or FAIL.

A consumed failed seed may not be rerun under this authority.

Any retry requires a separately created, independently verified, frozen recovery authority.

No automatic retry, resume, cleanup, overwrite, output reuse, or alternate seed is authorized.

## Freeze and worktree guards

Every exact wrapper must fail closed unless:

1. `P3W7_A0_AUTHORITY_FREEZE` is lowercase 40-hex;
2. the supplied SHA resolves to a Git commit;
3. current `HEAD` equals that SHA;
4. the tracked worktree is clean;
5. the Git index is clean;
6. the frozen tree contains this exact authority path;
7. the frozen tree contains the canonical dataset, P4-L sidecar, and P4-L provenance paths;
8. the corresponding worktree paths exist;
9. no exact output collision exists.

Unrelated untracked files are ignored by the tracked-worktree guard and do not authorize their staging, deletion, modification, or interpretation.

## Command identity contract

Each command SHA256 is computed over the complete one-line freeze-bound wrapper command below, encoded as UTF-8, followed by exactly one final LF byte.

The hash excludes:

- markdown fence bytes;
- any leading LF;
- display wrapping;
- trailing spaces;
- any extra final blank line.

The trainer-only suffix is not the hashed object.

Seed 180 command SHA256:

`dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e`

Seed 181 command SHA256:

`3794fbdcb9e347a13aef02a258bab2a7a597d49acee12686d363cb178e5ae1ea`

Seed 182 command SHA256:

`5c0a7609069f8c6e4a5ae4c27bda7c9cbd1be6f3cbb35a0b42d18acfd7dd1fac`

## Exact authorized command — seed 180

```bash
P3W7_A0_AUTHORITY_FREEZE="$(git rev-parse HEAD)" bash -lc 'set -euo pipefail; fail(){ printf "%s\n" "P3W7_A0_WRAPPER_REJECTED:$1" >&2; exit 64; }; AUTHORITY="reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md"; DATA="reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"; SIDECAR="reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"; PROVENANCE="reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"; OUTDIR="/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0"; REPORT="$OUTDIR/training_report.json"; PREDJSON="$OUTDIR/clean_dev_predictions.json"; PREDJSONL="$OUTDIR/training_report_predictions.jsonl"; CKPT="$OUTDIR/selected_checkpoint.pt"; [[ "${P3W7_A0_AUTHORITY_FREEZE:-}" =~ ^[0-9a-f]{40}$ ]] || fail FREEZE_NOT_LOWERCASE_40_HEX; [[ "$(git cat-file -t "$P3W7_A0_AUTHORITY_FREEZE" 2>/dev/null)" == "commit" ]] || fail FREEZE_NOT_COMMIT; [[ "$(git rev-parse HEAD)" == "$P3W7_A0_AUTHORITY_FREEZE" ]] || fail HEAD_MISMATCH; [[ -z "$(git status --short --untracked-files=no)" ]] || fail TRACKED_WORKTREE_DIRTY; [[ -z "$(git diff --cached --name-status)" ]] || fail INDEX_DIRTY; for p in "$AUTHORITY" "$DATA" "$SIDECAR" "$PROVENANCE"; do git cat-file -e "$P3W7_A0_AUTHORITY_FREEZE:$p" || fail "FREEZE_TREE_MISSING:$p"; [[ -e "$p" ]] || fail "WORKTREE_PATH_MISSING:$p"; done; for p in "$OUTDIR" "$REPORT" "$PREDJSON" "$PREDJSONL" "$CKPT"; do [[ ! -e "$p" ]] || fail "OUTPUT_COLLISION:$p"; done; python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --epochs 20 --max-length 128 --dev-ratio 0.2 --seed 180 --split-seed 174 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --ranking-weight 0.0 --class-weighting none --stage174c-clean-pairwise-mode off --stage174c-clean-pairwise-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --stage175b-support-anchor-mode off --stage175b-support-anchor-weight 0.0 --stage177c-frame-pairwise-mode off --stage177c-frame-pairwise-weight 0.0 --compatible-positive-margin-logit 0.0 --lr 0.001 --reason-router-arm A0 --reason-router-mode explicit_product --gradient-ownership-mode joint --controlled-integrity-sidecar-path reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08 --compatible-positive-margin-weight 0.0 --save-selected-checkpoint --selected-checkpoint-filename selected_checkpoint.pt --output-json /kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json --output-predictions-json /kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json'
```

## Exact authorized command — seed 181

```bash
P3W7_A0_AUTHORITY_FREEZE="$(git rev-parse HEAD)" bash -lc 'set -euo pipefail; fail(){ printf "%s\n" "P3W7_A0_WRAPPER_REJECTED:$1" >&2; exit 64; }; AUTHORITY="reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md"; DATA="reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"; SIDECAR="reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"; PROVENANCE="reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"; OUTDIR="/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed181/A0"; REPORT="$OUTDIR/training_report.json"; PREDJSON="$OUTDIR/clean_dev_predictions.json"; PREDJSONL="$OUTDIR/training_report_predictions.jsonl"; CKPT="$OUTDIR/selected_checkpoint.pt"; [[ "${P3W7_A0_AUTHORITY_FREEZE:-}" =~ ^[0-9a-f]{40}$ ]] || fail FREEZE_NOT_LOWERCASE_40_HEX; [[ "$(git cat-file -t "$P3W7_A0_AUTHORITY_FREEZE" 2>/dev/null)" == "commit" ]] || fail FREEZE_NOT_COMMIT; [[ "$(git rev-parse HEAD)" == "$P3W7_A0_AUTHORITY_FREEZE" ]] || fail HEAD_MISMATCH; [[ -z "$(git status --short --untracked-files=no)" ]] || fail TRACKED_WORKTREE_DIRTY; [[ -z "$(git diff --cached --name-status)" ]] || fail INDEX_DIRTY; for p in "$AUTHORITY" "$DATA" "$SIDECAR" "$PROVENANCE"; do git cat-file -e "$P3W7_A0_AUTHORITY_FREEZE:$p" || fail "FREEZE_TREE_MISSING:$p"; [[ -e "$p" ]] || fail "WORKTREE_PATH_MISSING:$p"; done; for p in "$OUTDIR" "$REPORT" "$PREDJSON" "$PREDJSONL" "$CKPT"; do [[ ! -e "$p" ]] || fail "OUTPUT_COLLISION:$p"; done; python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --epochs 20 --max-length 128 --dev-ratio 0.2 --seed 181 --split-seed 174 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --ranking-weight 0.0 --class-weighting none --stage174c-clean-pairwise-mode off --stage174c-clean-pairwise-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --stage175b-support-anchor-mode off --stage175b-support-anchor-weight 0.0 --stage177c-frame-pairwise-mode off --stage177c-frame-pairwise-weight 0.0 --compatible-positive-margin-logit 0.0 --lr 0.001 --reason-router-arm A0 --reason-router-mode explicit_product --gradient-ownership-mode joint --controlled-integrity-sidecar-path reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08 --compatible-positive-margin-weight 0.0 --save-selected-checkpoint --selected-checkpoint-filename selected_checkpoint.pt --output-json /kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed181/A0/training_report.json --output-predictions-json /kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed181/A0/clean_dev_predictions.json'
```

## Exact authorized command — seed 182

```bash
P3W7_A0_AUTHORITY_FREEZE="$(git rev-parse HEAD)" bash -lc 'set -euo pipefail; fail(){ printf "%s\n" "P3W7_A0_WRAPPER_REJECTED:$1" >&2; exit 64; }; AUTHORITY="reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md"; DATA="reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"; SIDECAR="reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"; PROVENANCE="reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"; OUTDIR="/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0"; REPORT="$OUTDIR/training_report.json"; PREDJSON="$OUTDIR/clean_dev_predictions.json"; PREDJSONL="$OUTDIR/training_report_predictions.jsonl"; CKPT="$OUTDIR/selected_checkpoint.pt"; [[ "${P3W7_A0_AUTHORITY_FREEZE:-}" =~ ^[0-9a-f]{40}$ ]] || fail FREEZE_NOT_LOWERCASE_40_HEX; [[ "$(git cat-file -t "$P3W7_A0_AUTHORITY_FREEZE" 2>/dev/null)" == "commit" ]] || fail FREEZE_NOT_COMMIT; [[ "$(git rev-parse HEAD)" == "$P3W7_A0_AUTHORITY_FREEZE" ]] || fail HEAD_MISMATCH; [[ -z "$(git status --short --untracked-files=no)" ]] || fail TRACKED_WORKTREE_DIRTY; [[ -z "$(git diff --cached --name-status)" ]] || fail INDEX_DIRTY; for p in "$AUTHORITY" "$DATA" "$SIDECAR" "$PROVENANCE"; do git cat-file -e "$P3W7_A0_AUTHORITY_FREEZE:$p" || fail "FREEZE_TREE_MISSING:$p"; [[ -e "$p" ]] || fail "WORKTREE_PATH_MISSING:$p"; done; for p in "$OUTDIR" "$REPORT" "$PREDJSON" "$PREDJSONL" "$CKPT"; do [[ ! -e "$p" ]] || fail "OUTPUT_COLLISION:$p"; done; python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --epochs 20 --max-length 128 --dev-ratio 0.2 --seed 182 --split-seed 174 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --ranking-weight 0.0 --class-weighting none --stage174c-clean-pairwise-mode off --stage174c-clean-pairwise-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --stage175b-support-anchor-mode off --stage175b-support-anchor-weight 0.0 --stage177c-frame-pairwise-mode off --stage177c-frame-pairwise-weight 0.0 --compatible-positive-margin-logit 0.0 --lr 0.001 --reason-router-arm A0 --reason-router-mode explicit_product --gradient-ownership-mode joint --controlled-integrity-sidecar-path reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08 --compatible-positive-margin-weight 0.0 --save-selected-checkpoint --selected-checkpoint-filename selected_checkpoint.pt --output-json /kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/training_report.json --output-predictions-json /kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/clean_dev_predictions.json'
```

## Required post-freeze local gates

After this candidate is independently verified and frozen, and before any Kaggle execution:

```text
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=no
git diff --cached --name-status
git diff --check
python -m py_compile scripts/train_controlled_v6b_minimal.py tests/test_reason_router_p4x_trainer_rebind.py tests/test_reason_router_p2_contract.py
pytest tests/test_reason_router_p4x_trainer_rebind.py
pytest tests/test_reason_router_p2_contract.py
pytest tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py
```

Per-seed preflight must then verify:

- frozen HEAD identity;
- frozen authority path;
- canonical dataset identity;
- canonical P4-L sidecar identity;
- canonical P4-L provenance identity;
- exact command SHA256;
- tracked-clean worktree and clean index;
- output-collision boundary.

Only after those checks may Kaggle bootstrap pin the exact authority freeze and the authorized seed command be executed.

## Execution and scientific boundaries

A0 is a baseline/control arm.

A0 execution success does not establish:

- effectiveness of conditional first-blocker routing;
- effectiveness of reason-specific primary CE;
- effectiveness of explicit-local gradient ownership;
- superiority of the full P2 mechanism;
- any A1/A2/A3 scientific conclusion.

Code correctness, execution success, artifact/provenance validity, and scientific conclusion are separate statuses.

Scientific interpretation requires valid collection/import and subsequent analysis under applicable authority.

No A1, A2, or A3 execution is authorized by this file.

## Failure boundary

Any prelaunch guard failure is an execution blocker, not automatically a scientific failure.

Any launched A0 run that fails consumes that seed's attempt and must stop for failure classification.

The failure must not be silently repaired inside the run.

Recovery, if scientifically and operationally justified, requires a new bounded recovery authority.

## Candidate freeze rule

This file must not predict its own future commit SHA.

Its content hash may be computed after materialization for review, but that content hash is not the Git authority freeze.

Only after independent verification PASS and explicit user commit/push does the resulting immutable commit become the P3-W7-A0 execution authority freeze.

Candidate materialization, independent verification, authority freeze, local gates, Kaggle bootstrap, execution, collection/import, and scientific interpretation remain separate workflow states.
