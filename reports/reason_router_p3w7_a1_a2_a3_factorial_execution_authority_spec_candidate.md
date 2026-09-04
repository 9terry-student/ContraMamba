# P3-W7 Matched A1/A2/A3 Factorial Execution Authority Specification Candidate

## Verdict

Status: `PASS_READY_FOR_INDEPENDENT_VERIFICATION`.

This candidate releases exactly the bounded matched P3-W7 A1/A2/A3 factorial execution for seeds `180`, `181`, and `182`, using the accepted common current-lineage reason-loss weight and the exact same-seed admitted A0 prediction references.

This authoring task performs no training, evaluation, Kaggle execution, model loading, checkpoint loading, staging, commit, or push.

## 1. Authority Release Decision

Historical `UNRESOLVED_REASON_LOSS_WEIGHT` is resolved for the current lineage by calibration-result acceptance commit:

`1221588b78d02900ee93cff36cf37b2202e04aea`

The exact common positive reason-loss weight is:

`0.6202430063306562`

This value is shared by A1 and A3 only. No seed-specific reason weight is permitted. A2 remains exactly `0.0`.

Calibration result acceptance and factorial execution release remain separate states. The calibration acceptance at `1221588b78d02900ee93cff36cf37b2202e04aea` recorded `A1_A3_released=false`; it accepted calibration evidence and resolved the common weight, but did not authorize A1/A2/A3 execution. This new authority candidate may release the bounded matched factorial only after independent verification and the Section 8 pushed frozen-authority gate.

## 2. Resolved Authority Chain

Authority precedence consumed:

1. Current controller instruction.
2. Frozen A0 N=3 validated-evidence report: `52e024b6a1389fb3dd46d1ec58ad8b4b99c86c6b`, `reports/reason_router_p3w7_a0_n3_validated_evidence_analysis_report.md`.
3. Frozen A0 N=3 analysis authority: `052d38bae4f795b0fef5ae802f35151a730a8126`, `reports/reason_router_p3w7_a0_n3_validated_evidence_analysis_authority_spec_candidate.md`.
4. Accepted current-lineage calibration result: `1221588b78d02900ee93cff36cf37b2202e04aea`, `reports/reason_router_p3w7_current_lineage_reason_weight_calibration_result_acceptance_authority_spec_candidate.md`.
5. Frozen current-lineage calibration authority: `ba3fd1a82cba029dac05ba38a86d518252ab858f`, `reports/reason_router_p3w7_current_lineage_reason_weight_calibration_execution_authority_spec_candidate.md`.
6. Formal P3-W7 A0 execution authority: `2737c3c6116ae3766b469801f990e2c45ba9a55e`, `reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`.
7. Exact admitted A0 evidence commits:
   - seed180: `b32d73dfa49b6b9dfabf3093802904323cf679cd`
   - seed181 `REPLACEMENT_R1`: `fb4f0e2c2a8382a642f1272b66f29552adaecb0e`
   - seed182: `82739bdfc8eee184de10ed8f55434f203a6d59a5`
8. Historical factorial execution design, semantic authority only: `72cfdd3d551832e33799ca0399a6d6bf0c431901`, `reports/reason_router_p2_p3_execution_spec.md`.
9. Current implementation/tests/contracts at `52e024b6a1389fb3dd46d1ec58ad8b4b99c86c6b`, especially `scripts/train_controlled_v6b_minimal.py`, `tests/test_reason_router_p2_contract.py`, `tests/test_reason_router_p3w1_calibration.py`, and `tests/test_reason_router_p4x_trainer_rebind.py`.
10. `AGENTS.md`.

## 3. Historical Blocker Disposition

`UNRESOLVED_REASON_LOSS_WEIGHT` is resolved only for the current P4-L lineage by `1221588b78d02900ee93cff36cf37b2202e04aea`.

`P3_BLOCKED_BY_MISSING_EXECUTION_OBSERVABILITY` remains an interpretation boundary. It does not block authoring or running this bounded factorial if frozen code contracts and tests authorize the arm semantics, but it blocks stronger later claims requiring unobserved per-run gradient-ownership behavior.

The historical dataset SHA `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640` and historical sidecar semantic SHA `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc` are historical only and must not be consumed.

## 4. Current P4-L Data Contract

The current-lineage data contract is frozen as:

| Field | Value |
|---|---|
| Dataset path | `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` |
| Dataset physical SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| Dataset semantic SHA256 | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| Integrity sidecar path | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl` |
| Sidecar physical SHA256 | `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1` |
| Sidecar semantic SHA256 | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` |
| Sidecar provenance path | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json` |
| Provenance physical SHA256 | `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2` |
| Total rows | `3600` |
| Train rows | `2880` |
| Dev rows | `720` |
| Split seed | `174` |
| Dev ratio | `0.2` |
| Ordered train row identity hash | `cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784` |

Any drift blocks execution.

## 5. Exact A0 Reference Membership And Identities

Exactly these three A0 reference members are frozen. No cross-seed A0 reference consumption is permitted. No A0 selected checkpoint, logits, metrics, or checkpoint loading may substitute for the required same-seed `training_report_predictions.jsonl`. `clean_dev_predictions.json` must not be substituted.

| Seed | Admitted member | Commit | Path | SHA256 | Bytes |
|---:|---|---|---|---|---:|
| 180 | seed180 | `b32d73dfa49b6b9dfabf3093802904323cf679cd` | `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl` | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` | `3934123` |
| 181 | `REPLACEMENT_R1` only | `fb4f0e2c2a8382a642f1272b66f29552adaecb0e` | `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/training_report_predictions.jsonl` | `b186af00684279095a2257ef46f826be281a92bb2da9b0b9ee8f157f5bdbc13c` | `3936706` |
| 182 | seed182 | `82739bdfc8eee184de10ed8f55434f203a6d59a5` | `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/training_report_predictions.jsonl` | `95aa57b9f14ff7119b19f0ec8e412bf5b4494ae325e73d4c0ef0df5b24e050e5` | `3938064` |

Seed180 caveat is preserved:

- `standard_cm_wrapper_provenance = INCOMPLETE`
- disposition: `RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE`

This candidate does not upgrade, erase, normalize, or reclassify the seed180 caveat.

Seed181 original consumed result remains inadmissible. Only `REPLACEMENT_R1` is admissible.

## 6. Exact Arm Contract

The four-arm interpretation is frozen as:

| Arm | Status | Router | Gradient ownership | Reason supervision | Reason-loss weight | A0 reference predictions |
|---|---|---|---|---|---:|---|
| A0 | baseline/control; already completed; do not rerun | `explicit_product` | `joint` | disabled | `0.0` | `null` |
| A1 | newly authorized by this candidate after the Section 8 pushed frozen-authority gate | `conditional_first_blocker` | `joint` | enabled | `0.6202430063306562` | exact same-seed admitted A0 `training_report_predictions.jsonl` |
| A2 | newly authorized by this candidate after the Section 8 pushed frozen-authority gate | `explicit_product` | `explicit_local` | disabled | `0.0` | exact same-seed admitted A0 `training_report_predictions.jsonl` |
| A3 | newly authorized by this candidate after the Section 8 pushed frozen-authority gate | `conditional_first_blocker` | `explicit_local` | enabled | `0.6202430063306562` | exact same-seed admitted A0 `training_report_predictions.jsonl` |

Existing semantic definitions are preserved:

- `conditional_first_blocker`
- `explicit_product`
- `joint`
- `explicit_local`
- primary reason order: `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`
- secondary reasons diagnostic-only
- router-only final 3-way CE ownership
- explicit-local detach semantics
- frozen encoder

No new reason-router mechanism is invented. A1 bundles conditional-first-blocker routing with reason supervision under joint ownership; this design does not separately identify router and reason-supervision effects.

## 7. Exact Nine-Run Matrix

Exactly nine new training runs are authorized after this candidate is independently verified and frozen:

| Order | Seed | Arm | Reason-loss weight | A0 reference binding |
|---:|---:|---|---:|---|
| 1 | 180 | A1 | `0.6202430063306562` | seed180 A0 reference |
| 2 | 180 | A2 | `0.0` | seed180 A0 reference |
| 3 | 180 | A3 | `0.6202430063306562` | seed180 A0 reference |
| 4 | 181 | A1 | `0.6202430063306562` | seed181 `REPLACEMENT_R1` A0 reference |
| 5 | 181 | A2 | `0.0` | seed181 `REPLACEMENT_R1` A0 reference |
| 6 | 181 | A3 | `0.6202430063306562` | seed181 `REPLACEMENT_R1` A0 reference |
| 7 | 182 | A1 | `0.6202430063306562` | seed182 A0 reference |
| 8 | 182 | A2 | `0.0` | seed182 A0 reference |
| 9 | 182 | A3 | `0.6202430063306562` | seed182 A0 reference |

Training seeds are exactly `180`, `181`, and `182`. Split seed is exactly `174`. Dev ratio is exactly `0.2`.

No A0 rerun, no E0 execution, no A4, and no other arm is authorized. E0 remains algebraic-equivalence diagnostic context only.

A nonzero exit, provenance mismatch, A0-reference mismatch, dataset or sidecar mismatch, NaN/nonfinite loss, or contract violation stops subsequent execution pending controller review. Do not silently skip a failed cell and continue the batch.

## 8. Execution Authority Commit And Push Gate

Authoring base and prerequisite evidence commit:

`52e024b6a1389fb3dd46d1ec58ad8b4b99c86c6b`

Static audit of the current trainer/tests/contracts at this commit confirms the required current trainer, P4-L bindings, arm resolver, calibration-consumer CLI surface, prediction-reference handling, explicit-local semantics, and required tests are present.

This base commit remains this candidate's authoring base and prerequisite evidence identity. It is not the eventual factorial runtime execution identity.

Future factorial runtime execution identity:

```text
FACTORIAL_EXECUTION_AUTHORITY_COMMIT = <EXACT_FULL_40_HEX_SHA_OF_FROZEN_AUTHORITY_COMMIT>
```

This value is filled only after this candidate passes independent verification and is committed/frozen by a dedicated authority commit. A1/A2/A3 execution is forbidden until all of the following are true:

1. this candidate passes independent verification;
2. this candidate is frozen in Git by a dedicated authority commit;
3. that exact authority commit is pushed to the remote;
4. remote presence of the exact full authority commit is verified;
5. runtime is prepared from that exact frozen authority commit.

Freeze alone is insufficient. Commit alone is insufficient. Local-only authority is insufficient.

Execution runtime HEAD must equal the exact full 40-character `FACTORIAL_EXECUTION_AUTHORITY_COMMIT`. Branch name is not execution identity. Short SHA is not sufficient. The authoring base commit `52e024b6a1389fb3dd46d1ec58ad8b4b99c86c6b` is not a substitute for the later frozen authority commit. All nine runs in this batch must use the same exact frozen authority commit unless a later explicit recovery authority says otherwise.

## 9. Trainer And CLI Contract

The current trainer inspected is `scripts/train_controlled_v6b_minimal.py`.

Common CLI values for every authorized A1/A2/A3 run:

```text
python scripts/train_controlled_v6b_minimal.py
  --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl
  --architecture v6b_minimal
  --backbone mamba
  --model-name state-spaces/mamba-130m-hf
  --freeze-encoder true
  --frame-downstream-gradient-mode joint
  --epochs 20
  --max-length 128
  --dev-ratio 0.2
  --seed <180|181|182>
  --split-seed 174
  --device cuda
  --flag-source controlled_heuristic
  --select-metric final_macro_f1
  --ranking-weight 0.0
  --class-weighting none
  --stage174c-clean-pairwise-mode off
  --stage174c-clean-pairwise-weight 0.0
  --stage174c-clean-polarity-preservation-weight 0.0
  --stage175b-support-anchor-mode off
  --stage175b-support-anchor-weight 0.0
  --stage177c-frame-pairwise-mode off
  --stage177c-frame-pairwise-weight 0.0
  --compatible-positive-margin-logit 0.0
  --lr 0.001
  --controlled-integrity-sidecar-path reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl
  --expected-integrity-sidecar-semantic-sha256 0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08
  --compatible-positive-margin-weight 0.0
  --save-selected-checkpoint
  --selected-checkpoint-filename selected_checkpoint.pt
  --reason-router-arm <A1|A2|A3>
  --reason-router-a0-reference-predictions <exact same-seed provisioned path>
  --output-json /kaggle/working/ContraMamba/reports/reason_router_p3w7_a1_a2_a3_factorial_runs/seed<seed>/<arm>/training_report.json
  --output-predictions-json /kaggle/working/ContraMamba/reports/reason_router_p3w7_a1_a2_a3_factorial_runs/seed<seed>/<arm>/clean_dev_predictions.json
```

A1 and A3 additionally require:

```text
--reason-loss-weight 0.6202430063306562
```

A2 uses the trainer-supported zero-weight representation by omitting `--reason-loss-weight`. The inspected resolver requires A0/A2 effective `reason_loss_weight == 0`; the parser default is `0.0`, and no unsupported flag form is invented.

All other auxiliary, ranking, intervention, pairwise, threshold-tuning, external-data, OOD, and external-metric objectives remain neutral or absent as in the current A0/current-lineage contract.

## 10. Command Registration And Run Names

The exact approved execution command or wrapper invocation must be saved or registered against the exact full `FACTORIAL_EXECUTION_AUTHORITY_COMMIT` after that authority commit exists and before Kaggle execution. The registered command must be reviewed as part of preflight. A command/run registry entry associated with another commit must not be reused. HEAD/hash mismatch between runtime, registered command metadata, and frozen authority is fail-closed. Branch name alone may not satisfy registry provenance. Changing any execution-semantic argument after registration requires new controller authorization and, where semantically material, a new authority.

Each authorized execution must use a descriptive, non-recycled run name that uniquely identifies at minimum the P3-W7 factorial stage, arm, training seed, and authority lineage/version sufficiently to avoid collision. A generic or default reusable run name such as `single` must not serve as the provenance-level run identity. No existing completed or failed run name from another commit may be reused. Silent recycling across authority commits is forbidden. Run-name collision is a preflight blocker. Wrapper or internal trainer `run_name` defaults do not override the outer provenance and run-registry naming requirement.

## 11. Output Namespace

New noncolliding namespace:

`reports/reason_router_p3w7_a1_a2_a3_factorial_runs/seed{seed}/{arm}/`

Each run must produce normal trainer artifacts required for later validated analysis, including at least:

- `training_report.json`
- `training_report_predictions.jsonl`
- `clean_dev_predictions.json`
- `selected_checkpoint.pt`
- `run_provenance.json`

If the current standard wrapper emits additional mandatory provenance artifacts, include them. Do not overwrite existing A0 or historical P3 output. Output collision blocks the run.

## 12. A0 Provisioning Design

The admitted A0 prediction files are immutable commit-addressed evidence and need not be merged or cherry-picked into the factorial branch.

Future execution must use deterministic fail-closed provisioning:

1. Read each reference from its exact Git commit and exact path.
2. Write it only to a dedicated execution-input location.
3. Verify exact SHA256 and byte count before trainer invocation.
4. Keep the execution Git worktree clean.
5. Never overwrite immutable source evidence.
6. Never substitute `clean_dev_predictions.json`.

Preferred Kaggle external input root:

`/kaggle/working/contramamba_factorial_a0_references/`

Resolved destinations:

| Seed | Destination |
|---:|---|
| 180 | `/kaggle/working/contramamba_factorial_a0_references/seed180/training_report_predictions.jsonl` |
| 181 | `/kaggle/working/contramamba_factorial_a0_references/seed181/training_report_predictions.jsonl` |
| 182 | `/kaggle/working/contramamba_factorial_a0_references/seed182/training_report_predictions.jsonl` |

If exact Git-object provisioning cannot be made deterministic and fail-closed, report `BLOCKED` rather than inventing a weaker reference path.

## 13. Required Run Provenance

Each A1/A2/A3 run must make later validation possible. Provenance must establish:

- exact execution commit, equal to the full 40-character `FACTORIAL_EXECUTION_AUTHORITY_COMMIT`
- seed
- split seed
- arm
- router mode
- gradient ownership mode
- effective reason-loss weight
- exact dataset identity
- exact sidecar identity
- exact A0 reference source commit/path
- exact provisioned A0-reference SHA256
- exact same-seed binding
- model/backbone/architecture
- encoder frozen
- selection metric
- epochs/max length/dev ratio/LR
- neutral auxiliary objectives
- selected checkpoint identity
- prediction artifact identity

Current trainer provenance records the run, trainer/source, dataset, P4-L sidecar/provenance identities, resolved arm/router/ownership/reason-loss metadata, selected checkpoint identity, and prediction/report artifact identities. The CLI records the provisioned A0-reference path. To bind immutable source commit/path/SHA to that provisioned file without trainer modification, the future wrapper must emit a separate wrapper/sidecar provenance artifact per run that records the source commit, source path, source SHA256, byte count, provisioned destination, and post-copy SHA256/byte verification.

Provenance sufficiency verdict: sufficient without trainer implementation change only if the future wrapper/sidecar provenance is emitted and imported with each run. If that wrapper/sidecar binding is omitted, the run must return `BLOCKED_NEEDS_FACTORIAL_PROVENANCE_IMPLEMENTATION_AUTHORITY` or an equivalent controller review blocker before scientific use.

This authoring task does not modify trainer implementation.

## 14. Observability Boundary

Preserve `P3_BLOCKED_BY_MISSING_EXECUTION_OBSERVABILITY` as an interpretation boundary where applicable.

Do not infer `ownership_violation_count == 0` merely from config/tests. Do not claim full causal mechanism support from successful training alone. Do not treat exact CLI gradient mode as empirical gradient-flow observation.

The current tests include static/contractual autograd ownership checks for the arm semantics, but no current per-run authoritative observability field was identified that resolves runtime ownership observation for these future runs. Missing execution observability therefore remains only an interpretation boundary for this bounded factorial execution authority, and a blocker for later stronger claims requiring unobserved per-run gradient-ownership behavior.

## 15. Scientific Interpretation Boundary

This execution authority defines intended later matched comparisons:

- `A1 - A0`
- `A2 - A0`
- `A3 - A1`
- `A3 - A2`
- `A3 - A0`

The matched 2x2 interaction may be analyzed later only if preserved by the frozen historical factorial semantics and validated imported evidence.

This authority must not aggregate results, calculate factorial effects, select a winning arm, declare causal mechanism support, promote A3, set a new threshold, or make a scientific conclusion. Those require validated imported evidence and a separate analysis/interpretation authority.

Do not overclaim independent identification of router versus reason-supervision effects because the frozen arm design bundles those changes in A1 and A3.

## 16. Failure Policy

Fail closed on at least:

- execution HEAD mismatch against the full 40-character `FACTORIAL_EXECUTION_AUTHORITY_COMMIT`
- missing remote verification of the full frozen authority commit
- missing or mismatched command registration for the full frozen authority commit
- command/run registry reuse from another commit
- generic, recycled, or colliding provenance-level run name
- dirty execution repo
- dataset SHA mismatch
- sidecar semantic, physical, or provenance mismatch where frozen
- split mismatch
- A0 source commit/path mismatch
- A0 provisioned SHA/byte mismatch
- cross-seed reference use
- original seed181 reference use
- seed180 caveat being erased or reclassified
- arm contract mismatch
- reason weight mismatch
- output collision
- nonfinite loss
- missing required run provenance
- checkpoint or prediction artifact collision/corruption

Do not auto-repair. Do not auto-rerun under a new run identity. Return to controller review after a failed authorized run.

## 17. Kaggle And Execution Boundary

This candidate defines the future exact commands/wrappers required for the nine bounded runs, but this authoring task does not execute them.

Even after this authority candidate is independently verified, actual Kaggle execution requires a separate controller handoff after the dedicated authority commit is frozen, pushed, verified on the remote by exact full SHA, command-registered against that exact full SHA, and prepared as runtime HEAD. This authority must not tell the user to run Kaggle merely because the report exists.

No GPU use is authorized or performed in this authoring task.

## 18. No Reinterpretation Of A0

Preserve:

- seed180 historical standard-CM wrapper provenance = `INCOMPLETE`
- seed181 only admissible reference = `REPLACEMENT_R1`
- seed182 = primary admissible

Do not merge or cherry-pick result commits merely to consume references. Do not rerun A0. Do not normalize seed180 provenance. Do not reinstate original seed181. Do not treat A0 N=3 descriptive metrics as factorial evidence.

## 19. Future Wrapper Requirements

Every future run wrapper must fail closed before trainer invocation unless:

- execution checkout HEAD equals the full 40-character `FACTORIAL_EXECUTION_AUTHORITY_COMMIT`
- the exact full `FACTORIAL_EXECUTION_AUTHORITY_COMMIT` has verified remote presence
- the exact approved execution command/wrapper is registered against that exact full commit
- registered command metadata, runtime HEAD, and frozen authority commit match exactly
- the provenance-level run name is descriptive, non-recycled, and noncolliding
- tracked worktree is clean
- Git index is clean
- current P4-L dataset, sidecar, and provenance identities match this authority
- provisioned same-seed A0 reference matches exact source commit/path/SHA/bytes
- output directory and required target artifacts do not already exist
- arm, seed, reason-loss weight, and output namespace match the nine-run matrix

The wrapper must write per-run immutable A0-reference binding provenance outside the trainer if the trainer's `run_provenance.json` only records the provisioned path.

## 20. Non-Authorization

This candidate does not authorize during authoring:

- training
- evaluation
- inference
- model loading
- checkpoint loading or deserialization
- Kaggle execution
- GPU execution
- source modification
- test modification
- dataset or sidecar mutation
- A0 report/result mutation
- calibration artifact mutation
- historical factorial spec mutation
- staging
- commit
- push

## 21. Candidate Materialization Notes

Candidate materialization target:

`reports/reason_router_p3w7_a1_a2_a3_factorial_execution_authority_spec_candidate.md`

Expected materialization state:

- Worktree: `C:\p3w7-a0-n3-validated-evidence-analysis`
- Branch: `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`
- HEAD: `52e024b6a1389fb3dd46d1ec58ad8b4b99c86c6b`
- Initial worktree/index: clean

Final candidate SHA256, byte count, line-ending facts, `git diff --check`, `git diff --name-status`, `git diff --cached --name-status`, and `git status --short` are intentionally reported outside this file to avoid self-referential candidate content.
