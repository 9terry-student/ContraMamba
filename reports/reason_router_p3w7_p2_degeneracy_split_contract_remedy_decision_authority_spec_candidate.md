# P3-W7 P2 Degeneracy Split-Contract Remedy Decision Authority Content

## 1. Verdict

Lifecycle status:

`FINAL_SPLIT_CONTRACT_REMEDY_DECISION_AUTHORITY_CONTENT`

Freeze readiness:

`PASS_READY_FOR_FREEZE`

Activation condition:

`ON_EXACT_COMMIT_PUSH_REMOTE_VERIFICATION`

Decision phase:

`SCIENTIFIC_CONTRACT_REMEDY_DECISION_AUTHORITY_FINALIZED_PRE_ACTIVATION_CONTENT`

Selected remedy class:

`SELECTED_REMEDY_CLASS = SPLIT_CONTRACT_CHANGE`

Decision structure:

`DECISION_STRUCTURE = MULTIPLE_NONDOMINATED_SCIENTIFIC_BRANCHES`

User-selected branch:

`USER_SELECTED_BRANCH = SPLIT_CONTRACT_CHANGE`

Execution:

`CURRENT_EXECUTION_STATE = BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`

Training/Evaluation/Kaggle/CUDA:

`NOT_AUTHORIZED`

This finalized pre-activation content records only the user's explicit remedy-class selection. It does not select an exact replacement split seed, generate a new split, regenerate sidecars, rerun A0, recalibrate, implement preflight logic, authorize execution, authorize training/evaluation/Kaggle/CUDA, stage, commit, push, or modify any existing artifact.

## 2. Authority Used

Highest authority for this materialization:

1. Current explicit user scientific decision: `SELECTED_REMEDY_CLASS = SPLIT_CONTRACT_CHANGE`.
2. Active P2 root-cause authority: `eea0714904ea1f95c42da48e85cd1af4bad23123`.
3. Active authority-lineage reconciliation: `1bb08179adb38637e9391491ba72cfd7e9bff3b3`.
4. Active unauthorized-execution incident correction: `0f6e00642fb6126ec86d7b7dde4b84626befca67`.
5. Final verified decision-support evidence:
   `reports/reason_router_p3w7_p2_degeneracy_scientific_contract_remedy_decision_spec_candidate.md`
   with lifecycle `FINAL_SCIENTIFIC_CONTRACT_REMEDY_DECISION_SUPPORT_EVIDENCE`,
   independent verdict `PASS_READY_FOR_SCIENTIFIC_DECISION`,
   and finalized SHA256 `dfea152d952ca610ea3a218931122f9f9760625ac23cdcf20a68d9aef3ea3fe7`.
6. Repository contract: `AGENTS.md`.

Support evidence path:
`reports/reason_router_p3w7_p2_degeneracy_scientific_contract_remedy_decision_spec_candidate.md`

Support evidence lifecycle:
`FINAL_SCIENTIFIC_CONTRACT_REMEDY_DECISION_SUPPORT_EVIDENCE`

Support evidence independent verdict:
`PASS_READY_FOR_SCIENTIFIC_DECISION`

Support evidence finalized SHA256:
`dfea152d952ca610ea3a218931122f9f9760625ac23cdcf20a68d9aef3ea3fe7`

The decision-support report is evidence/support only, not authority. It established:

`DECISION_STRUCTURE_MULTIPLE_NONDOMINATED_SCIENTIFIC_BRANCHES`

and independent verifier verdict:

`PASS_READY_FOR_SCIENTIFIC_DECISION`

The current user's explicit selection resolves the scientific branch choice in favor of `SPLIT_CONTRACT_CHANGE`. This finalized pre-activation content does not claim that the split-contract branch was uniquely dominant.

## 2.1 Activation Preconditions

This exact finalized authority content becomes eligible to be treated as active only after all of the following are complete:

1. The finalized decision-support report and this finalized authority report are explicitly staged together.
2. Their exact staged Git blobs are independently byte-verified.
3. A dedicated freeze/activation commit is created with exactly the intended two-file delta.
4. That exact full commit SHA is pushed.
5. The remote branch tip is verified.
6. The parent commit is verified.
7. Both remote blobs are verified against the finalized local identities.
8. Body-level lifecycle/status content is independently remote-verified.

`COMMIT_MESSAGE_DOES_NOT_OVERRIDE_BODY_LEVEL_AUTHORITY_STATUS`

Commit subject alone cannot activate authority. Before those conditions complete:

`ACTIVE_SPLIT_CONTRACT_REMEDY_DECISION_AUTHORITY = NONE_YET`

After successful exact remote verification, the controller may recognize the dedicated commit as the active split-contract remedy decision authority.

## 3. Direct Objective

Resolved by class-level decision only:

`FROZEN_SEED174_SPLIT_CONTRACT_INCOMPATIBLE_WITH_A1_A3_DEV_POLARITY_BINARY_READINESS`

The selected scientific remedy class is to revise only the pair-level train/dev split contract while preserving the substantive A1/A3 reason-supervision and dev-readiness semantics.

## 4. Preserved Scientific Invariants

A future split-contract revision must preserve the following unless a later explicit authority says otherwise:

1. Dataset content/semantic identity: no dataset regeneration, row addition/removal, or text/content changes.
2. Labels: no label changes.
3. Reason eligibility: no eligibility changes.
4. P2 applicability semantics: no applicability changes.
5. Polarity mapping: `REFUTE -> 0`, `SUPPORT -> 1`.
6. Pair-group split behavior: canonical pair groups remain intact, with no train/dev pair leakage.
7. Dev ratio: `0.2`.
8. Training seeds: `180 / 181 / 182`; these remain factorial training seeds and are not a substitute for the split seed.
9. A1/A3 reason-supervision semantics: remain enabled and must not be weakened to solve the blocker.
10. A2 reason supervision: remains disabled.
11. Current A1/A3 fail-closed minimum-count/readiness semantics: preserve current guard semantics; do not suppress or bypass the exception; a future revised split must satisfy the guard.
12. Dev reason-validation semantics: preserve current semantics; do not adopt V1/V2/V3 through this authority.
13. Router / gradient-ownership semantics: no change.
14. Final 3-way CE ownership semantics: no change.
15. Primary reason order: `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`.
16. Secondary reasons: diagnostic only.

## 5. Changed Contract

The frozen split seed `174` is no longer eligible as the future A1/A2/A3 factorial split because the active root-cause authority establishes that it deterministically violates A1/A3 dev polarity readiness.

Exact replacement split status:

`EXACT_REPLACEMENT_SPLIT_SEED = NOT_YET_SELECTED`

This authority does not select seed `0` or any other candidate. The bounded scan over `0..9999` is not permission to choose an arbitrary feasible seed.

## 6. Future Split-Selection Constraints

A subsequent dedicated split-contract revision/design authority must choose the exact replacement split using a pre-registered criterion that:

- is independent of model training/evaluation outcomes;
- does not cherry-pick `final_macro_f1` or any model metric;
- uses the frozen dataset and current P2 applicability/readiness semantics only;
- preserves pair integrity;
- satisfies all current A1/A3 applicable-cohort minimum-count conditions in both train and dev;
- records exact train/dev identities;
- records all relevant reason/applicable binary counts;
- is reproducible and deterministic;
- explicitly justifies the selection/tie-break criterion;
- is independently verified before any regeneration or execution.

The exact seed-selection criterion is not selected here unless already logically forced by active authority. This finalized pre-activation content does not invent one.

## 7. Empirical Decision-Support Context

The following is recorded only as verified decision-support evidence:

- scanned split seeds: `0..9999`;
- feasible: `9991 / 10000 = 99.91%`;
- seed `174` is the sole polarity-degenerate split in that bounded scan;
- eight additional seeds failed a different current dev `PREDICATE` minimum-readiness condition:
  `1617`, `1746`, `2929`, `4751`, `6125`, `6185`, `7907`, `9753`.

No population-wide probability claim is made from this bounded scan. No scanned seed is promoted to selected status.

## 8. Dependency Cascade

A split-contract change invalidates or requires replacement/revalidation of split-conditioned evidence:

| Dependency | Classification | Required later handling |
|---|---|---|
| Current P4-L sidecar | `INCOMPATIBLE_WITH_REVISED_SPLIT` | Requires regeneration/reconstruction under separate authority. |
| Current P4-L sidecar provenance | `INCOMPATIBLE_WITH_REVISED_SPLIT` | Requires new provenance. |
| Ordered train/dev identity hashes | `REQUIRES_RECOMPUTATION` | Must be recomputed for the finalized split. |
| Current same-seed A0 reference prediction artifacts | `NOT_ADMISSIBLE_AS_REVISED_SPLIT_REFERENCES` | Must not be reused as exact revised-split references. |
| A0 baseline comparability | `REQUIRES_NEW_REVISED_SPLIT_A0_EXECUTION/EVIDENCE` | Requires separate explicit authority. |
| Existing accepted reason-loss calibration `0.6202430063306562` | `REQUIRES_RECALIBRATION` | Must not be automatically carried forward. |
| Existing A1/A2/A3 factorial matrix | `SEMANTIC_ARM_DEFINITIONS_MAY_BE_PRESERVED` | Exact split-bound execution contract requires a new authority. |
| Existing promotion/comparison contract | `REQUIRES_EXPLICIT_REBIND_OR_REVALIDATION` | Must bind to revised split evidence before use. |
| Current source/tests | `MAY_REMAIN_PRESERVED_WHERE_EXISTING_SEMANTICS_ARE_ENCODED` | No implementation conclusion is authorized here. |
| Current factorial execution authority | `NOT_VALID_FOR_REVISED_SPLIT_EXECUTION` | Cannot authorize revised-split execution. |

## 9. A0 Boundary

Existing A0 artifacts remain valid historical scientific evidence for their original frozen split lineage. This authority does not invalidate their historical claims.

They must not be reused as exact same-seed A0 references for the revised split factorial. A revised split therefore requires a separately authorized A0 baseline evidence path before A1/A2/A3 revised-split execution.

## 10. Calibration Boundary

The accepted calibration value:

`0.6202430063306562`

remains valid historical/current-seed174-lineage accepted calibration evidence only.

It is not the authorized future A1/A3 reason-loss weight for a revised split until a new calibration process is explicitly authorized, executed, validated, and accepted.

Classification:

`REQUIRES_RECALIBRATION`

No new weight is chosen here.

## 11. Prelaunch Companion-Control Boundary

The active root-cause authority also established:

`MISSING_PRELAUNCH_P2_APPLICABLE_COHORT_FEASIBILITY_CHECK`

As a non-scientific companion control, any future revised-split execution authority must include a static prelaunch P2 applicable-cohort feasibility check before trainer launch.

This check:

- does not constitute the scientific root-cause remedy;
- must verify the exact finalized split;
- must fail closed before `TRAINER_PROCESS_LAUNCH_BEGIN`;
- must use the same current readiness semantics as the trainer;
- must not weaken the trainer guard;
- must not authorize execution by itself.

This authority does not implement the check.

## 12. Non-Selected Alternatives

`VALIDATION_OR_REASON_SUPERVISION_CONTRACT_CHANGE = NOT_SELECTED`

`DATASET_OR_ELIGIBILITY_CONTRACT_CHANGE = NOT_SELECTED`

`PRELAUNCH_FEASIBILITY_VALIDATION_ONLY = NOT_A_STANDALONE_ROOT_CAUSE_REMEDY`

The validation/reason-supervision branch was scientifically viable and non-dominated, but the user did not choose it. This finalized pre-activation content does not characterize that branch as invalid.

## 13. Prohibited Non-Remedies

The following remain prohibited as substitutes for the selected scientific remedy:

- run A2 first;
- skip seed180;
- retry A1 unchanged;
- recovery4 unchanged;
- change training seed but retain split174;
- add only another marker;
- suppress current exception;
- weaken current guard.

## 14. Execution Boundary

`ACTIVE_SPLIT_CONTRACT_REMEDY_DECISION_AUTHORITY = NONE_YET`

`CURRENT_EXECUTION_STATE = BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`

`A1/A2/A3 execution = BLOCKED`

`Training = NOT_AUTHORIZED`

`Evaluation = NOT_AUTHORIZED`

`Kaggle = NOT_AUTHORIZED`

`CUDA = NOT_AUTHORIZED`

No run name may be created. No recovery4 may be created.

## 15. Future Authority Sequence

After this decision authority is independently verified and truly made authoritative by the required later process, the next sequence is:

1. Dedicated revised split-contract design/selection authority: exact replacement split, exact deterministic selection rule, exact train/dev row/pair identities, exact readiness counts, no model execution.
2. P4-L revised-split regeneration/rebinding authority as required.
3. Revised-split A0 baseline execution/evidence authority.
4. Revised-split reason-loss recalibration authority and acceptance.
5. Revised factorial scientific/execution contract.
6. Any required implementation/prelaunch-control authority.
7. Validation/freeze.
8. New explicit execution authority.
9. Only then Kaggle/training.

These phases must not be collapsed automatically if their evidence dependencies require separation. This finalized pre-activation content authorizes none of them.

## 16. Critical Non-Inheritance

This selection is not inherited from:

- premature factorial-v2 split-contract artifact;
- premature root-cause reports/candidates;
- any non-authorized historical remedy proposal.

The `SPLIT_CONTRACT_CHANGE` selection comes from the current explicit user decision, informed by the independently verified fresh remedy audit.

## 17. Finalized Content Notes

Finalized content path:

`reports/reason_router_p3w7_p2_degeneracy_split_contract_remedy_decision_authority_spec_candidate.md`

Expected finalization state:

- Branch: `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`
- HEAD: `eea0714904ea1f95c42da48e85cd1af4bad23123`
- Initial worktree/index: exactly two pre-existing untracked reports, no tracked modifications, no staged changes.
- Final worktree/index: exactly two untracked reports, no tracked modifications, no staged changes.

Training/evaluation/model loading/checkpoint loading/CUDA/Kaggle:

`NOT_RUN_NOT_AUTHORIZED`

Final file SHA256, byte count, line-ending facts, trailing-whitespace status, `git status --short`, `git diff --name-status`, `git diff --cached --name-status`, and `git diff --check` are reported outside this file to avoid self-referential candidate content.
