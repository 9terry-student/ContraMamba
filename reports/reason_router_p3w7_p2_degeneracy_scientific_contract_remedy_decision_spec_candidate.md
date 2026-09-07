# P3-W7 P2 Degeneracy Scientific-Contract Remedy Decision Support Evidence

## 1. Verdict

Lifecycle status: `FINAL_SCIENTIFIC_CONTRACT_REMEDY_DECISION_SUPPORT_EVIDENCE`

Freeze readiness: `PASS_READY_FOR_FREEZE`

Independent verification: `INDEPENDENT_VERIFICATION = PASS_READY_FOR_SCIENTIFIC_DECISION`

Decision phase: `SCIENTIFIC_REMEDY_SELECTION_NOT_MADE_BY_THIS_SUPPORT_REPORT`

Execution: `BLOCKED`

Training/Evaluation/Kaggle: `NOT_AUTHORIZED`

Decision-structure classification:

`DECISION_STRUCTURE_MULTIPLE_NONDOMINATED_SCIENTIFIC_BRANCHES`

This report is final verified decision-support evidence only. It determines viable remedy classes and their dependency cascades, but it does not itself select or activate a remedy. The later user selection is recorded by the separate split-contract remedy decision authority document. This report is not execution authority and authorizes no implementation, training, evaluation, Kaggle, recalibration, A0 rerun, split generation, staging, commit, push, or change to split/data/sidecar/labels/applicability/guards.

## 2. Authority Used

Current controller instruction is highest authority for this report.

Primary active root-cause authority:

`eea0714904ea1f95c42da48e85cd1af4bad23123`

Active authority-lineage reconciliation:

`1bb08179adb38637e9391491ba72cfd7e9bff3b3`

Active unauthorized-execution incident correction:

`0f6e00642fb6126ec86d7b7dde4b84626befca67`

Active root cause consumed without reopening:

`FROZEN_SEED174_SPLIT_CONTRACT_INCOMPATIBLE_WITH_A1_A3_DEV_POLARITY_BINARY_READINESS`

Secondary prevention defect consumed:

`MISSING_PRELAUNCH_P2_APPLICABLE_COHORT_FEASIBILITY_CHECK`

Current branch/HEAD gate:

| Field | Required | Observed | Result |
|---|---|---|---|
| Branch | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` | PASS |
| HEAD | `eea0714904ea1f95c42da48e85cd1af4bad23123` | `eea0714904ea1f95c42da48e85cd1af4bad23123` | PASS |
| Initial `git status --short` | empty | empty | PASS |

## 3. Files And Artifacts Inspected

- `AGENTS.md`
- `reports/reason_router_p3w7_a1_seed180_p2_degeneracy_fresh_read_only_root_cause_audit_spec_candidate.md`
- `reports/reason_router_p3w7_a1_seed180_authority_lineage_reconciliation_spec_candidate.md`
- `reports/reason_router_p3w7_a1_seed180_recovery3_unauthorized_execution_incident_authority_correction_spec_candidate.md`
- `reports/reason_router_p3w7_a1_a2_a3_factorial_execution_authority_spec_candidate.md`
- `reports/reason_router_p3w7_a0_n3_validated_evidence_analysis_authority_spec_candidate.md`
- `reports/reason_router_p3w7_a0_n3_validated_evidence_analysis_report.md`
- `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`
- `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`
- `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`
- `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`
- `scripts/train_controlled_v6b_minimal.py`
- `scripts/build_controlled_v5.py`
- `tests/test_reason_router_p2_contract.py`
- Commit-addressed current-lineage calibration acceptance at `1221588b78d02900ee93cff36cf37b2202e04aea`
- Commit-addressed current-lineage calibration authority at `ba3fd1a82cba029dac05ba38a86d518252ab858f`
- Commit-addressed admitted A0 reference files for seeds `180`, `181`, and `182`

## 4. Immutable Baseline Contract

Scientific contract preserved absent remedy:

| Field | Frozen/current value |
|---|---|
| Dataset | `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` |
| Dataset rows | `3600` |
| Dataset physical SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| Dataset semantic SHA256 | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| P4-L sidecar | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl` |
| Sidecar rows | `3600` |
| Sidecar physical SHA256 | `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1` |
| Sidecar semantic SHA256 | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` |
| Sidecar provenance physical SHA256 | `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2` |
| Split seed | `174` |
| Dev ratio | `0.2` |
| Split semantics | sorted unique `pair_id`, `random.Random(seed).shuffle`, `round(pair_count * dev_ratio)` dev pairs, all rows in a pair remain together |
| Train/dev size | `240`/`60` pairs; `2880`/`720` rows |
| A1/A3 reason supervision | enabled |
| A2 reason supervision | disabled |
| Reason-loss weight A1/A3 | `0.6202430063306562` |
| Reason-loss weight A2 | `0.0` |
| A0 reference binding | exact same-seed admitted `training_report_predictions.jsonl`; no cross-seed substitution |
| Training seeds | `180`, `181`, `182` |
| Current selection metric | `final_macro_f1`, clean-dev internal only |
| P2 applicability | frame: eligible; predicate: eligible and frame compatible; sufficiency: eligible, frame compatible, predicate covered; polarity: eligible, frame compatible, predicate covered, sufficient, final label in `{REFUTE, SUPPORT}` |
| A1/A3 binary readiness guard | primary reason min counts `train >= 50`, `dev >= 20`; applicable binary cohorts require both classes with minimum `>= 1` in train and dev |

Scientific contract is the frozen data/sidecar/split/reference/supervision/calibration/selection meaning above. Execution authority is the separate permission layer that would allow a concrete trainer launch. Current execution authority is blocked by the active root-cause and incident-correction lineage.

## 5. Split-Seed Feasibility Scan

Read-only deterministic scan scope:

- Reconstructed seed `174` using `scripts/build_controlled_v5.py::split_by_pair_id` semantics.
- Scanned split seeds `0..9999` inclusive.
- Used `dev_ratio=0.2`, `300` sorted pairs, `60` dev pairs, `240` train pairs.
- Used current row-level P2 eligibility/applicability from the frozen dataset plus current P4-L sidecar fields.
- For alternate splits, this scan is distributional feasibility only. The existing P4-L sidecar embeds seed174 split labels and would not remain unchanged for execution under a replacement split.

Seed174 reconstruction:

| Quantity | Value |
|---|---:|
| Train pairs / rows | `240` / `2880` |
| Dev pairs / rows | `60` / `720` |
| Pair leakage | `0` |
| Train primary counts | FRAME `726`, PREDICATE `121`, SUFFICIENCY `242`, AUTHORIZED `361` |
| Dev primary counts | FRAME `174`, PREDICATE `29`, SUFFICIENCY `58`, AUTHORIZED `58` |
| Train polarity applicable | REFUTE `119`, SUPPORT `242` |
| Dev polarity applicable | REFUTE `0`, SUPPORT `58` |
| Failure payload | `{'dev': {'polarity': {0: 0, 1: 58}}}` |

Scan results:

| Metric | Value |
|---|---:|
| Seeds scanned | `10000` |
| Structurally valid pair splits | `10000` |
| Feasible seeds | `9991` |
| Feasible percentage | `99.91%` |
| Infeasible seeds | `9` |
| Infeasible specifically due dev polarity degeneracy | `1` |
| Other infeasible seeds | `8` |

The one dev-polarity-degeneracy seed in the bounded scan is seed `174`. The eight other infeasible seeds fail the current dev primary-reason minimum for `PREDICATE` (`17..19 < 20`) while their polarity cohorts are binary-present.

First 10 feasible seeds:

| Seed | Dev REFUTE | Dev SUPPORT | Train REFUTE | Train SUPPORT | Dev pair overlap with seed174 dev |
|---:|---:|---:|---:|---:|---:|
| 0 | 19 | 68 | 100 | 232 | 15 |
| 1 | 23 | 60 | 96 | 240 | 13 |
| 2 | 21 | 62 | 98 | 238 | 14 |
| 3 | 27 | 60 | 92 | 240 | 10 |
| 4 | 20 | 64 | 99 | 236 | 13 |
| 5 | 18 | 74 | 101 | 226 | 13 |
| 6 | 28 | 56 | 91 | 244 | 8 |
| 7 | 19 | 62 | 100 | 238 | 16 |
| 8 | 28 | 58 | 91 | 242 | 8 |
| 9 | 24 | 54 | 95 | 246 | 14 |

Conclusion for remedy class 1 feasibility: `SPLIT_CONTRACT_CHANGE` can make the frozen dataset and current A1/A3 row-level reason-supervision/applicability contract distributionally feasible by changing the pair split, but not while preserving the current seed174 split-bound sidecar/provenance, ordered identity hashes, A0 dev-reference files, accepted calibration, or execution authority unchanged.

## 6. Split-Change Dependency Cascade

For a hypothetical valid replacement split:

| Dependency | Classification | Evidence basis |
|---|---|---|
| Frozen P4-L sidecar | `INCOMPATIBLE_WITH_CURRENT_ARTIFACT` | P4-L sidecar rows include `split`; P4-L provenance binds `split_rule.shuffle_seed=174`; trainer checks sidecar split against runtime split. |
| Sidecar provenance | `INCOMPATIBLE_WITH_CURRENT_ARTIFACT` | Provenance binds source dataset, row order, split rule seed174, sidecar physical/semantic hashes, and seed174-derived artifact identities. |
| Ordered train/dev identity hashes | `REQUIRES_RECOMPUTATION` | Trainer computes P2 train/dev identity from ordered `(id, pair_id)` rows; calibration aggregate binds ordered train identity for seed174. |
| Exact same-seed A0 prediction reference files | `REQUIRES_RERUN` | Existing admitted references are 720-row seed174 dev populations. Static check showed seed0 would miss 540 required dev rows in every same-seed A0 reference file. |
| A0 baseline scientific comparability | `REQUIRES_RERUN` | A1/A2/A3 comparisons require same-seed A0 references over the same dev universe; current A0 descriptive baseline is seed174-bound. |
| Accepted reason-loss calibration weight `0.6202430063306562` | `REQUIRES_RECALIBRATION` | Acceptance binds `split_seed=174`, `ordered_train_row_count=2880`, and ordered train row identity `cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784`; calibration scope is train-only. |
| Existing A1/A2/A3 factorial matrix | `REQUIRES_NEW_AUTHORITY_ONLY` | Arm definitions can remain conceptually identical, but the current authority releases exact seed174/same-reference/current-calibration execution. |
| Promotion/comparison criteria | `REQUIRES_NEW_AUTHORITY_ONLY` | Criteria can be restated, but comparisons would be over a new split/reference/calibration contract. |
| Current tests | `PRESERVED_EXACTLY` | P2 tests enforce source semantics, A0-reference requirement, and fail-closed degenerate-cohort behavior; split replacement does not require weakening them. Hash-bound future tests would need separate authority if introduced. |
| Execution authority | `INCOMPATIBLE_WITH_CURRENT_ARTIFACT` | Current factorial authority explicitly binds split seed174, current P4-L identities, current calibration, and admitted seed174 A0 references. |

## 7. Validation/Reason-Supervision Branch Analysis

Frozen under this remedy class:

- dataset
- P4-L sidecar
- split seed `174`
- dev ratio `0.2`

Current guard protection:

- `_p2_prepare_reason_supervision` builds train and dev P2 supervision targets before model construction/training.
- For A1/A3, `require_min_counts=True`.
- Primary reason classes must satisfy `min_train_count=50` and `min_dev_count=20`.
- Each applicable binary cohort must contain both class `0` and class `1` in train and dev; the source minimum is `>=1`, not `>1`.
- The fail-closed exception prevents a run whose dev polarity applicable cohort is REFUTE0/SUPPORT58 from being treated as normal A1/A3 executable evidence.
- Training reason loss uses train tensors. Dev reason tensors feed loss/diagnostic exports. Checkpoint selection remains `final_macro_f1` from clean-dev final-label metrics unless separately changed.

Conceptual sub-classes:

| Sub-class | Viability | Training objective changes? | Dev evaluation semantics change? | Model selection changes? | A0 references usable? | Calibration applicable? | A1/A3 interpretation changes? | Safeguard weakened? |
|---|---|---|---|---|---|---|---|---|
| V1: train applicable binary readiness mandatory; dev single-class reason cohort permitted | `VIABLE_REQUIRES_NEW_AUTHORITY` | No, if train tensors/loss remain unchanged | Yes. Dev polarity is single-class but still accepted as evaluable/diagnostic input | No under current `final_macro_f1`; yes if future selector consumed dev reason metrics | Yes, because seed174 dev universe stays unchanged | Yes, because train split identity and calibration scope remain unchanged | Limited/material boundary: A1/A3 would no longer require binary dev reason-readiness | Yes |
| V2: dev reason polarity supervision/metric excluded or marked non-evaluable when single-class | `VIABLE_REQUIRES_NEW_AUTHORITY_AND_IMPLEMENTATION` | No, if exclusion is dev-only and train loss remains unchanged | Yes. Dev polarity reason validation becomes explicitly non-evaluable for that cohort | No under current `final_macro_f1`; must be blocked for any reason-metric selector unless separately specified | Yes | Yes | Limited: preserves A1/A3 train objective but narrows dev reason-validation claim | Yes, though less than a broad suppression if explicitly non-evaluable |
| V3: broader binary readiness criterion change | `UNKNOWN_OR_MATERIAL_REDESIGN` | Unknown; could change train objective if train readiness is weakened | Yes | Unknown; could alter selection if reason metrics become selectable or gate-like | Likely yes if seed174 dev remains, but depends on exact redesign | Likely yes only if train identity/calibration inputs remain unchanged | Material if it changes train readiness, applicability, labels, or reason loss | Yes |

Conclusion: `VALIDATION_OR_REASON_SUPERVISION_CONTRACT_CHANGE` can resolve the direct seed174 incompatibility while preserving dataset, sidecar, seed174 dev set, admitted A0 references, and accepted train-only calibration if constrained to dev-readiness semantics. Its cost is scientific: it changes the original A1/A3 dev reason-validation contract and weakens an intentional fail-closed safeguard.

## 8. Dataset/Eligibility Branch Analysis

This branch keeps seed174 if possible but modifies dataset composition, labels, eligibility, applicability semantics, or sidecar annotations.

Consequences:

| Item | Classification |
|---|---|
| P4-L sidecar/provenance | `REQUIRES_REGENERATION` or `INCOMPATIBLE_WITH_CURRENT_ARTIFACT` |
| A0 references | `REQUIRES_RERUN` if row identities, labels, applicability, or dev universe change |
| Calibration | `REQUIRES_RECALIBRATION` if train identity, labels, eligibility, or reason-loss population changes |
| Train/dev identities | `REQUIRES_RECOMPUTATION` |
| Factorial comparability | `REQUIRES_NEW_AUTHORITY_ONLY` at minimum; likely `REQUIRES_RERUN` for all affected arms |
| Existing scientific claims | `MATERIAL` scope change; current P4-L/A0/calibration claims no longer directly transfer |

Relative scientific scope: broader than split-only and validation-only remedies. Split-only changes the sample partition while preserving row semantics. Validation-only preserves row/split/reference identity but changes the readiness/evaluation contract. Dataset/eligibility changes alter the evidence universe or row-level entitlement semantics themselves; this is the broadest scientific question change among direct remedy classes.

## 9. Prelaunch-Validation Assessment

Adding a static prelaunch feasibility check alone does not resolve the direct root cause.

Evidence:

- The active root cause is seed174 split incompatibility with A1/A3 dev polarity binary readiness.
- Static reconstruction still yields seed174 dev polarity REFUTE0/SUPPORT58.
- `_p2_prepare_reason_supervision` intentionally raises `P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE` for A1/A3 under that state.

Classification:

| Question | Assessment |
|---|---|
| Standalone remedy to direct root cause? | `NO` |
| Companion control for future execution authority? | Strongly justified. It would prevent unauthorized/wasted trainer launch by detecting the same static infeasibility before model/trainer execution. |

Prelaunch validation is a prevention remedy for the secondary control defect, not a scientific remedy for seed174 A1/A3 executability.

## 10. Non-Remedy Assessment

| Candidate | Remedy? | Reason |
|---|---|---|
| Run A2 first | `NO` | A2 reason supervision is disabled and is not subject to the same A1/A3 guard, but current factorial progression is blocked by authority; running A2 does not make A1/A3 seed174 dev polarity binary-ready. |
| Skip seed180 only | `NO` | The incompatibility is split seed174, independent of training seed; A1/A3 seeds `181` and `182` would face the same dev cohort. |
| Retry A1 unchanged | `NO` | Deterministic seed174 reconstruction reproduces REFUTE0/SUPPORT58 and the same guard exception. |
| recovery4 unchanged | `NO` | A new run name or recovery wrapper without scientific-contract change still reaches the same infeasible contract. |
| Change training seed `180` to another training seed while keeping split seed174 | `NO` | Training seed changes initialization/order randomness, not the deterministic split-defined dev applicable polarity counts. |
| Simply add another preflight marker | `NO` | A marker does not change split, validation semantics, data, eligibility, or the underlying guard condition. |
| Suppress the exception without changing the underlying scientific contract | `NO` | This bypasses the intentional fail-closed safeguard and creates invalid evidence under the current contract. |

## 11. Scientific Preservation Matrix

| Remedy class | Fixes direct seed174 incompatibility? | Preserves frozen dataset? | Preserves seed174 dev set? | Preserves P4-L sidecar? | Preserves A0 references? | Preserves accepted calibration? | Preserves original A1/A3 training objective? | Preserves original dev reason-validation semantics? | Preserves original factorial interpretation? | Requires new A0 execution? | Requires recalibration? | Requires data/sidecar regeneration? | Authority cascade breadth | Scientific question changed? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| split-contract change | YES | YES | NO | NO | NO | NO | YES | YES | LIMITED/UNKNOWN until new A0/calibration are rebuilt | YES | YES | YES for sidecar/provenance | BROAD | LIMITED |
| validation/reason-supervision contract change | YES | YES | YES | YES | YES | YES if train-only calibration untouched | YES for V1/V2 dev-only changes | NO | LIMITED; original dev-readiness claim changes | NO | NO | NO | MODERATE | MATERIAL |
| dataset/eligibility contract change | YES possible, not designed here | NO/UNKNOWN | UNKNOWN | NO | NO/UNKNOWN | NO/UNKNOWN | UNKNOWN | UNKNOWN | NO/UNKNOWN | YES likely | YES likely | YES | BROAD | MATERIAL |
| prelaunch feasibility validation only | NO | YES | YES | YES | YES | YES | YES | YES | YES, but remains blocked | NO | NO | NO | NARROW | NO |

Every non-UNKNOWN cell above is supported by the inspected source/contracts/artifacts named in Sections 2-3. UNKNOWN marks cases where the exact result depends on a future not-yet-authorized design.

## 12. Decision-Structure Classification

`DECISION_STRUCTURE_MULTIPLE_NONDOMINATED_SCIENTIFIC_BRANCHES`

Static evidence does not establish a single dominant remedy class.

- Split-contract change resolves the direct blocker while preserving original A1/A3 supervision/readiness semantics, but it invalidates seed174 dev identity, current P4-L split-bound sidecar/provenance, exact A0 references, accepted train-only calibration, and current execution authority.
- Validation/reason-supervision contract change resolves the direct blocker while preserving frozen dataset, seed174 dev set, P4-L sidecar, A0 references, and accepted calibration, but changes the original dev reason-validation/readiness semantics and weakens an intentional fail-closed safeguard.
- Dataset/eligibility change is scientifically broader and changes the evidence universe or row-level semantics.
- Prelaunch feasibility validation alone prevents the secondary launch-control failure but cannot make seed174 A1/A3 executable.

Because the direct viable branches trade off preservation of split/data/reference/calibration identity against preservation of original dev reason-validation semantics, they are non-dominated scientific choices rather than a routine engineering preference.

## 13. User-Decision Table

| Branch | Scientific contract changes | Preserved | Must rerun/regenerate/recalibrate | Main validity risk | Main reproducibility/provenance cost |
|---|---|---|---|---|---|
| Split-contract change | Replace seed174 pair split with a feasible split under the same P2 readiness semantics | Dataset row content; A1/A3 train objective; binary dev-readiness principle; arm concepts | Regenerate sidecar/provenance for new split; rerun A0 same-seed references; recalibrate reason weight; re-authorize factorial execution | New split may answer a slightly different sampled-dev question and loses direct seed174 comparability | Broad artifact cascade across sidecar, A0, calibration, and execution authority |
| Validation/reason-supervision contract change | Keep seed174 but permit or mark non-evaluable the single-class dev polarity reason cohort | Dataset; seed174 dev set; current P4-L sidecar; admitted A0 references; accepted train-only calibration; A1/A3 train objective if dev-only | New scientific-contract authority; likely implementation/tests if V2 or explicit non-evaluable export is chosen | Weakens intentional fail-closed dev reason-readiness safeguard; narrows interpretation of dev polarity reason diagnostics | Moderate; fewer artifacts rerun, but authority must document changed validation semantics |
| Dataset/eligibility contract change | Modify row universe, labels, eligibility, applicability, or sidecar annotations | Possibly seed174 split seed only, depending on design | Regenerate data/sidecar/provenance; likely rerun A0 and A1/A2/A3; recalibrate; revalidate claims | Changes evidence-entitlement question itself and risks label/applicability leakage or semantic drift | Broadest cascade and hardest provenance comparison to current line |

No question is asked here; controller selection remains outside this report.

## 14. Authority Cascade After Any Future Choice

Required layering after any future scientific choice:

1. Scientific-contract remedy decision authority: explicitly choose the remedy class and define the scientific question, preservation/invalidation policy, and artifact cascade.
2. Implementation authority if code/data/sidecar/test changes are needed: whitelist exact files/symbols and forbid unrelated refactors.
3. Implementation validation/freeze: verify source/contracts/artifacts, line endings, hashes, and tests under the chosen scope.
4. New explicit execution authority: bind exact commit, command, data/sidecar/reference/calibration identities, run names, output namespace, and prelaunch feasibility checks.
5. Only then Kaggle/training: no current execution, CUDA, evaluation, or Kaggle run is authorized.

## 15. Blockers And Unknowns

Blockers:

- `Execution = BLOCKED`
- `Training/Evaluation/Kaggle = NOT_AUTHORIZED`
- No remedy class is selected by this report.

Unknowns:

- Exact implementation surface for V2/V3 is not designed or authorized.
- Exact replacement split seed, if any, is not selected or authorized.
- Exact regenerated sidecar/provenance hashes for any split/data/eligibility branch are unknown by design.
- Exact recalibrated reason-loss weight under any changed train split or changed eligibility population is unknown until separately authorized recalibration.

## 16. Validation Performed

Commands actually executed:

- `git rev-parse --abbrev-ref HEAD`
- `git rev-parse HEAD`
- `git status --short`
- `git show --stat --oneline --decorate --no-renames HEAD`
- targeted `Get-Content`, `rg`, and `git show` reads over the files/artifacts listed in Section 3
- read-only Python split reconstruction and seed scan over frozen JSONL data/sidecar
- read-only commit-addressed A0 reference coverage checks

Training/evaluation/model loading/checkpoint loading/CUDA/Kaggle: `NOT_RUN_NOT_AUTHORIZED`

## 17. Final Lifecycle Status

Final report identity, newline shape, `git status` after, `git diff --name-status`, `git diff --cached --name-status`, and `git diff --check` are intentionally not self-embedded in this support-evidence body. They are reported in the completion note for this finalization task.

Exact next authorized action:

`NEXT_LIFECYCLE_ACTION = COORDINATED_FREEZE_WITH_REFERENCING_DECISION_AUTHORITY_AFTER_FINAL_VERIFICATION`
