# P3-W7 A1 Seed180 P2 Degeneracy Fresh Read-Only Root-Cause Audit Candidate

Authority status:

`FINAL_P2_DEGENERACY_ROOT_CAUSE_AUDIT_CONTENT`

`PASS_READY_FOR_FREEZE`

`ON_EXACT_COMMIT_PUSH_REMOTE_VERIFICATION`

Audit result: `PASS_ROOT_CAUSE_RESOLVED`

Resolved root cause: `FROZEN_SEED174_SPLIT_CONTRACT_INCOMPATIBLE_WITH_A1_A3_DEV_POLARITY_BINARY_READINESS`

Secondary prevention defect: `MISSING_PRELAUNCH_P2_APPLICABLE_COHORT_FEASIBILITY_CHECK`

This report is activation-safe final root-cause audit content for future exact commit/push/remote verification. It is not active authority in the working tree, is not execution authority, does not select a scientific remedy, and does not authorize implementation, training, evaluation, Kaggle use, staging, commit, or push.

This exact finalized byte content is eligible to become active only when it is explicitly staged, committed, pushed, and independently remote-verified at the expected full commit SHA, parent, branch tip, file delta, and body content. Commit subject alone cannot activate authority. Body-level status is controlling. Until remote verification completes, execution remains blocked.

## 1. Authority Boundary

Active authority-lineage reconciliation commit:

`1bb08179adb38637e9391491ba72cfd7e9bff3b3`

Active unauthorized-execution incident-correction authority commit:

`0f6e00642fb6126ec86d7b7dde4b84626befca67`

Current repository HEAD observed for this audit:

`0f6e00642fb6126ec86d7b7dde4b84626befca67`

Current branch observed for this audit:

`p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`

Before future exact commit/push/remote verification occurs:

`ACTIVE_P2_ROOT_CAUSE_AUTHORITY = NONE_YET`

`CURRENT_EXECUTION_STATE = BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`

The substantive conclusions of commit `270df96f9f217cc7c1aad49d2c73ae551e560e28`, premature root-cause candidate SHA256 `5fe9782596aa5643827e7166ef02dc9b1453944ca71dddb1808a8a69b352590c`, premature root-cause candidate SHA256 `2f30584da3ab2d2ed52950780f0a494fd476d4376f95471af8715d886b5a98f7`, and premature factorial-v2 split-contract artifact SHA256 `c725570b08b583ae23766ad6ed1399671d63d8337846c85f2c4032f7f7c70601` are not inherited. They are provenance/history only.

## 2. Authenticated Recovery3 Incident

The authenticated incident evidence records:

| Field | Value |
|---|---|
| Run | `p3w7-factorial-a1-seed180-recovery3-auth98723fe` |
| Executed HEAD | `98723fe27ba71a97cd0b0a1986590295faaa424c` |
| Command SHA256 | `8f83b8e7deabdb7076cb6a0cb80bf10099f164f58e51a9df6a1946725f27fc05` |
| Exit | `1` |
| Started | `2026-09-05T13:35:50Z` |
| Finished | `2026-09-05T13:36:26Z` |
| ZIP SHA256 | `d0b43eb73ed5504c835c0c694bc48f18d9373614a6c58efe3139c2f5c66ee90c` |
| run.log SHA256 | `242dd312630be5c7d320f68a7ffeec2a424ed02ea6d2f160cc0d40f0a0356d24` |
| run.meta SHA256 | `0184e7f8462b24d2b97b21dedd90e68123562b06bcf0da14510742266375183f` |
| Prelaunch markers | `P4L_SEMANTIC_BINDING_PREFLIGHT=PASS`; `CUDA_PREFLIGHT=PASS`; `RECOVERY_PREFLIGHT_PASS`; `TRAINER_PROCESS_LAUNCH_BEGIN` |
| Trainer | `scripts/train_controlled_v6b_minimal.py` |
| Trainer launched | `TRUE` |
| Exact failure | `P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE: {'dev': {'polarity': {0: 0, 1: 58}}}` |
| Scientific disposition | `NO_VALID_A1_SCIENTIFIC_EVIDENCE` |
| Run classification | `UNAUTHORIZED_TRAINER_LAUNCH_PROVENANCE_INCIDENT` |

Recovery3 is provenance evidence only. It is not a valid A1 scientific run, not a completed A1 replicate, not factorial evidence, not promotion evidence, not winner evidence, not mechanism evidence, and not significance evidence.

## 3. Frozen Data And Sidecar Identity

Frozen dataset:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

| Field | Value |
|---|---|
| Rows | `3600` |
| Frozen Git/LF physical SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| Semantic SHA256 | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| DATA_SEMANTIC_MATCH | `True` |
| Previously observed Windows working-tree physical SHA256 | `eedbf93cf7fc3e141c4a49511750cbe4d8b0443e7de3463ea7e77696aca2c572` |

The frozen Git/LF physical identity is distinct from the Windows CRLF working-tree physical identity. This audit did not normalize the frozen file.

Frozen P4-L sidecar:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`

| Field | Value |
|---|---|
| Rows | `3600` |
| Frozen physical SHA256 | `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1` |
| Semantic SHA256 | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` |
| SIDECAR_SEMANTIC_MATCH | `True` |
| Previously observed Windows working-tree physical SHA256 | `a04f991554876cd6fea049d8ed494cd4a2f548ee5f69d08c8eacd9db6293389a` |
| Provenance frozen physical SHA256 | `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2` |
| P4-L authority commit | `80cb034792f03226cf6e22c196c1229ed4e6dd62` |
| Builder | `2f9e6076791358922e3ebd70e89533d9cb83b458` |

The sidecar frozen physical identity is distinct from the Windows CRLF working-tree physical identity. This audit did not normalize the sidecar or provenance file.

Stable identity and join checks:

| Check | Value |
|---|---|
| DATA_ROWS | `3600` |
| DATA_SEMANTIC_SHA256 | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| DATA_SEMANTIC_MATCH | `True` |
| SIDECAR_ROWS | `3600` |
| SIDECAR_SEMANTIC_SHA256 | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` |
| SIDECAR_SEMANTIC_MATCH | `True` |
| ROW_ID_UNIQUE | `True` |
| SIDECAR_ROW_ID_UNIQUE | `True` |
| STABLE_JOIN_KEYSET_MATCH | `True` |
| STABLE_JOIN_ORDER_MATCH | `True` |

## 4. Frozen Seed174 Split Reconstruction

| Field | Value |
|---|---|
| Split seed | `174` |
| Dev ratio | `0.2` |
| TOTAL_PAIRS | `300` |
| DEV_PAIR_TARGET | `60` |
| TRAIN_ROWS | `2880` |
| DEV_ROWS | `720` |
| TRAIN_PAIRS | `240` |
| DEV_PAIRS | `60` |
| PAIR_LEAKAGE_COUNT | `0` |
| SIDECAR_SPLIT_MISMATCH_COUNT | `0` |
| TRAIN_CANONICAL_PAIRS | `240` |
| DEV_CANONICAL_PAIRS | `60` |

The split reconstruction is deterministic. Split nondeterminism is `NO`. Split implementation defect is `NO`.

## 5. P2 Reason-Supervision Eligibility And Applicability

Eligibility:

| Field | Value |
|---|---|
| TRAIN eligible | `1450` |
| DEV eligible | `319` |
| TRAIN_SIDECAR_REASON_ELIGIBLE | `1450` |
| TRAIN_RECON_REASON_ELIGIBLE | `1450` |
| TRAIN_ELIGIBILITY_COUNT_MATCH | `True` |
| DEV_SIDECAR_REASON_ELIGIBLE | `319` |
| DEV_RECON_REASON_ELIGIBLE | `319` |
| DEV_ELIGIBILITY_COUNT_MATCH | `True` |

P2 applicability semantics verified from current source:

| Cohort | Applicability |
|---|---|
| Frame | `reason-supervision eligible` |
| Predicate | `eligible AND frame compatible` |
| Sufficiency | `eligible AND frame compatible AND predicate covered` |
| Polarity | `eligible AND frame compatible AND predicate covered AND sufficient AND final label directional` |

Polarity mapping:

| Label | Target |
|---|---:|
| `REFUTE` | `0` |
| `SUPPORT` | `1` |

Train primary counts:

| Primary reason | Count |
|---|---:|
| `FRAME` | `726` |
| `PREDICATE` | `121` |
| `SUFFICIENCY` | `242` |
| `AUTHORIZED` | `361` |

Train applicable binary counts:

| Cohort | 0 | 1 |
|---|---:|---:|
| frame | `726` | `724` |
| predicate | `121` | `603` |
| sufficiency | `242` | `361` |
| polarity | `119` | `242` |

Train direct polarity:

| Label | Count |
|---|---:|
| `REFUTE` | `119` |
| `SUPPORT` | `242` |

`REFUTE` exists in global/train data. This is not global polarity-class loss; it is the A1/A3 applicable polarity cohort readiness check.

Dev primary counts:

| Primary reason | Count |
|---|---:|
| `FRAME` | `174` |
| `PREDICATE` | `29` |
| `SUFFICIENCY` | `58` |
| `AUTHORIZED` | `58` |

Dev applicable binary counts:

| Cohort | 0 | 1 |
|---|---:|---:|
| frame | `174` | `145` |
| predicate | `29` | `116` |
| sufficiency | `58` | `58` |
| polarity | `0` | `58` |

Dev direct polarity:

| Label | Count |
|---|---:|
| `REFUTE` | `0` |
| `SUPPORT` | `58` |

Fresh reconstructed dictionary:

```text
{'dev': {'polarity': {0: 0, 1: 58}}}
```

Authenticated recovery3 exception payload:

```text
{'dev': {'polarity': {0: 0, 1: 58}}}
```

`EXACT_FAILURE_PAYLOAD_MATCH = True`

`STATIC_RECONSTRUCTION = PASS`

## 6. Guard And Test Behavior

Inspected source:

- `scripts/train_controlled_v6b_minimal.py`
- `tests/test_reason_router_p2_contract.py`

Verified source behavior:

- `_p2_prepare_reason_supervision` constructs P2 reason-supervision targets and applicable binary cohorts from source records plus sidecar identity/split checks.
- Reason minimum-count and applicable-binary-cohort enforcement is enabled for A1/A3 by `require_min_counts=args.reason_router_arm in {"A1", "A3"}`.
- A degenerate applicable binary cohort intentionally raises `P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE`.
- The test contract explicitly covers this fail-closed behavior in `test_normal_p2_full_helper_rejects_degenerate_polarity_cohort`.

Required conclusions:

| Conclusion | Value |
|---|---|
| validation-scope defect | `NO` |
| trainer-guard defect | `NO` |
| intentional A1/A3 fail-closed behavior | `YES` |

A2 is not claimed to be subject to this same reason-supervision blocker. A2 progression is nevertheless blocked by current authority.

## 7. Defect Matrix

| Classification | Value |
|---|---|
| dataset corruption/defect | `NO` |
| sidecar/provenance defect | `NO` |
| split nondeterminism | `NO` |
| split implementation defect | `NO` |
| split-contract feasibility issue | `YES` |
| label-semantics defect | `NO` |
| polarity mapping defect | `NO` |
| applicability/cohort-construction defect | `NO` |
| validation-scope defect | `NO` |
| trainer-guard defect | `NO` |
| configuration incompatibility | `YES` |
| missing pre-execution feasibility check | `YES` |
| insufficient evidence / unresolved ambiguity | `NO` |

Direct root cause:

- split-contract feasibility issue
- configuration incompatibility

Secondary prevention/control gap:

- missing pre-execution feasibility check

## 8. Causal Chain

1. Frozen dataset semantic identity is intact.
2. Frozen P4-L sidecar semantic identity is intact.
3. Seed174 pair split is deterministic and matches current implementation.
4. P2 eligibility/applicability reconstruction matches the sidecar.
5. Train applicable polarity contains REFUTE119/SUPPORT242.
6. Frozen seed174 dev applicable polarity contains REFUTE0/SUPPORT58.
7. A1/A3 require both binary classes in train/dev applicable cohorts under the intentional fail-closed reason-supervision readiness contract.
8. Therefore the exact degeneracy exception occurs deterministically.
9. The failure is a frozen scientific split-contract / A1-A3 readiness-contract incompatibility, not trainer corruption, dataset corruption, or split nondeterminism.

## 9. Secondary Prevention Finding

Secondary prevention defect:

`MISSING_PRELAUNCH_P2_APPLICABLE_COHORT_FEASIBILITY_CHECK`

The class-distribution infeasibility was statically computable before trainer launch from the frozen dataset, frozen P4-L sidecar, seed174 pair split, and current P2 applicability semantics. Recovery3 nevertheless passed preflight through:

`TRAINER_PROCESS_LAUNCH_BEGIN`

This is a prevention/control defect. It is not the direct cause of the class distribution.

## 10. Remedy Boundary

No remedy is selected, ranked, recommended, authorized, or implemented by this report.

Only future scientific-contract decision classes are identified:

- split contract
- validation/supervision contract
- dataset/eligibility contract
- prelaunch feasibility validation

This report does not authorize or recommend a new split seed, new dev ratio, stratification, MILP, resampling, regeneration, label changes, applicability changes, guard weakening, gate skipping, factorial v2, recalibration, retry, or recovery4.

## 11. Execution And Scientific Boundary

`ACTIVE_P2_ROOT_CAUSE_AUTHORITY = NONE_YET`

Recovery3:

`NO_VALID_A1_SCIENTIFIC_EVIDENCE`

Execution:

`BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`

`A2_A3_PROGRESSION = BLOCKED`

`TRAINING_EVALUATION_KAGGLE = NOT_AUTHORIZED`

Launch budget:

`FUTURE_AUTHORIZED_REPLACEMENT_LAUNCH_BUDGET_REQUIRES_NEW_EXPLICIT_AUTHORITY`

This report makes no claim of winner, promotion, mechanism, significance, completed A1 replicate, or factorial conclusion.

## 12. Non-Modification Statement

This audit did not change production source, tests, dataset, sidecar, provenance, split, labels, applicability semantics, trainer guard, authority artifacts, existing reports, git configuration, staging area, commits, branch, or HEAD.

Training/evaluation allowed: `NO`

Kaggle: `NO`

Commit/push: `NO`

Next authorized action:

`FINAL_INDEPENDENT_VERIFICATION_OF_THIS_FINALIZED_REPORT_CONTENT_ONLY`
