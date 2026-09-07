# Seed8192 Revised P4-L Implementation-Delta Authority Spec Candidate

## 1. Verdict

`FINAL_REVISED_P4L_IMPLEMENTATION_DELTA_AUTHORITY_CONTENT`

`PASS_READY_FOR_FREEZE`

`INDEPENDENT_VERIFICATION = PASS_READY_FOR_FINALIZATION`

`ACTIVATION_CONDITION = ON_EXACT_COMMIT_PUSH_REMOTE_VERIFICATION`

`ACTIVE_REVISED_P4L_IMPLEMENTATION_DELTA_AUTHORITY = NONE_YET`

`IMPLEMENTATION = NOT_AUTHORIZED_BY_THIS_CANDIDATE`

`P4L_RECONSTRUCTION = NOT_AUTHORIZED_BY_THIS_CANDIDATE`

`CURRENT_EXECUTION_STATE = BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`

`Training/Evaluation/CUDA/Kaggle = NOT_AUTHORIZED`

This candidate defines the minimum future implementation delta and dependency ordering required to make the seed8192 revised P4-L lineage reconstructable and later consumable. It does not edit source, edit tests, generate a sidecar, generate provenance, bind nonexistent future hashes, run training, run evaluation, use CUDA, use Kaggle, stage, commit, or push.

Finalization alone does not activate this authority. `ACTIVE_REVISED_P4L_IMPLEMENTATION_DELTA_AUTHORITY = NONE_YET` remains in force until every fail-closed activation condition in Section 31 is independently verified.

## 2. Authority Chain

| Role | Identity |
|---|---|
| Active revised P4-L reconstruction/rebinding/provenance authority | `ff181f565cefa0a28280c084246862286daf1f2d` |
| Active revised split design/selection authority | `b4fbb5666d796161f95ae23612ce2448c25063ee` |
| Active split-contract remedy decision authority | `c82a164ac460599c68318a3b29180303f12cbc1a` |
| Active P2 root-cause authority | `eea0714904ea1f95c42da48e85cd1af4bad23123` |
| Authority-lineage reconciliation | `1bb08179adb38637e9391491ba72cfd7e9bff3b3` |
| Unauthorized-execution incident correction | `0f6e00642fb6126ec86d7b7dde4b84626befca67` |
| Historical P4-L artifact-contract authority, precedent only | `80cb034792f03226cf6e22c196c1229ed4e6dd62` |
| Historical P4-L builder lineage | `2f9e6076791358922e3ebd70e89533d9cb83b458` |
| Repository contract | `AGENTS.md` |

The current task is report-only specification. The active `ff181f5...` authority is a constraint source, not permission to implement or materialize artifacts.

## 3. Repository Preconditions

| Check | Required | Observed | Result |
|---|---|---|---|
| Branch | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` | PASS |
| HEAD | `ff181f565cefa0a28280c084246862286daf1f2d` | `ff181f565cefa0a28280c084246862286daf1f2d` | PASS |
| `git status --short` | empty | empty before this file | PASS |
| `git diff --name-status` | empty | empty | PASS |
| `git diff --cached --name-status` | empty | empty | PASS |
| `git diff --check` | exit 0, no output | exit 0, no output | PASS |

No untracked files existed before this candidate was written.

## 4. Active ff181f5 Authority Authentication

Tracked report `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_reconstruction_rebinding_provenance_authority_spec_candidate.md` authenticates both required body-level decisions:

- `IMPLEMENTATION_DELTA_REQUIRED_BEFORE_REVISED_P4L_RECONSTRUCTION`
- `NO_SCHEMA_CHANGE_REQUIRED_FOR_REVISED_P4L_RECONSTRUCTION`

It also preserves `SELECTED_REVISED_SPLIT_SEED = 8192`, classifies historical seed174 P4-L as a valid historical lineage artifact, and rejects historical seed174 P4-L as compatible with seed8192 revised execution.

## 5. Exact Current Producer Discovery

Current producer file:

`scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`

Relevant present bindings:

| Symbol/function | Present behavior | Seed8192 incompatibility |
|---|---|---|
| `AUTHORITY_VERSION`, `NAMESPACE`, `SIDECAR_SCHEMA_VERSION`, `PROVENANCE_SCHEMA_VERSION`, `RULE_VERSION` | Current-lineage P3W6F2/P4-L names and versions | Revised lineage must distinguish seed8192 without changing row-field semantics |
| `P4L_AUTHORITY_PATH`, `P4L_AUTHORITY_COMMIT` | Historical P4-L authority path and `80cb0347...` | Revised provenance must bind `ff181f565cefa0a28280c084246862286daf1f2d` |
| `SOURCE_DATASET_PATH`, `SOURCE_DATASET_SHA256`, `SOURCE_DATASET_SEMANTIC_SHA256` | Frozen P4-B regenerated dataset | Compatible and must be preserved |
| `SPLIT_SEED` | `174` | Revised lineage requires explicit seed8192; no silent seed174 fallback |
| `DEV_RATIO` | `0.2` | Compatible and must be explicitly validated |
| `deterministic_pair_split` | sorted pair IDs, `random.Random(seed).shuffle`, first rounded 20% dev pairs; default seed is `SPLIT_SEED` | Algorithm is compatible, default binding is not |
| `assemble_sidecar_rows` | emits same JSONL row schema in source order; split-bound fields derive from `split_by_pair` | Must receive seed8192 split map and revised namespace/version values without schema change |
| `build_provenance` | records historical authority, builder commit, source identities, sidecar/provenance paths, and `split_rule.shuffle_seed = SPLIT_SEED` | Must record revised authority, seed8192 split identities, and revised paths |
| `canonical_output_dir` | historical namespace keyed only by builder commit | Must use non-destructive revised namespace including full authority commit and future builder commit |
| `build_sidecar_artifacts` | calls `deterministic_pair_split(source_rows)` without seed/dev-ratio arguments | Cannot materialize seed8192 without code delta |
| `parse_args`, `run`, `main` | expose repo root, builder commit, created-at, materialize only | No explicit revised mode, seed, dev-ratio, authority, or fail-closed seed8192 validation |

The producer imports no trainer, model, torch, checkpoint, network, CUDA, or Kaggle dependency. Existing atomic publication uses no-overwrite directory publication and must be preserved.

## 6. Exact Current Consumer Discovery

Current consumer file:

`scripts/train_controlled_v6b_minimal.py`

Relevant present bindings:

| Symbol/function | Present behavior | Phase-III impact |
|---|---|---|
| `_STAGE187_AUTHORITATIVE_DATA` | frozen dataset path | Preserve |
| `_P4X_CANONICAL_DIR`, `_STAGE187_AUTHORITATIVE_SIDECAR`, `_P4X_CANONICAL_PROVENANCE` | historical seed174 sidecar/provenance directory and filenames | Must be rebound only after revised artifacts exist |
| `_P4X_SIDECAR_PHYSICAL_SHA256`, `_STAGE187_SIDECAR_SEMANTIC_SHA256`, `_P4X_PROVENANCE_PHYSICAL_SHA256` | historical artifact hashes | Cannot be replaced until Phase II computes exact revised hashes |
| `_P4X_P4L_AUTHORITY_COMMIT`, `_P4X_BUILDER_SOURCE_COMMIT` | historical authority and builder commits | Must bind revised authority and frozen revised producer commit after materialization |
| `_p4x_validate_provenance` | exact fail-closed field checks for schema, authority, builder, dataset, sidecar hashes, row count, and authority booleans | Good architecture, but values are historical |
| `_p4x_validate_canonical_integrity_binding` | rejects wrong canonical data/sidecar/provenance paths, symlinks, missing files, and hash mismatches | Good architecture, but canonical identities are historical |
| `_p4x_validate_sidecar_rows` | validates required fields, schema version, dataset identities, booleans, reason codes, and historical count constants | Count constants must be revised after Phase II |
| `_stage187_load_integrity_sidecar`, `_p2_load_reason_integrity_sidecar` | both share canonical P4-X gate | Future rebind must preserve shared gate |
| `_p2_prepare_reason_supervision`, `_p2_prepare_reason_supervision_train_only` | derive primary reason, secondary reasons, applicability masks, pair leakage checks, and train/dev split checks from source plus sidecar | No trainer semantic change is required |
| `_p2_row_identity_hash` | hashes ordered rows as `id<TAB>pair_id<LF>` | Compatible with active seed8192 ordered row identity contract |

Relevant tests:

- `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py` covers historical split algorithm behavior, schema/hash serialization, and no-overwrite publication.
- `tests/test_reason_router_p4x_trainer_rebind.py` covers trainer canonical binding, hash mismatch rejection, provenance schema/authority rejection, stable join, and shared reason/positive-margin gate.

## 7. Exact Hard Bindings

Current hard bindings that must not leak into revised identity:

- producer `SPLIT_SEED = 174`;
- producer `P4L_AUTHORITY_COMMIT = 80cb034792f03226cf6e22c196c1229ed4e6dd62`;
- producer historical output directory `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_<builder_commit>/`;
- producer filenames `p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl` and provenance equivalent;
- trainer `_P4X_CANONICAL_DIR` with builder commit `2f9e6076791358922e3ebd70e89533d9cb83b458`;
- trainer historical sidecar physical `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`;
- trainer historical sidecar semantic `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`;
- trainer historical provenance physical `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`;
- A0 recovery `SPLIT_SEED = 174` and historical sidecar/provenance constants;
- calibration aggregate `EXPECTED_SPLIT_SEED = 174`.

## 8. Dependency Graph

Required sequence:

| Phase | Classification | Purpose |
|---|---|---|
| Phase I | `PHASE_I_PRE_MATERIALIZATION_PRODUCER_ENABLEMENT` | Make producer capable of deterministic seed8192 revised lineage generation without overwriting historical artifacts |
| Phase II | `PHASE_II_AUTHORIZED_RECONSTRUCTION_MATERIALIZATION` | Under a separate authority, generate and validate revised sidecar/provenance and compute exact physical/semantic/provenance hashes |
| Phase III | `PHASE_III_POST_MATERIALIZATION_CONSUMER_REBIND` | Under separate authority, bind consumers to exact frozen revised paths/hashes/commits |
| Phase IV | `PHASE_IV_PRELAUNCH_STATIC_CONTROL` | Add or activate fail-closed prelaunch static controls before trainer process launch |

No evidence requires a different ordering. Collapsing Phase I and Phase III would require inventing nonexistent revised hashes; therefore it is forbidden.

## 9. Phase-I Producer-Enablement Scope

Minimum future Phase-I behavior:

1. Explicit seed8192 binding.
2. No silent default back to seed174 for revised mode.
3. Preserve exact pair-level split algorithm.
4. Explicit dev ratio `0.2`.
5. Validate seed8192 split identities from active authority.
6. Bind revised provenance authority `ff181f565cefa0a28280c084246862286daf1f2d`.
7. Use a new non-destructive output namespace.
8. Preserve sidecar row schema and field semantics.
9. Preserve historical seed174 behavior and historical tests.
10. Fail closed for unsupported or mismatched revised parameters.

Phase I may make temporary pytest outputs only in a later authorized implementation phase; this report-authoring phase does not.

## 10. Producer API/Mode Decision

Decision:

`PRESERVE_HISTORICAL_MODE_AND_ADD_EXPLICIT_REVISED_MODE`

Justification: the current producer has meaningful historical behavior and tests. Generalizing the whole producer as a single implicit parameterized path risks changing what old historical commands mean. The narrowest defensible API is an explicit revised mode, for example `--lineage-mode revised-seed8192` or an equivalent fail-closed config object, where seed, dev ratio, authority commit, output namespace, expected split hashes, and output names are all explicitly bound.

Historical mode may retain its current default seed174 behavior. Revised mode must require explicit authority-bound settings and reject omissions, unsupported seeds, wrong dev ratio, wrong authority commit, historical output paths, and historical artifact hashes.

## 11. Historical Compatibility Contract

Historical seed174 P4-L remains:

`VALID_HISTORICAL_SEED174_LINEAGE_ARTIFACT`

For revised seed8192 execution it remains:

`INCOMPATIBLE_WITH_SEED8192_REVISED_SPLIT_LINEAGE`

Existing historical commands and tests must remain meaningful. A future patch must not delete, overwrite, rename, reinterpret, or mutate the historical seed174 sidecar/provenance directory or artifacts.

## 12. Seed8192 Split-Binding Contract

Future Phase-I implementation must be statically capable of reproducing exactly:

| Identity | Required value |
|---|---|
| split seed | `8192` |
| dev ratio | `0.2` |
| train pairs | `240` |
| dev pairs | `60` |
| train rows | `2880` |
| dev rows | `720` |
| pair leakage | `0` |
| complete pair universe | `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2` |
| selected dev pairs | `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4` |
| selected train pairs | `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049` |
| seed8192 shuffled sequence | `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55` |
| ordered train rows | `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8` |
| ordered dev rows | `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4` |
| historical seed174 dev reference | `259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d` |

Hash canonicalization must not be redefined.

## 13. Frozen Dataset Contract

The implementation must preserve the exact dataset:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

| Identity | Required value |
|---|---|
| Git/LF SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| semantic SHA256 | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| rows | `3600` |
| pairs | `300` |

Forbidden: dataset regeneration, row edits, row reorder, label edits, pair_id edits, eligibility semantic changes, and applicability semantic changes.

## 14. Schema Verdict

`NO_SCHEMA_CHANGE_REQUIRED_FOR_REVISED_P4L_RECONSTRUCTION`

The existing sidecar JSONL row-field contract is sufficient. Revised mode may change provenance and row values required by the revised lineage, such as namespace tokens, rule/version tokens, split assignments, split-bound eligibility/exclusion fields, builder hash, authority commit, output paths, and hashes. It may not add, remove, rename, or semantically redefine sidecar fields. If a future implementation proves schema change unavoidable, the correct verdict becomes `BLOCKED_SCHEMA_CHANGE_REQUIRES_SEPARATE_EXPLICIT_AUTHORITY`.

## 15. Revised Output Naming Design

`FINAL_REVISED_P4L_OUTPUT_PATH = NOT_YET_MATERIALIZED`

Future revised namespace:

```text
reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_<authority_commit>_<builder_commit>/
  p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl
  p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json
```

`<authority_commit>` must be `ff181f565cefa0a28280c084246862286daf1f2d`. `<builder_commit>` is unknown until implementation freeze and must not be guessed. The producer must reject historical output paths, ambiguous names such as latest/final/new/run1, and any overwrite of an existing path.

## 16. Provenance Implementation Design

Future revised provenance must preserve existing provenance structure where possible and rebind values without recursive self-reference. Required bindings include:

- revised P4-L authority commit `ff181f565cefa0a28280c084246862286daf1f2d`;
- exact future producer/builder commit, unknown until implementation freeze;
- frozen dataset physical and semantic identities;
- active split authority `b4fbb5666d796161f95ae23612ce2448c25063ee`;
- split seed `8192` and dev ratio `0.2`;
- split algorithm identity;
- complete/dev/train/shuffled pair hashes;
- ordered train/dev row hashes;
- sidecar schema identity/version;
- generated sidecar physical and semantic SHA256 after Phase II;
- generated provenance physical SHA256 where existing architecture can report it without self-reference.

The present provenance uses `provenance_physical_sha256_self_certified = False`; future implementation must preserve that non-self-certifying behavior unless a separate authority changes the architecture.

## 17. Future-Hash Boundary

Before Phase II:

`REVISED_P4L_SIDECAR_PHYSICAL_SHA256 = TO_BE_COMPUTED_BY_AUTHORIZED_RECONSTRUCTION`

`REVISED_P4L_SIDECAR_SEMANTIC_SHA256 = TO_BE_COMPUTED_BY_AUTHORIZED_RECONSTRUCTION`

`REVISED_P4L_PROVENANCE_PHYSICAL_SHA256 = TO_BE_COMPUTED_BY_AUTHORIZED_RECONSTRUCTION`

No Phase-I code may contain guessed placeholder hashes that can pass validation. If placeholder slots are technically necessary, they must fail closed and be incapable of masquerading as frozen artifact identities.

## 18. Phase-II Reconstruction Boundary

This candidate does not authorize reconstruction. Later reconstruction requires:

1. Phase-I implementation authority activated.
2. Implementation completed.
3. Focused tests/static validation pass.
4. Implementation independently verified.
5. Implementation frozen at exact commit.
6. Separate explicit P4-L reconstruction/materialization authority activated.
7. Only then may producer execution materialize revised sidecar/provenance.

No sidecar/provenance generation or dry-run artifact writing is authorized here.

## 19. Phase-III Consumer-Rebind Decision

Decision:

`A_SEPARATE_POST_MATERIALIZATION_CONSUMER-REBIND_IMPLEMENTATION_AUTHORITY`

Justification: `scripts/train_controlled_v6b_minimal.py` currently uses exact canonical paths and exact physical/semantic/provenance hashes. The revised hashes and final builder commit do not exist before Phase II. Pre-authorizing a consumer rebind mechanism before materialization would either require runtime substitution or unfrozen identity slots, which would weaken the current fail-closed architecture. Therefore Phase III must occur only after Phase II computes and validates exact artifacts.

## 20. Exact Trainer Bindings Deferred to Phase III

Deferred trainer bindings:

- `_P4X_CANONICAL_DIR`;
- `_STAGE187_AUTHORITATIVE_SIDECAR`;
- `_P4X_CANONICAL_PROVENANCE`;
- `_P4X_SIDECAR_PHYSICAL_SHA256`;
- `_STAGE187_SIDECAR_SEMANTIC_SHA256`;
- `_P4X_PROVENANCE_PHYSICAL_SHA256`;
- `_P4X_P4L_AUTHORITY_COMMIT`;
- `_P4X_BUILDER_SOURCE_COMMIT`;
- `_P4X_PROVENANCE_SCHEMA_VERSION`;
- `_P4X_SIDECAR_SCHEMA_VERSION` if revised tokens are used without row-field schema change;
- `_P4X_EXPECTED_REASON_ELIGIBLE_ROWS`;
- `_P4X_EXPECTED_REASON_INELIGIBLE_ROWS`;
- `_P4X_EXPECTED_INTEGRITY_COUNTS`;
- `_STAGE187_EXPECTED_ELIGIBLE_ROWS`;
- `_P4X_EXPECTED_POSITIVE_MARGIN_INELIGIBLE_ROWS`;
- provenance/path/hash checks inside `_p4x_validate_provenance` and `_p4x_validate_canonical_integrity_binding`;
- count validation in `_p4x_validate_sidecar_rows` and `_stage187_load_integrity_sidecar`.

No optimizer, model, router, loss, gradient, sampler, batch, schedule, metric, A1/A2/A3, or training semantic changes are authorized in this lineage.

## 21. A0 Boundary

`scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py` is historical seed174 A0 recovery. It binds `SPLIT_SEED = 174`, historical sidecar/provenance path and hashes, and A0 seed180 provenance. It is:

`OUT_OF_SCOPE_SEPARATE_AUTHORITY`

`HISTORICAL_A0 = NOT_ADMISSIBLE_AS_SEED8192_REVISED_SPLIT_REFERENCE`

`seed8192 = REQUIRES_NEW_A0_BASELINE_EVIDENCE`

## 22. Calibration Boundary

`scripts/aggregate_reason_router_p3w1_calibration.py` binds `EXPECTED_SPLIT_SEED = 174` and validates the historical train-only calibration line. The historical accepted reason-loss weight `0.6202430063306562` remains:

`HISTORICAL_SEED174_LINEAGE_ONLY`

Seed8192 requires recalibration under separate authority. Calibration code changes are:

`OUT_OF_SCOPE_SEPARATE_AUTHORITY`

## 23. Readiness Validation Targets

Future revised reconstruction must validate:

| Split | FRAME | PREDICATE | SUFFICIENCY | AUTHORIZED |
|---|---:|---:|---:|---:|
| train primary | 714 | 119 | 238 | 338 |
| dev primary | 186 | 31 | 62 | 81 |

| Split | Cohort | Class 0 | Class 1 |
|---|---|---:|---:|
| train | frame | 714 | 695 |
| train | predicate | 119 | 576 |
| train | sufficiency | 238 | 338 |
| train | polarity | 100 | 238 |
| dev | frame | 186 | 174 |
| dev | predicate | 31 | 143 |
| dev | sufficiency | 62 | 81 |
| dev | polarity | 19 | 62 |

These are validation constants, not generated data and not provenance-only metadata. Later validation belongs in producer/reconstruction tests and Phase-III static trainer compatibility, not manual sidecar patching.

## 24. Prelaunch-Control Phase Decision

`MISSING_PRELAUNCH_P2_APPLICABLE_COHORT_FEASIBILITY_CHECK` remains open.

Decision:

`PHASE_IV_PRELAUNCH_STATIC_CONTROL`

The check must occur before `TRAINER_PROCESS_LAUNCH_BEGIN`. It must not be silently folded into Phase I merely because seed8192 counts are available, and P4-L reconstruction success alone does not fix the defect.

## 25. Phase-I Exact File Set

| FILE | PHASE | EXPECTED CHANGE | WHY REQUIRED | NOT AUTHORIZED YET |
|---|---|---|---|---|
| `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py` | `PHASE_I_PRE_MATERIALIZATION_PRODUCER_ENABLEMENT` | Add explicit revised seed8192 mode/config and fail-closed bindings | Producer is hard-bound to seed174/history and old output namespace | YES |
| `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py` or a focused revised successor test file | `PHASE_I_PRE_MATERIALIZATION_PRODUCER_ENABLEMENT` | Cover historical preservation plus revised-mode seed/split/provenance/path/hash-boundary behavior | Producer delta needs executable contract | YES |
| `scripts/train_controlled_v6b_minimal.py` | `PHASE_III_POST_MATERIALIZATION_CONSUMER_REBIND` | Rebind canonical consumer identities after exact hashes exist | Current consumer pins historical path/hash/commit values | YES |
| `tests/test_reason_router_p4x_trainer_rebind.py` or focused revised successor | `PHASE_III_POST_MATERIALIZATION_CONSUMER_REBIND` | Cover revised exact bindings and historical hash rejection | Consumer rebind must remain fail-closed | YES |
| `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py` | `OUT_OF_SCOPE_SEPARATE_AUTHORITY` | No Phase-I change | Historical A0 is not seed8192 evidence | YES |
| `scripts/aggregate_reason_router_p3w1_calibration.py` | `OUT_OF_SCOPE_SEPARATE_AUTHORITY` | No Phase-I change | Seed8192 requires recalibration authority | YES |

Strong Phase-I preference: only the producer and directly associated tests.

## 26. Phase-I Exact Symbol/Function Changes

| Symbol/function | Phase-I required future behavior | Delta classification |
|---|---|---|
| `SPLIT_SEED` | Preserve historical constant for historical mode; revised mode must explicitly bind `8192` | `HISTORICAL_COMPATIBILITY_PRESERVATION`, `EXPLICIT_REVISED_SPLIT_PARAMETERIZATION` |
| `DEV_RATIO` | Preserve `0.2`; revised mode validates exact `0.2` | `FAIL_CLOSED_VALIDATION` |
| `deterministic_pair_split` | Preserve algorithm byte-for-byte where practical; call with explicit revised seed/dev ratio | `EXPLICIT_REVISED_SPLIT_PARAMETERIZATION` |
| `build_sidecar_artifacts` | Accept explicit lineage-mode/config and thread seed/dev-ratio/authority/output bindings into split, sidecar, provenance | `EXPLICIT_REVISED_SPLIT_PARAMETERIZATION`, `AUTHORITY_BINDING` |
| `assemble_sidecar_rows` | Use revised namespace/rule/version values in revised mode while preserving row fields and semantics | `PROVENANCE_REBINDING`, `HISTORICAL_COMPATIBILITY_PRESERVATION` |
| `build_provenance` | Record revised authority, split seed8192, dev ratio, pair/row identity hashes, revised output path, future builder commit, and computed artifact hashes | `PROVENANCE_REBINDING`, `AUTHORITY_BINDING` |
| `canonical_output_dir` | Build non-destructive revised path with full authority commit and future builder commit | `NON_DESTRUCTIVE_OUTPUT_NAMESPACE` |
| `parse_args` | Add explicit revised-mode arguments/config; reject ambiguous revised invocation | `FAIL_CLOSED_VALIDATION` |
| `run` | Route historical vs revised mode explicitly; no revised materialization without later authority | `FAIL_CLOSED_VALIDATION` |
| `main` | Preserve fail-closed exit behavior and make revised failures unmistakable | `FAIL_CLOSED_VALIDATION` |

## 27. Test-Delta Specification

Existing tests already cover historical pair-level seeded split, semantic hash excluding only `created_at`, LF/no-BOM/final-LF JSONL serialization, exact binary rejection, historical no-overwrite publication, and platform fail-closed publication behavior.

Future Phase-I tests must add or revise coverage for:

1. historical seed174 mode remains reproducible;
2. revised seed8192 mode requires explicit authority-bound seed;
3. revised mode cannot silently default to seed174;
4. exact seed8192 pair split;
5. exact pair hash identities;
6. exact ordered row identities;
7. schema preservation;
8. source dataset identity preservation;
9. revised provenance records seed8192;
10. revised provenance binds `ff181f565cefa0a28280c084246862286daf1f2d`;
11. historical output path remains immutable;
12. revised output path is non-destructive;
13. duplicate/overwrite publication fails closed;
14. semantic hash canonicalization unchanged;
15. physical bytes remain canonical LF/no-BOM/final-LF;
16. unsupported seed/authority combination fails closed;
17. no historical seed174 artifact hash is accepted as revised identity.

Historical tests must not be weakened merely to make revised mode pass.

## 28. Static Validation Plan

No exact registered `cm gate` for seed8192 revised P4-L implementation was found in current source/test references. Do not invent one.

Future implementation validation should include:

- `git diff --check`;
- focused pytest for `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py` and any revised producer successor tests;
- focused pytest for Phase-III trainer rebind tests only after Phase II/III authority exists;
- static source audit proving no revised P4-L path/hash/provenance field uses historical seed174 identities as current revised identities;
- exact diff review;
- no trainer process, model/checkpoint loading, CUDA, Kaggle, or sidecar generation unless a separate Phase-II authority explicitly authorizes materialization.

## 29. Expected Implementation Hunk Classifications

Permitted future classifications:

- `EXPLICIT_REVISED_SPLIT_PARAMETERIZATION`
- `AUTHORITY_BINDING`
- `NON_DESTRUCTIVE_OUTPUT_NAMESPACE`
- `PROVENANCE_REBINDING`
- `FAIL_CLOSED_VALIDATION`
- `HISTORICAL_COMPATIBILITY_PRESERVATION`
- `TEST_COVERAGE`
- `POST_MATERIALIZATION_CONSUMER_REBIND` for Phase III only
- `PRELAUNCH_STATIC_CONTROL` for Phase IV only

## 30. Forbidden Delta Classifications

Forbidden:

- `DATASET_SEMANTICS_CHANGE`
- `LABEL_CHANGE`
- `PAIR_ID_CHANGE`
- `ELIGIBILITY_CHANGE`
- `APPLICABILITY_CHANGE`
- `SPLIT_ALGORITHM_CHANGE`
- `P2_READINESS_RULE_CHANGE`
- `ROUTER_CHANGE`
- `LOSS_CHANGE`
- `GRADIENT_CHANGE`
- `A1_A2_A3_CHANGE`
- `TRAINING_HYPERPARAMETER_CHANGE`
- `EVALUATION_CHANGE`
- `SCHEMA_CHANGE`
- `HISTORICAL_ARTIFACT_MUTATION`
- `FUTURE_HASH_INVENTION`

## 31. Implementation Lifecycle

This is finalized freeze-ready authority content. Finalization does not activate implementation authority.

Activation is fail closed and occurs only after all of the following are independently verified:

1. the exact finalized candidate is the file staged;
2. staged raw Git blob bytes match the finalized candidate identity;
3. the staged delta contains exactly this one authority file;
4. a dedicated freeze/activation commit is created;
5. the exact full activation-commit SHA is obtained;
6. the activation-commit parent is exactly `ff181f565cefa0a28280c084246862286daf1f2d`;
7. the activation commit is pushed to `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`;
8. the remote branch tip equals the exact activation commit;
9. the remote file/blob identity equals the verified finalized staged blob; and
10. the remote body-level lifecycle/authority status is independently verified.

None of the following alone activates authority: candidate existence; local finalization; `PASS_READY_FOR_FREEZE` wording; staging; a commit subject or message; local commit existence; or push success without remote verification.

`COMMIT_MESSAGE_DOES_NOT_OVERRIDE_BODY_LEVEL_AUTHORITY_STATUS`

Until all activation conditions pass, `ACTIVE_REVISED_P4L_IMPLEMENTATION_DELTA_AUTHORITY = NONE_YET` and no wording in this finalized candidate activates implementation authority.

## 32. Execution Boundary

| Activity | Status |
|---|---|
| Phase-I source implementation | `NOT_AUTHORIZED_UNTIL_THIS_AUTHORITY_IS_ACTIVATED` |
| Phase-II P4-L reconstruction | `NOT_AUTHORIZED` |
| Phase-III consumer rebind | `NOT_AUTHORIZED` |
| Phase-IV prelaunch implementation | `NOT_AUTHORIZED_UNLESS_SEPARATELY_AND_EXPLICITLY_COVERED_LATER` |
| A0 | `NOT_AUTHORIZED` |
| calibration | `NOT_AUTHORIZED` |
| A1/A2/A3 | `BLOCKED` |
| training | `NOT_AUTHORIZED` |
| evaluation | `NOT_AUTHORIZED` |
| CUDA | `NOT_AUTHORIZED` |
| Kaggle | `NOT_AUTHORIZED` |

No run name. No recovery4.

## 33. Candidate Path

`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_implementation_delta_authority_spec_candidate.md`

This is the only file written by this task.

## 34. Candidate SHA/Shape

Candidate SHA256, byte count, LF/CR/CRLF counts, UTF-8 BOM status, final LF status, and trailing-whitespace line count are measured after materialization and reported with final repository validation. They are not guessed inside this body.

## 35. Git State Before/After

Before candidate creation:

| Command | Observed |
|---|---|
| `git branch --show-current` | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` |
| `git rev-parse HEAD` | `ff181f565cefa0a28280c084246862286daf1f2d` |
| `git status --short` | empty |
| `git diff --name-status` | empty |
| `git diff --cached --name-status` | empty |
| `git diff --check` | PASS |

Expected after finalization:

- same branch;
- same HEAD;
- exactly one untracked finalized candidate, this file;
- no tracked modifications;
- no staged changes;
- `git diff --check` PASS.

## 36. Blockers/Mismatches

No precondition blocker, authority mismatch, dependency graph conflict, schema-change requirement, dataset semantic-change requirement, split-algorithm-change requirement, trainer semantic-change requirement, historical compatibility blocker, non-destructive output blocker, or Phase-I file-set bounding blocker was found.

The remaining execution state is intentionally blocked:

`CURRENT_EXECUTION_STATE = BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`

## 37. Exact Next Authorized Action

`FRESH_FINAL_INDEPENDENT_VERIFICATION_OF_THE_FINALIZED_CANDIDATE_THEN_EXACT_FREEZE_ACTIVATION_WORKFLOW_ONLY_IF_PASS`

The next verifier should perform fresh final independent verification of this finalized candidate. Only if that verification passes may the exact freeze/activation workflow begin. The verifier should confirm this candidate is the only untracked file, no tracked or staged changes exist, active `ff181f5...` decisions are authenticated, producer/consumer findings are source-grounded, Phase I is bounded to producer enablement plus tests, Phase III waits for exact revised hashes, and all execution remains blocked.
