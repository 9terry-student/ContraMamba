# P3-W7 Seed8192 Revised P4-L Reconstruction/Rebinding Provenance Authority Spec Candidate

## 1. Verdict

`FINAL_REVISED_P4L_RECONSTRUCTION_REBINDING_PROVENANCE_AUTHORITY_CONTENT`

`PASS_READY_FOR_FREEZE`

`INDEPENDENT_VERIFICATION = PASS_READY_FOR_FINALIZATION`

`ACTIVATION_CONDITION = ON_EXACT_COMMIT_PUSH_REMOTE_VERIFICATION`

`ACTIVE_REVISED_P4L_RECONSTRUCTION_REBINDING_AUTHORITY = NONE_YET`

`CURRENT_EXECUTION_STATE = BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`

`Training/Evaluation/Kaggle/CUDA = NOT_AUTHORIZED`

This finalized authority content defines how the historical seed174-bound P4-L sidecar/provenance lineage must be reconstructed or rebound for the active revised split seed8192 after activation. It does not generate a revised sidecar, generate provenance JSON, copy historical artifacts, modify source, run a trainer, run evaluation, use CUDA, use Kaggle, stage, commit, push, reset, clean, or restore.

`COMMIT_MESSAGE_DOES_NOT_OVERRIDE_BODY_LEVEL_AUTHORITY_STATUS`

Before exact commit, push, remote-tip verification, remote-blob verification, and remote body-level verification, this file is freeze-ready content only and must not be treated as the active revised P4-L reconstruction/rebinding authority.

## 2. Authority Used

Authority precedence consumed:

| Role | Identity |
|---|---|
| Active revised split design/selection authority | `b4fbb5666d796161f95ae23612ce2448c25063ee` |
| Active split-contract remedy decision authority | `c82a164ac460599c68318a3b29180303f12cbc1a` |
| Active P2 root-cause authority | `eea0714904ea1f95c42da48e85cd1af4bad23123` |
| Active authority-lineage reconciliation | `1bb08179adb38637e9391491ba72cfd7e9bff3b3` |
| Active unauthorized-execution incident correction | `0f6e00642fb6126ec86d7b7dde4b84626befca67` |
| Historical P4-L artifact-contract authority, schema/provenance precedent only | `80cb034792f03226cf6e22c196c1229ed4e6dd62` |
| Historical P4-L builder lineage | `2f9e6076791358922e3ebd70e89533d9cb83b458` |
| Repository rules | `AGENTS.md` |

The current HEAD is exactly the active revised split design/selection authority. Its body authenticates `SELECTED_REVISED_SPLIT_SEED = 8192` and the exact seed8192 split identities recorded below.

## 3. Repository Preconditions

Opening repository state:

| Check | Required | Observed | Result |
|---|---|---|---|
| Branch | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` | PASS |
| HEAD | `b4fbb5666d796161f95ae23612ce2448c25063ee` | `b4fbb5666d796161f95ae23612ce2448c25063ee` | PASS |
| `git status --short` | empty | empty | PASS |
| `git diff --name-status` | empty | empty | PASS |
| `git diff --cached --name-status` | empty | empty | PASS |
| `git diff --check` | no output, exit 0 | no output, exit 0 | PASS |

No tracked modifications, staged changes, or untracked files existed before this candidate was written.

## 4. Historical P4-L Five-Part Identity

Historical P4-L is classified as:

`VALID_HISTORICAL_SEED174_LINEAGE_ARTIFACT`

and simultaneously:

`INCOMPATIBLE_WITH_SEED8192_REVISED_SPLIT_LINEAGE`

Five-part historical identity:

| Part | Historical value |
|---|---|
| Historical authority | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md` at `80cb034792f03226cf6e22c196c1229ed4e6dd62` |
| Producer/builder | `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py` at `2f9e6076791358922e3ebd70e89533d9cb83b458` |
| Frozen source dataset | `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` |
| Historical split binding | pair-level split, `shuffle_seed = 174`, `dev_ratio = 0.2` |
| Historical artifacts | sidecar JSONL plus provenance JSON under `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/` |

## 5. Historical Sidecar/Provenance Authentication

Authenticated from tracked Git bytes:

| Artifact | Required SHA256 | Observed SHA256 | Result |
|---|---|---|---|
| Historical sidecar physical | `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1` | `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1` | PASS |
| Historical sidecar semantic | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` | PASS |
| Historical provenance physical | `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2` | `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2` | PASS |

The historical provenance records `split_rule.shuffle_seed = 174`; therefore the historical sidecar/provenance must not be treated as valid for seed8192 execution.

## 6. Exact Producer/Builder Discovery

Producer discovery is unambiguous.

| Item | Evidence |
|---|---|
| Historical artifact freeze commit | `a93c291` added the historical sidecar JSONL and provenance JSON |
| Historical artifact contract commit | `80cb034792f03226cf6e22c196c1229ed4e6dd62` |
| Initial builder implementation commit | `25a85015fb344552b48e57b6dd92f3b0320d37d1` |
| Builder lineage used by frozen artifact | `2f9e6076791358922e3ebd70e89533d9cb83b458` |
| Builder source path | `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py` |
| Main builder entry points | `build_sidecar_artifacts`, `run`, `main` |
| Sidecar row assembly | `assemble_sidecar_rows` |
| Provenance assembly | `build_provenance` |
| Split producer function | `deterministic_pair_split` |
| Canonical-row producer functions | `canonical_row_ids`, `validate_canonical_lineage` |
| Semantic sidecar hash | `semantic_sidecar_sha256`, excluding only `created_at` |
| Physical sidecar bytes | `compact_jsonl_bytes`, compact sorted-key JSONL, LF, no BOM, final LF |
| Provenance physical bytes | `deterministic_json_bytes`, sorted-key indented JSON, LF, no BOM, final LF |

Inputs consumed by the historical builder:

| Input | Binding |
|---|---|
| P4-L authority path/commit | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`, `80cb034792f03226cf6e22c196c1229ed4e6dd62` |
| P4-B regenerated dataset | physical `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`, semantic `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| P4-B compatibility rows/summary/provenance | rows `59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f`, summary `ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8`, provenance `09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6` |
| Historical Stage185 sidecar/source | historical sidecar semantic `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`, source SHA `11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc` |

Hard-coded seed174 assumption discovered:

| Source | Evidence | Impact |
|---|---|---|
| `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py` | `SPLIT_SEED = 174` | Default builder split is seed174 |
| `deterministic_pair_split` | default parameter `seed: int = SPLIT_SEED` | `build_sidecar_artifacts` cannot pass seed8192 without code change |
| `build_provenance` | `split_rule.shuffle_seed = SPLIT_SEED` | provenance would record seed174 |
| `canonical_output_dir` and names | old `p3w6f2_p4l_current_lineage...` namespace | future seed8192 lineage needs a new non-destructive revised namespace/path |

## 7. Exact Consumer/Validator Discovery

Current relevant consumers/validators:

| Path | Symbol(s) | Required artifact fields or identities |
|---|---|---|
| `scripts/train_controlled_v6b_minimal.py` | `_STAGE187_AUTHORITATIVE_DATA`, `_P4X_CANONICAL_DIR`, `_STAGE187_AUTHORITATIVE_SIDECAR`, `_P4X_CANONICAL_PROVENANCE` | exact dataset path, exact historical sidecar path, exact provenance path |
| `scripts/train_controlled_v6b_minimal.py` | `_P4X_SIDECAR_PHYSICAL_SHA256`, `_P4X_PROVENANCE_PHYSICAL_SHA256`, `_STAGE187_SIDECAR_SEMANTIC_SHA256` | exact historical sidecar/provenance hashes |
| `scripts/train_controlled_v6b_minimal.py` | `_P4X_P4L_AUTHORITY_COMMIT`, `_P4X_BUILDER_SOURCE_COMMIT` | historical authority and builder commit identities |
| `scripts/train_controlled_v6b_minimal.py` | `_p4x_validate_provenance` | provenance `schema_version`, `sidecar_schema_version`, `p4l_authority_commit`, `builder_source_commit`, `row_count`, dataset hashes, sidecar hashes, boolean authority fields, artifact paths |
| `scripts/train_controlled_v6b_minimal.py` | `_p4x_validate_stable_join` | source `id`, sidecar `row_id`, one-to-one coverage, no duplicates, exact source order |
| `scripts/train_controlled_v6b_minimal.py` | `_p4x_validate_sidecar_rows` | `P2_SIDE_CAR_REQUIRED_FIELDS`, `schema_version`, source dataset path/hash/semantic hash, `frame_compatible_label`, sorted unique `reason_codes`, `integrity_status`, `p2_reason_supervision_eligible`, `eligible_for_positive_margin`, fixed historical counts |
| `scripts/train_controlled_v6b_minimal.py` | `_stage187_load_integrity_sidecar` | `eligible_for_positive_margin`, `split`, `frame_compatible_label`, `integrity_status`, `time_swap_status`, `dataset_source_status`, fixed historical eligible count |
| `scripts/train_controlled_v6b_minimal.py` | `_p2_load_reason_integrity_sidecar` | canonical P4-X binding plus sidecar rows by `row_id` |
| `scripts/train_controlled_v6b_minimal.py` | `_p2_prepare_reason_supervision` | row `id`, `pair_id`, source labels, sidecar split/canonical/status fields, binary labels, primary reason and applicable binary cohorts |
| `scripts/train_controlled_v6b_minimal.py` | `_p2_prepare_reason_supervision_train_only` | train-only calibration reason supervision; same source/sidecar field contract, train split only |
| `scripts/train_controlled_v6b_minimal.py` | `_p2_row_identity_hash` | ordered rows hashed as `id<TAB>pair_id<LF>` |
| `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py` | `EXPECTED_SIDECAR_PATH`, `EXPECTED_SIDECAR_PHYSICAL_SHA256`, `EXPECTED_SIDECAR_SEMANTIC_SHA256`, `EXPECTED_P4L_PROVENANCE_PHYSICAL_SHA256`, split checks | historical seed174 recovery constants; not admissible as seed8192 validation authority without revision |
| `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py` | builder tests | seed174 deterministic split, semantic hash excluding only `created_at`, atomic publication, no overwrite |
| `tests/test_reason_router_p2_contract.py` and `tests/test_reason_router_p3w1_calibration.py` | P2 readiness and calibration tests | P2 degenerate cohort rejection, split seed174 calibration assumptions |

No consumers were modified.

## 8. Sidecar Schema

The historical P4-L sidecar is a JSONL file with exactly 3600 JSON objects, one per dataset row in source row order. The observed material row-field union contains exactly these fields:

`canonical_row_id`, `canonical_status`, `created_at`, `dataset_source_status`, `eligible_for_positive_margin`, `family_contract_id`, `frame_compatible_label`, `generator_source_path`, `generator_source_sha256`, `grammar_status`, `historical_audit_changed_axes`, `historical_audit_expected_axes`, `historical_audit_pair_failure_scope`, `historical_audit_preserved_axes`, `historical_bridge_status`, `historical_stage185_bridge_status`, `historical_stage185_used_as_current_source_identity`, `integrity_builder_sha256`, `integrity_status`, `intervention_contract_status`, `intervention_type`, `namespace`, `p2_primary_reason`, `p2_reason_exclusion_codes`, `p2_reason_supervision_eligible`, `p2_secondary_reasons_3`, `p4b_effective_compatibility_status`, `p4b_effective_reason_codes`, `pair_id`, `polarity_contamination_status`, `reason_codes`, `row_id`, `rule_version`, `schema_status`, `schema_version`, `source_dataset_path`, `source_dataset_semantic_sha256`, `source_dataset_sha256`, `source_order_index`, `split`, `stage182a_report_sha256`, `stage184a_report_sha256`, `time_swap_status`.

There is no sidecar aggregate object or metadata header in the JSONL sidecar.

## 9. Provenance Schema

The historical provenance is one JSON object. Its observed field set is:

`a0_execution_authorized`, `artifact_materialization_authorized_by_p4l`, `authority_version`, `blockers`, `builder_source_commit`, `builder_source_path`, `builder_source_sha256`, `canonical_row_rule`, `evaluation_authorized`, `failure_reasons`, `gpu_authorized`, `historical_stage185_sidecar_path`, `historical_stage185_sidecar_semantic_sha256`, `historical_stage185_source_path`, `historical_stage185_source_sha256`, `historical_stage185_used_as_current_source_identity`, `implementation_authorized`, `kaggle_authorized`, `one_to_one_row_coverage`, `p4b_compatibility_authorized_scope`, `p4b_compatibility_provenance_path`, `p4b_compatibility_provenance_sha256`, `p4b_compatibility_rows_path`, `p4b_compatibility_rows_sha256`, `p4b_compatibility_summary_path`, `p4b_compatibility_summary_sha256`, `p4h_authority_commit`, `p4h_result_freeze_commit`, `p4h_verification_attestation_freeze_commit`, `p4k_freeze_commit`, `p4l_authority_commit`, `p4l_authority_path`, `provenance_path`, `provenance_physical_sha256_self_certified`, `row_count`, `row_order_rule`, `schema_version`, `sidecar_path`, `sidecar_physical_sha256`, `sidecar_schema_version`, `sidecar_semantic_sha256`, `source_dataset_path`, `source_dataset_semantic_sha256`, `source_dataset_sha256`, `split_rule`, `training_admission_released`, `training_authorized`, `unique_row_id`.

## 10. Field-By-Field Sidecar Classification

| Sidecar row field | Classification | Required seed8192 treatment |
|---|---|---|
| `canonical_row_id` | PRESERVED_SPLIT_INDEPENDENT | Recompute/verify from same-pair `intervention_type == "none"` rule; value should be unchanged because dataset rows/pairs are unchanged |
| `canonical_status` | PRESERVED_SPLIT_INDEPENDENT | Carry only if same row/canonical topology validates |
| `created_at` | NEW_PROVENANCE_BINDING | New authorized reconstruction timestamp; excluded from semantic hash |
| `dataset_source_status` | PRESERVED_SPLIT_INDEPENDENT | Recompute against frozen dataset identity; expected `PASS` |
| `eligible_for_positive_margin` | RECOMPUTED_SPLIT_BOUND | Depends on `split == "train"` and must be recomputed under seed8192 |
| `family_contract_id` | PRESERVED_SPLIT_INDEPENDENT | Preserve schema/intervention binding unless schema changes, which is not authorized |
| `frame_compatible_label` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact source row integer |
| `generator_source_path` | PRESERVED_SPLIT_INDEPENDENT | Historical observation may be carried as historical source path only |
| `generator_source_sha256` | PRESERVED_SPLIT_INDEPENDENT | Preserve `HISTORICAL_OBSERVATION_ONLY` or authenticated historical observation semantics |
| `grammar_status` | PRESERVED_SPLIT_INDEPENDENT | Preserve/rederive from historical bridge plus P4-B compatibility, independent of train/dev assignment |
| `historical_audit_changed_axes` | PRESERVED_SPLIT_INDEPENDENT | Historical observation carried only after row identity match |
| `historical_audit_expected_axes` | PRESERVED_SPLIT_INDEPENDENT | Historical observation carried only after row identity match |
| `historical_audit_pair_failure_scope` | PRESERVED_SPLIT_INDEPENDENT | Historical observation carried only after row identity match |
| `historical_audit_preserved_axes` | PRESERVED_SPLIT_INDEPENDENT | Historical observation carried only after row identity match |
| `historical_bridge_status` | PRESERVED_SPLIT_INDEPENDENT | Revalidate historical bridge by row/pair/intervention identity |
| `historical_stage185_bridge_status` | PRESERVED_SPLIT_INDEPENDENT | Revalidate historical Stage185 bridge by row/pair/intervention identity |
| `historical_stage185_used_as_current_source_identity` | PRESERVED_SPLIT_INDEPENDENT | Must remain `false` |
| `integrity_builder_sha256` | NEW_PROVENANCE_BINDING | Must bind to revised authorized builder source SHA |
| `integrity_status` | PRESERVED_SPLIT_INDEPENDENT | Recompute from component statuses; component statuses are row-semantic, not split-derived |
| `intervention_contract_status` | PRESERVED_SPLIT_INDEPENDENT | Preserve/revalidate from row-level historical bridge |
| `intervention_type` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact source row value |
| `namespace` | NEW_PROVENANCE_BINDING | Must distinguish revised seed8192 lineage from historical seed174 lineage without schema break |
| `p2_primary_reason` | PRESERVED_SPLIT_INDEPENDENT | Recompute from source binary labels, primary order `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED` |
| `p2_reason_exclusion_codes` | RECOMPUTED_SPLIT_BOUND | Recompute under seed8192 sidecar split/canonical checks |
| `p2_reason_supervision_eligible` | RECOMPUTED_SPLIT_BOUND | Recompute after seed8192 split/canonical/status checks |
| `p2_secondary_reasons_3` | PRESERVED_SPLIT_INDEPENDENT | Recompute from source binary labels; diagnostic only |
| `p4b_effective_compatibility_status` | PRESERVED_SPLIT_INDEPENDENT | Preserve/rederive for authorized P4-B rows only |
| `p4b_effective_reason_codes` | PRESERVED_SPLIT_INDEPENDENT | Preserve/rederive for authorized P4-B rows only |
| `pair_id` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact source row value |
| `polarity_contamination_status` | PRESERVED_SPLIT_INDEPENDENT | Preserve/revalidate row-level status |
| `reason_codes` | RECOMPUTED_SPLIT_BOUND | Recompute because split-derived codes such as `DEV_SPLIT_EXCLUDED` change |
| `row_id` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact source `id` |
| `rule_version` | NEW_PROVENANCE_BINDING | Must bind to revised authority/schema/version token while preserving schema semantics |
| `schema_status` | PRESERVED_SPLIT_INDEPENDENT | Preserve/revalidate schema status |
| `schema_version` | NEW_PROVENANCE_BINDING | Revised lineage may use a revised token but must not change fields/semantics without separate authority |
| `source_dataset_path` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact frozen dataset path |
| `source_dataset_semantic_sha256` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact frozen semantic SHA |
| `source_dataset_sha256` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact frozen tracked Git-byte SHA |
| `source_order_index` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact JSONL row order index |
| `split` | RECOMPUTED_SPLIT_BOUND | Must be recomputed from seed8192 pair assignment |
| `stage182a_report_sha256` | PRESERVED_SPLIT_INDEPENDENT | Historical observation carried only after row identity match |
| `stage184a_report_sha256` | PRESERVED_SPLIT_INDEPENDENT | Historical observation carried only after row identity match |
| `time_swap_status` | PRESERVED_SPLIT_INDEPENDENT | Preserve/revalidate row-level status |

No P2 eligibility/applicability field was found to depend on train/dev assignment except through split consistency and positive-margin/train applicability gates. Row-semantic primary reason, secondary reasons, labels, and applicability formulas remain source-derived.

## 11. Field-By-Field Provenance Classification

| Provenance field | Classification | Required seed8192 treatment |
|---|---|---|
| `a0_execution_authorized` | PRESERVED_SPLIT_INDEPENDENT | Must remain `false` |
| `artifact_materialization_authorized_by_p4l` | PRESERVED_SPLIT_INDEPENDENT | Must remain `false` for authority provenance |
| `authority_version` | NEW_PROVENANCE_BINDING | Bind to revised P4-L reconstruction/rebinding authority after activation |
| `blockers` | NEW_PROVENANCE_BINDING | Recompute for revised reconstruction result |
| `builder_source_commit` | NEW_PROVENANCE_BINDING | Bind to exact revised builder commit |
| `builder_source_path` | NEW_PROVENANCE_BINDING | Bind to revised producer path if changed, otherwise same path |
| `builder_source_sha256` | NEW_PROVENANCE_BINDING | Bind to revised builder source bytes |
| `canonical_row_rule` | PRESERVED_SPLIT_INDEPENDENT | Preserve same-pair `none` self-anchor rule |
| `evaluation_authorized` | PRESERVED_SPLIT_INDEPENDENT | Must remain `false` |
| `failure_reasons` | NEW_PROVENANCE_BINDING | Recompute for revised reconstruction result |
| `gpu_authorized` | PRESERVED_SPLIT_INDEPENDENT | Must remain `false` |
| `historical_stage185_sidecar_path` | HISTORICAL_ONLY_NOT_CARRIED | May appear only as historical bridge input, never current seed8192 artifact identity |
| `historical_stage185_sidecar_semantic_sha256` | HISTORICAL_ONLY_NOT_CARRIED | May appear only as historical bridge input |
| `historical_stage185_source_path` | HISTORICAL_ONLY_NOT_CARRIED | May appear only as historical source input |
| `historical_stage185_source_sha256` | HISTORICAL_ONLY_NOT_CARRIED | May appear only as historical source input |
| `historical_stage185_used_as_current_source_identity` | PRESERVED_SPLIT_INDEPENDENT | Must remain `false` |
| `implementation_authorized` | NEW_PROVENANCE_BINDING | Must reflect the separate revised implementation/materialization authority, not this candidate |
| `kaggle_authorized` | PRESERVED_SPLIT_INDEPENDENT | Must remain `false` |
| `one_to_one_row_coverage` | PRESERVED_SPLIT_INDEPENDENT | Must remain `true` after validation |
| `p4b_compatibility_authorized_scope` | PRESERVED_SPLIT_INDEPENDENT | Preserve 119-pair/357-member scope |
| `p4b_compatibility_provenance_path` | PRESERVED_SPLIT_INDEPENDENT | Preserve existing P4-B provenance path |
| `p4b_compatibility_provenance_sha256` | PRESERVED_SPLIT_INDEPENDENT | Preserve existing P4-B provenance SHA |
| `p4b_compatibility_rows_path` | PRESERVED_SPLIT_INDEPENDENT | Preserve existing P4-B rows path |
| `p4b_compatibility_rows_sha256` | PRESERVED_SPLIT_INDEPENDENT | Preserve existing P4-B rows SHA |
| `p4b_compatibility_summary_path` | PRESERVED_SPLIT_INDEPENDENT | Preserve existing P4-B summary path |
| `p4b_compatibility_summary_sha256` | PRESERVED_SPLIT_INDEPENDENT | Preserve existing P4-B summary SHA |
| `p4h_authority_commit` | HISTORICAL_ONLY_NOT_CARRIED | Historical seed174 chain only unless explicitly retained as precedent, not revised authority |
| `p4h_result_freeze_commit` | HISTORICAL_ONLY_NOT_CARRIED | Historical seed174 chain only |
| `p4h_verification_attestation_freeze_commit` | HISTORICAL_ONLY_NOT_CARRIED | Historical seed174 chain only |
| `p4k_freeze_commit` | HISTORICAL_ONLY_NOT_CARRIED | Historical seed174 chain only |
| `p4l_authority_commit` | NEW_PROVENANCE_BINDING | Must bind to the activated revised authority commit |
| `p4l_authority_path` | NEW_PROVENANCE_BINDING | Must bind to revised authority path |
| `provenance_path` | NEW_PROVENANCE_BINDING | Must be new seed8192 lineage path |
| `provenance_physical_sha256_self_certified` | PRESERVED_SPLIT_INDEPENDENT | Must remain `false` unless separately authorized |
| `row_count` | PRESERVED_SPLIT_INDEPENDENT | Must remain `3600` |
| `row_order_rule` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact P4-B regenerated JSONL physical row order |
| `schema_version` | NEW_PROVENANCE_BINDING | Revised lineage provenance schema token may change only as a binding token, not field semantics |
| `sidecar_path` | NEW_PROVENANCE_BINDING | Must be new seed8192 lineage path |
| `sidecar_physical_sha256` | RECOMPUTED_SPLIT_BOUND | Compute only after authorized reconstruction |
| `sidecar_schema_version` | NEW_PROVENANCE_BINDING | Bind revised schema/version token while preserving schema semantics |
| `sidecar_semantic_sha256` | RECOMPUTED_SPLIT_BOUND | Compute only after authorized reconstruction |
| `source_dataset_path` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact frozen dataset path |
| `source_dataset_semantic_sha256` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact frozen semantic SHA |
| `source_dataset_sha256` | PRESERVED_SPLIT_INDEPENDENT | Preserve exact frozen tracked Git-byte SHA |
| `split_rule` | RECOMPUTED_SPLIT_BOUND | Replace seed174 binding with exact seed8192 binding |
| `training_admission_released` | PRESERVED_SPLIT_INDEPENDENT | Must remain `false` |
| `training_authorized` | PRESERVED_SPLIT_INDEPENDENT | Must remain `false` |
| `unique_row_id` | PRESERVED_SPLIT_INDEPENDENT | Must remain `true` |

## 12. Preserved Dataset/Row Semantics

Frozen dataset binding:

| Field | Required value | Observed/static status |
|---|---|---|
| Path | `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` | PASS |
| Tracked/Git-byte SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` | PASS |
| Semantic SHA256 | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` | PASS |
| Rows | `3600` | PASS |
| Pairs | `300` | PASS |

The revised P4-L lineage must preserve exact row content, row order, `id`, `pair_id`, `final_label`, binary labels, eligibility semantics, and applicability semantics. No dataset regeneration, row modification, or label modification is authorized. Windows working-tree CRLF representation must not replace the authoritative tracked Git blob identity.

## 13. Exact Seed8192 Split Binding

The future revised P4-L lineage must bind to active authority `b4fbb5666d796161f95ae23612ce2448c25063ee`:

| Identity | Value |
|---|---|
| `SELECTED_REVISED_SPLIT_SEED` | `8192` |
| dev ratio | `0.2` |
| train pairs | `240` |
| dev pairs | `60` |
| train rows | `2880` |
| dev rows | `720` |
| pair leakage | `0` |
| split implementation | `scripts/build_controlled_v5.py::split_by_pair_id` semantics: sorted pair IDs, `random.Random(seed).shuffle`, first 60 dev pairs |

## 14. Pair/Ordered-Row Identity Compatibility

Seed8192 identities from active authority:

| Identity | SHA256 |
|---|---|
| complete pair universe | `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2` |
| selected dev pairs | `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4` |
| selected train pairs | `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049` |
| seed8192 shuffled sequence | `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55` |
| historical seed174 dev reference | `259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d` |
| ordered train rows | `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8` |
| ordered dev rows | `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4` |

Compatibility finding:

`PAIR_AND_ORDERED_ROW_IDENTITY_CONVENTIONS_COMPATIBLE_WITH_P4L_CANONICALIZATION_AFTER_BOUNDED_REBIND`

Reason: active seed8192 authority uses the same pair-level sorted/shuffled split convention and the trainer `_p2_row_identity_hash` convention (`id<TAB>pair_id<LF>`) for ordered row identities. The historical P4-L builder has the same split algorithm, but its seed value is hard-coded to 174 and therefore requires bounded rebinding.

## 15. Seed8192 Readiness Binding

Required primary reason counts:

| Split | FRAME | PREDICATE | SUFFICIENCY | AUTHORIZED |
|---|---:|---:|---:|---:|
| train | `714` | `119` | `238` | `338` |
| dev | `186` | `31` | `62` | `81` |

Required applicable binary counts:

| Split | Cohort | Class 0 | Class 1 |
|---|---|---:|---:|
| train | frame | `714` | `695` |
| train | predicate | `119` | `576` |
| train | sufficiency | `238` | `338` |
| train | polarity | `100` | `238` |
| dev | frame | `186` | `174` |
| dev | predicate | `31` | `143` |
| dev | sufficiency | `62` | `81` |
| dev | polarity | `19` | `62` |

These counts are validation targets for future reconstruction, not fields to be manually patched into a sidecar. Every current A1/A3 readiness condition must remain PASS.

## 16. Builder Reuse vs Implementation-Delta Verdict

`IMPLEMENTATION_DELTA_REQUIRED_BEFORE_REVISED_P4L_RECONSTRUCTION`

Evidence:

| Requirement | Existing builder/consumer status | Verdict |
|---|---|---|
| Split seed parameterized for artifact generation | `deterministic_pair_split` accepts `seed`, but `build_sidecar_artifacts` calls it without exposing seed; module constant `SPLIT_SEED = 174` controls build | FAIL for unchanged reuse |
| Seed8192 can be supplied explicitly | No CLI argument or `build_sidecar_artifacts` parameter supplies split seed8192 | FAIL |
| Dataset remains unchanged | Builder binds correct frozen dataset | PASS |
| Output schema remains unchanged | Existing sidecar/provenance schema can be preserved | PASS |
| Provenance records revised split | `build_provenance` records `shuffle_seed: SPLIT_SEED` with seed174 | FAIL |
| Output location can be new non-destructive lineage | Existing `canonical_output_dir` writes old current-lineage namespace keyed only by builder commit | FAIL for revised naming |
| No seed174-only assumptions remain | Builder, trainer P4-X constants, calibration aggregation, and A0 recovery constants contain seed174 assumptions | FAIL |

The historical builder must not be executed unchanged for seed8192 reconstruction.

## 17. Exact Implementation Delta If Required

Minimum bounded implementation delta before revised reconstruction:

| File/symbol | Required bounded change | Must not change |
|---|---|---|
| `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py::SPLIT_SEED` and split flow | Replace hard-coded seed174 build binding with explicit required split seed parameter, with seed8192 authority validation | Dataset content, row order, source identity, split algorithm |
| `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py::parse_args` | Add explicit `--split-seed 8192` or equivalent fail-closed parameter under revised authority | Do not allow silent default to training seed |
| `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py::build_sidecar_artifacts` | Thread split seed/dev ratio into `deterministic_pair_split` and provenance | Sidecar row schema semantics |
| `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py::build_provenance` | Bind revised authority commit/path, seed8192 split identity, pair hashes, ordered row hashes, future sidecar/provenance hashes after materialization | No historical seed174 hashes as revised identities |
| `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py::canonical_output_dir` | Use a new non-destructive revised seed8192 lineage naming rule | Never overwrite historical seed174 directory |
| `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py` or revised successor | Add static tests proving explicit seed binding, provenance seed binding, no overwrite, semantic hash rule, and schema preservation | Do not weaken historical tests without replacement |
| `scripts/train_controlled_v6b_minimal.py` P4-X constants/checks | Rebind canonical revised sidecar/provenance path and post-reconstruction hashes after materialization | Do not launch trainer; preserve fail-closed canonical checks |
| `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py` and tests | Do not use historical A0 recovery constants for seed8192; revise only under later A0 authority if needed | Do not make historical A0 admissible as seed8192 evidence |
| `scripts/aggregate_reason_router_p3w1_calibration.py` and tests | Seed174 calibration verifier must not be reused for seed8192 recalibration without separate calibration authority | Do not carry historical weight `0.6202430063306562` |

Required static validation for the delta:

- `git diff --check`
- focused builder unit tests for explicit seed8192 binding and no seed174 default leakage
- static source grep/audit proving no revised P4-L path/hash/provenance field uses historical seed174 identities as current revised identities
- later authorized artifact validation contract in Section 21

This candidate does not implement the delta.

## 18. Schema-Change Verdict

`NO_SCHEMA_CHANGE_REQUIRED_FOR_REVISED_P4L_RECONSTRUCTION`

The existing sidecar row fields and provenance field concepts are sufficient. A provenance value changing from seed174 to seed8192 is a rebinding, not a schema change. If future implementation finds that a field must be added, removed, renamed, or semantically changed, then:

`SCHEMA_CHANGE_REQUIRES_SEPARATE_EXPLICIT_AUTHORITY`

## 19. Future Artifact Naming Contract

Historical seed174 paths are immutable and must not be overwritten.

`FINAL_REVISED_P4L_OUTPUT_PATH = NOT_YET_MATERIALIZED`

Deterministic naming rule/template:

```text
reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_<authority_commit>_<builder_commit>/
  p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl
  p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json
```

`<authority_commit>` must be the full 40-character activated revised P4-L authority commit. `<builder_commit>` must be the full 40-character producer commit actually used for authorized reconstruction. If repository precedent later selects a stricter naming token, the name must remain non-destructive, unambiguous, and seed8192-bound.

Future artifact identities:

| Identity | Status |
|---|---|
| `REVISED_P4L_SIDECAR_PHYSICAL_SHA256` | `TO_BE_COMPUTED_BY_AUTHORIZED_RECONSTRUCTION` |
| `REVISED_P4L_SIDECAR_SEMANTIC_SHA256` | `TO_BE_COMPUTED_BY_AUTHORIZED_RECONSTRUCTION` |
| `REVISED_P4L_PROVENANCE_PHYSICAL_SHA256` | `TO_BE_COMPUTED_BY_AUTHORIZED_RECONSTRUCTION` |
| Additional content-derived identities | `TO_BE_COMPUTED_BY_AUTHORIZED_RECONSTRUCTION` |

## 20. Future Provenance Chain

Minimum future provenance chain:

```text
revised P4-L authority commit
-> exact producer/builder commit
-> frozen dataset physical + semantic identities
-> exact split seed8192 authority b4fbb5666d796161f95ae23612ce2448c25063ee
-> exact split algorithm identity
-> exact pair identities/hashes
-> exact ordered train/dev row identities
-> sidecar schema identity/version
-> sidecar physical SHA256
-> sidecar semantic SHA256
-> provenance physical SHA256
-> static consumer/validator compatibility result
```

The provenance must fail closed if any binding is missing or mismatched. It must clearly distinguish `historical_seed174_p4l_lineage` from `revised_seed8192_p4l_lineage`.

## 21. Future Reconstruction Validation Contract

A later authorized reconstruction must pass all checks below:

1. source dataset tracked identity exact match;
2. dataset semantic identity exact match;
3. 3600 dataset rows;
4. 3600 sidecar rows;
5. one-to-one dataset to sidecar row mapping;
6. exact dataset row order preserved where schema requires it;
7. no missing IDs;
8. no duplicate IDs;
9. no extra IDs;
10. exact `pair_id` preservation;
11. exact label preservation;
12. exact eligibility preservation;
13. exact applicability preservation;
14. split field exactly matches seed8192 pair assignment;
15. train rows = 2880;
16. dev rows = 720;
17. train pairs = 240;
18. dev pairs = 60;
19. pair leakage = 0;
20. pair hashes match `b4fbb5666d796161f95ae23612ce2448c25063ee`;
21. ordered train hash = `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8`;
22. ordered dev hash = `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4`;
23. exact primary-reason counts match authority;
24. exact applicable binary counts match authority;
25. A1/A3 static readiness PASS;
26. historical seed174 artifact remains unchanged;
27. new sidecar physical SHA computed;
28. new sidecar semantic SHA computed;
29. new provenance SHA computed;
30. provenance binds all required authority/source/builder/split identities;
31. current trainer/validator consumers accept schema and identities under static/prelaunch validation;
32. no trainer/model process launched.

No successful artifact generation alone constitutes scientific execution authority.

## 22. Static Consumer Compatibility Plan

Existing registered exact seed8192 P4-L validation gate was not found. Do not invent a `cm gate` target.

Future authorized static plan:

1. Run revised builder unit tests that do not materialize artifacts unless the implementation authority explicitly allows temporary outputs outside tracked lineage.
2. After authorized reconstruction, run a narrow direct validator that imports no model/checkpoint and checks Sections 20 and 21 against the new sidecar/provenance. Preferred implementation surface is a non-training validation function adjacent to `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py` or a dedicated revised validator.
3. Statically exercise the trainer's P4-X/P2 loader compatibility only up to sidecar/provenance/schema/readiness validation, without reaching tokenizer/model construction or `TRAINER_PROCESS_LAUNCH_BEGIN`.
4. If no direct non-training entry point exists after implementation, require one before reconstruction can be admitted for execution-lineage use.

The current `scripts/train_controlled_v6b_minimal.py::_p4x_validate_canonical_integrity_binding` cannot accept a future revised seed8192 artifact unchanged because it pins historical path, hashes, authority commit, builder commit, and count constants.

## 23. Historical Seed174 P4-L Boundary

Historical seed174 sidecar/provenance remain valid historical artifacts within their original lineage.

They must not be deleted, overwritten, renamed, edited, relabeled as seed8192, or retroactively invalidated.

For future revised factorial use:

`HISTORICAL_SEED174_P4L = NOT_ADMISSIBLE_AS_SEED8192_SPLIT_BOUND_ARTIFACT`

## 24. A0 Boundary

Existing A0 artifacts remain historical seed174-lineage evidence only. A revised split requires a separately authorized seed8192 A0 baseline execution/evidence authority after revised P4-L reconstruction and independent validation.

`A0 revised-split execution = NOT_AUTHORIZED`

## 25. Calibration Boundary

Historical calibration:

`0.6202430063306562`

remains historical seed174-lineage evidence only.

`Seed8192 = REQUIRES_RECALIBRATION`

Historical same-seed A0 references:

`NOT_ADMISSIBLE_AS_SEED8192_REVISED_SPLIT_REFERENCES`

Calibration execution is not authorized by this candidate.

## 26. Factorial/Execution Boundary

After revised P4-L reconstruction and independent validation, the downstream sequence remains:

1. revised-split A0 baseline execution/evidence authority;
2. revised-split reason-loss recalibration authority and acceptance;
3. revised factorial scientific/execution contract;
4. required implementation/prelaunch-control authority;
5. validation/freeze;
6. separate explicit execution authority;
7. only then Kaggle/training.

`A1/A2/A3 = BLOCKED`

## 27. Prelaunch-Control Boundary

The secondary prevention defect remains:

`MISSING_PRELAUNCH_P2_APPLICABLE_COHORT_FEASIBILITY_CHECK`

Future revised-split execution authority must still require a fail-closed static P2 applicable-cohort feasibility check before:

`TRAINER_PROCESS_LAUNCH_BEGIN`

A reconstructed P4-L artifact passing static validation does not replace that future execution-time check. This candidate does not implement it.

## 28. Execution Boundary

`ACTIVE_REVISED_P4L_RECONSTRUCTION_REBINDING_AUTHORITY = NONE_YET`

`CURRENT_EXECUTION_STATE = BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`

| Activity | Status |
|---|---|
| Implementation | `NOT_AUTHORIZED_BY_THIS_AUTHORITY` |
| P4-L reconstruction | `NOT_AUTHORIZED_BY_THIS_AUTHORITY` |
| P4-L provenance generation | `NOT_AUTHORIZED_BY_THIS_AUTHORITY` |
| A0 revised-split execution | `NOT_AUTHORIZED` |
| Calibration execution | `NOT_AUTHORIZED` |
| A1/A2/A3 | `BLOCKED` |
| Training | `NOT_AUTHORIZED` |
| Evaluation | `NOT_AUTHORIZED` |
| CUDA | `NOT_AUTHORIZED` |
| Kaggle | `NOT_AUTHORIZED` |

No run name. No recovery4.

## 29. Non-Inheritance

The future revised lineage must not inherit revised P4-L identities from:

- historical seed174 sidecar hashes;
- premature factorial-v2 artifacts;
- non-activated recovery proposals;
- unauthorized execution outputs.

It must derive solely from:

1. active `b4fbb5666d796161f95ae23612ce2448c25063ee` seed8192 authority;
2. frozen dataset identity;
3. authenticated historical P4-L schema/producer semantics;
4. this new authority after it becomes active;
5. a separately authorized reconstruction/materialization step.

## 30. Activation Condition

`ACTIVATION_CONDITION = ON_EXACT_COMMIT_PUSH_REMOTE_VERIFICATION`

Activation may occur only after all of:

1. exact finalized file explicitly staged;
2. staged Git blob independently byte-verified;
3. staged delta verified as exactly one intended file;
4. dedicated freeze/activation commit created;
5. exact full commit SHA obtained;
6. expected parent verified as `b4fbb5666d796161f95ae23612ce2448c25063ee`;
7. exact commit pushed;
8. remote branch tip verified to equal that commit;
9. remote blob identity verified against staged blob;
10. remote body-level lifecycle/status and substantive authority contents independently verified.

File existence does not activate authority. Staging does not activate authority. Commit subject does not activate authority. Local commit alone does not activate authority. Push alone without remote verification does not activate authority.

## 31. Candidate Path

`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_reconstruction_rebinding_provenance_authority_spec_candidate.md`

This is the only file written by this task.

## 32. Candidate SHA/Shape

Candidate SHA256, byte count, LF/CR/CRLF counts, BOM status, final LF status, and trailing whitespace lines are reported after file materialization. They are not predicted inside this body.

## 33. Git State Before/After

Before candidate creation:

| Command | Observed |
|---|---|
| `git branch --show-current` | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` |
| `git rev-parse HEAD` | `b4fbb5666d796161f95ae23612ce2448c25063ee` |
| `git status --short` | empty |
| `git diff --name-status` | empty |
| `git diff --cached --name-status` | empty |
| `git diff --check` | PASS |

Expected after candidate creation:

- same branch;
- same HEAD;
- `git status --short` shows exactly one untracked file, this candidate;
- `git diff --name-status` remains empty;
- `git diff --cached --name-status` remains empty;
- `git diff --check` remains PASS.

## 34. Blocker/Mismatch

No repository precondition, dataset identity, historical artifact identity, historical producer discovery, schema discovery, field classification, or seed8192 split identity blocker was found.

Blocking condition for reconstruction itself:

`IMPLEMENTATION_DELTA_REQUIRED_BEFORE_REVISED_P4L_RECONSTRUCTION`

Reason: the existing historical P4-L builder and current trainer consumer bindings are seed174-bound and historical-path/hash-bound. This is a bounded implementation/rebinding blocker, not a schema-change blocker.

## 35. Exact Next Authorized Action

Exact next authorized action:

`FRESH_FINAL_INDEPENDENT_VERIFICATION_OF_FINALIZED_AUTHORITY_CONTENT_ONLY`

Verification should confirm:

1. this candidate is the only untracked file;
2. no tracked or staged changes exist;
3. historical P4-L authentication remains exact;
4. active seed8192 split identity remains exact;
5. builder/consumer discovery and implementation-delta verdict are source-grounded;
6. lifecycle/status content is freeze-ready but not active before exact remote verification;
7. execution remains blocked.

No reconstruction, provenance generation, implementation, A0, calibration, factorial execution, training, evaluation, CUDA, Kaggle, staging, commit, or push is authorized by this candidate.
