# P3-W6-F2-P4-T P4-B Schema Recovery Validation Correction Execution Authority Specification

Authority/version:

`P3W6F2P4T_P4B_SCHEMA_RECOVERY_VALIDATION_CORRECTION_EXECUTION_AUTHORITY_V1`

This document is a bounded authority-spec candidate only. Candidate creation
does not authorize Python execution, pytest, Kaggle, builder execution,
materialization, canonical artifact mutation, recovery validation execution,
staging, commit, or push.

After targeted independent static verification PASS and immutable P4-T freeze,
this authority may authorize exactly one future CPU-only read-only recovery
validation of the already published immutable canonical P4-L bytes. The only
semantic correction from frozen P4-S is the P4-B summary/provenance schema
assertion correction defined here.

## 1. Candidate Creation State

Candidate creation authority:

- Current controller instruction.
- Expected local HEAD:
  `2faa789c35f7ff9258fb7b005a92890da17d04be`.
- Frozen P4-S:
  `2faa789c35f7ff9258fb7b005a92890da17d04be`.
- Frozen P4-R:
  `d80ed289273763c1c90f2fba14ca796c604c9529`.
- Execution HEAD:
  `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- Frozen P4-L:
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`.

Candidate creation requires:

- `git rev-parse HEAD ==
  2faa789c35f7ff9258fb7b005a92890da17d04be`.
- tracked worktree clean.
- index clean.
- candidate path absent before creation:
  `reports/reason_router_p2_p3w6f2_p4t_p4b_schema_recovery_validation_correction_execution_authority_spec.md`.
- pre-existing unrelated untracked files untouched.
- exactly one new untracked P4-T authority spec created.

Current candidate creation evidence:

- `git rev-parse HEAD` returned
  `2faa789c35f7ff9258fb7b005a92890da17d04be`.
- `git diff --quiet` exited `0`.
- `git diff --cached --quiet` exited `0`.
- `Test-Path -LiteralPath <candidate path>` returned `False`.
- `git status --short --branch` reported branch `main...origin/main`, no
  tracked modifications, and pre-existing unrelated untracked files.

The pre-existing unrelated untracked files are not part of P4-T and must remain
untouched.

## 2. Prior Disposition

P4-T records and preserves these historical dispositions:

- P4-Q Gate 1 = PASS.
- P4-Q overall = FAIL due filename-order authority defect.
- P4-R = FAIL due source-semantic validator defect.
- P4-S = FAIL with exact observed token:
  `P4S_FAIL:P4B_SUMMARY_PAIR_SCOPE_MISMATCH`.

P4-S did not establish artifact validity. P4-T must not retroactively mark
P4-Q, P4-R, or P4-S PASS.

## 3. Scope

P4-T corrects only P4-S's incorrect executable assertions about the P4-B
summary and P4-B provenance schemas.

P4-T does not:

- modify P4-S.
- rerun P4-S.
- materialize artifacts.
- execute the builder.
- mutate canonical P4-L bytes.
- change source, sidecar, or provenance bytes.
- change the P4-L artifact contract.
- authorize trainer rebind, A0, training, or evaluation.

The already published canonical P4-L artifact/provenance bytes are read-only
validation inputs.

## 4. Runtime Authority Boundary

After P4-T freeze, the workflow controller must supply exactly one runtime
parameter:

`P4T_AUTHORITY_FREEZE`

The future validator must require:

- `P4T_AUTHORITY_FREEZE` is set.
- `P4T_AUTHORITY_FREEZE` is exactly 40 lowercase hex characters.
- `git cat-file -e "${P4T_AUTHORITY_FREEZE}^{commit}"` succeeds.
- the frozen P4-T authority path exists at that commit.
- the validator does not check out the P4-T freeze commit.
- `git rev-parse HEAD ==
  2f9e6076791358922e3ebd70e89533d9cb83b458`.
- CPU-only operation; GPU disabled before and after.

The sole future validation is read-only. It must not write, unlink, rename,
replace, chmod, clean, move-aside, or create repository artifact paths.

## 5. Exact Root Cause

Frozen P4-S incorrectly checked P4-B summary fields:

- `pair_count`
- `pairs`
- `member_count`
- `members`

Those fields are not the frozen P4-B compatibility summary schema.

Frozen P4-S also incorrectly treated the P4-B provenance manifest as a source
of pair/member count fields. The frozen P4-B provenance schema does not contain
`pair_count`, `pairs`, `member_count`, or `members`.

Correct P4-T behavior is to validate P4-B count scope from:

- P4-B compatibility rows.
- P4-B compatibility summary.
- canonical P4-L provenance
  `p4b_compatibility_authorized_scope`.

It must not require count fields from the P4-B provenance manifest.

## 6. Frozen P4-B Summary Schema Proof

Actual frozen P4-B summary path:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json`

Actual frozen P4-B summary SHA256:

`ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8`

Static read of the frozen JSON established these actual keys and values:

- `schema_version =
  P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1`
- `authorized_member_count = 357`
- `authorized_pair_count = 119`
- `compatibility_fail_count = 0`
- `compatibility_gate_status = "PASS"`
- `compatibility_pass_count = 357`
- `compatibility_rule_version =
  P3W6F2P4B_R1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY_V1`
- `compatibility_unresolved_count = 0`
- `failure_reasons = []`
- `historical_authority_weakened = false`
- `permitted_predicate_realization_delta_count = 357`
- `raw_stage185_predicate_axis_observation_count = 238`
- `row_count = 357`
- `stage185_v1_mutated = false`
- `training_admission_released = false`

The summary schema does not contain `pair_count`, `pairs`, `member_count`, or
`members`.

## 7. Corrected Summary Assertions

The future P4-T validator must assert the frozen P4-B summary fields exactly:

```text
summary["schema_version"] == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1"
summary["compatibility_rule_version"] == "P3W6F2P4B_R1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY_V1"
summary["row_count"] == 357
summary["authorized_pair_count"] == 119
summary["authorized_member_count"] == 357
summary["compatibility_gate_status"] == "PASS"
summary["training_admission_released"] is false
```

The future validator may also preserve P4-S/P4-R physical SHA checks for the
summary file, but count scope must use the exact frozen fields above.

## 8. Frozen P4-B Provenance Schema Proof

Actual frozen P4-B provenance path:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json`

Actual frozen P4-B provenance SHA256:

`09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6`

Static read of the frozen JSON established these actual keys:

- `base_form_coverage_path`
- `base_form_coverage_sha256`
- `compatibility_rows_path`
- `compatibility_rows_sha256`
- `compatibility_rule_version`
- `compatibility_summary_path`
- `compatibility_summary_sha256`
- `created_at_utc`
- `historical_stage185_authority`
- `historical_stage185_authority_sha256`
- `regenerated_dataset_path`
- `regenerated_dataset_sha256`
- `schema_version`
- `stage185_source_script`
- `stage185_source_script_sha256`
- `structured_source_producer`
- `structured_source_producer_sha256`

The provenance schema does not contain `pair_count`, `pairs`,
`member_count`, or `members`.

## 9. Corrected Provenance Assertions

The future P4-T validator must assert the frozen P4-B provenance fields exactly:

```text
provenance["schema_version"] == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_PROVENANCE_V1"
provenance["compatibility_rows_sha256"] == "59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f"
provenance["compatibility_summary_sha256"] == "ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8"
provenance["regenerated_dataset_path"] == "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
provenance["regenerated_dataset_sha256"] == "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
provenance["stage185_source_script_sha256"] == "11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc"
```

Do not require P4-B pair/member count fields from the P4-B provenance manifest.

## 10. Removed Nonexistent Count-Field Assertions

P4-T explicitly removes the P4-S requirement that P4-B summary or P4-B
provenance contain any of:

- `pair_count`
- `pairs`
- `member_count`
- `members`

These fields are not part of the frozen P4-B summary/provenance schema.

Generic fallback forms are forbidden, including:

```text
summary.get("pair_count", summary.get("pairs"))
summary.get("member_count", summary.get("members"))
provenance.get("pair_count", provenance.get("pairs"))
provenance.get("member_count", provenance.get("members"))
```

Every executable P4-B field assertion must use an actually frozen field name.

## 11. Correct 119/357 Scope Validation

P4-T requires all three independent scope surfaces.

Surface A: compatibility rows:

- physical rows SHA256 equals
  `59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f`.
- exactly `357` JSONL row objects.
- every row has schema
  `P3W6F2P4B_R1_STAGE185_COMPATIBILITY_ROW_V1`.
- every row has
  `compatibility_rule_version ==
  P3W6F2P4B_R1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY_V1`.
- every row has `effective_compatibility_status == "PASS"`.
- every row has
  `training_admission_effect.training_admission_released is false`.
- every `member_id` is non-empty and unique.
- exactly `119` unique non-empty `pair_id` values.

Surface B: compatibility summary:

- `row_count == 357`.
- `authorized_pair_count == 119`.
- `authorized_member_count == 357`.

Surface C: canonical P4-L provenance:

- `p4b_compatibility_authorized_scope` is a dict.
- `p4b_compatibility_authorized_scope["pair_count"] == 119`.
- `p4b_compatibility_authorized_scope["member_count"] == 357`.

Do not infer full-3600 compatibility from scoped P4-B artifacts.

## 12. Builder Equivalence

Exact builder at execution HEAD
`2f9e6076791358922e3ebd70e89533d9cb83b458` contains
`load_p4b_compatibility(repo_root, rows_path, summary_path, provenance_path)`.

P4-T corrected assertions are builder-equivalent to that function's P4-B checks:

```text
summary.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1"
summary.get("compatibility_rule_version") == P4B_COMPATIBILITY_RULE_VERSION
summary.get("row_count") == P4B_AUTHORIZED_MEMBER_COUNT
summary.get("authorized_pair_count") == P4B_AUTHORIZED_PAIR_COUNT
summary.get("authorized_member_count") == P4B_AUTHORIZED_MEMBER_COUNT
summary.get("compatibility_gate_status") == "PASS"
summary.get("training_admission_released") is False
provenance.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_PROVENANCE_V1"
provenance.get("compatibility_rows_sha256") == P4B_COMPATIBILITY_ROWS_SHA256
provenance.get("compatibility_summary_sha256") == P4B_COMPATIBILITY_SUMMARY_SHA256
provenance.get("regenerated_dataset_path") == SOURCE_DATASET_PATH
provenance.get("regenerated_dataset_sha256") == SOURCE_DATASET_SHA256
provenance.get("stage185_source_script_sha256") == HISTORICAL_STAGE185_SOURCE_SHA256
len(by_member) == P4B_AUTHORIZED_MEMBER_COUNT
len(pair_ids) == P4B_AUTHORIZED_PAIR_COUNT
```

The builder performs P4-B count checks from summary fields and from the
compatibility rows' member/pair universe, not from P4-B provenance count fields.

## 13. Source Semantic Regression Guard

P4-T preserves P4-S's corrected source semantic hash exactly:

- whole ordered projected source list.
- projection fields, in order:
  `id`, `pair_id`, `claim`, `evidence`, `final_label`,
  `frame_compatible_label`, `predicate_covered_label`, `sufficiency_label`,
  `polarity_label`, `primary_failure_type`, `intervention_type`.
- canonical JSON serialized once.
- JSON settings: `sort_keys=true`, `separators=(",", ":")`,
  `ensure_ascii=false`, non-finite values rejected.
- SHA256 over the single canonical JSON byte string.

Expected:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

No JSONL semantic hashing, newline-concatenated rows, row-by-row hashing, row
sorting, missing-field fallback, or final newline is authorized for this
semantic hash.

## 14. Sidecar Semantic Regression Guard

P4-T preserves P4-S's corrected sidecar semantic hash exactly:

- whole ordered sidecar list.
- remove `created_at` only.
- remove no other field.
- sort keys within each object.
- canonical JSON serialized once.
- JSON settings: `sort_keys=true`, `separators=(",", ":")`,
  `ensure_ascii=false`, non-finite values rejected.
- SHA256 over the single canonical JSON byte string.

Expected:

`0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

No JSONL semantic hashing, newline-concatenated rows, row-by-row hashing, row
sorting, extra field deletion, or final newline is authorized for this semantic
hash.

## 15. Preserved P4-S Pass-Grade Checks

P4-T preserves P4-S's pass-grade checks without redesign, except for the P4-B
schema correction explicitly defined here:

- immutable recovery hashes.
- exact detached `2f9e6076791358922e3ebd70e89533d9cb83b458` execution.
- sole runtime parameter `P4T_AUTHORITY_FREEZE`.
- CPU only and GPU off.
- no builder/materialization/rerun.
- exact two-file check.
- path/symlink safety.
- serialization and non-finite checks.
- source 3600/order/schema checks.
- split replay.
- canonical replay.
- primary reason replay.
- positive-margin replay.
- unresolved reporting.
- P4-B physical SHA checks.
- canonical provenance P4-B path/SHA bindings.
- source provenance bindings.
- sidecar physical/semantic hash.
- provenance physical hash.
- provenance self-certified false.
- P4-L intrinsic false flags.
- postconditions.
- no artifact writes or cleanup.
- no trainer, A0, training, or evaluation.

## 16. Immutable/No-Materialization Boundary

P4-T validation input is the existing canonical P4-L directory:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458`

The future validation must verify, read, and hash only. It must not:

- create that directory.
- delete or replace that directory.
- rewrite the sidecar.
- rewrite the provenance.
- materialize a new sidecar/provenance pair.
- invoke the builder.
- invoke P4-Q Gate 1.
- invoke P4-R or P4-S validators.

If canonical bytes are absent or inconsistent, P4-T FAILS or BLOCKS according
to the failure mapping; it must not repair the bytes.

## 17. Observations And PASS Token

The future P4-T validator must print observations JSON immediately before the
PASS token.

The observations JSON must include at least:

- authority freeze.
- execution HEAD.
- P4-B summary schema proof.
- P4-B provenance schema proof.
- compatibility row count and unique pair count.
- summary row/pair/member counts.
- canonical provenance P4-B authorized scope.
- source semantic SHA.
- sidecar semantic SHA.
- sidecar physical SHA.
- provenance physical SHA.
- GPU observation.
- post-HEAD and tracked/index state.

Only after every read-only assertion and postcondition passes may it print:

`P3W6F2P4T_P4B_SCHEMA_RECOVERY_VALIDATION_CORRECTION_PASS`

## 18. Interpretation On P4-T PASS

On P4-T PASS:

- canonical P4-L artifact/provenance integrity = ESTABLISHED.
- P4-Q overall remains FAIL.
- P4-R overall remains FAIL.
- P4-S overall remains FAIL.
- trainer rebind remains NOT AUTHORIZED.
- A0 remains NOT AUTHORIZED.
- training remains NOT AUTHORIZED.
- evaluation remains NOT AUTHORIZED.
- scientific conclusion remains NOT_ESTABLISHED.

P4-T establishes artifact/provenance integrity only for the frozen P4-L
contract and immutable canonical bytes. It does not establish model efficacy.

## 19. Failure Mapping

P4-T FAILS for:

- P4-B rows, summary, or provenance hash mismatch.
- P4-B summary schema mismatch.
- P4-B provenance schema mismatch.
- any use of nonexistent P4-B summary/provenance count fields.
- any generic fallback field-name assertion.
- compatibility row count not `357`.
- compatibility unique pair count not `119`.
- summary `row_count`, `authorized_pair_count`, or
  `authorized_member_count` mismatch.
- canonical P4-L provenance
  `p4b_compatibility_authorized_scope` missing or mismatched.
- source semantic hash regression.
- sidecar semantic hash regression.
- canonical artifact/provenance bytes mismatch.
- mutation/materialization attempt.
- trainer/A0/training/evaluation widening.

P4-T BLOCKS for:

- P4-T authority freeze unavailable.
- execution HEAD mismatch.
- local environment/tooling inability that prevents safe read-only validation
  without evidence of artifact defect.
- canonical path ambiguity that cannot be safely inspected read-only.

## 20. Static Verification Requirement

Before freeze, require one targeted independent static verifier.

Verifier scope is only:

- actual P4-B summary schema versus executable assertions.
- actual P4-B provenance schema versus executable assertions.
- builder `load_p4b_compatibility()` equivalence at
  `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- correct 119/357 three-surface scope validation.
- source semantic-hash regression check.
- sidecar semantic-hash regression check.
- no mutation/materialization.
- authority boundaries.

No broad re-audit is authorized by this requirement.

The verifier must reject any executable assertion using guessed or fallback
P4-B field names.

## 21. Final State Requirements

Candidate creation final state requires:

- HEAD unchanged:
  `2faa789c35f7ff9258fb7b005a92890da17d04be`.
- tracked worktree clean.
- index clean.
- one new untracked P4-T spec only, with pre-existing unrelated untracked
  files untouched.
- `git diff --check` PASS.
- candidate SHA256 computed and reported.

No commit or push is authorized.

## 22. Candidate Path

Candidate path:

`reports/reason_router_p2_p3w6f2_p4t_p4b_schema_recovery_validation_correction_execution_authority_spec.md`

Candidate SHA256 is computed after file creation. It is a property of this
authority-spec file only and is not a predicted artifact hash.

## 23. Blockers

Current candidate-creation blockers: none identified.

Independent targeted static verification is required before freeze.

Success token:

`P3W6F2P4T_P4B_SCHEMA_RECOVERY_VALIDATION_CORRECTION_EXECUTION_AUTHORITY_CANDIDATE_READY_FOR_TARGETED_VERIFICATION`
