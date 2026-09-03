# P3-W7 A0 Seed180 Reference Recovery Execution Authority Audit-Schema Correction Spec Candidate

Status: READY candidate, static audit-schema correction only.

This candidate authorizes no recovery execution during candidate writing. It corrects the frozen v2 recovery execution authority's `A0_REFERENCE_AUDIT` schema binding after independently reconciling the frozen v2 requirements against the exact corrected helper implementation.

## Authority Basis

Current authoring state verified for this candidate:

- HEAD: `39dade8cdeb74edc2fba4a1376334af8a9d61478`
- Branch: `p3w7-a0-seed180-reference-recovery-execution-authority-audit-schema-correction`
- Frozen but execution-blocked v2 authority: `39dade8cdeb74edc2fba4a1376334af8a9d61478`
- Frozen v2 authority file: `reports/reason_router_p3w7_a0_seed180_reference_recovery_execution_authority_spec_candidate.md`
- Exact corrected helper implementation commit: `812c82d96b2461ed7ae236f6c3ba6d0cf775a182`
- Corrected helper path: `scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py`
- Corrected helper committed LF SHA256: `bf63b73d9aac2f2546dc3182599e41cf611470f65f17b61fadd6d11fab450f30`
- Frozen sidecar-semantic correction authority: `3e1bb765883f2d2bad9a77e67dd58b0a691cfc22`
- Frozen upstream retained-artifact recovery authority: `ceaee6236340ef7006f7004d910f388ec565db0e`
- Source seed180 execution: `2737c3c6116ae3766b469801f990e2c45ba9a55e`
- Historical provenance recovery authority: `233ed0be080e1d30dd47de2e66136475ec2ede76`

AGENTS.md was inspected. This task is static execution-authority correction only. No implementation change, helper execution, dedicated execution worktree creation, ZIP extraction or materialization, dataset or sidecar mutation, checkpoint or model deserialization, training, evaluation, Kaggle use, commit, or push is authorized.

## Exact V2 Defect

The frozen v2 authority at `39dade8cdeb74edc2fba4a1376334af8a9d61478` requires this persisted audit key/value:

| Field | Required value |
|---|---|
| `manifest_sha256` | `69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed` |

The exact corrected helper implementation at `812c82d96b2461ed7ae236f6c3ba6d0cf775a182` does not emit `manifest_sha256` in `build_audit()`. It emits this key/value:

| Field | Required value |
|---|---|
| `recovery_manifest_sha256` | `69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed` |

The manifest SHA256 value is unchanged and correct. The defect is only the frozen v2 execution authority's required field name.

Verdict: v2 remains immutable provenance, but v2 is execution-blocked and is not admissible as runtime recovery execution authority.

## Audit-Schema Reconciliation

The exact corrected helper `build_audit()` was inspected at `812c82d96b2461ed7ae236f6c3ba6d0cf775a182`. It constructs the persisted audit dictionary with these static and path-bound fields before adding the independently recomputed dev-identity fields:

| Helper-produced field | Helper-produced value or binding | V2 reconciliation |
|---|---|---|
| `audit_id` | `p3_seed180_A0_REFERENCE_AUDIT` | Compatible; not contradicted by v2. |
| `run_id` | `p3_seed180_A0` | Compatible; not contradicted by v2. |
| `seed` | `180` | Compatible; preserves seed180. |
| `status` | `PASS` | Matches v2 required value. |
| `errors` | empty list | Compatible with `PASS`. |
| `execution_commit` | `2737c3c6116ae3766b469801f990e2c45ba9a55e` | Compatible with source execution binding. |
| `p2_implementation_tested_commit` | `2737c3c6116ae3766b469801f990e2c45ba9a55e` | Compatible; not contradicted by v2. |
| `output_dir` | `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0` | Compatible with v2 destination. |
| `reference_prediction_path` | destination `training_report_predictions.jsonl` path | Compatible with v2 artifact table. |
| `prediction_sha256` | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` | Compatible with v2 retained ZIP member table. |
| `selected_checkpoint_path` | destination `selected_checkpoint.pt` path | Compatible with v2 artifact table and Git exclusion boundary. |
| `selected_checkpoint_sha256` | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` | Compatible with v2 retained ZIP member table. |
| `report_path` | destination `training_report.json` path | Compatible with v2 artifact table. |
| `report_sha256` | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` | Compatible with v2 retained ZIP member table. |
| `selected_epoch` | selected epoch reread from `training_report.json` | Compatible; helper requires non-null before `PASS`. |
| `selected_epoch_source` | epoch source path in `training_report.json` | Compatible; helper requires non-null before `PASS`. |
| `data_path` | canonical dataset path | Compatible with v2 dataset identity. |
| `dataset_sha256_expected` | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` | Matches v2 required dataset physical SHA. |
| `dataset_sha256_observed` | runtime SHA256 of canonical dataset path | Matches v2 when observed value equals expected value. |
| `sidecar_path` | canonical sidecar path | Compatible with v2 sidecar identity. |
| `sidecar_semantic_sha256_expected` | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` | Matches v2 required sidecar semantic SHA. |
| `sidecar_semantic_sha256_observed` | runtime semantic SHA256 of canonical sidecar path | Matches v2 when observed value equals expected value. |
| `split_seed` | `174` | Compatible; preserves split contract. |
| `split_policy` | `fixed_explicit_split_seed` | Compatible; preserves split contract. |
| `dev_ratio` | `0.2` | Compatible; preserves split contract. |
| `source_execution_commit` | `2737c3c6116ae3766b469801f990e2c45ba9a55e` | Matches v2 required value. |
| `recovery_authority_commit` | `233ed0be080e1d30dd47de2e66136475ec2ede76` | Matches v2 required value. |
| `retained_zip_path` | exact runtime ZIP path | Compatible with v2 retained ZIP contract. |
| `retained_zip_sha256` | `6bbd0e89a5858d7c68b1eecc1cf44911cc415c7411670a94605c56c08e955966` | Matches v2 required value. |
| `recovery_manifest_sha256` | `69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed` | Correct helper key; replaces void v2 key `manifest_sha256`. |
| `standard_cm_wrapper_provenance` | `INCOMPLETE` | Matches v2 required value. |
| `provenance_disposition` | `RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE` | Matches v2 required value. |
| `recovery_reference_status` | `RECOVERY_REFERENCE_AUDIT_PASS` | Matches v2 required value. |

The helper then adds the `validate_dev_identity()` result fields:

- `authoritative_dev_row_count`
- `authoritative_dev_row_identity_hash`
- `prediction_joined_dev_row_identity_hash`
- `gold_counts`
- `prediction_counts`
- `a0_false_entitlement_count`
- `a0_stable_true_support_count`
- `row_count`
- `unique_row_id_count`
- `unique_row_pair_count`

The helper requires these fields to be non-null before allowing `PASS`:

- `selected_epoch`
- `selected_epoch_source`
- `prediction_sha256`
- `selected_checkpoint_sha256`
- `report_sha256`
- `authoritative_dev_row_identity_hash`
- `prediction_joined_dev_row_identity_hash`

Persisted-audit validation was inspected. `validate_persisted_audit()` requires:

- persisted root is an object;
- persisted `status` is `PASS`;
- persisted payload equals the exact in-memory `expected_audit` dictionary.

`validate_published_reference()` rereads `A0_REFERENCE_AUDIT.json` and revalidates exact equality, dataset identity drift, sidecar semantic identity drift, and prediction identity drift.

Complete reconciliation verdict: PASS. The only execution-blocking schema discrepancy between the frozen v2 authority requirements and the corrected helper output schema is `manifest_sha256` versus `recovery_manifest_sha256`.

## Corrected Authority Semantics

The frozen v2 authority commit `39dade8cdeb74edc2fba4a1376334af8a9d61478` remains immutable provenance.

The frozen v2 authority is not admissible as runtime recovery execution authority because it requires the wrong persisted audit field name.

The v2 `manifest_sha256` key requirement is VOID.

The intended frozen manifest identity remains unchanged:

`69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed`

The admissible exact key/value is:

| Field | Required value |
|---|---|
| `recovery_manifest_sha256` | `69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed` |

All other v2 constraints are preserved. No helper code modification is authorized or required.

## Runtime Self-Binding

This candidate defines:

`RECOVERY_EXECUTION_AUTHORITY_COMMIT`

as the immutable Git commit that first freezes this exact independently verified audit-schema correction candidate.

Before freeze, the literal 40-hex commit is unknown. This is an intentional future binding, not a runtime placeholder.

After freeze:

- the dedicated LF execution worktree HEAD must equal the new correction freeze commit;
- the helper CLI must receive that same literal through `--expected-recovery-execution-authority-commit`;
- corrected helper implementation commit `812c82d96b2461ed7ae236f6c3ba6d0cf775a182` must be an ancestor of runtime HEAD;
- v2 authority commit `39dade8cdeb74edc2fba4a1376334af8a9d61478` is provenance and defective predecessor only and is not runtime HEAD;
- `98dfe3ee25c266ad0e12e2215f8ca68ea499fdda` remains defective helper predecessor only.

Verdict: PASS. The runtime authority self-binding is fail-closed and points to the future correction freeze commit, not the defective v2 predecessor.

## Preserved V2 Execution Contract

This correction preserves the frozen v2 execution contract without broadening:

- dedicated LF worktree requirement;
- `core.autocrlf=false` and `core.eol=lf` checkout-time setup;
- actual byte preflight;
- canonical dataset physical SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`;
- canonical dataset size `1879593`;
- dataset CR `0` and CRLF `0`;
- corrected helper committed LF SHA256 `bf63b73d9aac2f2546dc3182599e41cf611470f65f17b61fadd6d11fab450f30`;
- sidecar semantic SHA256 `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`;
- exact retained ZIP path `C:\Users\Home1\Downloads\seed180_a0_6189be22715e.zip`;
- retained ZIP SHA256 `6bbd0e89a5858d7c68b1eecc1cf44911cc415c7411670a94605c56c08e955966`;
- exact retained ZIP member table;
- destination first-execution absence requirement;
- exact materialize-reference CLI shape;
- exact six final writes plus helper-owned staging;
- `SUCCESS`, `HELPER_BLOCKED`, `EXECUTION_ENVIRONMENT_BLOCKED`, `ENVIRONMENT_FAILURE`, and `PARTIAL_PUBLICATION` distinctions;
- fail-closed retry semantics;
- no retry after `SUCCESS`;
- historical standard-CM provenance remains `INCOMPLETE`;
- no training, evaluation, A1-A3 execution, or scientific conclusion;
- `selected_checkpoint.pt` is not Git-eligible;
- independent result and artifact verification is required after any future `SUCCESS`.

The exact future command remains the v2 command shape, with only the future frozen correction authority commit substituted as the literal runtime self-binding:

```bash
python scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py materialize-reference --zip C:\Users\Home1\Downloads\seed180_a0_6189be22715e.zip --expected-recovery-execution-authority-commit <RECOVERY_EXECUTION_AUTHORITY_COMMIT>
```

## Explicit Non-Authorization

This candidate does not authorize:

- helper execution;
- dedicated execution worktree creation;
- ZIP extraction or materialization;
- dataset mutation;
- sidecar mutation;
- model or checkpoint deserialization;
- training;
- evaluation;
- Kaggle;
- commit;
- push.

## Candidate-Writing Validation Summary

Read-only validation performed before creating this candidate:

- verified current HEAD equals `39dade8cdeb74edc2fba4a1376334af8a9d61478`;
- verified current branch equals `p3w7-a0-seed180-reference-recovery-execution-authority-audit-schema-correction`;
- inspected AGENTS.md;
- inspected frozen v2 authority exact text at `39dade8cdeb74edc2fba4a1376334af8a9d61478`;
- inspected exact corrected helper at `812c82d96b2461ed7ae236f6c3ba6d0cf775a182`;
- inspected `build_audit()`;
- inspected `validate_persisted_audit()` and the reread/equality path;
- verified corrected helper committed LF SHA256 is `bf63b73d9aac2f2546dc3182599e41cf611470f65f17b61fadd6d11fab450f30`;
- verified cited authority objects are present;
- verified corrected helper implementation commit `812c82d96b2461ed7ae236f6c3ba6d0cf775a182` is an ancestor of current HEAD;
- verified `39dade8cdeb74edc2fba4a1376334af8a9d61478`, `3e1bb765883f2d2bad9a77e67dd58b0a691cfc22`, `ceaee6236340ef7006f7004d910f388ec565db0e`, `2737c3c6116ae3766b469801f990e2c45ba9a55e`, and `233ed0be080e1d30dd47de2e66136475ec2ede76` are ancestors of current HEAD.

Final verdict: READY.

The exact v2 defect is `manifest_sha256` versus helper-produced `recovery_manifest_sha256`.

The complete audit-schema reconciliation found no additional contradiction.

The corrected exact audit key/value is `recovery_manifest_sha256 = 69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed`.

V2 remains immutable provenance but is not admissible as runtime recovery execution authority.

Future runtime authority must be the immutable Git commit that first freezes this exact audit-schema correction candidate.
