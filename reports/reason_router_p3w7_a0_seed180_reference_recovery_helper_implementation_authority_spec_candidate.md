# P3-W7 A0 Seed180 Reference Recovery Helper Implementation Authority Specification Candidate

Authority/version:

`P3W7_A0_SEED180_REFERENCE_RECOVERY_HELPER_IMPLEMENTATION_AUTHORITY_V1_CANDIDATE`

## Status

READY.

This is a static helper implementation-authority candidate only. It authorizes no retained ZIP extraction, no seed180 artifact materialization, no `A0_REFERENCE_AUDIT.json` creation, no training, no evaluation, no model load, no checkpoint deserialization, no Kaggle execution, no run-registry mutation, no commit, and no push.

If independently verified and frozen, this candidate may authorize a later bounded helper implementation. That later implementation still does not by itself authorize recovery execution/import; a separate recovery execution authority remains required.

## Authority Basis

Authority freeze commit for this candidate:

`ceaee6236340ef7006f7004d910f388ec565db0e`

Frozen retained-artifact reference-recovery authority:

`reports/reason_router_p3w7_a0_seed180_retained_artifact_reference_recovery_authority_spec_candidate.md`

Existing recovery implementation inspected:

`scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`

Existing recovery tests inspected:

`tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`

P3 A0 reference contract inspected:

- `reports/reason_router_p2_p3_execution_spec.md`
- `scripts/train_controlled_v6b_minimal.py`
- `tests/test_reason_router_p2_contract.py`

Repository-wide `AGENTS.md` applies.

Authority identities that must remain distinct:

1. authority freeze commit: `ceaee6236340ef7006f7004d910f388ec565db0e`;
2. future implementation commit: the commit that implements this helper after this candidate is frozen;
3. later recovery execution authority: a separate authority that may authorize running the helper against the retained ZIP.

This helper implementation authority does not authorize the later recovery execution.

## Selected Implementation Shape

Selected shape: add one new narrowly scoped helper script.

Future source file permitted to change:

`scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py`

Future test file permitted to change:

`tests/test_reason_router_p3w7_a0_seed180_reference_recovery_helper.py`

No trainer, model, checkpoint-loader, dataset-builder, existing recovery script, existing recovery test, run registry, retained ZIP, dataset, sidecar, seed180 artifact, or existing authority/report modification is authorized.

Rationale: the existing `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py` already has bounded `collect` and `audit-import` semantics for a recovery provenance package. Its `audit-import` command intentionally writes an audit outside the repository and does not materialize normal seed-local A0 reference artifacts. Extending that command would couple external provenance-audit semantics to repo-local materialization and the normal P3 dependency gate. A new static helper keeps the recovery bridge auditable and reduces regression risk to the existing provenance recovery workflow.

If future implementation proves trainer modification is required, the implementation is BLOCKED. The authorized response is to stop and draft a new authority question, not to broaden scope.

## Frozen Retained ZIP Contract

Retained ZIP path:

`C:\Users\Home1\Downloads\seed180_a0_6189be22715e.zip`

Retained ZIP SHA256:

`6bbd0e89a5858d7c68b1eecc1cf44911cc415c7411670a94605c56c08e955966`

The exact ZIP entry set is:

| ZIP entry | Size | SHA256 |
| --- | ---: | --- |
| `recovery_manifest.json` | 2144 | `69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json` | 306114 | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json` | 4838225 | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl` | 3934123 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/selected_checkpoint.pt` | 518269815 | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json` | 68429 | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` |

The helper must fail closed before materialization on missing entry, unexpected entry, duplicate entry, duplicate logical entry, directory entry, encrypted entry, symlink or special-file entry, absolute path, drive path, traversal or `..`, dot path, backslash path ambiguity, NUL path, malformed path, or noncanonical entry name.

The helper must validate the ZIP SHA256 before opening entries for materialization. Reading `selected_checkpoint.pt` as raw bytes for SHA256/size is allowed. Deserializing or loading the checkpoint is forbidden.

## Required Helper CLI Contract

The future helper must expose exactly one materialization command:

```text
python scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py materialize-reference --zip C:\Users\Home1\Downloads\seed180_a0_6189be22715e.zip --expected-authority-freeze-commit ceaee6236340ef7006f7004d910f388ec565db0e
```

The command must:

- resolve the repository root from Git;
- require current `HEAD` to equal `--expected-authority-freeze-commit`;
- reject malformed commit values;
- perform static file and JSON validation only;
- never import `torch`, `transformers`, `mamba_ssm`, `scripts.train_controlled_v6b_minimal`, or trainer-local model/runtime modules;
- never execute the trainer;
- never mutate the source ZIP;
- return nonzero with a blocker-style diagnostic on any failed gate.

Optional test-only dependency injection is allowed inside functions, but no production CLI option may broaden the ZIP identity, destination namespace, dataset identity, sidecar identity, split policy, seed, or artifact allowlist.

## Manifest And Provenance Gates

`recovery_manifest.json` and packaged `run_provenance.json` must be parsed with duplicate-key rejection.

Before writing anything, the helper must validate all frozen manifest/provenance bindings, including:

- `schema = contramamba-seed180-a0-provenance-recovery-v1`;
- `seed = 180`;
- `attempt_disposition = CONSUMED`;
- `execution_status = completed`;
- `original_authorized_wrapper_sha256 = dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e`;
- `original_execution_commit = 2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- `recovery_authority_commit = 233ed0be080e1d30dd47de2e66136475ec2ede76`;
- `source_run_provenance_sha256 = 4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b`;
- `source_trainer_git_commit = 2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- `standard_cm_wrapper_provenance = missing/incomplete`;
- `scientific_conclusion = NOT_ESTABLISHED`;
- packaged artifact paths, sizes, and SHA256 values exactly match the frozen table.

Historical provenance must not be reinterpreted. The helper must never emit `standard_cm_wrapper_provenance = PASS` or any equivalent value. The normal audit must preserve `standard_cm_wrapper_provenance = INCOMPLETE`.

## Destination And Materialization Contract

Required destination namespace:

`reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/`

The helper may materialize only:

- `training_report.json`;
- `clean_dev_predictions.json`;
- `training_report_predictions.jsonl`;
- `selected_checkpoint.pt`;
- `run_provenance.json`.

No-overwrite semantics:

- absent destination: may create through the transaction strategy below;
- existing byte-identical destination: reread and accept without rewrite;
- existing nonidentical destination: fail closed before any write;
- existing `A0_REFERENCE_AUDIT.json`: fail closed before any write;
- never delete, truncate, rename over, or silently replace an existing file;
- never clean user-existing files.

Transaction strategy:

1. Prevalidate the ZIP hash, ZIP entry set, entry metadata, entry content SHA256/size, duplicate-key JSON, manifest bindings, run provenance bindings, canonical dataset SHA, sidecar semantic SHA, destination directory ancestry, and all destination collisions before the first write.
2. Create a helper-owned staging directory under the destination parent, for example `.seed180_reference_recovery_staging.<pid>.<nonce>`, using exclusive creation. If the staging path already exists, fail.
3. Stream each absent artifact from the ZIP entry into a staging file opened exclusively. Hash and size-check every staged file, then reread every staged file from disk and reverify.
4. Publish each staged artifact to its final path using an atomic exclusive operation that cannot overwrite an existing path. Preferred publication is same-volume hardlink from staged file to final path, failing if the final path exists. If the host/filesystem cannot provide atomic exclusive publication, the helper must block rather than downgrade to overwrite-prone rename semantics.
5. After publication, remove only helper-created staging links/files/directories. Cleanup must never target user-existing files and must never recursively delete a computed path unless the path is verified to be the helper-owned staging directory.
6. After materialization, reread every destination artifact from disk and verify exact SHA256 and size.

This strategy is not a claim of crash-atomic all-or-nothing recovery. A crash may leave a subset of final artifacts, but each published final artifact must be complete and byte-verified. A later rerun may accept existing byte-identical artifacts without rewrite.

## Canonical Dataset, Sidecar, And Split Contract

Canonical dataset:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Dataset SHA256:

`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

Canonical sidecar:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`

Sidecar semantic SHA256:

`0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

Split seed: `174`

Split policy: `fixed_explicit_split_seed`

Dev ratio: `0.2`

The helper must implement the production split statically: load JSONL records, validate required row structure consistently with `scripts/build_controlled_v5.py`, sort unique `pair_id` values, shuffle with `random.Random(174)`, select `round(pair_count * 0.2)` dev pairs bounded to at least one and at most `pair_count - 1`, and preserve source row order for dev rows.

The helper must implement the production identity rules statically:

- row identity: first nonempty string among `stable_id`, `row_id`, `source_id`, `id`;
- pair identity: first nonempty string from `pair_id`;
- prediction gold source: `gold_label`, then `gold_final_label`, then `final_label`;
- prediction label source: `pred_label`, then `prediction`, then `pred_final_label`;
- label normalization must match the P2 external labels `REFUTE`, `NOT_ENTITLED`, `SUPPORT`.

Exact dev identity serialization:

```text
row_id<TAB>pair_id<TAB>normalized_gold_label<NEWLINE>
```

Required gates:

- authoritative dev count equals prediction count;
- no missing or duplicate source row IDs;
- no missing or duplicate prediction row IDs;
- source and prediction row-ID sets equal;
- `(row_id, pair_id)` sets equal;
- normalized gold labels equal;
- predictions limited to `REFUTE`, `NOT_ENTITLED`, `SUPPORT`;
- authoritative and joined identity hashes equal.

## A0_REFERENCE_AUDIT Generation Contract

The helper must generate exactly one seed-local audit:

`reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/A0_REFERENCE_AUDIT.json`

The audit must be generated from reread on-disk materialized artifacts, canonical dataset, and canonical sidecar. It must not be copied from ZIP metadata.

Audit creation rules:

- fail if `A0_REFERENCE_AUDIT.json` already exists;
- write atomically/exclusively using the same staging plus exclusive-publication discipline;
- after write, reread the audit JSON from disk with duplicate-key rejection;
- validate all expected fields and identities from the reread JSON;
- require `status == PASS`;
- if audit status would be non-PASS, fail closed and do not publish the final audit.

Required normal fields include at least:

- `audit_id`;
- `run_id`;
- `seed`;
- `status`;
- `errors`;
- `execution_commit`;
- `p2_implementation_tested_commit`;
- `output_dir`;
- `reference_prediction_path`;
- `prediction_sha256`;
- `selected_checkpoint_path`;
- `selected_checkpoint_sha256`;
- `report_path`;
- `report_sha256`;
- `selected_epoch`;
- `selected_epoch_source`;
- `row_count`;
- `unique_row_id_count`;
- `unique_row_pair_count`;
- `authoritative_dev_row_count`;
- `authoritative_dev_row_identity_hash`;
- `prediction_joined_dev_row_identity_hash`;
- `gold_counts`;
- `prediction_counts`;
- `a0_false_entitlement_count`;
- `a0_stable_true_support_count`;
- `data_path`;
- `dataset_sha256_expected`;
- `dataset_sha256_observed`;
- `sidecar_path`;
- `sidecar_semantic_sha256_expected`;
- `sidecar_semantic_sha256_observed`;
- `split_seed`;
- `split_policy`;
- `dev_ratio`.

Required additive recovery fields:

- `source_execution_commit = 2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- `recovery_authority_commit = 233ed0be080e1d30dd47de2e66136475ec2ede76`;
- `retained_zip_path = C:\Users\Home1\Downloads\seed180_a0_6189be22715e.zip`;
- `retained_zip_sha256 = 6bbd0e89a5858d7c68b1eecc1cf44911cc415c7411670a94605c56c08e955966`;
- `recovery_manifest_sha256 = 69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed`;
- `standard_cm_wrapper_provenance = INCOMPLETE`;
- `provenance_disposition = RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE`;
- `recovery_reference_status = RECOVERY_REFERENCE_AUDIT_PASS`.

Normal audit values:

- `audit_id = p3_seed180_A0_REFERENCE_AUDIT`;
- `run_id = p3_seed180_A0`;
- `seed = 180`;
- `status = PASS`;
- `errors = []`;
- `execution_commit = 2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- `output_dir` is the required destination namespace;
- `reference_prediction_path` points to seed180 `training_report_predictions.jsonl`;
- `selected_checkpoint_path` points to seed180 `selected_checkpoint.pt`;
- `report_path` points to seed180 `training_report.json`;
- `selected_epoch` is read from `runs.single.best_epoch` first, then top-level `best_epoch`;
- `selected_epoch_source` records the field used.

`p2_implementation_tested_commit` must follow the P3 contract value already required by the frozen A0 reference-audit authority. If that value cannot be determined from the inspected authority/contract without changing scientific semantics, implementation must block.

No normal-reference eligibility claim may be made beyond producing a valid recovery audit. Subsequent artifact freeze/import and later factorial authority remain required.

## Required Future Test Matrix

The future implementation must add focused tests in `tests/test_reason_router_p3w7_a0_seed180_reference_recovery_helper.py` covering:

- exact happy-path retained ZIP behavior using synthetic byte fixtures and constants matching the frozen retained ZIP contract;
- ZIP hash mismatch;
- entry set mismatch;
- duplicate ZIP member;
- traversal, absolute, drive-path, dot-path, NUL, and backslash entries;
- encrypted entry rejection where testable;
- symlink and special-file entry rejection where testable;
- malformed or noncanonical entry name;
- manifest duplicate JSON key;
- provenance duplicate JSON key;
- manifest binding mismatch;
- run provenance binding mismatch;
- source artifact hash mismatch;
- destination nonidentical collision;
- existing identical artifact acceptance without rewrite;
- prevalidation-before-write behavior;
- dataset hash mismatch;
- sidecar semantic hash mismatch;
- prediction/source row-count mismatch;
- row-ID set mismatch;
- pair mismatch;
- gold mismatch;
- source duplicate row IDs;
- prediction duplicate row IDs;
- missing row IDs;
- invalid prediction class;
- checkpoint reread SHA mismatch;
- report reread SHA mismatch;
- prediction reread SHA mismatch;
- `A0_REFERENCE_AUDIT.json` preexistence rejection;
- audit status non-PASS rejection before final audit publication;
- persisted audit reread validation;
- historical `INCOMPLETE` provenance preserved;
- `standard_cm_wrapper_provenance = PASS` never emitted;
- no trainer/model/checkpoint deserialization imports;
- no source ZIP mutation;
- no deletion, truncation, overwrite, or rename-over semantics;
- cleanup limited to helper-created staging paths;
- CLI rejects wrong authority freeze commit;
- CLI rejects malformed authority freeze commit.

Allowed validation for future implementation:

- targeted pytest for the new helper tests;
- existing recovery tests;
- relevant pure/static P2 contract tests;
- `py_compile` and static syntax checks;
- direct file SHA tests;
- import-forbidden AST checks.

Forbidden validation/execution for future implementation:

- model loading;
- checkpoint deserialization;
- trainer execution;
- training;
- evaluation;
- Kaggle.

## Confirmation Of Trainer Boundary

Direct inspection found no need to modify `scripts/train_controlled_v6b_minimal.py`. The normal dependency gate can be satisfied by a static helper that reproduces the documented row identity, split, sidecar semantic hash, artifact hash, selected-epoch, and audit-field contracts from reread files.

If additive recovery fields are rejected by a future normative consumer, the path is BLOCKED pending a new schema/consumer authority. This candidate does not authorize weakening or removing those recovery fields.

## Explicit Non-Authorizations

This candidate does not authorize:

- implementation in this task;
- changing existing source files;
- changing existing tests;
- modifying the retained ZIP;
- extracting or materializing seed180 artifacts;
- creating `A0_REFERENCE_AUDIT.json`;
- changing dataset or sidecar files;
- changing existing authorities or reports;
- changing the run registry;
- changing Git history;
- training;
- evaluation;
- model load;
- checkpoint deserialization;
- Kaggle;
- commit;
- push.

## Validation Performed For This Candidate

Static inspection performed:

- `git status --short --branch`;
- `git rev-parse HEAD`;
- `AGENTS.md`;
- `reports/reason_router_p3w7_a0_seed180_retained_artifact_reference_recovery_authority_spec_candidate.md`;
- `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`;
- `tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`;
- `reports/reason_router_p2_p3_execution_spec.md`;
- `scripts/train_controlled_v6b_minimal.py`;
- `scripts/build_controlled_v5.py`;
- `tests/test_reason_router_p2_contract.py`.

Validation still required after writing this candidate:

- `git diff --check`;
- exactly one new untracked Markdown candidate;
- candidate SHA256, byte size, CR/LF counts, final LF, and trailing-whitespace check.

Not performed:

- implementation;
- source/test modification;
- retained ZIP extraction;
- seed180 artifact materialization;
- `A0_REFERENCE_AUDIT.json` creation;
- training;
- evaluation;
- model load;
- checkpoint deserialization;
- Kaggle;
- commit;
- push.
