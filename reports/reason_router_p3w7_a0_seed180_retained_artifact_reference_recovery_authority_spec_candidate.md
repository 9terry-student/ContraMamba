# P3-W7 A0 Seed180 Retained-Artifact Reference Recovery Authority Specification Candidate

Authority/version:

`P3W7_A0_SEED180_RETAINED_ARTIFACT_REFERENCE_RECOVERY_AUTHORITY_V1_CANDIDATE`

## Status

READY.

This is a static retained-artifact recovery-authority materialization candidate only. It does not authorize extraction, repository import, audit generation, training, evaluation, model load, checkpoint load, Kaggle execution, A1, A2, A3, commit, push, or mutation of existing seed180 bytes.

If independently verified and frozen, this candidate may authorize a later bounded implementation/execution authority to materialize a same-seed seed180 A0 reference package from the exact retained ZIP identified below. The future recovery must preserve the historical statement that standard `cm` wrapper provenance for the original seed180 run is `missing/incomplete`.

## Authority Basis

Current materialization basis:

`1221588b78d02900ee93cff36cf37b2202e04aea`

Formal A0 execution authority:

`reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`

Formal original seed180 execution commit:

`2737c3c6116ae3766b469801f990e2c45ba9a55e`

Original seed180 authorized wrapper SHA256:

`dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e`

Existing seed180 recovery-authority chain inspected by Git object/history:

- `233ed0be080e1d30dd47de2e66136475ec2ede76` - seed180 A0 provenance recovery authority.
- `cdd71ea4f556392eab594ebb5df8258355610e01` - seed180 recovery tooling implementation authority.
- `1b6516d16596d1169ff2fa4fd8d8c8f8adb80450` - external `cm` drift reconciliation authority.
- `15387eb6fd2af9b1171b8b988a64cfcf4417c1cd` - seed180 provenance recovery tooling implementation.
- `8752646b106eb5b11d2de5241fce874edae75087` - prediction row-count reconciliation authority.
- `9a9b11a3212fb0073d3f3678875bc2a3ae003501` - row-count remediation implementation.
- `3de16c2215fe50e6f17aabe5ae33da3eab3f8540` - dataset semantic SHA forensic attestation.
- `a5ffa107882947842be3d04993d3a534c6909490` - dataset semantic SHA reconciliation authority.
- `6189be22715e435ddc3247271e4966bb3d3b526d` - semantic SHA recovery remediation implementation.

Validated-evidence analysis authority/report:

- `reports/reason_router_p3w7_a0_validated_evidence_analysis_authority_spec_candidate.md`
- `reports/reason_router_p3w7_a0_validated_evidence_analysis_report.md`

Normative A0 reference-audit authority inspected:

- `reports/reason_router_p2_p3_execution_spec.md`
- `scripts/train_controlled_v6b_minimal.py`
- `tests/test_reason_router_p2_contract.py`
- `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`
- `tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`

Repository `AGENTS.md` applies.

## Retained ZIP Identity

Source ZIP exact path:

`C:\Users\Home1\Downloads\seed180_a0_6189be22715e.zip`

ZIP SHA256:

`6bbd0e89a5858d7c68b1eecc1cf44911cc415c7411670a94605c56c08e955966`

ZIP entries were independently stream-hashed without extraction into the repository.

| ZIP entry | Size | SHA256 |
| --- | ---: | --- |
| `recovery_manifest.json` | 2144 | `69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json` | 306114 | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json` | 4838225 | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl` | 3934123 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/selected_checkpoint.pt` | 518269815 | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json` | 68429 | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` |

The manifest binds:

- `schema = contramamba-seed180-a0-provenance-recovery-v1`
- `seed = 180`
- `attempt_disposition = CONSUMED`
- `execution_status = completed`
- `implementation_commit = 6189be22715e435ddc3247271e4966bb3d3b526d`
- `original_authorized_wrapper_sha256 = dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e`
- `original_execution_commit = 2737c3c6116ae3766b469801f990e2c45ba9a55e`
- `recovery_authority_commit = 233ed0be080e1d30dd47de2e66136475ec2ede76`
- `source_run_provenance_sha256 = 4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b`
- `source_trainer_git_commit = 2737c3c6116ae3766b469801f990e2c45ba9a55e`
- `standard_cm_wrapper_provenance = missing/incomplete`
- `scientific_conclusion = NOT_ESTABLISHED`

## Authority Interpretation

The existing seed180 recovery authority chain authorized a truthful recovery-only provenance package for already-existing exact seed180 artifacts. It explicitly withheld training rerun, resume, retry, checkpoint regeneration, artifact mutation, scientific interpretation, result promotion, and fabrication of historical standard `cm` wrapper records.

The implemented recovery tooling can validate a retained ZIP and produce a recovery audit with `standard_cm_wrapper_provenance = INCOMPLETE`, `recovered_artifact_identity = VALIDATED`, and `scientific_conclusion = NOT_ESTABLISHED`. That is sufficient to prove retained recovery-package identity, but it is not itself the normal seed-local P3 `A0_REFERENCE_AUDIT.json` required for future A1/A2/A3 dependency gates.

The validated-evidence analysis authority did not permanently disqualify seed180. It held seed180 as `CAVEATED_ADMISSIBLE_ONLY_IF_EXPLICITLY_LABELED` because no repository result-import commit or recovery audit-output artifact analogous to seed181 R1/seed182 had been found. This candidate resolves only the authority question for a later explicit recovery bridge; it does not itself admit seed180 into any aggregate or release A1/A2/A3.

Static retained-artifact recovery is legal if and only if the future operation:

- extracts only the exact retained ZIP entries after all hashes and manifest bindings pass;
- preserves `standard_cm_wrapper_provenance = missing/incomplete`;
- creates a distinct normal seed-local A0 reference audit from reread extracted artifacts;
- makes no claim that standard historical `cm` wrapper provenance existed;
- performs no trainer execution, model load, checkpoint load, evaluation, data regeneration, or byte mutation.

## Seed181 And Seed182 Precedent

Seed181 `REPLACEMENT_R1` result commit `fb4f0e2c2a8382a642f1272b66f29552adaecb0e` persists these files in Git under `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/`:

- `training_report.json`
- `clean_dev_predictions.json`
- `training_report_predictions.jsonl`
- `run_provenance.json`

Seed182 result commit `82739bdfc8eee184de10ed8f55434f203a6d59a5` persists these files in Git under `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/`:

- `training_report.json`
- `clean_dev_predictions.json`
- `training_report_predictions.jsonl`
- `run_provenance.json`

Neither precedent commits `selected_checkpoint.pt`. Both bind selected checkpoint identity by `run_provenance.json:finalization.selected_checkpoint.sha256` and `size_bytes`. Therefore the retained seed180 checkpoint does not need to be committed as a Git blob merely to match result-import precedent. The checkpoint must remain cryptographically bound by SHA256/size and must be available from the retained ZIP, or materialized as an untracked/external file when a future dependency gate requires rereading it from disk.

No reachable Git history contains a committed `A0_REFERENCE_AUDIT.json`. Therefore the minimum Git precedent for existing validated A0 evidence is the four small files above plus checkpoint SHA/size in `run_provenance.json`. For normal future A1/A2/A3 reference consumption, the stricter P3 reference-audit contract additionally requires a persisted seed-local `A0_REFERENCE_AUDIT.json`.

## Normative A0 Reference Audit Requirements

The P3 execution spec requires a seed-local:

`A0_REFERENCE_AUDIT.json`

The normal audit status must be:

`PASS`

Required standardized fields:

- `audit_id`
- `run_id`
- `seed`
- `status`
- `errors`
- `execution_commit`
- `p2_implementation_tested_commit`
- `output_dir`
- `reference_prediction_path`
- `prediction_sha256`
- `selected_checkpoint_path`
- `selected_checkpoint_sha256`
- `report_path`
- `report_sha256`
- `selected_epoch`
- `selected_epoch_source`
- `row_count`
- `unique_row_id_count`
- `unique_row_pair_count`
- `authoritative_dev_row_count`
- `authoritative_dev_row_identity_hash`
- `prediction_joined_dev_row_identity_hash`
- `gold_counts`
- `prediction_counts`
- `a0_false_entitlement_count`
- `a0_stable_true_support_count`
- `data_path`
- `dataset_sha256_expected`
- `dataset_sha256_observed`
- `sidecar_path`
- `sidecar_semantic_sha256_expected`
- `sidecar_semantic_sha256_observed`
- `split_seed`
- `split_policy`
- `dev_ratio`

Fields that must not be null:

- `selected_epoch`
- `selected_epoch_source`
- `prediction_sha256`
- `selected_checkpoint_sha256`
- `report_sha256`
- `authoritative_dev_row_identity_hash`
- `prediction_joined_dev_row_identity_hash`

Required identity gates:

- same seed `180`;
- `execution_commit = 2737c3c6116ae3766b469801f990e2c45ba9a55e`, unless a later frozen reference authority explicitly defines a recovery execution commit field separately from the source execution commit;
- dataset path `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`;
- dataset physical SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`;
- sidecar path `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`;
- sidecar semantic SHA256 `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`;
- split seed `174`;
- split policy `fixed_explicit_split_seed`;
- dev ratio `0.2`;
- production dev split row count equals prediction row count;
- no missing source or prediction row ID;
- no duplicate source or prediction row ID;
- source and prediction row-ID sets equal;
- `(row_id, pair_id)` sets equal;
- normalized gold labels equal;
- predictions in `REFUTE`, `NOT_ENTITLED`, `SUPPORT`;
- authoritative and prediction-joined row identity hashes equal.

Dev identity hash serialization:

```text
row_id<TAB>pair_id<TAB>normalized_gold_label<NEWLINE>
```

The normal audit should include additive recovery provenance fields, because the standard schema does not otherwise encode the seed180 caveat:

- `source_execution_commit = 2737c3c6116ae3766b469801f990e2c45ba9a55e`
- `recovery_authority_commit = 233ed0be080e1d30dd47de2e66136475ec2ede76`
- `retained_zip_path = C:\Users\Home1\Downloads\seed180_a0_6189be22715e.zip`
- `retained_zip_sha256 = 6bbd0e89a5858d7c68b1eecc1cf44911cc415c7411670a94605c56c08e955966`
- `recovery_manifest_sha256 = 69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed`
- `standard_cm_wrapper_provenance = INCOMPLETE`
- `provenance_disposition = RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE`
- `recovery_reference_status = RECOVERY_REFERENCE_AUDIT_PASS`

The additive fields must not replace `status = PASS`; they must preserve the normal P3 dependency gate while honestly recording the caveat. If a future verifier determines that the consumer rejects extra fields or requires standard wrapper provenance PASS, this recovery path is blocked until a new schema/consumer authority is created.

## Required Future Recovery Procedure

Destination namespace:

`reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/`

This namespace is required because the P3 same-seed A1/A2/A3 dependency gate expects `RUN_ROOT / seed180 / A0 / A0_REFERENCE_AUDIT.json` and `training_report_predictions.jsonl` in the normal seed-local A0 run directory. A separate recovery namespace may be used only for auxiliary recovery reports/audits, not as the normal reference path consumed by A1/A2/A3.

Collision guards and no-overwrite semantics:

- repository tracked and index state must be clean before recovery;
- target namespace must not contain any nonidentical file at any destination path;
- any existing identical file must be reported and reread, not overwritten;
- `A0_REFERENCE_AUDIT.json` must be created exclusively and must fail if it exists;
- no retained ZIP entry may be extracted outside the destination namespace;
- symlinks, absolute paths, backslashes, dot paths, duplicate entries, encrypted entries, directory entries, and unexpected ZIP entries must fail closed;
- retained ZIP and existing seed180 bytes must not be modified.

Required source entries:

- `training_report.json`
- `clean_dev_predictions.json`
- `training_report_predictions.jsonl`
- `selected_checkpoint.pt`
- `run_provenance.json`

Required post-recovery artifact hashes:

| Destination file | Required SHA256 | Required size |
| --- | --- | ---: |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json` | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` | 306114 |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json` | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` | 4838225 |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl` | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` | 3934123 |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/selected_checkpoint.pt` | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` | 518269815 |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json` | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` | 68429 |

Eligible for Git commit after future recovery/audit PASS:

- `training_report.json`
- `clean_dev_predictions.json`
- `training_report_predictions.jsonl`
- `run_provenance.json`
- `A0_REFERENCE_AUDIT.json`
- optional small recovery-import audit/report explicitly authorized by the future recovery execution authority

Not eligible for normal Git commit under observed precedent:

- `selected_checkpoint.pt`

The checkpoint remains external/retained unless a later storage authority explicitly authorizes large-binary persistence. Its identity remains cryptographically bound by the retained ZIP SHA256, ZIP entry SHA256/size, `run_provenance.json:finalization.selected_checkpoint.sha256`, `run_provenance.json:finalization.selected_checkpoint.size_bytes`, and `A0_REFERENCE_AUDIT.json:selected_checkpoint_sha256`.

## Required Helper Implementation

Existing commands/scripts are not sufficient for the full normal-reference recovery path.

The existing `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py audit-import` validates the retained recovery ZIP and can write a recovery audit outside the repository, but it intentionally does not extract artifacts into the repository and does not generate the seed-local normal `A0_REFERENCE_AUDIT.json`. The P3 normal dependency gate requires that audit to be reread from disk and requires artifact hashes, row identity hashes, and checkpoint/report/prediction paths.

A separate implementation authority is required before recovery execution. The bounded helper may be either a new script or a new subcommand, but it must be static-only and must:

- validate the retained ZIP by stream hashing before extraction;
- validate `recovery_manifest.json` and packaged `run_provenance.json` with duplicate-key rejection;
- extract only exact allowlisted files into the normal seed180 namespace under exclusive/no-overwrite semantics;
- generate `A0_REFERENCE_AUDIT.json` by rereading the extracted prediction JSONL, report, dataset, sidecar, and checkpoint bytes from disk;
- compute the P3 dev identity hashes using the production split contract;
- record the additive recovery provenance fields above;
- fail if the resulting normal audit is not `status = PASS`;
- avoid importing trainer/model/GPU libraries unless a later authority explicitly proves a safe static-only import is necessary.

No helper implementation is authorized by this candidate itself.

## Eligibility Boundary

Seed180 becomes normal same-seed A0-reference eligible only after all of the following occur:

1. this candidate is independently verified and frozen;
2. a separate bounded helper implementation authority is created, verified, frozen, implemented, and validated;
3. a separate recovery execution/import authority authorizes use of the exact retained ZIP `6bbd0e89a5858d7c68b1eecc1cf44911cc415c7411670a94605c56c08e955966`;
4. the helper extracts/materializes the exact retained artifacts with no overwrite or byte mutation;
5. a seed-local `A0_REFERENCE_AUDIT.json` is generated by rereading the materialized files and returns `status = PASS`;
6. the resulting small-file package and audit are imported/frozen as required;
7. a subsequent factorial A1/A2/A3 authority explicitly consumes seed180 as a same-seed recovered A0 reference and preserves the historical provenance caveat.

This candidate does not release A1/A2/A3.

## Explicit Non-Authorizations

This candidate does not authorize:

- training;
- evaluation;
- model load;
- checkpoint load;
- Kaggle execution;
- retained ZIP modification;
- seed180 artifact modification;
- extraction into the repository;
- normal `A0_REFERENCE_AUDIT.json` creation;
- helper implementation;
- standard `cm` wrapper-provenance rewrite;
- run-registry mutation;
- history rewrite;
- commit;
- push.

## Validation Performed For This Candidate

Performed static validation:

- `git status --short --branch`
- `git rev-parse HEAD`
- direct ZIP SHA256 hash
- direct ZIP entry listing
- direct ZIP-stream SHA256/size verification for all required entries
- direct `recovery_manifest.json` inspection from ZIP stream
- Git object/history inspection of the seed180 recovery chain
- seed181 `REPLACEMENT_R1` and seed182 Git tree inspection
- P3 A0 reference-audit schema/gate inspection
- trainer/source/tests inspection for `--reason-router-a0-reference-predictions` and A0 reference join requirements

Not performed:

- training;
- evaluation;
- model load;
- checkpoint load;
- Kaggle;
- extraction into repository;
- retained ZIP mutation;
- commit;
- push.
