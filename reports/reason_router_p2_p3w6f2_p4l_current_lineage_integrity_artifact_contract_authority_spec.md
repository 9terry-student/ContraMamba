# P3-W6-F2-P4-L Current-Lineage Integrity Artifact Contract Authority Specification

Authority/version:

`P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_ARTIFACT_CONTRACT_AUTHORITY_V1`

This document is a candidate frozen-authority specification only. It becomes
canonical only after independent static verification PASS and immutable freeze.
Candidate verification does not authorize materialization, implementation,
trainer modification, manifest modification, parameter adoption, A0, training,
evaluation, Kaggle, or GPU use. No future result/attestation provenance
contract is implied or required by this specification unless an existing higher
authority explicitly requires it.

## A. Candidate Creation State

Candidate creation requires:

- HEAD exactly
  `703b861ab738b1cfdf73121de23ca07b6bbb9e48`.
- Tracked worktree and index clean.
- Pre-existing untracked files, if any, must remain untouched.
- Exactly one new untracked file may be created:
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`.

Current-authority creation evidence:

- `git rev-parse HEAD` returned
  `703b861ab738b1cfdf73121de23ca07b6bbb9e48`.
- `git status --short --branch` reported branch `main...origin/main`, no
  tracked modifications, and pre-existing untracked local files.
- The candidate path was absent before creation.

Warnings from inaccessible local pytest/cache directories during `git status`
and broad repository search are environmental read warnings only; they are not
functional validation PASSes and do not authorize ignoring tracked dirt.

## B. Namespace And History Search

Repository content and Git history were independently searched for:

- `P4-L`
- `P4L`
- `CURRENT_LINEAGE_INTEGRITY_ARTIFACT`
- `current-lineage integrity`
- `lineage integrity`
- `integrity artifact contract`
- `integrity-sidecar or equivalent`

Current tracked content found no existing P4-L/P4L canonical namespace and no
existing authority that already defines this exact current-lineage integrity
artifact contract.

The material relevant hit was P4-K Section I, item 10, which requires later
verification of a "derived current-lineage integrity-sidecar or equivalent
artifact contract." That is an upstream requirement and provenance pointer, not
an already-applicable P4-L contract.

Git history search found no existing P4-L/P4L report/spec namespace. Pickaxe
search for `integrity-sidecar or equivalent artifact contract` found P4-K
freeze commit `13e7b0d7e229aa678e791e06b2e1d7de26474414`, again as an upstream
requirement rather than this contract.

If a verifier finds an existing applicable frozen authority that already
resolves this same contract, P4-L is BLOCKED. If a verifier finds namespace
collision, P4-L is BLOCKED.

## C. Authority Chain Consumed

P4-L consumes the final verified P4-H authority chain:

- P4-H authority commit:
  `368d3b6991389aa6b6fd80f421c73565b562e290`
- P4-K freeze:
  `13e7b0d7e229aa678e791e06b2e1d7de26474414`
- P4-H result freeze:
  `b3626ae80ecf0664433821a772be28a56c6409da`
- P4-H verification attestation freeze:
  `703b861ab738b1cfdf73121de23ca07b6bbb9e48`
- final verification token:
  `P3W6F2P4H_A0_CURRENT_LINEAGE_REBIND_AUDIT_PRIMARY_PASS_INDEPENDENT_VERIFICATION_PASS`

P4-H established that a new current-lineage effective integrity sidecar or a
semantically equivalent deterministic artifact is required before current-
lineage A0 can be considered. P4-L resolves that class into the concrete
canonical artifact representation below.

## D. Chosen Canonical Artifact Representation

The canonical P4-L artifact class is exactly:

`CURRENT_LINEAGE_EFFECTIVE_INTEGRITY_SIDECAR_JSONL`

It is a new current-lineage effective integrity sidecar, represented as a
deterministic JSONL file, with exactly one JSON object per regenerated dataset
row in exact regenerated dataset row order.

This representation is chosen because current frozen trainer contracts consume
an ordered Stage185-compatible integrity sidecar through:

- `P2_SIDE_CAR_REQUIRED_FIELDS`
- `_stage187_load_integrity_sidecar`
- `_p2_load_reason_integrity_sidecar`
- `_p2_prepare_reason_supervision`
- `_p2_prepare_reason_supervision_train_only`
- Stage187/191/193/195 path, SHA, split, and sidecar checks

Any alternative artifact that is not directly sidecar-compatible would require
a sidecar-compatible projection before the trainer could preserve current
fail-fast semantics. Such a representation is therefore dominated for this
contract. No multiple non-dominated artifact representations remain unresolved.

The artifact itself is not created by P4-L.

## E. Source Dataset Identity

The source dataset bound by this contract is the P4-B R1 regenerated full
dataset:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Required physical SHA256:

`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

Required semantic SHA256:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

The historical dataset:

`f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`

and historical Stage185 sidecar semantic SHA:

`5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`

remain immutable historical evidence only. They are never current-lineage
source identity.

Every artifact row and every provenance record that contains a current source
dataset field must bind:

`source_dataset_sha256 = eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

The historical `f552...` value must never appear as the current source identity
in the P4-L artifact. It may appear only in explicitly named historical input
or comparison fields.

## F. Full-Row Universe And Ordering

The artifact is defined over the complete regenerated trainer row universe:

- row count: exactly `3600`
- source: the P4-B R1 regenerated dataset above
- row order: exact JSONL source order of the regenerated full dataset
- coverage: exact one-to-one coverage of all regenerated rows
- uniqueness: unique non-empty `row_id`
- no missing rows
- no extra rows
- no duplicate rows

Frozen P4-B evidence confirms row count and row order:

- P4-B full-output isolation reports `row_count_historical = 3600`,
  `row_count_regenerated = 3600`, and `row_order_identical = true`.
- P4-B regeneration summary reports `authorized_member_count = 357` and
  `unchanged_non_f2_row_count = 3243`, totaling 3600 rows.

The regenerated dataset row schema is exactly:

- `id`
- `pair_id`
- `claim`
- `evidence`
- `final_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `polarity_label`
- `primary_failure_type`
- `intervention_type`

The sidecar must use source row `id` as `row_id`.

Defense-in-depth identity checks must include all applicable available
identities:

- zero-based source row position derived from regenerated JSONL order
- `row_id`
- source row `id`
- `pair_id`
- `canonical_row_id`
- `split`
- source dataset path
- source dataset physical SHA256
- source dataset semantic SHA256
- sidecar row order equality to regenerated source order

The controlled dataset has no explicit persisted `source_order` field. P4-L
therefore freezes source/order identity as the deterministic row position in
the regenerated JSONL file. A future builder may record an additive
`source_order_index`, but the canonical ordering authority remains physical
JSONL row order.

## G. Required Sidecar Row Schema

Every P4-L sidecar row must include at minimum all fields required by current
trainer `P2_SIDE_CAR_REQUIRED_FIELDS` exactly as named:

- `row_id`
- `split`
- `pair_id`
- `canonical_row_id`
- `canonical_status`
- `intervention_contract_status`
- `integrity_status`
- `schema_status`
- `dataset_source_status`
- `grammar_status`
- `polarity_contamination_status`
- `time_swap_status`
- `reason_codes`
- `source_dataset_path`
- `source_dataset_sha256`
- `frame_compatible_label`

Every row must also include these Stage185-compatible sidecar fields because
they are part of the existing trainer-compatible sidecar representation and
preserve deterministic bridge/provenance semantics:

- `intervention_type`
- `eligible_for_positive_margin`
- `family_contract_id`
- `rule_version`
- `generator_source_sha256`
- `integrity_builder_sha256`
- `created_at`

The following audit/provenance fields may be included when deterministically
derived, and must be included if a future builder consumes historical Stage185
observations for that row:

- `audit_changed_axes`
- `audit_expected_axes`
- `audit_pair_failure_scope`
- `audit_preserved_axes`
- `generator_source_path`
- `stage182a_report_sha256`
- `stage184a_report_sha256`

No claim or evidence text is required or authorized as a sidecar field. A
future builder may include additional deterministic provenance fields only if
they are documented, stable, non-secret, and excluded from no required loader
check.

## H. Trainer-Required Source Metadata

P4-L also freezes the source metadata fields required by current reason
supervision preparation. The sidecar builder does not add them to the dataset;
instead it must validate exactly the trainer `P2_SOURCE_REQUIRED_FIELDS` in
the regenerated source rows before emitting the sidecar:

- `id`
- `pair_id`
- `intervention_type`
- `final_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `polarity_label`
- `primary_failure_type`

Exact binary fields must be JSON integers `0` or `1`, not booleans:

- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`

Canonical final labels remain exactly:

- `REFUTE`
- `NOT_ENTITLED`
- `SUPPORT`

Canonical polarity normalization remains:

- integer `0` maps to `NONE`
- integer `1` maps to `REFUTE`
- integer `2` maps to `SUPPORT`
- string values are stripped and uppercased

Allowed primary failure values are:

- `none`
- `frame`
- `predicate`
- `sufficiency`
- `polarity`

## I. Deterministic Status And Eligibility Semantics

Criterion status fields use the existing fail-closed enum surface:

- `PASS`
- `FAIL`
- `UNRESOLVED`
- `NOT_APPLICABLE`

`integrity_status` is exactly one of:

- `ELIGIBLE`
- `INELIGIBLE`
- `UNRESOLVED`

Integrity composition is frozen:

1. Dataset/SHA/identity/join/split/topology/provenance contradictions block
   the whole build.
2. Any deterministic criterion `FAIL` makes the row `INELIGIBLE`.
3. With no failure, any required `UNRESOLVED` makes the row `UNRESOLVED`.
4. Only all required criterion statuses `PASS` makes the row `ELIGIBLE`.

`eligible_for_positive_margin` is a JSON boolean and is true exactly when all
of the following are true:

- `integrity_status == "ELIGIBLE"`
- `split == "train"`
- `frame_compatible_label == 1`
- `time_swap_status == "PASS"`
- `dataset_source_status == "PASS"`

Every other row must have `eligible_for_positive_margin = false`.

If `eligible_for_positive_margin` is true, trainer-compatible defense in depth
also requires:

- `split == "train"`
- `frame_compatible_label == 1`
- `integrity_status == "ELIGIBLE"`
- `time_swap_status == "PASS"`
- `dataset_source_status == "PASS"`

Rows that are not positive-margin eligible may still be reason-supervision
eligible if they pass the P2 reason-supervision gates below. Positive-margin
eligibility and P2 reason-supervision eligibility are distinct.

## J. Reason Supervision Semantics

Primary reason derivation is deterministic:

- if `frame_compatible_label == 0`, primary reason is `FRAME`
- else if `predicate_covered_label == 0`, primary reason is `PREDICATE`
- else if `sufficiency_label == 0`, primary reason is `SUFFICIENCY`
- else primary reason is `AUTHORIZED`

Expected primary reason from source row is deterministic:

- `primary_failure_type == "frame"` expects `FRAME`
- `primary_failure_type == "predicate"` expects `PREDICATE`
- `primary_failure_type == "sufficiency"` expects `SUFFICIENCY`
- `primary_failure_type in {"none", "polarity"}` expects `AUTHORIZED`

P4-L preserves primary reason order:

`FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`

Reason-supervision exclusion codes are accumulated deterministically. A row is
P2 reason-supervision eligible exactly when the accumulated code list is empty.
The exact code names preserved from trainer contracts are:

- `P2_NON_CANONICAL_SOURCE`
- `P2_SIDECAR_MISSING`
- `P2_SPLIT_MISMATCH`
- `P2_CANONICAL_ROW_ID_MISMATCH`
- `P2_SIDECAR_SOURCE_BINARY_MISMATCH`
- `P2_POLARITY_INTERVENTION_CONTRACT_FAIL`
- `P2_INTEGRITY_SOURCE_REQUIRED`
- `P2_GENERATOR_STATUS_DEFECT`
- `P2_PRIMARY_REASON_AXIS_CONFLICT`
- `P2_FAILURE_FINAL_LABEL_MISMATCH`
- `P2_AUTHORIZED_FINAL_LABEL_MISMATCH`
- `P2_POLARITY_TARGET_FINAL_MISMATCH`

Sidecar component status normalization is deterministic over:

- `schema_status`
- `dataset_source_status`
- `grammar_status`
- `canonical_status`
- `intervention_contract_status`
- `polarity_contamination_status`
- `time_swap_status`

Normalization:

- `CLEAN` means all component status fields are `PASS`.
- `DEFECT` means at least one component status field is `FAIL` and no field is
  missing, non-string, or unsupported.
- `UNRESOLVED` means one or more component fields are missing, non-string, or
  outside `PASS`/`FAIL`.

Reason-supervision fields attached to source records by trainer preparation
remain:

- `p2_primary_reason`
- `p2_primary_reason_target_4`
- `p2_secondary_reasons_3`
- `p2_reason_supervision_eligible`
- `p2_reason_exclusion_codes`
- `p2_frame_applicable`
- `p2_predicate_applicable`
- `p2_sufficiency_applicable`
- `p2_polarity_applicable`
- `p2_polarity_target_2`
- `intervention_contract_pass`
- `generator_integrity_status`

Tensor/input field names remain:

- `p2_primary_reason_targets_4`
- `p2_secondary_reason_targets_3`
- `p2_reason_supervision_eligible`
- `p2_frame_applicability_mask`
- `p2_predicate_applicability_mask`
- `p2_sufficiency_applicability_mask`
- `p2_polarity_applicability_mask`
- `p2_polarity_targets_2`

Applicability masks are deterministic:

- `frame`: reason-supervision eligible
- `predicate`: eligible and `frame_compatible_label == 1`
- `sufficiency`: eligible and `frame_compatible_label == 1` and
  `predicate_covered_label == 1`
- `polarity`: eligible and `frame_compatible_label == 1` and
  `predicate_covered_label == 1` and `sufficiency_label == 1` and
  `final_label in {REFUTE, SUPPORT}`

Polarity targets are:

- `REFUTE -> 0`
- `SUPPORT -> 1`
- otherwise `-100`

Secondary reasons remain diagnostic-only as `[1 - frame, 1 - predicate,
1 - sufficiency]`. They must not be duplicated into the external class target
or loss.

All unresolved or defective rows fail closed for the affected eligibility and
mask. No unresolved status may be promoted to eligibility.

## K. Split And Canonical Lineage

The split is pair-level, deterministic, and must be reproduced exactly from the
historical Stage185 split contract unless a higher later authority explicitly
supersedes it:

- sorted pair IDs
- shuffle with seed `174`
- dev ratio `0.2`
- all rows of a pair remain in the same split

P4-L does not establish A0 execution parameters. The split rule above is an
artifact-integrity compatibility rule needed to reproduce the existing sidecar
contract, not A0 parameter authority.

Canonical lineage must satisfy:

- `canonical_row_id` is a non-empty string.
- For each pair and split, all rows have exactly one canonical row ID.
- The canonical target row exists in the same source row universe.
- The canonical target has the same `pair_id`.
- The canonical target has the same `split`.
- The canonical target sidecar row has `row_id == canonical_row_id`.
- The canonical target sidecar row has `canonical_row_id == row_id`.
- For controlled-v5 rows, the canonical row is the same-pair row with
  `intervention_type == "none"`.

Any pair ID mismatch, split mismatch, missing canonical target, multiple
canonical targets, or non-self-anchored canonical target is a fail-closed
lineage error.

## L. P4-B Compatibility Evidence Consumption

P4-B artifacts 8-10 may support current-lineage predicate-realization
realization for the authorized F2 scope:

- rows artifact:
  `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl`
- rows SHA256:
  `59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f`
- summary artifact:
  `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json`
- summary SHA256:
  `ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8`
- provenance artifact:
  `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json`
- provenance SHA256:
  `09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6`

These artifacts prove only:

- 119-pair/357-member predicate-realization compatibility
- historical Stage185-v1 non-mutation
- structured semantic predicate preservation in the scoped F2 universe
- compatibility PASS in defined scope
- `training_admission_released = false`

They must not be treated as proof of:

- the full 3600-row sidecar
- exact regenerated row-order identity
- current `source_dataset_sha256` binding
- complete current split identity
- every P2 sidecar field
- A0 manifest/reference-audit data and sidecar identity bindings

The builder must consume artifacts 8-10 as scoped evidence only. It must still
independently construct and validate the full-row current-lineage sidecar.

## M. Historical Stage185 Bridge Rules

Historical Stage185-v1 remains immutable. It is historical input/comparison
evidence only. It may contribute observations where authority-valid. It is
never rewritten and never silently relabeled as current lineage.

Fields that may be reused from historical Stage185 when, and only when, the
regenerated row is byte-identical for the relevant semantics and all
current-lineage identity checks pass:

- `row_id`
- `pair_id`
- `intervention_type`
- `split`
- `canonical_row_id`
- `canonical_status`
- `intervention_contract_status`
- `schema_status`
- non-F2 `grammar_status`
- non-F2 `polarity_contamination_status`
- `time_swap_status`
- non-F2 `reason_codes`
- `family_contract_id`
- historical audit observations explicitly labeled as historical

Fields that must be recomputed or rebound from regenerated current-lineage
data:

- `source_dataset_path`
- `source_dataset_sha256`
- current source dataset semantic SHA provenance
- `dataset_source_status`
- `eligible_for_positive_margin`
- `integrity_status`
- F2 `grammar_status`
- F2 `reason_codes`
- any effective F2 predicate-realization compatibility status
- sidecar physical SHA256
- sidecar semantic SHA256
- provenance manifest current dataset identity fields

For authorized F2 rows, raw Stage185 predicate-axis observations may be
recorded as historical observations, but the effective current-lineage
compatibility status must be derived from P4-B artifacts 8-10 and regenerated
row identity. Stage185-v1 output must not be edited to make this true.

For any row whose bridge condition cannot be proven deterministically, the
builder must mark the affected status `UNRESOLVED` or block the artifact if the
unresolved condition concerns global identity/order/provenance.

## N. Namespace, Paths, And Hashes

New current-lineage namespace:

`P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR`

Canonical sidecar schema version:

`P3W6F2P4L_CURRENT_LINEAGE_EFFECTIVE_INTEGRITY_SIDECAR_V1`

Canonical artifact path pattern:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_<builder_commit>/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`

where `<builder_commit>` is the full 40-character source commit of the future
builder execution state. P4-L does not predict that commit.

Canonical provenance manifest schema version:

`P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR_PROVENANCE_V1`

Canonical provenance manifest path pattern:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_<builder_commit>/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`

Physical SHA256 computation:

- compute SHA256 over exact artifact bytes
- UTF-8
- LF line endings
- no BOM
- exactly one compact JSON object per line for JSONL
- final newline required
- no NaN, Infinity, or non-JSON values

Sidecar semantic SHA256 computation:

1. Parse the JSONL into an ordered list of JSON objects.
2. Preserve source row order.
3. For each row, remove `created_at`.
4. Remove no identity, status, eligibility, reason, or provenance hash field.
5. Serialize the ordered list with JSON `sort_keys = true`,
   `separators = (",", ":")`, and `ensure_ascii = false`.
6. Hash the UTF-8 bytes of that canonical payload with SHA256.

This matches current trainer `_stage187_semantic_sidecar_sha256` semantics and
therefore preserves existing fail-fast comparison behavior.

P4-L does not predict future sidecar or provenance SHA values.

## O. Future Builder Contract

P4-L does not implement the builder.

A future builder must have deterministic inputs sufficient for two independent
implementations to produce byte-identical or canonically identical outputs:

- P4-L authority path
- P4-L authority commit after freeze
- P4-L authority/version token
- builder source path
- builder source commit
- builder source SHA256
- P4-H authority commit
  `368d3b6991389aa6b6fd80f421c73565b562e290`
- P4-K freeze
  `13e7b0d7e229aa678e791e06b2e1d7de26474414`
- P4-H result freeze
  `b3626ae80ecf0664433821a772be28a56c6409da`
- P4-H verification attestation freeze
  `703b861ab738b1cfdf73121de23ca07b6bbb9e48`
- P4-B regenerated dataset path, physical SHA256, and semantic SHA256
- P4-B artifacts 8-10 paths and SHA256 values
- historical Stage185 sidecar path and semantic SHA256 as historical input only
- current regenerated dataset row order
- deterministic split rule
- deterministic canonical-row rule
- exact schema/status/eligibility rules in this specification

Required builder outputs:

- the sidecar JSONL at the canonical path pattern
- the provenance manifest at the canonical path pattern
- physical SHA256 of the sidecar
- semantic SHA256 of the sidecar
- physical SHA256 of the provenance manifest
- builder audit summary inside the provenance manifest
- complete blocker/failure list, empty only on successful materialization

Required provenance manifest fields:

- `schema_version`
- `authority_version`
- `p4l_authority_path`
- `p4l_authority_commit`
- `builder_source_path`
- `builder_source_commit`
- `builder_source_sha256`
- `p4h_authority_commit`
- `p4k_freeze_commit`
- `p4h_result_freeze_commit`
- `p4h_verification_attestation_freeze_commit`
- `source_dataset_path`
- `source_dataset_sha256`
- `source_dataset_semantic_sha256`
- `sidecar_path`
- `sidecar_physical_sha256`
- `sidecar_semantic_sha256`
- `sidecar_schema_version`
- `row_count`
- `row_order_rule`
- `one_to_one_row_coverage`
- `unique_row_id`
- `split_rule`
- `canonical_row_rule`
- `p4b_compatibility_rows_path`
- `p4b_compatibility_rows_sha256`
- `p4b_compatibility_summary_path`
- `p4b_compatibility_summary_sha256`
- `p4b_compatibility_provenance_path`
- `p4b_compatibility_provenance_sha256`
- `historical_stage185_sidecar_path`
- `historical_stage185_sidecar_semantic_sha256`
- `historical_stage185_used_as_current_source_identity`
- `training_admission_released`
- `implementation_authorized`
- `artifact_materialization_authorized_by_p4l`
- `a0_execution_authorized`
- `training_authorized`
- `evaluation_authorized`
- `kaggle_authorized`
- `gpu_authorized`
- `blockers`
- `failure_reasons`

The manifest field `historical_stage185_used_as_current_source_identity` must
be `false`.

## P. Future Validation Contract

P4-L defines but does not authorize future validators.

Future non-training validation must check at minimum:

- source dataset physical SHA256 equals
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- source dataset semantic SHA256 equals
  `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
- source dataset path equals the P4-B R1 regenerated dataset path
- sidecar row count exactly 3600
- source row count exactly 3600
- exact one-to-one row coverage
- unique non-empty `row_id`
- exact source order equality
- no missing/extra/duplicate rows
- per-row `source_dataset_sha256` equals `eb1e...`
- required sidecar fields are present exactly
- `reason_codes` is a sorted unique JSON array
- `frame_compatible_label` is exact integer 0/1, not bool
- split/pair/canonical consistency
- current-lineage source/order identity
- P4-B scoped compatibility consumption is limited to 119 pairs/357 members
- historical Stage185-v1 is not mutated
- historical `f552...` never appears as current source identity
- fail-closed unresolved count and unresolved reasons are reported
- deterministic semantic hash recomputation
- physical hash recomputation
- provenance manifest schema and hash recomputation
- trainer-loader compatibility for Stage187/P2 sidecar expectations
- no trainer modification, training, evaluation, Kaggle, or GPU side effects

Validation PASS may not be reported unless the future command actually ran
successfully under an authority that permits it.

## Q. Downstream Boundary

P4-L PASS/freeze authorizes only later specification or implementation work to
consume this artifact contract.

P4-L does not authorize:

- artifact materialization
- builder implementation
- trainer modification
- manifest modification
- parameter adoption
- A0
- A1/A2/A3
- training
- evaluation
- validators
- Stage185 rerun
- Kaggle
- GPU
- checkpoint creation or mutation
- promotion

Mandatory values:

`training_admission_released = false`

`implementation_authorized = false`

`artifact_materialization_authorized = false`

`a0_execution_authorized = false`

`training_authorized = false`

`evaluation_authorized = false`

`kaggle_authorized = false`

`gpu_authorized = false`

## R. Scientific Invariants

P4-L preserves without modification:

- Conditional First-Blocker Reason Router
- Reason-Specific Supervision
- Explicit Gradient Ownership
- FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED
- secondary reasons diagnostic-only
- router-only final 3-way CE
- detached F/P/S and polarity CE path
- EMA observer/baseline-only
- A0-A3
- E0

P4-L does not change dataset label semantics, split semantics, reason-router
semantics, loss semantics, gradient ownership, promotion criteria, or clean-vs-
external evaluation separation.

## S. Stop Conditions

P4-L must be BLOCKED if any of the following holds:

- namespace collision
- existing applicable authority already resolves this exact contract
- HEAD mismatch at candidate creation
- tracked worktree or index dirt at candidate creation
- more than one non-dominated artifact representation remains
- exact trainer-required schema cannot be resolved statically
- regenerated source identity is ambiguous
- deterministic historical/current bridge rules cannot be specified
- scientific semantics would need modification
- historical `f552...` would need to be treated as current source identity
- Stage185-v1 would need mutation or relabeling
- P4-B artifacts 8-10 would need to be treated as proof of full 3600-row
  current-lineage sidecar completeness

Missing authority is BLOCKED, not scientific FAIL.

## T. Candidate Readiness

Candidate path:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`

Candidate SHA256 is computed after file creation and is not predicted inside
this specification body.

Final candidate readiness token:

`P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_ARTIFACT_CONTRACT_AUTHORITY_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION`
