# ContraMamba P3-W6-F1 Stage185 Predicate-Realization Compatibility Specification

## Executive Decision

This specification defines the new versioned authority:

```text
P3W6F1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY_V1
```

The authority is approved for static specification review only. It does not implement analyzer behavior, does not regenerate artifacts, does not accept candidates, and does not release polarity supervision.

The compatibility mechanism replaces the unreachable P3-W6 requirement that repaired F1 rows must obtain raw Stage185-v1 `integrity_status = ELIGIBLE`. Historical Stage185-v1 remains immutable evidence and may continue to report `intervention_contract_status = FAIL`, `integrity_status = INELIGIBLE`, and `audit_changed_axes = ["polarity", "predicate"]` for successfully repaired authorized F1 rows.

The new acceptance model is:

```text
raw_stage185_v1_integrity_status = INELIGIBLE
predicate_realization_compatibility_status = PASS
effective_F1_repair_integrity_status = COMPATIBILITY_ELIGIBLE
```

This effective state is valid only when the apparent raw Stage185-v1 predicate-axis change is fully explained by the exact authorized generator realization:

```text
did not <inflected_predicate_surface>
->
did not <expected_base_predicate>
```

No other predicate equivalence is authorized.

## Authority Hierarchy

The authority hierarchy is frozen as follows:

```text
1. immediate compatibility-audit authority
2. prior P3-W6-F1 specification/target-scope authority
3. P3-W6-F1 implementation authority
4. P3-W5 remediation authority
5. P3-W4 grammar authority
6. immutable historical Stage185 evidence/dependencies
7. repository AGENTS.md
```

Frozen authority identities:

```text
immediate compatibility-audit authority:
d15fc541dc4d0e54296a5c0fdc6d3a34ef2551d8

P3-W6-F1 production implementation authority:
11102ea05b28f6638fdead205b4a9ee0f35ca0de

P3-W6-F1 target-scope/specification authority:
ff6929bf33693fb4e70bd9528551053f4402fe1c

P3-W5 separate-remediation authority:
01d983f8d09cacf0eddefd2014fc81a28771cf5e

P3-W4 implementation authority:
ca99038d812696467a4330cffc1c4c5b5f72cfe2

P3-W4 preservation/authority commit:
f0a9afddc5b93c54aa72b0335c5a1a2f517cf934
```

Authoritative findings to preserve:

```text
reports/reason_router_p2_p3w6f1_stage185_predicate_realization_compatibility_audit.md
reports/reason_router_p2_p3w6f1_stage185_predicate_realization_compatibility_manifest.json
```

Preserved authority families:

```text
P3-W6-F1
P3-W5
P3-W4
Stage184
historical Stage185
```

Frozen immutable historical artifact and dependency identities:

```text
Historical Stage185 baseline semantic SHA256:
5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc

Historical Stage185 generator source SHA256:
c41e6a52401bd8c83970286b176950fc751509bee6d797d5da9aea4262c72802

Historical Stage185 integrity-builder SHA256:
11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc

Authoritative clean dataset SHA256:
f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640
```

Historical Stage185 generator provenance is separate from compatibility base-form authority. The historical SHA256 `c41e6a52401bd8c83970286b176950fc751509bee6d797d5da9aea4262c72802` belongs only to historical Stage185 and baseline provenance.

Compatibility base-form authority is:

```text
commit:
11102ea05b28f6638fdead205b4a9ee0f35ca0de

path:
scripts/build_controlled_v5.py

symbol:
_BASE_PREDICATE_BY_INFLECTED
```

The exact source/blob identity for this compatibility base-form mapping must be resolved from that commit and bound in future execution provenance. No SHA256 is frozen here for `scripts/build_controlled_v5.py` at `11102ea05b28f6638fdead205b4a9ee0f35ca0de` because that file blob identity has not been authority-verified by this V2 correction.

The compatibility implementation must not silently change the generator or derive the mapping from the historical `c41e6a52401bd8c83970286b176950fc751509bee6d797d5da9aea4262c72802` source identity. If the generator path, symbol, or blob at the preserved P3-W6-F1 implementation authority does not match the expected mapping authority, compatibility is `MANUAL_REVIEW_REQUIRED` and cannot PASS.

The P3-W6-F1 production implementation authority commit `11102ea05b28f6638fdead205b4a9ee0f35ca0de` remains the authority for the currently blocked deterministic polarity-regeneration implementation only. It is not a compatibility implementation authority. This compatibility mechanism remains specification-only and has no implementation authority yet.

This specification supersedes only the explicitly listed unreachable P3-W6 integration requirements in the "Superseded P3-W6 Clauses" section. Everything else remains authoritative.

## Established 121-Row Evidence

The compatibility audit and focused runtime diagnostic established the exact authorized F1 aggregate:

```text
authorized F1 rows = 121

grammar_status:
PASS = 121

intervention_contract_status:
FAIL = 121

integrity_status:
INELIGIBLE = 121

audit_changed_axes:
["polarity", "predicate"] = 121

reason_codes:
INTERVENTION_CONTRACT_FAIL = 121
```

This is a deterministic incompatibility affecting all 121 authorized F1 repairs. It is not a row-specific exception and must not be treated as a manually tolerated anomaly.

The existing focused pytest truth remains:

```text
115 passed
2 failed
```

Those failures exposed the Stage185-v1 compatibility conflict. The implementation remains blocked for execution acceptance until this specification is implemented and validated.

## Problem Statement

The approved deterministic generator repair performs:

```text
did not <inflected predicate>
->
did not <generator-authorized base predicate>
```

Example:

```text
did not released
->
did not release
```

Historical Stage185-v1 derives predicate semantic state through literal presence of:

```text
fact["predicate"]
fact["alternate_predicate"]
```

in evidence. Therefore Stage185-v1 conflates semantic predicate identity with surface grammatical realization and observes the legitimate inflected-to-base grammatical repair as an additional predicate-axis change.

The root cause is:

```text
B. Stage185-v1 predicate-realization representation limitation
C. P3-W6 integration/specification defect
```

The generator repair itself is not the root defect.

## Raw-Vs-Effective Semantic Model

Two evidence layers are defined.

Historical Stage185-v1 Authority:

```text
immutable
retains original semantic SHA
retains original builder blob
retains baseline sidecar bytes
continues to report raw FAIL/INELIGIBLE where applicable
```

P3-W6-F1 Predicate-Realization Compatibility Authority:

```text
consumes historical Stage185-v1 results
adds a new narrowly scoped interpretation proof
does not mutate historical evidence
does not supersede historical Stage185 globally
```

Required terminology:

```text
semantic predicate identity
```

means the predicate fact identity that Stage184/Stage185 predicate-swap semantics protect.

```text
surface grammatical realization
```

means the grammatical surface form of the same generator-owned predicate within the exact authorized negative repair span.

The only authorized realization equivalence is:

```text
did not <inflected_predicate_surface>
->
did not <expected_base_predicate>
```

where all compatibility authority conditions pass. Generic lemmatization, stemming, morphology heuristics, external NLP morphology packages, LLM semantic judgment, learned semantic equivalence, fuzzy predicate matching, and arbitrary predicate matching are prohibited.

## Exact Compatibility Mechanism

Compatibility may be derived only for a row-level candidate satisfying all of the following:

```text
row_id is in exact authorized_F1_row_ids
intervention_type == polarity_flip
row is structurally a negative polarity-flip target
inflected_predicate_surface is authority-proven for that row
expected_base_predicate comes exactly from scripts/build_controlled_v5.py _BASE_PREDICATE_BY_INFLECTED
the replacement span is unique
original evidence contains exactly one authorized did not <inflected_predicate_surface> span
repaired evidence contains exactly one authorized did not <expected_base_predicate> span
outside-span evidence is byte-identical
claim is byte-identical
all labels are identical
all non-evidence fields are identical
generator replay identity passes
repair consumption passes
full-output isolation passes
```

The compatibility mechanism must record the source identity of `scripts/build_controlled_v5.py` and the symbol `_BASE_PREDICATE_BY_INFLECTED` by SHA-256. A mismatch is fail-closed.

The compatibility audit manifest is not the source authority for the exact `authorized_F1_row_ids` array. Exact target authority is the existing P3-W5/P3-W4 derivation:

```text
P3-W4 pair artifact
+
P3-W5 decision-supporting F1 pair authority
+
existing extraction contract:
extract_decision_supporting_pair_ids(...)
extract_authorized_f1_targets(...)
->
exact authorized_F1_row_ids
```

`authorized_F1_row_ids` must equal the exact 121 polarity member `source_row.id` values established by prior authority.

Compatibility row ordering is:

```text
authoritative clean baseline ID sequence
filtered by row_id in authorized_F1_row_ids
```

Lexical sorting, set iteration order, and compatibility-audit manifest order are not row-order authorities.

The compatibility mechanism may require generator and full-output validation statuses as prerequisites, but it must not duplicate, reinterpret, or weaken those validations.

## Phase Order

The execution order is frozen as:

```text
1. input authority validation
2. frozen artifact / dependency validation
3. authorized F1 target extraction
4. PRE target-scope membership validation
5. deterministic generator repair replay
6. repair-consumption audit
7. repaired-output replay identity validation
8. full-output isolation
9. historical Stage185-v1 runtime authority validation
10. baseline Stage185-v1 provenance reconstruction
11. repaired Stage185-v1 provenance reconstruction
12. P3W6F1 predicate-realization compatibility derivation
13. compatibility-aware effective F1 transition derivation
14. row-local semantic audit
15. execution provenance validation
16. global candidate finalization
17. accounting
18. execution decision
```

Phases 1 through 4 are `PRE / no repaired output required`.

Phase 5 is `GENERATION_REPLAY_PRODUCTION`. It deterministically produces the replayed repaired output.

Phases 6 through 18 are `POST_GENERATION_ONLY` and require the phase-5 repaired replay/output.

PRE target-scope membership validation and POST repair-consumption validation are separate phases and must not be merged. PRE membership answers whether a row is authorized to be considered. POST repair-consumption answers whether the generator consumed exactly the authorized repair set.

The compatibility mechanism must be established before the row-local semantic decision. It must not be appended as a late gate after a row has already been rejected or downgraded to `MANUAL` by the impossible raw Stage185 transition. A row rejected or downgraded to `MANUAL` before compatibility proof must not later be silently upgraded.

## Positive Compatibility Contract

A compatibility PASS requires all required facts to be simultaneously established:

```text
authorized_F1_membership = PASS
intervention_type = polarity_flip
generator_replay_identity_status = PASS
repair_consumption_status = PASS
full_output_isolation_status = PASS
baseline_stage185_v1_provenance_status = PASS
repaired_stage185_v1_provenance_status = PASS
stage185_v1_runtime_authority_status = PASS
base_form_source_identity_status = PASS
span_uniqueness_status = PASS
outside_span_byte_identity = PASS
claim_identity_status = PASS
label_identity_status = PASS
non_evidence_field_identity_status = PASS
```

Baseline raw Stage185-v1 state must exactly satisfy:

```text
grammar_status = FAIL
intervention_contract_status = PASS
integrity_status = INELIGIBLE
canonical_status = PASS
polarity_contamination_status = PASS
audit_expected_axes = ["polarity"]
audit_changed_axes = ["polarity"]
reason_codes contains approved grammar-defect evidence
```

Repaired raw Stage185-v1 state must exactly satisfy:

```text
grammar_status = PASS
intervention_contract_status = FAIL
integrity_status = INELIGIBLE
canonical_status = PASS
polarity_contamination_status = PASS
dataset_source_status = PASS
schema_status = PASS
time_swap_status = PASS
audit_expected_axes = ["polarity"]
audit_changed_axes = ["polarity", "predicate"]
reason_codes contains INTERVENTION_CONTRACT_FAIL
```

The only unexpected Stage185-v1 axis may be:

```text
predicate
```

The derived fields must be:

```text
predicate_semantic_identity_preserved = TRUE
surface_realization_changed = TRUE
compatibility_explained_stage185_axes = ["predicate"]
unexplained_stage185_axes = []
effective_semantic_changed_axes = ["polarity"]
effective_intervention_contract_status = COMPATIBILITY_PASS
effective_F1_repair_integrity_status = COMPATIBILITY_ELIGIBLE
ordered_compatibility_blockers = []
```

## Negative/Fail-Closed Contracts

Compatibility decisions are enums:

```text
PASS
MANUAL_REVIEW_REQUIRED
REJECTED
```

`MANUAL_REVIEW_REQUIRED` is used when authority or provenance is unresolved and no semantic contradiction has been proven. `REJECTED` is used when the row contradicts the compatibility contract or attempts to use the mechanism outside its authorized scope.

The cross-field status mapping is total:

| compatibility_status | compatibility_pass | effective_intervention_contract_status | effective_F1_repair_integrity_status |
| --- | --- | --- | --- |
| PASS | true | COMPATIBILITY_PASS | COMPATIBILITY_ELIGIBLE |
| MANUAL_REVIEW_REQUIRED | false | COMPATIBILITY_BLOCKED | COMPATIBILITY_BLOCKED |
| REJECTED | false | COMPATIBILITY_FAIL | COMPATIBILITY_INELIGIBLE |

Required negative cases:

| Case | Result | Reason |
| --- | --- | --- |
| wrong base predicate | REJECTED | The realized base form does not match `_BASE_PREDICATE_BY_INFLECTED`. |
| correct base predicate on non-authorized structural polarity row | REJECTED | Scope is limited to exact authorized F1 row IDs. |
| alternate-predicate substitution | REJECTED | This is a semantic predicate change, not surface realization. |
| unrelated third-predicate substitution | REJECTED | No arbitrary predicate equivalence is authorized. |
| multiple candidate replacement spans | REJECTED | The exact span is not unique. |
| missing original authorized span | REJECTED | The authorized source realization cannot be proven. |
| missing regenerated authorized span | REJECTED | The authorized repaired realization cannot be proven. |
| outside-span evidence mutation | REJECTED | Compatibility cannot explain non-span evidence mutation. |
| claim mutation | REJECTED | Claim identity is a hard invariant. |
| any label mutation | REJECTED | Label identity is a hard invariant. |
| any non-evidence field mutation | REJECTED | Non-evidence identity is a hard invariant. |
| generator replay failure | MANUAL_REVIEW_REQUIRED | Generator identity is unresolved, not semantically adjudicated by compatibility. |
| repair-consumption mismatch | MANUAL_REVIEW_REQUIRED | Repair set authority is unresolved. |
| full-output isolation failure | MANUAL_REVIEW_REQUIRED | Output-level isolation is unresolved. |
| baseline Stage185 provenance failure | MANUAL_REVIEW_REQUIRED | Historical baseline authority is unresolved. |
| repaired Stage185 provenance failure | MANUAL_REVIEW_REQUIRED | Repaired historical authority is unresolved. |
| Stage185 runtime authority mismatch | MANUAL_REVIEW_REQUIRED | Runtime dependency identity is unresolved. |
| repaired grammar_status still FAIL | REJECTED | The F1 grammar repair did not succeed. |
| repaired Stage185 changed axes missing polarity | REJECTED | The expected semantic polarity change was not observed. |
| repaired Stage185 changed axes containing anything beyond polarity+predicate | REJECTED | Additional axes are not compatibility-explainable. |
| raw Stage185 failure reason beyond exact compatibility-explainable predicate artifact | REJECTED | Compatibility explains only predicate realization. |
| generator base-form source identity mismatch | MANUAL_REVIEW_REQUIRED | Base-form authority is unresolved. |
| F2 row passed to compatibility mechanism | REJECTED | F2 remains manual-only. |
| unauthorized F1-like row | REJECTED | Scope is the exact authorized F1 target set. |
| predicate_swap or other semantic predicate change | REJECTED | Predicate-swap semantics are preserved. |

## Compatibility-Aware Transition

The old impossible transition:

```text
raw Stage185-v1:
INELIGIBLE -> ELIGIBLE
```

is replaced by:

```text
baseline raw Stage185-v1:
grammar_status = FAIL
intervention_contract_status = PASS
integrity_status = INELIGIBLE
audit_changed_axes = ["polarity"]

repaired raw Stage185-v1:
grammar_status = PASS
intervention_contract_status = FAIL
integrity_status = INELIGIBLE
audit_changed_axes = ["polarity", "predicate"]

P3W6F1 predicate-realization compatibility:
compatibility_status = PASS

effective F1 repair state:
effective_F1_repair_integrity_status = COMPATIBILITY_ELIGIBLE
effective_semantic_changed_axes = ["polarity"]
```

Downstream acceptance condition:

```text
effective_F1_repair_integrity_status == COMPATIBILITY_ELIGIBLE
effective_intervention_contract_status == COMPATIBILITY_PASS
effective_semantic_changed_axes == ["polarity"]
ordered_compatibility_blockers == []
```

Raw Stage185-v1 fields must remain visible and unchanged.

## Semantic-Audit Ownership

`DETERMINISTIC_POLARITY_REPAIR_PASS` is revised to require:

```text
exact local repair semantics = PASS
P3W6F1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY_V1 = PASS
```

It no longer requires repaired raw Stage185-v1 `integrity_status = ELIGIBLE`, because the compatibility audit proved that state is unreachable under historical Stage185-v1 for the approved repair.

A compatibility PASS must never override any independent hard failure, including claim mutation, label mutation, non-evidence mutation, span ambiguity, unauthorized row membership, predicate-swap evidence, generator replay failure, repair-consumption mismatch, full-output isolation failure, or provenance failure.

## Predicate-Swap Preservation

This mechanism cannot make a true predicate substitution appear preserved because it recognizes only:

```text
fact predicate -> expected authorized base realization
```

inside the exact approved F1 span, for an exact authorized row ID, using the generator-owned base-form mapping.

These remain semantic predicate changes:

```text
fact predicate -> alternate predicate
fact predicate -> unrelated predicate
fact predicate -> arbitrary lemmatized-looking surface
fact predicate -> expected base predicate outside the authorized span
fact predicate -> expected base predicate on an unauthorized row
```

Stage184 and Stage185 predicate-swap semantics are unaffected outside the exact authorized F1 repair span. Canonical rows, paraphrase rows, non-authorized structural polarity rows, F2 rows, and all predicate-swap cases remain protected by historical predicate-swap detection.

## Historical Authority Separation

The following must be preserved byte-for-byte or by existing recorded digest:

```text
historical Stage185-v1 builder blob
historical Stage185 baseline sidecar bytes
historical Stage185 baseline semantic SHA
Stage184 family contract matrix
exact 121 authorized F1 row IDs
F2 manual-only status
generator-owned base-form mapping
exact generator replay
repair_consumed_row_ids == authorized_F1_row_ids
full-output isolation
row count
row ID set
row ID sequence / row order
claim identity
all labels
all non-evidence fields
outside-authorized-replacement-span byte identity
predicate-swap detection
canonical rows unchanged
paraphrase rows unchanged
non-authorized structural polarity rows unchanged
F2 rows unchanged
A1/A2/A3 blocked
polarity supervision unreleased
```

The compatibility layer consumes raw Stage185-v1 records and emits a separate compatibility result. It must not modify historical sidecars, builder blobs, semantic SHA values, or raw Stage185-v1 fields.

## Generator/Full-Output Responsibility Boundary

Stage185 is not a raw evidence or raw generator-label identity oracle.

Exact generator and output identity remain owned by:

```text
validate_repaired_output_replay_identity
repair-consumption audit
full-output isolation
execution provenance
```

The compatibility mechanism requires these statuses as prerequisites. It must not duplicate their logic, weaken their checks, or accept a row when any of them is non-PASS.

## Provenance Contract

The compatibility record must bind:

```text
compatibility_rule_id
compatibility_rule_version
immediate_authority_commit
audit_report_path
audit_manifest_path
base_form_source_path
base_form_source_sha256
base_form_source_symbol
stage185_v1_runtime_dependency_id
stage185_v1_runtime_dependency_sha256
baseline_stage185_v1_sidecar_sha256
repaired_stage185_v1_sidecar_sha256
generator_replay_artifact_sha256
repair_consumption_artifact_sha256
full_output_isolation_artifact_sha256
```

All SHA fields are lowercase hex SHA-256 strings. Any missing required digest produces `MANUAL_REVIEW_REQUIRED`.

## Accounting Invariants

Required summary fields:

```text
target_count
compatibility_checked_count
compatibility_pass_count
compatibility_manual_count
compatibility_rejected_count
missing_count
unauthorized_count
compatibility_checked_row_ids
pass_row_ids
manual_row_ids
rejected_row_ids
missing_row_ids
unauthorized_row_ids
```

Invariants:

```text
target_count = 121
count == len(corresponding row-id array)
compatibility_checked_count == compatibility_pass_count + compatibility_manual_count + compatibility_rejected_count
compatibility_checked_row_ids == ordered partition union of pass_row_ids, manual_row_ids, rejected_row_ids
target_count == compatibility_checked_count + missing_count
pass_row_ids, manual_row_ids, rejected_row_ids are pairwise disjoint
checked and missing authorized IDs are disjoint and together equal exact authorized_F1_row_ids
compatibility_checked_row_ids and missing_row_ids contain authorized_F1_row_ids only
rejected_row_ids contains only authorized target rows whose compatibility result is REJECTED
unauthorized_count == len(unauthorized_row_ids)
unauthorized_row_ids are disjoint from compatibility_checked_row_ids and missing_row_ids
unauthorized rows are outside the authorized target partition and must be reported only in unauthorized_row_ids and unauthorized_count
every count equals the corresponding row-ID array length
row-id arrays preserve authoritative clean baseline filtered row order
```

If an unauthorized row is presented to the compatibility mechanism, its compatibility semantic result is `REJECTED` because use outside the exact F1 scope is prohibited. In execution-level accounting, that row must not be inserted into `compatibility_checked_row_ids`, `pass_row_ids`, `manual_row_ids`, `rejected_row_ids`, or `missing_row_ids`. Focused or unit validation may directly invoke the compatibility rule on an unauthorized row and expect `REJECTED`; that does not make the row part of the authorized execution accounting partition.

A successful future execution is expected to require:

```text
compatibility_checked_count = 121
compatibility_pass_count = 121
compatibility_manual_count = 0
compatibility_rejected_count = 0
missing_count = 0
unauthorized_count = 0
```

These values are not claimed to have already been achieved.

## Artifact Schema

Future execution artifacts:

```text
row-level compatibility JSONL
row-level compatibility CSV
compatibility summary JSON
compatibility execution report MD
compatibility provenance manifest JSON
```

Row-level compatibility records must be emitted in authorized F1 row order. JSON serialization must use UTF-8, sorted object keys, two-space indentation for JSON documents, no trailing whitespace, and newline termination. JSONL records must use one compact sorted-key JSON object per row and preserve row order. CSV must preserve the same row order and use the field order below.

Required row-level fields:

```text
compatibility_rule_id
compatibility_rule_version
row_id
pair_id
compatibility_status
compatibility_pass
authorized_F1_membership
intervention_type
inflected_predicate_surface
expected_base_predicate
base_form_derivation_method
base_form_source_path
base_form_source_sha256
base_form_source_symbol
base_form_source_identity_status
span_uniqueness_status
original_authorized_span
regenerated_authorized_span
outside_span_byte_identity
claim_identity_status
label_identity_status
non_evidence_field_identity_status
generator_replay_identity_status
repair_consumption_status
full_output_isolation_status
stage185_v1_runtime_authority_status
baseline_stage185_v1_provenance_status
repaired_stage185_v1_provenance_status
baseline_stage185_v1_grammar_status
baseline_stage185_v1_intervention_contract_status
baseline_stage185_v1_integrity_status
baseline_stage185_v1_canonical_status
baseline_stage185_v1_polarity_contamination_status
baseline_stage185_v1_audit_expected_axes
baseline_stage185_v1_audit_changed_axes
baseline_stage185_v1_reason_codes
repaired_stage185_v1_grammar_status
repaired_stage185_v1_intervention_contract_status
repaired_stage185_v1_integrity_status
repaired_stage185_v1_canonical_status
repaired_stage185_v1_polarity_contamination_status
repaired_stage185_v1_dataset_source_status
repaired_stage185_v1_schema_status
repaired_stage185_v1_time_swap_status
repaired_stage185_v1_audit_expected_axes
repaired_stage185_v1_audit_changed_axes
repaired_stage185_v1_reason_codes
predicate_semantic_identity_preserved
surface_realization_changed
compatibility_explained_stage185_axes
unexplained_stage185_axes
effective_semantic_changed_axes
effective_intervention_contract_status
effective_F1_repair_integrity_status
ordered_compatibility_blockers
```

Enums:

```text
compatibility_status = PASS | MANUAL_REVIEW_REQUIRED | REJECTED
compatibility_pass = true | false
authorized_F1_membership = PASS | FAIL
intervention_type = polarity_flip | other
base_form_derivation_method = GENERATOR_AUTHORITY_BASE_PREDICATE_BY_INFLECTED
base_form_source_identity_status = PASS | FAIL | NOT_RUN
span_uniqueness_status = PASS | FAIL | NOT_RUN
outside_span_byte_identity = PASS | FAIL
claim_identity_status = PASS | FAIL | NOT_RUN
label_identity_status = PASS | FAIL | NOT_RUN
non_evidence_field_identity_status = PASS | FAIL | NOT_RUN
generator_replay_identity_status = PASS | FAIL | NOT_RUN
repair_consumption_status = PASS | FAIL | NOT_RUN
full_output_isolation_status = PASS | FAIL | NOT_RUN
stage185_v1_runtime_authority_status = PASS | FAIL | NOT_RUN
baseline_stage185_v1_provenance_status = PASS | FAIL | NOT_RUN
repaired_stage185_v1_provenance_status = PASS | FAIL | NOT_RUN
grammar_status = PASS | FAIL
intervention_contract_status = PASS | FAIL
integrity_status = ELIGIBLE | INELIGIBLE
canonical_status = PASS | FAIL
polarity_contamination_status = PASS | FAIL
dataset_source_status = PASS | FAIL
schema_status = PASS | FAIL
time_swap_status = PASS | FAIL
reason_codes = array<string>
predicate_semantic_identity_preserved = TRUE | FALSE
surface_realization_changed = TRUE | FALSE
effective_intervention_contract_status = COMPATIBILITY_PASS | COMPATIBILITY_FAIL | COMPATIBILITY_BLOCKED
effective_F1_repair_integrity_status = COMPATIBILITY_ELIGIBLE | COMPATIBILITY_INELIGIBLE | COMPATIBILITY_BLOCKED
```

Required-key presence and unresolved-value semantics:

```text
required key presence is unconditional
a value may be null only when its owning prerequisite/status proves that the value could not be established
null means UNRESOLVED / NOT ESTABLISHED
null never means PASS, FAIL, TRUE, or FALSE
raw Stage185 binary observation fields use PASS | FAIL when observed
raw Stage185 binary observation fields use null when unavailable under owner/prerequisite rules
NOT_RUN is not a raw Stage185 observation value
compatibility PASS forbids null in every positive-contract field
compatibility PASS forbids null in every raw Stage185 signature field required for PASS
REJECTED must use concrete contradictory evidence for the blocker that caused rejection
unrelated downstream fields not evaluated after the decisive blocker may be null
MANUAL_REVIEW_REQUIRED may use null for fields whose authority/provenance or prerequisite execution was unresolved
ordered_compatibility_blockers must identify the owner/reason for each unresolved or decisive failure
a null field can never contribute evidence toward COMPATIBILITY_ELIGIBLE
```

Specific ownership rules:

```text
if baseline_stage185_v1_provenance_status != PASS, baseline raw Stage185 value fields may be null and compatibility cannot PASS
if repaired_stage185_v1_provenance_status != PASS, repaired raw Stage185 value fields may be null and compatibility cannot PASS
if base_form_source_identity_status != PASS, base-form-dependent derived values may be null and compatibility cannot PASS
if span_uniqueness_status is NOT_RUN because an earlier authority/provenance blocker prevented evaluation, span-derived fields may be null
```

Raw Stage185-v1 reason codes must not be replaced or reinterpreted. They remain historical observations, not compatibility-overridden values.

Artifact SHA accounting must include every generated compatibility artifact and its SHA-256 in the provenance manifest.

## Superseded P3-W6 Clauses

| Old requirement | Reason unreachable | Replacement requirement | Scope | Historical evidence preserved |
| --- | --- | --- | --- | --- |
| repaired Stage185-v1 `intervention_contract_status == PASS` | Stage185-v1 observes authorized base-form realization as predicate-axis change and emits `FAIL`. | repaired raw state must match the known compatibility signature and compatibility status must be `PASS`. | Exact authorized P3-W6-F1 rows only. | Yes, raw `FAIL` remains visible. |
| repaired Stage185-v1 `integrity_status == ELIGIBLE` | Raw repaired Stage185-v1 remains `INELIGIBLE` for all 121 authorized F1 repairs. | `effective_F1_repair_integrity_status == COMPATIBILITY_ELIGIBLE`. | Exact authorized P3-W6-F1 rows only. | Yes, raw `INELIGIBLE` remains visible. |
| repaired Stage185-v1 `audit_changed_axes == ["polarity"]` | Raw repaired Stage185-v1 deterministically reports `["polarity", "predicate"]`. | raw axes must be exactly `["polarity", "predicate"]`, and compatibility must explain only `predicate`. | Exact authorized P3-W6-F1 rows only. | Yes, raw axes remain visible. |
| `F1_integrity_transition == INELIGIBLE_TO_ELIGIBLE` | Historical Stage185-v1 cannot produce raw `ELIGIBLE`. | `F1_effective_transition == RAW_INELIGIBLE_TO_COMPATIBILITY_ELIGIBLE`. | Exact authorized P3-W6-F1 rows only. | Yes, raw transition remains `INELIGIBLE -> INELIGIBLE`. |

No other P3-W6, P3-W5, P3-W4, Stage184, or historical Stage185 requirements are superseded.

## Implementation Implications

The analyzer flow must be revised from:

```text
semantic_audit_record(...)
    ->
stage185_transition(...)
```

where `stage185_transition()` requires impossible raw Stage185-v1 acceptance, to a flow where phases 1 through 13 complete before row-local semantic audit consumes the effective compatibility-aware transition.

The compatibility-aware row-local semantic audit must consume:

```text
raw Stage185-v1 state
separate compatibility record
effective_F1_repair_integrity_status
effective_semantic_changed_axes
```

Current analyzer behavior is not compliant until this flow is implemented.

## Required Future Tests

Future implementation must add or update tests for:

```text
all 121 authorized F1 rows derive compatibility PASS under exact replay
row order and row ID sequence are preserved
raw Stage185-v1 fields remain unchanged and visible
effective state is COMPATIBILITY_ELIGIBLE only with compatibility PASS
wrong base predicate is rejected
correct base predicate on unauthorized structural polarity row is rejected
alternate-predicate substitution is rejected
unrelated third-predicate substitution is rejected
multiple spans are rejected
missing original span is rejected
missing regenerated span is rejected
outside-span mutation is rejected
claim mutation is rejected
label mutation is rejected
non-evidence field mutation is rejected
generator replay failure blocks
repair-consumption mismatch blocks
full-output isolation failure blocks
Stage185 provenance failure blocks
Stage185 runtime identity mismatch blocks
F2 input is rejected
predicate_swap remains rejected
additional changed axes beyond polarity+predicate are rejected
unresolved raw Stage185 provenance yields null rather than NOT_RUN for raw Stage185 binary fields
raw Stage185 binary fields reject NOT_RUN
focused pytest conflict is resolved by implementation rather than test weakening
```

## Explicit Non-Authorizations

This specification does not authorize:

```text
production code modification in this phase
test modification in this phase
data modification in this phase
checkpoint modification
historical authority artifact modification
production regeneration
candidate acceptance
polarity supervision release
A1/A2/A3
F2 automatic repair
generic lemmatization
generic stemming
arbitrary morphology heuristics
external NLP morphology packages
LLM semantic judgment
learned semantic equivalence
fuzzy predicate matching
predicate-swap weakening
silent upgrade after MANUAL downgrade or rejection
```

## Commands Deliberately Not Executed

Per task boundary, the following were deliberately not executed:

```text
python
pytest
analyzers
generators
regeneration
training
evaluation
git
Kaggle
```

No execution artifacts are generated by this specification-only task.

P3W6F1_STAGE185_COMPATIBILITY_SPEC_READY_FOR_STATIC_REVIEW
