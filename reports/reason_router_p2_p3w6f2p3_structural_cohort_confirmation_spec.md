# P3-W6-F2-P3 Structural Cohort Confirmation Spec

## Decision

`P3W6F2P3_STRUCTURAL_COHORT_CONFIRMATION_V2_BLOCKER_REPAIR_READY_FOR_INDEPENDENT_REREVIEW`

This document specifies the production infrastructure amendment for `P3W6F2_HYBRID_HUMAN_REVIEW_V2`. It is review infrastructure only. It does not create real WIP records, import the workbook, import AI prescreen output, execute cohort confirmation, finalize Level 1, remediate rows, mutate data, train, evaluate, commit, or push.

## Authority Identity

- Repository branch: `main`
- Authority HEAD: `f99e6f4bba00c6aaf9730d8326ce728831c2bd98`
- Historical human WIP protocol preserved: `P3W5_F2_MANUAL_REVIEW_V1`
- Active hybrid protocol: `P3W6F2_HYBRID_HUMAN_REVIEW_V2`
- Structural gate: `P3W6F2_STRUCTURAL_COHORT_GATE_V1`
- AI prescreen protocol: `P3W6F2_AI_ASSISTED_PRESCREEN_V1`

## Human Audit Evidence

The first 20 workbook pairs were individually inspected and explicitly confirmed as `V / V / V / M`, with `override_count = 0`, deriving `CANONICAL_REGENERATION_REQUIRED`.

The exact individually reviewed audit set is:

`generated_fact_152`, `generated_fact_154`, `generated_fact_156`, `generated_fact_157`, `generated_fact_159`, `generated_fact_160`, `generated_fact_162`, `generated_fact_163`, `generated_fact_164`, `generated_fact_167`, `generated_fact_168`, `generated_fact_169`, `generated_fact_170`, `generated_fact_171`, `generated_fact_173`, `generated_fact_175`, `generated_fact_177`, `generated_fact_178`, `generated_fact_179`, `generated_fact_180`.

Predicate coverage is `restored=4`, `selected=4`, `approved=4`, `delivered=3`, `published=2`, `opened=2`, `launched=1`. This is structural coverage, not statistical proof.

## Review Method

Every V2 WIP record must contain `review_method`.

Allowed values:

- `INDIVIDUAL_TRIPLE_REVIEW`
- `STRUCTURAL_COHORT_CONFIRMATION`

`INDIVIDUAL_TRIPLE_REVIEW` is created only by the existing `record` semantics and still requires `--ack-complete-triple-reviewed`. It must have empty `cohort_confirmation_id`.

`STRUCTURAL_COHORT_CONFIRMATION` is created only by `cohort-confirm`, never by `record`. It requires a nonempty `cohort_confirmation_id`, a valid cohort confirmation artifact, AI prescreen validation, and structural gate PASS.

Every V2 WIP record also contains `record_origin`.

Allowed method/provenance/origin combinations:

- CLI individual record: `INDIVIDUAL_TRIPLE_REVIEW`, `CAPTURED_IN_RECORD`, `CLI_INDIVIDUAL_RECORD`
- Imported workbook confirmation: `INDIVIDUAL_TRIPLE_REVIEW`, `NOT_CAPTURED_IN_XLSX`, `XLSX_CONFIRMED_IMPORT`
- Structural cohort record: `STRUCTURAL_COHORT_CONFIRMATION`, `NOT_APPLICABLE_STRUCTURAL_COHORT`, `STRUCTURAL_COHORT_CONFIRMATION`

No other method/provenance/origin combination is valid.

## Timestamp Provenance

V2 records include:

- `authority_recorded_at_utc`
- `reviewed_at_utc`
- `human_review_time_provenance`

For V2 compatibility, `reviewed_at_utc` equals `authority_recorded_at_utc` and means authoritative record creation/import time, not fabricated historical visual-review time.

Allowed `human_review_time_provenance` values:

- `CAPTURED_IN_RECORD`
- `NOT_CAPTURED_IN_XLSX`
- `NOT_APPLICABLE_STRUCTURAL_COHORT`

The existing 20 workbook confirmations must later import with `NOT_CAPTURED_IN_XLSX` if sourced from the current XLSX, because that workbook did not capture trustworthy visual-review timestamps.

Structural cohort records use `NOT_APPLICABLE_STRUCTURAL_COHORT`; their `authority_recorded_at_utc` and compatibility `reviewed_at_utc` are cohort authority record creation time, not individual visual-review time.

## Structural Audit Artifact

The cohort flow is two phase.

Phase 1, `cohort-audit`, loads source authority, validates V2 WIP, validates AI prescreen, validates the exact 20 individual audit records, runs `P3W6F2_STRUCTURAL_COHORT_GATE_V1`, computes eligible and exception pair sets, writes a stable external `P3W6F2_STRUCTURAL_COHORT_AUDIT_V1` artifact, and prints its path and SHA-256. It writes no human cohort authority and no structural WIP records.

The audit artifact includes `schema_version`, `structural_gate_version`, `audit_created_at_utc`, `source_authority_identity`, authorized pair/member counts, `source_record_sha256_by_pair`, `ai_prescreen_artifact_sha256`, `ai_prescreen_protocol_version`, `validated_individual_wip_state_sha256`, required 20-pair IDs, eligible pair IDs, exception pair IDs, per-pair structural gate results, `overall_structural_gate_result`, and `audit_payload_sha256`.

`audit_payload_sha256` is computed with deterministic canonical JSON: UTF-8, `sort_keys=True`, compact separators, no NaN, over the semantic audit payload excluding `audit_created_at_utc` and excluding `audit_payload_sha256` itself. Any source, WIP, AI prescreen, eligible set, exception set, or gate-result change changes the recomputed audit identity.

## Cohort Artifact

Phase 2, `cohort-confirm`, requires `--audit-path`, `--expected-audit-sha256`, and `--ack-structural-cohort-confirm`. It loads the previously reviewed audit artifact, verifies the supplied SHA, reloads current source/WIP/AI inputs, recomputes the canonical audit payload, and fails closed if the recomputed payload differs from the reviewed audit payload. It does not silently refresh a stale audit.

The strict external cohort confirmation artifact contains:

- `schema_version`
- `cohort_confirmation_protocol_version`
- `cohort_confirmation_id`
- `confirmation_payload_sha256`
- `authority_recorded_at_utc`
- `reviewer_id`
- `structural_audit_sha256`
- `structural_audit_path_identity`
- `ai_prescreen_artifact_sha256`
- `validated_individual_wip_state_sha256`
- `eligible_pair_count`
- `exception_pair_count`
- `eligible_pair_ids`
- `exception_pair_ids`
- `individually_reviewed_pair_ids`
- `cohort_confirmed_pair_ids`
- `structural_gate_version`
- `structural_gate_result`
- `ai_prescreen_protocol_version`
- `ai_prescreen_result_summary`
- `source_authority_identity`
- `human_action = COHORT_CONFIRM`

`confirmation_payload_sha256` is computed with deterministic canonical JSON over the immutable confirmation payload, excluding `cohort_confirmation_id` and excluding `confirmation_payload_sha256` itself. The canonical payload binds the reviewed audit SHA, structural gate version/result, source authority identity, AI prescreen identity/protocol, validated exact-20 WIP identity, exact pair membership sets, reviewer, authority timestamp, and `human_action`. `cohort_confirmation_id` is derived deterministically from the recomputed `confirmation_payload_sha256`; later loaders recompute both values and reject any mismatch before exposing the artifact to `status`, `next`, `show`, or `finalize`.

It is an external review artifact, analogous to WIP. Structural WIP validation preserves the loaded confirmation object and requires each structural record's `cohort_confirmation_id` to match the artifact and each structural `pair_id` to be present in `cohort_confirmed_pair_ids` and absent from `individually_reviewed_pair_ids`. A human-readable cohort ID alone is never sufficient authority, and an arbitrary nonempty loaded ID is never trusted.

The cohort confirmation operation stages the future confirmation artifact and future WIP ledger together, validates both staged payloads, promotes them in deterministic order, rolls back the first promotion if the second fails, verifies final hashes, and removes staging only after success. After any handled failure, final paths represent either the old consistent state or the new consistent state.

## AI Prescreen Input

The production script does not parse XLSX. Later workbook export must provide strict JSONL or JSON with one record per pair:

- `pair_id`
- `source_record_sha256`
- `ai_canonical_semantics_suggestion`
- `ai_paraphrase_semantics_suggestion`
- `ai_polarity_flip_semantics_suggestion`
- `ai_grammar_validity_suggestion`
- `ai_triage_status`
- `ai_prescreen_protocol_version`
- `ai_prescreen_model_or_system_id`
- `ai_prescreen_created_at_utc`

Structural eligibility requires exactly `V / V / V / M` and `CLEAR_SUGGESTION`.

## Structural Gate

`P3W6F2_STRUCTURAL_COHORT_GATE_V1` fails closed unless the pair satisfies the authorized F2 universe, role, label, proposition, negated-relation, affirmative-polarity, grammar, reason-code, `did not + inflected predicate`, source-hash, and AI-prescreen requirements.

Any failed pair remains individual-review-required. The gate never forces a classification.

## Cohort Command

`cohort-audit` and `cohort-confirm` are distinct from `record`. `cohort-confirm` uses `--ack-structural-cohort-confirm`, not `--ack-complete-triple-reviewed`.

The confirmation phase validates source authority, V2 WIP, AI prescreen identity, the exact reviewed audit SHA, the exact 20 audit records, seven-predicate coverage, and structural gate results before creating an external cohort artifact and V2 structural records. It must not overwrite existing individual records.

## Completion Accounting

V2 summary artifacts preserve:

- `individual_review_pair_count`
- `individual_review_pair_ids`
- `structural_cohort_confirmation_pair_count`
- `structural_cohort_confirmation_pair_ids`

The method counts must partition completed reviewed pairs and sum to 119 only when Level-1 completion is valid.

## Nonclaims

This protocol does not claim all 119 pairs were individually inspected, AI labels became human labels automatically, 20 examples statistically prove 99 examples, structural equivalence proves a general scientific claim, or this is a paper result.

Successful Level-1 completion remains hybrid human review completion only. It is not remediation completion, controlled-data integrity closure, training admission, or scientific hypothesis validation.
