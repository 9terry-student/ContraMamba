# P3-W4 Canonical Grammar Authority Audit Spec

## Static Status

Decision after static implementation: `P3W4_IMPLEMENTATION_READY_FOR_STATIC_REVIEW`.

V1 static review result was `P3W4_CANONICAL_GRAMMAR_AUTHORITY_V1_STATIC_REVIEW_BLOCKED`.

V2 static review result was `P3W4_CANONICAL_GRAMMAR_AUTHORITY_V2_STATIC_REVIEW_BLOCKED`.

V3 static review result was `P3W4_CANONICAL_GRAMMAR_AUTHORITY_V3_STATIC_REVIEW_BLOCKED`.

V4 static review result was `P3W4_CANONICAL_GRAMMAR_AUTHORITY_V4_STATIC_REVIEW_BLOCKED`.

V5 static review result was `P3W4_CANONICAL_GRAMMAR_AUTHORITY_V5_STATIC_REVIEW_BLOCKED`.

V6 static review result was `P3W4_CANONICAL_GRAMMAR_AUTHORITY_V6_STATIC_REVIEW_BLOCKED`.

V7 static review result was `P3W4_CANONICAL_GRAMMAR_AUTHORITY_V7_STATIC_REVIEW_BLOCKED`.

V8 static review result was `P3W4_CANONICAL_GRAMMAR_AUTHORITY_V8_STATIC_REVIEW_BLOCKED`.

Recorded V2 blockers:

- analyzer had a Python syntax error
- P3-W3 validator paths did not match the actual authority schema
- synthetic tests mirrored the wrong schema
- REFUTE JSONL validation still allowed missing authority fields
- fact authority generation used affected-count identity incorrectly
- grammar proof reimplemented and overclaimed the production rule
- scenario action and retained sets overlapped
- scenario totals omitted baseline SUPPORT authority
- 359 exported REFUTE rows were conflated with 478 affected members
- F2 propagation was treated as regeneration proof
- runtime blockers remained pre-execution values
- Git preservation prerequisite was documented but not enforced

Recorded V3 blockers:

- SHA mismatch test was intercepted by external-path gate
- U8 row-ID cardinality was not validated
- REFUTE evidence classes were not enforced per row
- integrity-builder SHA was required but not verified
- F2 manual remediation was still interpreted as regeneration
- manual F2 pairs were omitted from decision blockers
- R3 treated existing defective rows as regeneration yield
- complete fact authority was only checked as an affected-pair superset
- runtime analyzer schema remained v2

Recorded V4 blockers:

- three helper functions were referenced but undefined
- F1 incorrectly required a paraphrase member
- F1/F2 sidecar family contracts were not enforced during reconstruction
- an imported validator could be mislabeled as Stage185-local authority
- validator call-chain verification relied on substring presence
- validator signature check accepted incompatible required arguments
- potential authority yield remained partially class-driven
- semantic-conflict action could be reported as none
- test fixtures and deterministic error expectations were inconsistent

Recorded V5 blockers:

- two grammar reproduction tests used incomplete validator metadata and raised KeyError
- reproduce_grammar_rule did not validate its validator-record schema
- any grammar_anomaly call in any Stage185 function could be treated as production authority
- local and imported validator authorities were not explicitly disambiguated
- mixed F1/F2 remediation combinations could be collapsed into an incorrect single-family decision
- validator authority was not surfaced as a singleton runtime summary record

Recorded V6 blockers:

- grammar_validator_source_blob_identity was hard-coded to Stage182 even when the resolved production validator was Stage185-local
- AST call-site validation returned after the first authorized call and could miss later unauthorized calls
- run/main reachability alone could authorize a non-sidecar debug helper
- module-level calls embedded in assignments were not detected
- pair validator metadata comparison omitted validator_authority_function and validator_call_site_authorized
- validator-record validation checked field presence but not verified-record semantic consistency

Recorded V7 blockers:

- AST call-site collection walked overlapping parent and child roots, so each grammar_anomaly call was counted twice
- the two-call inventory test failed with observed count four instead of two
- top-level ClassDef scopes were skipped and class-method grammar_anomaly calls could escape inventory
- validator-record validation checked inventory length but did not validate each call-site entry or inventory uniqueness
Recorded V8 blockers:

- nested function decorators and default expressions inherited the enclosing build_sidecar scope and were incorrectly authorized
- nested class decorators, bases, and keywords inherited the enclosing build_sidecar scope and were incorrectly authorized
- lambda defaults inherited the enclosing build_sidecar scope and were incorrectly authorized
- function argument and return annotations were not visited, so grammar_anomaly calls in annotations escaped inventory
- async build_sidecar could inherit synchronous build_sidecar authorization
- validator-record validation did not validate reachable_from_run_or_main presence/type or representative reachability equality

## Authority Schema

P3-W3 summary validation uses only the preserved authority schema paths: `refute_row_count_exported`, `pair_level_canonical_comparison`, `final_label_overview`, `generator_evidence_class_counts`, `sidecar_semantic_interpretation_audit`, `counterfactual_eligibility_results.C5`, `candidate_universe_counts.U8_final_polarity_applicable_rows`, `A1_A3_released`, and `polarity_supervision_released`.

P3-W4 runtime analyzer schema is `reason_router_p3w4_canonical_grammar_authority_audit_v3`.

P3-W3 REFUTE JSONL rows require exact nonempty authority fields: `row_id`, `pair_id`, `intervention_type`, `final_label`, `canonical_row_id`, `canonical_counterpart_row_id`, `canonical_counterpart_final_label`, `canonical_counterpart_eligibility`, `ordered_exclusion_codes`, `generator_evidence_class`, `generator_source_sha256`, and `integrity_builder_sha256`.

## Commit And Artifact Gates

P3-W3 artifact authority is `summary["execution_commit"] == --expected-p3w3-execution-commit == 8a587a6f28a84a01237d81a47898ec4d5597ffc4`. P3-W4 runtime authority is `git HEAD == --execution-commit`.

The P3-W3 summary and REFUTE JSONL paths must be repository-internal, Git tracked, accessible at HEAD, and SHA-matched to the caller values. External or untracked artifacts are rejected. Runtime prerequisite marker: `P3W3_EXECUTION_ARTIFACTS_MUST_BE_GIT_PRESERVED_BEFORE_P3W4_EXECUTION`.

## Fact And Grammar Authority

Fact authority is reconstructed from the complete source dataset unique pair count, then filtered to affected pair IDs. It must contain every affected pair exactly once. P3-W3 row `generator_source_sha256` values must be singleton and match the current `scripts/build_controlled_v5.py` SHA, or the analyzer fails closed with `P3W4_GENERATOR_SOURCE_AUTHORITY_MISMATCH`.

Grammar proof inventories every Stage185 `grammar_anomaly` call-site with a single-pass, scope-aware AST collector. Each call-site identity includes `scope_path`, `context_path`, `lineno`, and `col_offset`, and the deterministic inventory is sorted by `(scope_path, context_path, lineno, col_offset)`. Definition-time expressions are explicitly separated using `definition_time_expression` and `scope_kind`; function decorators, defaults, annotations, return annotations, class decorators/bases/keywords, lambda defaults, and type parameters are unauthorized definition expressions. Authorization is limited to calls in the synchronous top-level `build_sidecar` function body with `scope_path == ["function:build_sidecar"]`, `scope_kind == "sync_function_body"`, and `definition_time_expression == false`; async `build_sidecar`, module-level expressions or assignments, helper functions, run/main-reachable helpers, nested functions, lambdas, and class scopes are unauthorized. The approved validator record is schema-validated before reproduction and validates every inventory entry, uniqueness, representative function/line/reachability, callable source path, signature, and call-chain status. The resolved validator authority source drives `grammar_validator_source_blob_identity`; Stage182 is separately recorded as `stage182_regression_oracle_blob_identity`. Runtime summary records a singleton `grammar_validator_authority`, and pair-level validator metadata plus resolved blob identity must match it exactly. F1 true defect requires the production validator to reproduce the failure plus generator SHA and fact identity authority.

## F2 Propagation And Remediation

`F2_CANONICAL_DEFECT_PROPAGATED_TO_DERIVATIVES` remains a lineage observation. It is not regeneration approval. F2 remediation state is recorded separately and defaults to `MANUAL_REVIEW_REQUIRED` without independent textual/semantic proof. Downstream yield, decision, supporting IDs, and blocking IDs use `remediation_state`, not root-cause class.

## Scenarios And Row Namespaces

Scenario action sets are disjoint: `action_review_row_ids`, `action_regenerate_row_ids`, `action_exclude_row_ids`, and `current_retained_row_ids`. Conditional potential admitted rows are recorded separately.

Baseline polarity authority is preserved in every scenario: SUPPORT 242, REFUTE 0. R2 contributes 238 REFUTE and 119 SUPPORT for the 119 F2 triples, so potential totals are SUPPORT 361 and REFUTE 238. R4 regenerates 478 rows and conditionally contributes 359 REFUTE and 119 SUPPORT, so potential totals are SUPPORT 361 and REFUTE 359. R5 records unknown available new REFUTE authority and minimum required new REFUTE rows 50.

Runtime output separates `p3w3_exported_REFUTE_row_count=359` from `p3w4_affected_member_row_count=478`. R3 confirmed contribution is zero; only conditional label-preservation contribution is recorded, and readiness remains unresolved.

## Runtime Truthfulness

Runtime artifacts use only these remaining blockers: `P2_POLARITY_LOCAL_SUPERVISION_NOT_TRAINING_READY` and `P3W4_RESULT_STATIC_REVIEW_PENDING`.

Static spec and manifest retain these blockers: `P2_POLARITY_LOCAL_SUPERVISION_NOT_TRAINING_READY`, `P3W3_EXECUTION_ARTIFACTS_NOT_GIT_PRESERVED`, `P3W4_CANONICAL_GRAMMAR_AUDIT_NOT_EXECUTED`, and `P3W4_STATIC_REVIEW_PENDING`.

Runtime fields include `audit_execution_completed=true`, `result_static_review_completed=false`, `human_review_required=true`, `production_repair_approved=false`, `polarity_supervision_released=false`, and `A1_A3_released=false`.

## Forbidden Claims

This static implementation does not claim `P3W4_AUDIT_PASS`, `F2_ROWS_RECOVERABLE`, `GRAMMAR_VALIDATOR_FALSE_POSITIVE`, `PRODUCTION_INTEGRITY_GATE_DEFECTIVE`, `P2_POLARITY_SUPERVISION_RESOLVED`, `A1_READY`, `A2_READY`, `A3_READY`, or `P3_PASS`.
