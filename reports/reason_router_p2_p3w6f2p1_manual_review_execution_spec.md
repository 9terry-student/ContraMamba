# P3-W6-F2-P1 F2 Manual Review Execution Specification

## Decision

`P3W6F2P1_MANUAL_REVIEW_EXECUTION_SPEC_READY_FOR_STATIC_REREVIEW`

This is a specification/documentation-only artifact. It does not modify code, data, source rows, tests, checkpoints, annotations, human fields, F1 artifacts, P3-W4/P3-W5/P3-W6-F1 authority artifacts, or training/evaluation state. It does not run analyzers, regenerate datasets, rematerialize Stage185 outputs, repair rows, commit, or push.

## Repository And Authority Identity

Repository: `9terry-student/ContraMamba`.

Expected branch: `main`.

Expected HEAD: `49d7c37cd307893bf8fbc96cd2b6730369fcd8d6`.

Expected HEAD message: `Freeze P3-W6-F1 final result authority`.

Expected parent: `35157bca7e34a36e1a398c1d419ce0473a109fd4`.

P3-W6-F1 is closed under `P3W6F1_FINAL_RESULT_REVIEW_PASS` and `P3W6F1_CLOSED`; it remains untouched. No F2 review artifact may modify any F1 source row, F1 sidecar, F1 analyzer output, dataset row, or F1 SHA at Level 1.

P3-W6-F2-P0 established `P3W6F2P0_ROOT_CAUSE_AUDIT_PASS` and `P3W6F2P0_NEXT_F2_MANUAL_REVIEW_EXECUTION_SPEC_READY`. P0 findings are structural context only, not human semantic or grammar decisions.

Primary F2 authority remains:

- P3-W5 specification: `reports/reason_router_p2_p3w5_separate_remediation_spec.md`
- P3-W5 manifest: `reports/reason_router_p2_p3w5_separate_remediation_manifest.json`
- P3-W5 authority commit: `01d983f8d09cacf0eddefd2014fc81a28771cf5e`
- P3-W4 execution commit: `ca99038d812696467a4330cffc1c4c5b5f72cfe2`
- P3-W4 result authority: `f0a9afddc5b93c54aa72b0335c5a1a2f517cf934`
- P3-W4 decision: `P3W4_RESULT_REVIEW_PASS_F1_REGENERATION_F2_MANUAL_REVIEW_REQUIRED`

F2 state entering this specification:

- `F2_target_pair_count = 119`
- `F2_target_member_count = 357`
- `F2_remediation_state = MANUAL_REVIEW_REQUIRED`
- `P3W5_F2_MANUAL_REVIEW_NOT_EXECUTED`

## Frozen Input Artifacts

Pair authority:

- Path: `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_pairs.jsonl`
- SHA-256: `850ac6e8924fe334fa7f18659d204f6e0546381b1c3d3eb601f893f3eb00a493`

Summary authority:

- Path: `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_summary.json`
- SHA-256: `7c0cc383dde38a1c564dae445a78eaf9171b8648d0720de3a2acc0ba68e68e80`

F2 manual-review template:

- Path: `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_f2_manual_review.csv`
- Canonical Git/LF SHA-256: `ccc539e743d1a4226391cdca1422bb0a1054c53fd7c53a4210a54271d1e9e8a5`
- Historical original execution CRLF SHA-256: `0c3c0f85bed08cedc9d664a3c685d1c40560dbe3f9bd5b8bb88543ed9e528515`

Only the canonical Git/LF identity is repository authority. The CRLF identity is historical/worktree projection evidence only.

## Review Unit Contract

One F2 review unit is exactly one complete pair family identified by `pair_id`, containing all three members together:

- canonical `none` with final label `REFUTE`
- `paraphrase` with final label `REFUTE`
- `polarity_flip` with final label `SUPPORT`

Reviewing 357 member rows independently is forbidden. No semantic decision may be recorded without viewing the complete triple and its labels, claims, evidence, grammar statuses, reason codes, text-diff summaries, automatic root-cause class, and automatic evidence together.

Review order is the source authority CSV row order from `p3w4_f2_manual_review.csv`. Future completed artifacts must preserve that order, or record an explicit authority-preserving order field if a tool presents rows differently. This specification does not reorder the template.

## Immutable Source Schema

The immutable source schema is exactly these 27 columns, in this order:

1. `pair_id`
2. `canonical_none_row_id`
3. `paraphrase_row_id`
4. `polarity_flip_row_id`
5. `canonical_final_label`
6. `paraphrase_final_label`
7. `polarity_flip_final_label`
8. `canonical_claim`
9. `paraphrase_claim`
10. `polarity_flip_claim`
11. `canonical_evidence`
12. `paraphrase_evidence`
13. `polarity_flip_evidence`
14. `canonical_grammar_status`
15. `paraphrase_grammar_status`
16. `polarity_flip_grammar_status`
17. `canonical_reason_codes`
18. `paraphrase_reason_codes`
19. `polarity_flip_reason_codes`
20. `canonical_claim_text_diff_summary`
21. `paraphrase_claim_text_diff_summary`
22. `polarity_flip_claim_text_diff_summary`
23. `canonical_evidence_text_diff_summary`
24. `paraphrase_evidence_text_diff_summary`
25. `polarity_flip_evidence_text_diff_summary`
26. `automatic_root_cause_class`
27. `automatic_evidence`

These source fields are immutable. The completed-review artifact must not alter any byte-level parsed field value under `F2_SOURCE_RECORD_HASH_V1`.

## Human Review Fields

The six human fields are exactly:

1. `human_canonical_semantics`
2. `human_paraphrase_semantics`
3. `human_polarity_flip_semantics`
4. `human_grammar_validity`
5. `human_authority_decision`
6. `human_notes`

No extra semantic-decision field is introduced. Existing P3-W5 authority keeps `human_authority_decision` as a recorded field; the completion checker must verify that the entered value exactly matches the deterministic `F2_REVIEW_COMPATIBILITY_V1` matrix result.

## Completed CSV Schema

The authoritative completed CSV is `p3w5_f2_review_completed.csv`. It must contain 37 columns in this exact order:

1. `pair_id`
2. `canonical_none_row_id`
3. `paraphrase_row_id`
4. `polarity_flip_row_id`
5. `canonical_final_label`
6. `paraphrase_final_label`
7. `polarity_flip_final_label`
8. `canonical_claim`
9. `paraphrase_claim`
10. `polarity_flip_claim`
11. `canonical_evidence`
12. `paraphrase_evidence`
13. `polarity_flip_evidence`
14. `canonical_grammar_status`
15. `paraphrase_grammar_status`
16. `polarity_flip_grammar_status`
17. `canonical_reason_codes`
18. `paraphrase_reason_codes`
19. `polarity_flip_reason_codes`
20. `canonical_claim_text_diff_summary`
21. `paraphrase_claim_text_diff_summary`
22. `polarity_flip_claim_text_diff_summary`
23. `canonical_evidence_text_diff_summary`
24. `paraphrase_evidence_text_diff_summary`
25. `polarity_flip_evidence_text_diff_summary`
26. `automatic_root_cause_class`
27. `automatic_evidence`
28. `human_canonical_semantics`
29. `human_paraphrase_semantics`
30. `human_polarity_flip_semantics`
31. `human_grammar_validity`
32. `human_authority_decision`
33. `human_notes`
34. `source_record_sha256`
35. `reviewer_id`
36. `review_protocol_version`
37. `reviewed_at_utc`

The arithmetic is fixed as `27 + 6 + 4 = 37`. The completed CSV must contain exactly 119 rows, 119 unique `pair_id` values, zero extra pairs, zero missing pairs, and zero duplicate pairs.

## F2_SOURCE_RECORD_HASH_V1

`source_record_sha256` is computed mechanically from the authority template before review values are accepted:

1. Read CSV using an RFC 4180-compatible parser.
2. Use the exact 27 immutable source columns in frozen order.
3. Use exact strings returned by the parser.
4. Do not trim.
5. Do not Unicode-normalize.
6. Do not reparse JSON-like cells.
7. Do not sort reason codes.
8. Preserve empty cells as `""`.
9. Serialize as one JSON array.
10. Encode as UTF-8 without BOM.
11. Use `ensure_ascii = false`.
12. Use `allow_nan = false`.
13. Use separators `[",", ":"]`.
14. Use no trailing newline.
15. Hash with SHA-256 lowercase hex.

Human fields and reviewer provenance fields are excluded from the source hash.

## Semantic Review Protocol

Allowed values for each semantic field are `VALID`, `INVALID`, and `UNCLEAR`.

Each semantic field evaluates both:

- whether that member text is semantically consistent with its final label
- whether that member is consistent with the intended family transformation relationship

For canonical `none` `REFUTE`, the reviewer must consider:

- Does the evidence express the negated proposition intended by `REFUTE`?
- Is the grammatical defect surface-only, or does it make intended semantics genuinely unclear?
- Does the evidence preserve the canonical fact identity represented by the claim/family?

For `paraphrase` `REFUTE`, the reviewer must consider:

- Does the paraphrase preserve the canonical `REFUTE` semantics?
- Does it preserve the intended paraphrase relationship rather than introduce a different fact?
- Is the malformed grammar separable from semantic meaning?

For `polarity_flip` `SUPPORT`, the reviewer must consider:

- Does the evidence support the claim?
- Is it the intended polarity inversion of the canonical `REFUTE` member?
- Is its lack of literal grammar anomaly semantically consistent with the family?

These questions guide human judgment. Grammar status is not mechanically translated into semantic judgment: grammar `FAIL` does not imply semantic `INVALID`, and grammar `PASS` does not imply semantic `VALID`.

## Human Grammar Validity

Allowed values for `human_grammar_validity` are:

- `CANONICAL_ONLY_DEFECT`: the reviewer judges the reproducible grammar/text defect to be isolated to the canonical member in the family relationship.
- `MULTI_MEMBER_DEFECT`: the reviewer judges a reproducible grammar/text defect to affect multiple members of the triple.
- `NO_REPRODUCIBLE_DEFECT`: the reviewer cannot reproduce a relevant grammar/text defect from the presented triple.
- `UNCLEAR`: the reviewer cannot determine the grammar-defect scope from the presented evidence.

P0 structural observations do not pre-populate this field. Although P0 found automatic grammar `FAIL` for all canonical and paraphrase members and `PASS` for all polarity-flip members, `human_grammar_validity` remains a human authority input.

## Compatibility Matrix

Compatibility matrix version: `F2_REVIEW_COMPATIBILITY_V1`.

Precedence:

1. `UNCLEAR`
2. `INVALID`
3. `ALL_VALID_BY_GRAMMAR_STATUS`

Mapping:

- If any semantic field is `UNCLEAR`, or `human_grammar_validity == UNCLEAR`, the required decision is `INSUFFICIENT_EVIDENCE_KEEP_BLOCKED`.
- Else if any semantic field is `INVALID`, the required decision is `SEMANTIC_CONFLICT`.
- Else if all semantic fields are `VALID` and `human_grammar_validity == CANONICAL_ONLY_DEFECT`, the required decision is `CANONICAL_TEXTUAL_REPAIR_CANDIDATE`.
- Else if all semantic fields are `VALID` and `human_grammar_validity == MULTI_MEMBER_DEFECT`, the required decision is `CANONICAL_REGENERATION_REQUIRED`.
- Else if all semantic fields are `VALID` and `human_grammar_validity == NO_REPRODUCIBLE_DEFECT`, the required decision is `NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED`.

Undefined combinations are invalid. The five decision enum values are exactly:

- `CANONICAL_TEXTUAL_REPAIR_CANDIDATE`
- `CANONICAL_REGENERATION_REQUIRED`
- `SEMANTIC_CONFLICT`
- `INSUFFICIENT_EVIDENCE_KEEP_BLOCKED`
- `NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED`

## Notes Rule

`human_notes`, after trim, must be nonempty when:

- any semantic field is `UNCLEAR`
- `human_grammar_validity == UNCLEAR`
- `human_authority_decision == SEMANTIC_CONFLICT`
- `human_authority_decision == INSUFFICIENT_EVIDENCE_KEEP_BLOCKED`
- `human_authority_decision == NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED`

P3-W5 does not require mandatory notes for positive `CANONICAL_TEXTUAL_REPAIR_CANDIDATE` or `CANONICAL_REGENERATION_REQUIRED` decisions. Optional explanatory notes are recommended for reviewer clarity, but are not an authority requirement for those two positive candidate decisions.

## Reviewer Provenance

Completed review records must add:

- `source_record_sha256`
- `reviewer_id`
- `review_protocol_version`
- `reviewed_at_utc`

`review_protocol_version` is frozen as `P3W5_F2_MANUAL_REVIEW_V1`.

`reviewer_id` must be a nonempty, trimmed, stable identifier supplied explicitly at execution time. This specification does not choose or hard-code the reviewer ID.

`reviewed_at_utc` must be an RFC 3339 UTC timestamp ending in `Z`.

One reviewer is sufficient under current authority. Independent second review may be scientifically desirable as future work, but is not mandatory authority here.

## Review Presentation Contract

A future review UI/helper should present one complete triple at a time, with:

- `pair_id`
- canonical final label, claim, evidence, grammar status, and reason codes
- paraphrase final label, claim, evidence, grammar status, and reason codes
- polarity-flip final label, claim, evidence, grammar status, and reason codes
- relevant claim/evidence text-diff summaries
- automatic root-cause class and automatic evidence

The tool then collects exactly the six human fields. A reviewer should not be required to inspect raw JSON/CSV syntax if a faithful read-only presentation is available. Any future UI/helper must preserve exact authority source fields and must not silently normalize or rewrite text.

## Partial Review And Resume Contract

A partial review artifact is work-in-progress only and cannot trigger remediation, result review, training admission, or polarity supervision release.

Safe resume requires:

- already reviewed pair source hashes are revalidated before resume
- no duplicate pair review rows are accepted
- immutable source fields are not edited
- reviewer provenance is retained per completed pair
- incomplete rows are not counted as reviewed
- compatibility decisions are revalidated
- final completion gate is checked only after 119 valid rows are present

## Correction Contract

A reviewer may correct an entered judgment before final completion under these rules:

- immutable source fields may never be edited
- changing any human field must trigger compatibility decision revalidation
- `reviewed_at_utc` for a revised record must be updated to the UTC time at which the corrected judgment is finalized
- `source_record_sha256` remains based only on immutable source fields
- corrections must not produce duplicate `pair_id` rows

Versioned edit history is recommended for traceability but is not mandatory under current authority.

## Automation Boundary

Future execution implementation may automate authority loading, target-set verification, pair presentation, source hashing, schema validation, enum validation, compatibility-matrix derivation, notes validation, provenance validation, source mutation detection, pair/count/set accounting, output generation, and completion-gate validation.

It must not automate semantic `VALID`/`INVALID`/`UNCLEAR` judgments, intended transformation semantic judgments, human grammar classification, free-form human notes, or replacement of human semantic authority with an LLM/model heuristic. If `human_authority_decision` is mechanically derived from human input fields through the frozen matrix, that derivation may be automated.

## Execution Artifact Contract

Future execution directory namespace:

`reports/reason_router_p2_p3w5_f2_manual_review_execution_<execution_commit_short>/`

Future output names:

- `p3w5_f2_review_completed.csv`: authoritative full source plus human review table.
- `p3w5_f2_review_summary.json`: authoritative machine-readable summary/accounting and completion gate state.
- `p3w5_f2_review_decisions.jsonl`: authoritative pair-level decision records.

F2 execution namespace fields remain:

- `F2_execution_status`
- `F2_execution_decision`
- `F2_artifact_paths`
- `F2_input_sha256`
- `F2_execution_commit`
- `F2_output_sha256`

Manual review execution complete is not F2 remediation complete and is not controlled-data integrity closure.

Minimum decision JSONL fields:

- `pair_id`
- `source_record_sha256`
- `human_canonical_semantics`
- `human_paraphrase_semantics`
- `human_polarity_flip_semantics`
- `human_grammar_validity`
- `human_authority_decision`
- `human_notes`
- `reviewer_id`
- `review_protocol_version`
- `reviewed_at_utc`
- `compatibility_matrix_version`
- `compatibility_matrix_expected_decision`
- `compatibility_matrix_match`
- `review_record_valid`
- `ordered_validation_errors`

Fields used only for deterministic validation may be added if clearly marked derived/non-authoritative.

Minimum summary fields:

- `authorized_F2_pair_count = 119`
- `authorized_F2_pair_ids`
- `reviewed_pair_count`
- `reviewed_pair_ids`
- `unreviewed_pair_count`
- `unreviewed_pair_ids`
- `textual_repair_candidate_count`
- `textual_repair_candidate_pair_ids`
- `regeneration_required_count`
- `regeneration_required_pair_ids`
- `semantic_conflict_count`
- `semantic_conflict_pair_ids`
- `insufficient_evidence_count`
- `insufficient_evidence_pair_ids`
- `no_reproducible_defect_keep_blocked_count`
- `no_reproducible_defect_keep_blocked_pair_ids`
- `invalid_review_count`
- `invalid_review_pair_ids`
- `invalid_combination_count`
- `invalid_combination_pair_ids`
- `missing_reviewer_provenance_count`
- `missing_reviewer_provenance_pair_ids`
- `source_hash_mismatch_count`
- `source_hash_mismatch_pair_ids`
- `missing_human_field_count`
- `missing_human_field_pair_ids`
- `invalid_enum_count`
- `invalid_enum_pair_ids`
- `missing_required_notes_count`
- `missing_required_notes_pair_ids`
- `completed_decision_pair_count`
- `completed_decision_pair_ids`

Every count must equal the length of its paired ID array.

Set-theoretic completion contract:

- `reviewed_pair_ids` and `unreviewed_pair_ids` are disjoint.
- `reviewed_pair_ids union unreviewed_pair_ids == authorized_F2_pair_ids`.
- `invalid_review_pair_ids` is a subset of `reviewed_pair_ids`.
- `completed_decision_pair_ids == reviewed_pair_ids - invalid_review_pair_ids`.
- The five completed decision arrays are pairwise disjoint.
- The union of the five completed decision arrays exactly equals `completed_decision_pair_ids`.
- Invalid and unreviewed records must not appear in completed decision categories.

## Completion Validator Contract

A future validator must fail closed for at least:

- `AUTHORITY_PAIR_UNIVERSE_MISMATCH`
- `SOURCE_SCHEMA_MISMATCH`
- `SOURCE_RECORD_HASH_MISMATCH`
- `SOURCE_FIELD_MUTATION`
- `MISSING_HUMAN_FIELD`
- `INVALID_SEMANTIC_ENUM`
- `INVALID_GRAMMAR_ENUM`
- `INVALID_AUTHORITY_DECISION_ENUM`
- `COMPATIBILITY_MATRIX_MISMATCH`
- `MISSING_REQUIRED_NOTES`
- `MISSING_REVIEWER_ID`
- `INVALID_REVIEW_PROTOCOL_VERSION`
- `INVALID_REVIEW_TIMESTAMP`
- `DUPLICATE_PAIR_ID`
- `MISSING_PAIR_ID`
- `UNAUTHORIZED_PAIR_ID`
- `COUNT_ARRAY_ASYMMETRY`
- `DECISION_PARTITION_MISMATCH`

Deterministic ordered validation precedence:

1. authority pair universe and schema checks
2. immutable source hash/mutation checks
3. pair identity coverage checks
4. human/provenance field presence checks
5. enum checks
6. compatibility matrix checks
7. required notes checks
8. count/array symmetry checks
9. decision partition checks

This stage does not implement the validator.

## Completion Gate

Manual-review execution is complete only if:

- `reviewed_pair_count = 119`
- `unreviewed_pair_count = 0`
- `invalid_review_count = 0`
- `completed_decision_pair_count = 119`
- 119 unique pair IDs
- zero missing pairs
- zero duplicate pairs
- all six human fields populated
- all provenance fields populated
- all enums valid
- all compatibility combinations valid
- all required notes present
- all 119 source hashes match
- no immutable source mutation
- no pair, row, or label mutation

The execution status is then `P3W5_F2_MANUAL_REVIEW_EXECUTION_COMPLETE_PENDING_RESULT_REVIEW`. This does not call F2 remediated or closed.

## Three-Level Lifecycle

Level 1, F2 manual review completion: all 119 pairs are reviewed validly with provenance.

Level 2, F2 remediation completion: all pairs routed to repair, regeneration, or conflict-resolution branches have been resolved under separate authority and validated.

Level 3, controlled-data integrity closure: no known unresolved F1/F2 integrity defect remains, post-remediation Stage185/provenance/isolation checks pass, independent result review passes, and final authority is frozen.

This P1 specification prepares Level 1 only.

## P0 Structural Context

P0 structural cluster: `F2-SC-01`, 119 pairs and 357 members.

Automatic pattern:

- canonical grammar `FAIL`
- paraphrase grammar `FAIL`
- polarity-flip grammar `PASS`
- canonical reasons: `DID_NOT_INFLECTED_PREDICATE`, `GRAMMAR_TEMPLATE_FAIL`
- paraphrase reasons: `CANONICAL_ROW_KNOWN_GENERATOR_DEFECT`, `DID_NOT_INFLECTED_PREDICATE`, `GRAMMAR_TEMPLATE_FAIL`
- polarity reasons: `CANONICAL_ROW_KNOWN_GENERATOR_DEFECT`

Seven malformed surfaces:

- `did not selected`: 20 pairs
- `did not restored`: 18 pairs
- `did not approved`: 17 pairs
- `did not opened`: 17 pairs
- `did not delivered`: 16 pairs
- `did not published`: 16 pairs
- `did not launched`: 15 pairs

These counts total 119. They are not automatic semantic judgments and do not authorize any bypass of human semantic review.

## Future Remediation Branches

`CANONICAL_TEXTUAL_REPAIR_CANDIDATE` requires future separate textual repair/revalidation authority.

`CANONICAL_REGENERATION_REQUIRED` requires future separate generator/regeneration authority.

`SEMANTIC_CONFLICT` requires future separate semantic-conflict resolution authority.

`INSUFFICIENT_EVIDENCE_KEEP_BLOCKED` remains blocked.

`NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED` remains blocked pending separate authority resolution.

Manual review completion alone does not resolve these branches.

## Non-Claims

This P1 specification does not establish:

- any F2 semantic judgment
- any F2 repair candidate
- any F2 regeneration candidate
- any semantic conflict
- any training admission
- polarity supervision release
- F2 remediation completion
- controlled-data integrity closure
- any model performance change
- any Reason Router experimental result
