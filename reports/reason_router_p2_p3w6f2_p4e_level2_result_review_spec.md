# P3-W6-F2-P4-E Gate 6 Level-2 Result Review Specification

Title: P3-W6-F2-P4-E LEVEL-2 RESULT REVIEW CONTRACT

## A. Authority And Prerequisite Freeze

This specification does not authorize Gate-6 review execution. Future Gate-6 static/read-only review requires an independently verified and frozen/committed P4-E specification plus subsequent explicit workflow authorization. Independent verification and commit alone do not release Gate 6.

Current workflow prerequisite:

- Official P4-D Gate 5 has PASSed and its handoff has been imported successfully.

Frozen P4-D Gate-5 specification:

- path: `reports/reason_router_p2_p3w6f2_p4d_controlled_data_integrity_gate_spec.md`
- authority commit: `1be4050dbadb0cb5ed2f6b55a2391051f5f6c07e`

Official Gate-5 execution:

- run: `p3w6f2-p4d-gate5-official-eced1d4`
- head: `eced1d46e8788e4372eca14dcf090c2840649399`
- command SHA256: `b2e1efae4c06ee9a312b0b7e0ca0a8b40701eca4e461a05e629769f9c553eecd`
- exit code: `0`
- PASS token: `P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_PASS`
- all Gate-5 statuses: `PASS`
- failure_reasons: `[]`
- validator_commit: `eced1d46e8788e4372eca14dcf090c2840649399`
- current_head: `eced1d46e8788e4372eca14dcf090c2840649399`
- authority_commit: `1be4050dbadb0cb5ed2f6b55a2391051f5f6c07e`
- training_admission_released: `false`

Imported Gate-5 handoff provenance:

- run: `p3w6f2-p4d-gate5-official-eced1d4`
- head: `eced1d46e8788e4372eca14dcf090c2840649399`
- ZIP SHA256: `4e42868c437eb361292a9123e37fbab1be7e12a3fb36297228624b19cf965666`
- command SHA256: `b2e1efae4c06ee9a312b0b7e0ca0a8b40701eca4e461a05e629769f9c553eecd`
- run log SHA256: `26161f680386a8048d942066accf5554aa887b694a04d6a2f1aeb1582484b58c`
- run meta SHA256: `c74992a686d7952144b4220c303d0eecd42227ab55ca6326600074c78c72c910`
- import: `PASS`

The imported handoff may record `VALIDATED=0`, `COPIED=0`, and `IDENTICAL=0` because the validator did not materialize a repo report artifact. That is not itself a Gate-5 failure. Gate 6 must still fail closed if Gate-5 provenance, imported provenance, or frozen input identities conflict.

Frozen P4-B R1 regeneration authority:

- path: `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_spec.md`
- authority commit: `fcc3b9ccaf2bbee33ac18dcef10d50acff54aab4`
- specification artifact SHA256 recorded by P4-B summary: `42c152a44f1bf81471d8fe566aee8388c17c576a53584848db6f7205e06b291e`

P4-B R1 execution artifact directory:

- `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/`

P4-B R1 artifact hashes frozen for Gate 6:

1. `controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
2. `p3w6f2_p4b_r1_regenerated_members.jsonl` SHA256 `3dd91d6e2888d50ccd45acb8243d2b8d47bd3476e2a76f5c2b9e7cd93b82bbf3`
3. `p3w6f2_p4b_r1_regeneration_audit.jsonl` SHA256 `17eaae3e20779fc6bbfe730222bd4e410d5bd03b108aaf7ea98214eaeb8d77a1`
4. `p3w6f2_p4b_r1_regeneration_summary.json` SHA256 `e09bcd09207d78a211a4b63a94af8db2a93b6c4c6b1d618e2a673168618f1157`
5. `p3w6f2_p4b_r1_full_output_isolation.json` SHA256 `c4b342d200757b4e330fd7b4bfb1b5550b3c74933bca5c99323eeac9c87ebb7e`
6. `p3w6f2_p4b_r1_deterministic_generator_invocation.json` SHA256 `2cbd5057cf89c3ba0a01bbaa1d6168b3bd595a1704bbcec50902de44066494f0`
7. `p3w6f2_p4b_r1_base_form_coverage.json` SHA256 `5d130c529e0ebcca9fa3f7137620222488a9eb8d2db137e5a7283c345a277bb3`
8. `p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl` SHA256 `59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f`
9. `p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json` SHA256 `ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8`
10. `p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json` SHA256 `09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6`

Applicable frozen Level-1 F2 authority resolved from repository state:

- Level-1 freeze commit: `acc078f8ddb5ba362d0c6861e23de21aad09cb8b`
- parent runtime authority commit: `cf80d52c222450cf84622a4f830b7331355bee07`
- decision: `P3W6F2P3_FINAL_RESULT_REVIEW_PASS`
- level1_decision: `P3W6F2P3_REAL_HYBRID_LEVEL1_REVIEW_COMPLETION_CONFIRMED`
- authorized_pair_count: `119`
- authorized_member_count: `357`
- reviewed_pair_count: `119`
- unreviewed_pair_count: `0`
- level2_remediation_complete: `false`
- level3_training_admission_released: `false`

Frozen Level-1 artifact set:

- `reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_cf80d52c/p3w6f2_hybrid_review_completed.csv` SHA256 `8c01bf4c4301382a28928543611fd1f78cb094810ed09d430b187da9bd4216c2`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_cf80d52c/p3w6f2_hybrid_review_decisions.jsonl` SHA256 `d2c845baa7316187466bd3a2352824a7821387136524a9ef5c03630f0b3c397f`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_cf80d52c/p3w6f2_hybrid_review_summary.json` SHA256 `5401f7e7fe1fb3cdd55802021b37cb33e6a7e3919faba85dbb34d8d5adbbffbc`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_final_result_review.json` SHA256 `a0656020bc62b1933350f114054b839028113d532a538cf0c82786e356e9040c`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_final_result_review.md`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_final_review_wip.jsonl` SHA256 `28792fe90a8470c0fb3fec2a134a61c9d6897c458f23bfc174175f5bd906bf6b`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_structural_cohort_audit_v1.json` SHA256 `dbe1c5a3dbe3ca76d2723ab62844774de92e2480c65bdff49228b1726a0df794`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_structural_cohort_confirmation_v1.json` SHA256 `b3686f732136bf3f3e5047ddf5123d5a78153abaf03cb397203704eeb5f25d06`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_reviewer_alias_evidence_v1.json` SHA256 `ecf77c655e0b8c8ab143fb5162422b9d93d37f0a5eac98cb7013799e1d28c919`

Frozen dataset identities:

- historical controlled dataset path: `data/controlled_v5_v3_without_time_swap.jsonl`
- historical controlled dataset hash: `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`
- regenerated dataset physical hash: `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- regenerated semantic hash: `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

Gate 6 must distinguish Gate-5 controlled-data integrity/provenance correctness from Gate-6 semantic and grammatical remediation correctness. Gate-5 PASS is a prerequisite; it is not a substitute for per-pair semantic disposition.

## B. Review Population

Gate 6 freezes the exact Level-2 review population as:

- 119 authorized F2 pairs
- 357 authorized members
- exactly three relevant members per pair:
  - `none` / canonical
  - `paraphrase`
  - `polarity_flip`
- 238 authorized evidence changes:
  - 119 canonical
  - 119 paraphrase
- 119 `polarity_flip` rows unchanged

The exact pair and member identities must come from the frozen P4-B artifacts listed above, especially artifacts 2, 3, 4, 5, and 8. No newly inferred, expanded, sampled, or substituted population is allowed.

No cohort substitution is allowed for Gate-6 semantic disposition. Every one of the 119 pairs must receive an explicit pair-level result.

## C. Semantic Remediation Criteria

For every authorized F2 pair, Gate 6 requires explicit verification that:

1. Canonical regenerated evidence is grammatical.
2. Canonical regenerated evidence realizes the intended base predicate under the approved R1 negation construction, `did not <base predicate>`, and does not preserve the historical inflection defect.
3. Paraphrase regenerated evidence is grammatical and semantically consistent with the same structured fact and predicate authority.
4. Regeneration does not change the intended subject, entity, object, role, or frame semantics of the underlying structured fact merely to make the sentence grammatical.
5. Canonical and paraphrase retain their intended negative/polarity role.
6. `polarity_flip` retains its intended affirmative role and is unchanged from the frozen historical row where required.
7. Labels, pair linkage, intervention identity, claim, and all fields other than the specifically authorized evidence realization remain consistent with frozen authority. Gate-5 PASS may be used as the integrity prerequisite for these structural facts; Gate 6 must not present them as newly discovered semantic evidence.
8. No new semantic contradiction, predicate substitution, entity drift, role reversal, temporal/frame drift, or other remediation-induced defect is present.
9. Stage185 predicate-realization compatibility remains supported by frozen artifacts 8-10 and Gate-5 PASS.
10. No row may be accepted merely because it matches a string template. Semantic correspondence to the frozen structured source must be part of the review basis.

The accepted structured predicates and base realizations remain exactly those frozen by P4-B:

| Structured semantic predicate | Required negative auxiliary base predicate |
| --- | --- |
| `approved` | `approve` |
| `delivered` | `deliver` |
| `launched` | `launch` |
| `opened` | `open` |
| `published` | `publish` |
| `restored` | `restore` |
| `selected` | `select` |

## D. Exhaustive Pair-Level Disposition

The future Gate-6 review must emit one pair-level record for each of the 119 authorized pair IDs. Missing records, duplicate records, aggregate-only acceptance, and implicit cohort acceptance are forbidden.

Allowed pair dispositions are exactly:

- `PASS`
- `FAIL`
- `BLOCKED`

`PASS` means the pair satisfies all semantic remediation criteria and all prerequisites for that pair are valid.

`FAIL` means valid evidence establishes that the pair does not satisfy one or more remediation criteria.

`BLOCKED` means the reviewer cannot make a valid scientific disposition because required authority, evidence, provenance, or structured-source interpretation is missing, conflicting, ambiguous, malformed, or incomplete.

The minimum pair-level record schema is:

- `schema_version`: must be `P3W6F2P4E_LEVEL2_PAIR_REVIEW_V1`
- `pair_id`
- `canonical_member_id`
- `paraphrase_member_id`
- `polarity_flip_member_id`
- `expected_structured_predicate`
- `expected_base_predicate`
- `canonical_remediation_status`: `PASS`, `FAIL`, or `BLOCKED`
- `paraphrase_remediation_status`: `PASS`, `FAIL`, or `BLOCKED`
- `polarity_flip_preservation_status`: `PASS`, `FAIL`, or `BLOCKED`
- `structured_fact_semantic_alignment_status`: `PASS`, `FAIL`, or `BLOCKED`
- `label_linkage_prerequisite_status`: `PASS`, `FAIL`, or `BLOCKED`
- `stage185_compatibility_prerequisite_status`: `PASS`, `FAIL`, or `BLOCKED`
- `canonical_grammar_status`: `PASS`, `FAIL`, or `BLOCKED`
- `paraphrase_grammar_status`: `PASS`, `FAIL`, or `BLOCKED`
- `canonical_negative_polarity_status`: `PASS`, `FAIL`, or `BLOCKED`
- `paraphrase_negative_polarity_status`: `PASS`, `FAIL`, or `BLOCKED`
- `polarity_flip_affirmative_role_status`: `PASS`, `FAIL`, or `BLOCKED`
- `unauthorized_mutation_status`: `PASS`, `FAIL`, or `BLOCKED`
- `defect_flags`: array of zero or more explicit reason codes
- `defect_reason`: nullable string, required non-empty when any status is `FAIL` or `BLOCKED`
- `final_pair_disposition`: `PASS`, `FAIL`, or `BLOCKED`

## E. Review Methodology

Gate 6 is static and read-only. A future reviewer may inspect:

- historical dataset `data/controlled_v5_v3_without_time_swap.jsonl`
- regenerated dataset artifact 1
- P4-B artifacts 1-10
- structured source authority in `scripts/build_controlled_v5.py`
- Level-1 review artifacts listed in section A
- imported Gate-5 provenance evidence listed in section A
- P4-B and P4-D frozen specifications

The future Gate-6 review must not:

- regenerate data
- modify artifacts
- execute the Stage185 builder
- load a model
- train
- evaluate a model
- change labels
- change promotion criteria
- create, mutate, or use checkpoints
- run Kaggle execution
- replace any dataset

Because semantic correctness is research-critical, Gate 6 requires:

1. one exhaustive primary reviewer over all 119 pairs; and
2. an independent verifier of the completed Gate-6 result artifact.

The independent verifier must inspect all 119 pair-level records and must not merely trust aggregate counts.

## F. Gate-6 Result Artifact Contract

This specification defines future result artifacts but does not create them.

Future Gate-6 review artifacts must be written under:

`reports/reason_router_p2_p3w6f2_p4e_level2_result_review_<review_commit>/`

`<review_commit>` must be the full 40-character commit SHA of the future committed Gate-6 review execution state. If the future workflow separates specification freeze, review execution, and verification commits, all such commits must be recorded in the summary.

Required future artifact 1:

`p3w6f2_p4e_level2_pair_dispositions.jsonl`

Purpose: one explicit disposition for each authorized F2 pair.

Schema version:

`P3W6F2P4E_LEVEL2_PAIR_REVIEW_V1`

Contract:

- exactly 119 JSONL records
- one record per authorized pair ID
- no duplicate pair IDs
- no pair outside the frozen authorized P4-B population
- each record must contain at least the fields listed in section D
- deterministic UTF-8 encoding with LF line endings

Required future artifact 2:

`p3w6f2_p4e_level2_result_review_summary.json`

Purpose: provenance, aggregate counts, blockers, and final decision token.

Schema version:

`P3W6F2P4E_LEVEL2_RESULT_REVIEW_SUMMARY_V1`

Required fields:

- `schema_version`
- `decision_token`
- `review_commit`
- `verification_commit`
- `p4d_authority_commit`
- `p4d_gate5_run`
- `p4d_gate5_head`
- `p4d_gate5_command_sha256`
- `p4d_gate5_pass_token`
- `imported_gate5_zip_sha256`
- `imported_gate5_run_log_sha256`
- `imported_gate5_run_meta_sha256`
- `p4b_authority_commit`
- `p4b_artifact_directory`
- `p4b_artifact_hashes`
- `level1_freeze_commit`
- `level1_artifact_hashes`
- `historical_dataset_sha256`
- `regenerated_dataset_sha256`
- `regenerated_dataset_semantic_sha256`
- `gate5_prerequisite_status`
- `imported_gate5_provenance_status`
- `pair_record_count`
- `duplicate_pair_record_count`
- `pass_pair_count`
- `fail_pair_count`
- `blocked_pair_count`
- `canonical_remediation_pass_count`
- `paraphrase_remediation_pass_count`
- `polarity_flip_preservation_pass_count`
- `structured_semantic_alignment_pass_count`
- `stage185_compatibility_prerequisite_status`
- `unresolved_semantic_issue_count`
- `unauthorized_field_or_data_mutation_discovered`
- `authority_provenance_ambiguity_count`
- `training_admission_released`
- `failure_reasons`
- `blockers`
- `pair_dispositions_path`
- `pair_dispositions_sha256`

Required future artifact 3:

`p3w6f2_p4e_level2_result_review_report.md`

Purpose: concise human-readable report summarizing provenance, methodology, counts, blockers, final token, and Level-3 boundary.

No future Gate-6 artifact may overwrite P4-B artifacts 1-10, P4-D validator/tests/spec, controlled datasets, Stage185 scripts, structured generator, Level-1 artifacts, model/training/evaluation code, unrelated patch files, or `reports/stage180a_pass2_annotations_completed.csv`.

## G. Aggregate Acceptance Requirements

Gate-6 PASS requires all of the following:

- Gate-5 prerequisite = `PASS`
- imported Gate-5 provenance = valid
- 119/119 pair records present
- duplicate pair records = 0
- PASS pair count = 119
- FAIL pair count = 0
- BLOCKED pair count = 0
- canonical remediation PASS = 119
- paraphrase remediation PASS = 119
- `polarity_flip` preservation PASS = 119
- structured semantic alignment PASS = 119
- Stage185 compatibility prerequisite PASS for the authorized population
- no unresolved semantic issue
- no unauthorized field/data mutation discovered
- no authority/provenance ambiguity
- `training_admission_released = false`

There is no partial-pass, majority, sampled, or cohort-level acceptance threshold.

## H. Decision Tokens

Gate 6 decision tokens are exactly:

- `P3W6F2P4E_LEVEL2_RESULT_REVIEW_PASS`
- `P3W6F2P4E_LEVEL2_RESULT_REVIEW_FAIL`
- `P3W6F2P4E_LEVEL2_RESULT_REVIEW_BLOCKED`

If an existing applicable repository authority later proves that equivalent canonical token names already existed before this specification, the reviewer must stop and report the conflict rather than creating or using a second namespace.

## I. PASS, FAIL, And BLOCKED Semantics

`P3W6F2P4E_LEVEL2_RESULT_REVIEW_PASS` means all 119 pairs satisfy the frozen Level-2 remediation criteria and all provenance prerequisites are valid.

`P3W6F2P4E_LEVEL2_RESULT_REVIEW_FAIL` means valid evidence establishes that at least one authorized pair does not satisfy the remediation criteria.

`P3W6F2P4E_LEVEL2_RESULT_REVIEW_BLOCKED` means the review cannot make a valid scientific disposition because required authority, evidence, provenance, structured source, artifact schema, or pair-level review evidence is missing, conflicting, ambiguous, malformed, or incomplete.

Missing evidence must be treated as BLOCKED, not FAIL.

## J. Gate Boundary

Gate-6 PASS may establish only:

- P3-W6-F2 Level-2 remediation review is closed/passed.

Gate-6 PASS may authorize only:

- preparation of a separate Level-3 admission/authority proposal, if the governing research authority permits it.

Gate-6 PASS must not itself authorize:

- training
- evaluation
- model loading
- checkpoint creation, use, or mutation
- Kaggle GPU execution
- dataset replacement
- promotion
- changing promotion criteria
- training admission

The Gate-6 result artifact must contain:

`training_admission_released = false`

Level-3 requires a separate explicit authority after Gate 6.

## K. Fail-Closed Rules

Gate 6 must return BLOCKED for at least:

- Gate-5 provenance mismatch
- imported Gate-5 provenance mismatch
- P4-B artifact hash mismatch
- Level-1 population mismatch
- missing pair disposition
- duplicate pair disposition
- pair record count not equal to 119
- unresolved semantic judgment
- regenerated/historical identity ambiguity
- structured-source ambiguity
- Stage185 compatibility prerequisite ambiguity
- any attempt to widen the authorized F2 population
- any attempt to infer Level-3 authority from Gate-6 PASS
- missing subsequent explicit workflow authorization for Gate-6 review
- malformed or incomplete result artifact schema
- missing independent verification of either this specification or the completed 119-pair review artifact

Gate 6 must return FAIL only when valid evidence establishes a semantic, grammatical, polarity, preservation, or unauthorized-mutation defect in at least one authorized pair.

## L. No Hidden Execution Authority

The P4-E specification itself does not authorize Gate-6 review execution. Future Gate-6 static/read-only review may begin only after all three of the following are true:

1. specification independent verification PASS;
2. specification freeze/commit;
3. subsequent explicit workflow authorization.

Independent verification plus commit alone are insufficient. This specification must not self-authorize its own Gate-6 execution.

The subsequent explicit workflow authorization is authorization only to perform the static/read-only Gate-6 Level-2 review. It does not imply authorization for Level-3, training, evaluation, model loading, checkpoint creation/use/mutation, Kaggle execution, dataset replacement, promotion, or training admission.

Creating this specification does not authorize:

- Python
- pytest
- py_compile
- Gate-5 re-execution
- regeneration
- Stage185 execution
- model code
- Kaggle
- training
- evaluation

## M. Future Verification Requirements

This specification requires independent verification before freeze. The completed future 119-pair Gate-6 review artifact also requires independent verification before final Gate-6 disposition.

The same review pass must not serve both roles without independent review.

The verifier of the completed Gate-6 result artifact must verify:

- all frozen authority identities and hashes
- exactly 119 pair records
- zero missing and zero duplicate pair IDs
- every pair-level disposition and defect reason
- aggregate counts against pair records
- Gate-5 provenance binding
- P4-B artifact binding
- Level-1 population binding
- Stage185 compatibility prerequisite binding
- `training_admission_released = false`
- no Level-3 authority implied or released

## N. Validation Boundary For This Specification Task

Validation for creating this specification is static only:

- repository inspection
- `git status`
- `git diff`
- `git diff --check`
- read-only Git history
- reading frozen artifacts/specifications

This specification task must not run Python, pytest, py_compile, Gate-5 again, regeneration, Stage185 execution, model code, training, evaluation, or Kaggle.

## O. Final Specification-Readiness Token

P3W6F2P4E_LEVEL2_RESULT_REVIEW_SPEC_READY_FOR_INDEPENDENT_VERIFICATION
