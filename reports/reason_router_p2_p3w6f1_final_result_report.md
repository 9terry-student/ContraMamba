# P3-W6-F1 Final Result Report / Authority Freeze

Decision: `P3W6F1_FINAL_RESULT_REVIEW_PASS`

Lifecycle state: `P3W6F1_CLOSED`

This is a documentation-only final authority freeze. It does not modify code, data, generators, tests, analyzer artifacts, checkpoints, or runtime outputs. It does not regenerate F1, rematerialize Stage185, run analyzers, train, evaluate, commit, or push.

## Repository Identity

- Branch: `main`
- HEAD: `35157bca7e34a36e1a398c1d419ce0473a109fd4`
- Commit message: `Fix P3-W6-F1 Stage184 Git-object authority portability`
- Parent: `90235a339714e40aff6ee0ead2256173891df685`

The P3-W6-F1 authority chain preserves two separate execution authorities:

- `F1_EXECUTION_COMMIT = dc8179e45f7c10416026acdadcbe5cbd8a78d37e`
- `MATERIALIZER_EXECUTION_COMMIT = 35157bca7e34a36e1a398c1d419ce0473a109fd4`

These commits are not collapsed. The historical materializer predecessor `90235a339714e40aff6ee0ead2256173891df685` remains classified as `IMPLEMENTATION_REVIEWED_BUT_RUNTIME_PORTABILITY_BLOCKED`.

## Decision Chain

The analyzer execution decision was:

`P3W5_F1_REGENERATION_COMPLETE_ALL_CANDIDATES_ACCEPTED_PENDING_RESULT_REVIEW`

The completed independent final result review promoted that pending analyzer decision to:

`P3W6F1_FINAL_RESULT_REVIEW_PASS`

The final lifecycle state is:

`P3W6F1_CLOSED`

Historical analyzer output is not rewritten by this report. The analyzer decision and the final P3-W6-F1 review decision remain distinct.

## Problem And Root Cause

The F1 defect was the approved `F1_TRUE_POLARITY_GENERATION_DEFECT`: negative `polarity_flip` evidence rendered do-support negation with an inflected predicate surface.

The exact defect class was:

```text
did not <inflected_predicate_surface>
```

where English realization required:

```text
did not <expected_base_predicate>
```

The root cause was surface rendering / morphology selection in the authorized F1 negative `polarity_flip` construction path. It was not a label reinterpretation, predicate-swap repair, paraphrase cleanup, or general grammar normalizer.

## Authorized Repair

The only authorized repair was:

```text
did not <inflected_predicate_surface>
->
did not <expected_base_predicate>
```

for the approved F1 negative `polarity_flip` rows only.

No other grammar normalization, predicate equivalence, paraphrase cleanup, label reinterpretation, semantic rewrite, or F2 remediation occurred.

Repair API:

`build_controlled_records_with_f1_polarity_repair_audit`

Repair mode:

`f1_authorized_polarity_negative_only`

Projection:

`baseline_id_sequence`

## Baseline Authority

- Baseline dataset: `data/controlled_v5_v3_without_time_swap.jsonl`
- Baseline dataset SHA256: `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`
- Historical baseline Stage185 sidecar: `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl`
- Historical baseline Stage185 semantic SHA256: `5bc03caa2a29f9b9176ab0201db57ebad516352797546db1a18e6ec3373fc`
- P3-W4 summary SHA256: `7c0cc383dde38a1c564dae445a78eaf9171b8648d0720de3a2acc0ba68e68e80`
- P3-W4 pairs SHA256: `850ac6e8924fe334fa7f18659d204f6e0546381b1c3d3eb601f893f3eb00a493`
- P3-W5 authority commit: `01d983f8d09cacf0eddefd2014fc81a28771cf5e`
- Trusted historical Stage185 dependency: `ff6929bf33693fb4e70bd9528551053f4402fe1c`
- Historical baseline generator SHA256: `c41e6a52401bd8c83970286b176950fc751509bee6d797d5da9aea4262c72802`

## F1 Accounting

- `F1_target_pair_count = 121`
- `F1_generated_candidate_count = 121`
- `F1_accepted_candidate_count = 121`
- `F1_manual_review_required_count = 0`
- `F1_rejected_candidate_count = 0`
- `F1_missing_candidate_count = 0`
- `F1_unauthorized_candidate_count = 0`
- `F1_execution_blockers = []`

F1 regeneration authority:

- F1 execution commit: `dc8179e45f7c10416026acdadcbe5cbd8a78d37e`
- Repaired JSONL SHA256: `d403c437d982e7d61e524fe21ed24391c84d5103be75890a61be7aa40d942833`
- Rows: `3600`
- Changed rows: `121`
- Authorized F1 row IDs SHA256: `386c0c5d5ed80e699f1607f94c2f8ba2861fa0cb1216d5d421f66e62c03d8c64`
- Baseline ID sequence SHA256: `898070cd6718f9c677ba68442ee8ed9010200363df01d147528779306917c0eb`
- Pair count: `300`
- Authorized F1 row count: `121`
- Structural negative polarity_flip row count: `150`
- Repaired generator SHA256: `37e47a3ef60b26c7186d37367d59db158c28c6b9c9eb9e25a13927fc85810684`

## Full-Output Isolation

The final full-output isolation result is frozen as:

```text
changed_ids == authorized_F1_row_ids
repair_consumed_row_ids == authorized_F1_row_ids
evidence_changed_row_ids == authorized_F1_row_ids
missing_ids == []
added_ids == []
duplicate_ids == []
unauthorized_changed_row_ids == []
F2_changed_row_ids == []
unaffected_changed_row_ids == []
canonical_changed_row_ids == []
paraphrase_changed_row_ids == []
claim_changed_row_ids == []
non_text_field_changed_row_ids == []
baseline_row_count = 3600
repaired_row_count = 3600
row_order_changed = false
```

This proves repair isolation only for the exact authorized F1 row set. It does not generalize beyond the P3-W6-F1 authority set.

## Compatibility Accounting

- `target_count = 121`
- `compatibility_checked_count = 121`
- `compatibility_pass_count = 121`
- `compatibility_manual_count = 0`
- `compatibility_rejected_count = 0`
- `missing_count = 0`
- `unauthorized_count = 0`

Core gates:

- `authority_cardinality_pass = true`
- `target_scope_membership_pass = true`
- `base_form_coverage_pass = true`
- `full_output_isolation_pass = true`
- `stage185_provenance_pass = true`
- `execution_provenance_pass = true`
- `compatibility_provenance_pass = true`

## F2 Preservation

P3-W6-F1 success does not close F2.

Inherited F2 authority remains:

- `F2_target_pair_count = 119`
- `F2_target_member_count = 357`
- `F2_remediation_state = MANUAL_REVIEW_REQUIRED`
- `P3W5_F2_MANUAL_REVIEW_NOT_EXECUTED`

F2 was not modified. `F2_changed_row_ids == []`. P3-W6-F1 does not constitute F2 remediation. F2 remains a separately authorized unresolved path. This report does not create an F2 plan.

## Stage184 Portability Authority

Stage184 contract matrix:

`reports/stage184a_controlled_train_integrity_mask_spec_20260715_134538/stage184a_family_contract_matrix.csv`

Final executable canonical raw Git-object SHA256:

`4287bf1ca7f1f2b08e5de53d24ad4019ca5ddff8a16db2dbb65727a5189e96fa`

Historical accidental Windows CRLF worktree SHA256:

`e5f61ac8d0ca3de3dd43767b83bec8c2c171a1635d419466c98d8d32ec2f38e5`

The old value is diagnostic evidence only and is not executable authority. The portability correction changed no scientific semantics.

Runtime validation previously reported:

- focused materializer: `57 passed`
- regeneration wrapper: `22 passed`
- existing P3-W6-F1 analyzer/regression: `206 passed`

## Repaired Stage185 Materialization Authority

- Materializer execution commit: `35157bca7e34a36e1a398c1d419ce0473a109fd4`
- Status: `P3W6F1_REPAIRED_STAGE185_MATERIALIZATION_PASS`
- Independent gate: `P3W6F1_REPAIRED_STAGE185_MATERIALIZATION_INDEPENDENT_OUTPUT_GATE_PASS`
- Sidecar physical SHA256: `2d7323bb4dccf5bb8c68b18ffce1041a601fd8383590aa6902557db44ce810cf`
- Sidecar semantic SHA256: `fd8f71beaaae028e65fa477726fefa9fde08c8767c6abdd6b9c58c6d3fae9938`
- Authorized F1 row count: `121`
- Total/train/dev: `3600 / 2880 / 720`
- Split seed: `174`
- Dev ratio: `0.2`
- Rule version: `stage185a_v1`
- Historical Stage185 binary executed: `false`
- Provenance validation status: `PASS`

## Raw Stage185 State And Compatibility Interpretation

Raw repaired Stage185 state remains immutable:

- `grammar_status = PASS`
- `intervention_contract_status = FAIL`
- `integrity_status = INELIGIBLE`
- `canonical_status = PASS`
- `polarity_contamination_status = PASS`
- `dataset_source_status = PASS`
- `schema_status = PASS`
- `time_swap_status = PASS`
- `audit_expected_axes = ["polarity"]`
- `audit_changed_axes = ["polarity", "predicate"]`

Historical/raw Stage185 evidence is not rewritten.

The separate compatibility rule is:

`P3W6F1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY`

Version:

`V1`

For exact authorized F1 compatibility rows, this rule may derive:

```text
effective_intervention_contract_status = COMPATIBILITY_PASS
effective_F1_repair_integrity_status = COMPATIBILITY_ELIGIBLE
```

This must not be represented as rewriting raw Stage185 to:

```text
intervention_contract_status = PASS
integrity_status = ELIGIBLE
```

## Analyzer Authority

Analyzer execution was performed under:

`F1_EXECUTION_COMMIT = dc8179e45f7c10416026acdadcbe5cbd8a78d37e`

This is required because the analyzer execution provenance contract requires runtime HEAD to equal the F1 execution commit.

Analyzer source was byte-identical between:

- `dc8179e45f7c10416026acdadcbe5cbd8a78d37e`
- `35157bca7e34a36e1a398c1d419ce0473a109fd4`

Successful analyzer invocation used module mode:

```text
python -m scripts.analyze_reason_router_p3w6f1_deterministic_polarity_regeneration
```

The deterministic invocation/configuration inputs were supplied as canonical JSON payload strings, matching the analyzer's `parse_json_arg` contract.

Earlier failed launches were invocation ergonomics failures only:

1. direct script package import failure
2. JSON path passed where analyzer requires a JSON payload string

Neither failed launch produced final result artifacts. They are not scientific blockers.

Independent final analyzer gate:

`P3W6F1_AUTHORITATIVE_ANALYZER_INDEPENDENT_RESULT_GATE_PASS`

## Final Analyzer Artifact SHA Freeze

| artifact | SHA256 |
|---|---|
| `p3w6f1_full_output_isolation.json` | `4561ae365ae912ecf55a56a1a77bcf487342c44b18e6da9eb45ac2c376050e3b` |
| `p3w6f1_regenerated_rows.jsonl` | `62ec4840df8b494ed08d9020d3cdf24db47edca026cead058a3a9d03bd349325` |
| `p3w6f1_regeneration_audit.jsonl` | `62ec4840df8b494ed08d9020d3cdf24db47edca026cead058a3a9d03bd349325` |
| `p3w6f1_regeneration_summary.json` | `e99e991580329b3e7b6a48d9c5b0165ef01693699e049fe0a365a7b157d9f612` |
| `p3w6f1_stage185_predicate_realization_compatibility_provenance_manifest.json` | `31297c32655ecf25618a8a0ec030fae7f43e6d1815e8f86e09064b86ef330ebe` |
| `p3w6f1_stage185_predicate_realization_compatibility_report.md` | `dea1265714174e39864a582a1fa57e054e8ea33108a785839bf6c3cbc50552c9` |
| `p3w6f1_stage185_predicate_realization_compatibility_rows.csv` | `861d62c3b486e216832b2bdadd507e17b5a662b00bb29fa014b5d0042a3eae97` |
| `p3w6f1_stage185_predicate_realization_compatibility_rows.jsonl` | `16b1556ca036ceca471cef71c2c35e656a2db2f542a435dee7e1da933c271929` |
| `p3w6f1_stage185_predicate_realization_compatibility_summary.json` | `0d1ea255b3f1269b97fb94a8ba3b00262a9973d472aadc788ee519f24b284323` |

No runtime-generated analyzer artifacts are copied by this final documentation freeze.

## Post-Execution State

Kaggle returned successfully to:

- Branch: `main`
- HEAD: `35157bca7e34a36e1a398c1d419ce0473a109fd4`
- Gate: `P3W6F1_POST_ANALYZER_RETURN_TO_MATERIALIZER_HEAD_PASS`

## Final Result Review Authority

The completed independent Final Result Review found:

- authority chain PASS
- F1 target totality PASS
- causal repair scope PASS
- full-output isolation PASS
- F2 preservation PASS
- raw Stage185 preservation PASS
- compatibility legitimacy PASS
- provenance PASS
- Stage184 portability scientific non-change PASS
- analyzer invocation failures non-scientific
- analyzer accounting consistency PASS
- no remaining P3-W6-F1 blocker

Final decision:

`P3W6F1_FINAL_RESULT_REVIEW_PASS`

Final state:

`P3W6F1_CLOSED`

## Limitations And Non-Claims

P3-W6-F1 does not establish:

- F2 remediation
- general grammar normalization
- arbitrary predicate equivalence
- general semantic paraphrase equivalence
- model-training improvement
- Reason Router experimental benefit
- A1/A2/A3 comparative benefit
- generalization beyond the controlled P3-W6-F1 authority set

It closes only the exact deterministic F1 repair and its authority/provenance validation.

## Exact Next Research State

Closed:

`P3-W6-F1 deterministic F1 polarity repair`

Open:

`P3-W5 F2 manual remediation authority: 119 target pairs; 357 target members; MANUAL_REVIEW_REQUIRED; P3W5_F2_MANUAL_REVIEW_NOT_EXECUTED`

Separately possible after this freeze:

`return to the larger P3 Reason Router experimental path`

This report does not choose or execute either future path.
