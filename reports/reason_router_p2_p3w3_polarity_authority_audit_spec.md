# P3-W3 Polarity Supervision Authority Audit Spec

## Static Review Status

`P3W3_POLARITY_AUTHORITY_V1_STATIC_REVIEW_BLOCKED`, `P3W3_POLARITY_AUTHORITY_V2_STATIC_REVIEW_BLOCKED`, and `P3W3_POLARITY_AUTHORITY_V3_STATIC_REVIEW_BLOCKED` have been addressed in this static implementation. The static implementation decision remains `P3W3_IMPLEMENTATION_READY_FOR_STATIC_REVIEW`.

V2 blockers recorded for truthfulness:

- Unknown intervention membership gate was declared but not applied.
- Narrow-repair reason-code and preserved-axis authority was not repository-backed.
- Production supervision audit counts were not retained.
- Pair eligibility was mislabeled as clean evidence.
- No-missing-REFUTE decision branch was incorrect.

V3 blockers recorded for truthfulness:

- Unsupported generator statuses were overclassified as independent defects.
- Runtime output retained an audit-not-executed blocker after execution.
- Mixed evidence classes did not always produce mixed remediation.
- Production-count equality test used row-derived self-comparison.
- Main control-flow order did not match documented fail-closed order.

## Observed P3-W2 Evidence

- Implementation authority commit: `e8124806dd5644c9713c0afd5cc9af8bc041eff4`
- Preserved calibration authority commit: `76d068cd9bc3b888d101e0cf2b7a3ded82578077`
- Resolved reason loss weight: `0.6518018402446165`
- Eligible polarity observed in P3-W2: `REFUTE 0 / SUPPORT 242`
- Normal A1/A3 training readiness: `false`
- A1/A3 released: `false`

## Production Helper Authority

The analyzer fail-closes identity and lineage first, then deep-copies authoritative train records and calls `scripts/train_controlled_v6b_minimal.py::_p2_prepare_reason_supervision_train_only` with `train_inputs={}`, `train_source_labels=["clean_main"] * n`, `require_min_counts=False`, `min_train_count=50`, and `torch.device("cpu")`.

Analyzer row authority comes only from production-written fields: `p2_primary_reason`, `p2_primary_reason_target_4`, `p2_reason_supervision_eligible`, `p2_reason_exclusion_codes`, `p2_frame_applicable`, `p2_predicate_applicable`, `p2_sufficiency_applicable`, `p2_polarity_applicable`, `p2_polarity_target_2`, `intervention_contract_pass`, and `generator_integrity_status`. The analyzer does not replay or reorder P2 exclusion semantics.

The runtime analyzer schema is `reason_router_p3w3_polarity_authority_audit_v3`. The production helper return value is retained as `production_supervision_audit`. Summary authority counts come from `train_reason_counts`, `target_class_counts.train_applicable_binary`, and `train_exclusion_counts`. Row-derived reconstruction is diagnostic and must exactly match those production counts or the analyzer fails closed with `P3W3_PRODUCTION_AUDIT_RECONSTRUCTION_MISMATCH`.

## Authority Inputs And Gates

- Data: `data/controlled_v5_v3_without_time_swap.jsonl`
- Data SHA-256: `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`
- Sidecar: `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl`
- Sidecar semantic SHA-256: `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`
- Split seed `174`, dev ratio `0.2`, train rows `2880`, dev rows `720`
- Ordered train identity `cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784`
- Intervention type authority: `scripts/build_controlled_v5.py::INTERVENTION_TYPES`, defined from `src/contramamba/labels.py::InterventionType` and enforced by `scripts/build_controlled_v5.py::validate_record`

Fail-closed gates include data SHA, sidecar semantic SHA, execution commit, dirty tracked tree, split count and identity, duplicate row ID, train/dev pair leakage, source metadata, sidecar row/pair/split identity, canonical lineage, exact binary labels, unknown final labels, empty intervention types, and unknown nonempty intervention types.

## Nested U1-U8 Funnel

- U1: directional final-label rows
- U2: U1 plus all three authorization axes pass
- U3: U2 plus raw polarity candidate contract
- U4: U3 plus source/sidecar row identity, pair identity, split, and canonical lineage
- U5: U4 plus no `P2_POLARITY_INTERVENTION_CONTRACT_FAIL`
- U6: U5 plus production normalized generator integrity status `CLEAN`
- U7: U6 plus production `p2_reason_supervision_eligible` and `p2_primary_reason == AUTHORIZED`
- U8: U7 plus production `p2_polarity_applicable` and target in `{0, 1}`

The analyzer asserts U1 >= U2 >= U3 >= U4 >= U5 >= U6 >= U7 >= U8 by label and row-id subset, and attrition may not be negative.

## Sidecar Interpretation Audit

Production functions and fields used in the summary are explicit:

- Semantic sidecar SHA: `scripts/train_controlled_v6b_minimal.py::_stage187_semantic_sidecar_sha256`
- Sidecar loader: `scripts/train_controlled_v6b_minimal.py::_p2_load_reason_integrity_sidecar`
- Production row authority: `scripts/train_controlled_v6b_minimal.py::_p2_prepare_reason_supervision_train_only`
- Generator normalizer: `scripts/train_controlled_v6b_minimal.py::_p2_normalized_generator_status`
- Canonical lineage: `scripts/train_controlled_v6b_minimal.py::_p2_resolve_canonical_lineage_for_split`
- Generator component fields: `schema_status`, `dataset_source_status`, `grammar_status`, `canonical_status`, `intervention_contract_status`, `polarity_contamination_status`, `time_swap_status`
- Diagnostic provenance fields: `reason_codes`, `audit_changed_axes`, `audit_expected_axes`, `audit_preserved_axes`, `generator_source_path`, `generator_source_sha256`, `integrity_builder_sha256`, `stage182a_report_sha256`, `stage184a_report_sha256`

Static repository review found diagnostic sidecar provenance fields and builder validation, but did not find an explicit repository-backed polarity-only mismatch proof contract defining accepted reason codes plus required preserved axes. Therefore `generator_evidence_proof_contract_available=false` and `generator_evidence_proof_contract_source=null` by default.

## Generator Evidence Classifier

The classifier is diagnostic only. It emits `PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY`, `INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE`, `AMBIGUOUS_INTEGRITY_EVIDENCE`, or `CLEAN`.

Component statuses are first partitioned using production status authority from `trainer.P2_GENERATOR_COMPONENT_STATUS_FIELDS`, `trainer.P2_GENERATOR_CLEAN_STATUSES`, and `trainer._p2_normalized_generator_status`. Exact `PASS` is clean; exact `FAIL` may be evidence; missing, `None`, non-string values, `UNKNOWN`, `UNRESOLVED`, `NOT_APPLICABLE`, and other unsupported statuses are `AMBIGUOUS_INTEGRITY_EVIDENCE`, not independent defects. Only exact `FAIL` in `schema_status`, `dataset_source_status`, `grammar_status`, `canonical_status`, `polarity_contamination_status`, or `time_swap_status` is independent generator or semantic defect evidence. `intervention_contract_status == FAIL` alone is ambiguous unless a repository-backed proof contract is available and exactly matched. Generator defect status alone is never sufficient for narrow repair.

`PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY` is possible only when a repository-backed proof authority is available and the row exactly matches its accepted reason codes, required preserved axes, identity/lineage, REFUTE final/polarity target, `polarity_flip`, and no-independent-defect clauses. Synthetic tests may inject an explicit proof authority object to exercise the pure classifier, but production default does not invent proof tokens.

Runtime output is distinct from this static spec. Executed summaries use `P3W3_AUDIT_EXECUTION_COMPLETE_PENDING_RESULT_REVIEW`, set `audit_execution_completed=true`, `result_static_review_completed=false`, `production_behavior_modified=false`, `polarity_supervision_released=false`, and `A1_A3_released=false`. Runtime remaining blockers are `P2_POLARITY_LOCAL_SUPERVISION_NOT_TRAINING_READY` and `P3W3_RESULT_STATIC_REVIEW_PENDING`; the runtime artifact must not retain `P3W3_POLARITY_AUTHORITY_AUDIT_NOT_EXECUTED`.

## Counterfactuals

C0-C4 operate on production exclusion codes and remain diagnostic only. C5 may admit only REFUTE rows satisfying axis authorization, REFUTE final/polarity match, `polarity_flip`, source/sidecar/split/canonical lineage validity, no production independent exclusion, and generator evidence class `PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY`. With no repository-backed proof authority, C5 admits zero generator-defect REFUTE rows. Newly admitted C5 rows include `generator_evidence_class`, `generator_evidence_reasons`, `proof_contract_available`, `proof_contract_source`, and `proof_contract_clause`.

## Pair-Level Canonical Comparison

Pair aggregation is by unique `pair_id`. It separately reports `refute_row_count`, `unique_refute_pair_count`, and `multi_refute_row_pair_count`. `pairs_with_eligible_canonical_SUPPORT_row` is eligibility-only. Evidence cleanliness is reported separately as `pairs_with_generator_evidence_clean_canonical_SUPPORT_row`, alongside REFUTE evidence classes.

## Provisional Decision Rules

Decision branch order is deterministic:

- `axis_authorized_refute_count == 0` emits `P3W3_NEW_REFUTE_AUTHORITY_REQUIRED`.
- If all axis-authorized REFUTE rows are already eligible and at least one eligible REFUTE exists, emit `P3W3_AUDIT_BLOCKED` with reason `polarity supervision blocker is not reproduced`.
- Only missing axis-authorized REFUTE rows are then classified for remediation.

Missing axis-authorized REFUTE evidence classes map to remediation classes: `PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY` to `NARROW_CONTRACT_REPAIR`, `AMBIGUOUS_INTEGRITY_EVIDENCE` to `INTEGRITY_REANNOTATION`, and `INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE` to `NEW_REFUTE_AUTHORITY`. More than one remediation class emits `P3W3_MIXED_REMEDIATION_REQUIRED`. A `CLEAN` missing REFUTE emits `P3W3_AUDIT_BLOCKED` because it is inconsistent with production eligibility. Single-class decisions emit the corresponding narrow repair, integrity reannotation, or new authority decision. Human static review remains required.

## Forbidden Claims

This static implementation does not claim `P3W3_AUDIT_PASS`, `P2_POLARITY_SUPERVISION_RESOLVED`, `P2_INTEGRITY_GATE_DEFECTIVE`, `P2_REFUTE_ROWS_RECOVERABLE`, `NEW_REFUTE_DATA_REQUIRED`, `A1_READY`, `A2_READY`, `A3_READY`, or `P3_PASS`.

## Remaining Blockers

- `P2_POLARITY_LOCAL_SUPERVISION_NOT_TRAINING_READY`
- `P3W3_POLARITY_AUTHORITY_AUDIT_NOT_EXECUTED`
- `P3W3_STATIC_REVIEW_PENDING`
