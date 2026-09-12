# ContraMamba Gen4 Operator-Cell Generator-Structure Sidecar Materialization Result/Provenance Recording Authority Specification — Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_OPERATOR_CELL_GENERATOR_STRUCTURE_SIDECAR_RESULT_PROVENANCE_RECORDING
- Frozen execution authority: aaefde1ad6a795b68a5e0e53b3e3d05ec2fd81d5
- Frozen implementation commit: ac75e91ac281c984a2de0540491120847adff0c5
- Generator semantic source authority: 91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea
- When this candidate is frozen, result/provenance report creation authorized: YES
- New materialization execution authorized: NO
- Sidecar modification authorized: NO
- Training authorized: NO
- Evaluation authorized: NO
- Model inference authorized: NO
- Tokenizer execution authorized: NO
- Kaggle authorized: NO
- Commit/Push of result artifacts: NO until separate manual review

This authority permits only static recording of the already-completed,
already-validated Gen4 sidecar materialization result and provenance.

It does not authorize rerunning the canonical scientific materialization.

## 2. Completed execution identity

The execution was authorized by:

aaefde1ad6a795b68a5e0e53b3e3d05ec2fd81d5

The materializer implementation identity was:

ac75e91ac281c984a2de0540491120847adff0c5

The generator semantic authority embedded in sidecar rows is:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

## 3. Canonical source identity

Exact source path:

reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl

Physical SHA256:

eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3

Total source rows:

3600

The execution confirmed that source bytes were identical before and after
materialization.

## 4. Canonical sidecar identity

Exact sidecar path:

reports/reason_router_gen4_operator_cell_generator_structure_materialization_ac75e91ac281c984a2de0540491120847adff0c5/gen4_operator_cell_generator_structure_sidecar.jsonl

Physical SHA256:

e371aa7b2768ee1aa97b9a4e5479e246b2317a52011e00a75d9681bb92b2d913

Physical byte size:

454540

Output rows:

1200

Unique output row IDs:

1200

Duplicate output row IDs:

0

Unique represented pair IDs:

300

## 5. Validated operator counts

entity_swap = 300

role_swap = 300

title_name_swap = 300

predicate_swap = 300

Total = 1200

No non-target intervention was accepted.

## 6. Validated schema/provenance contract

Every accepted sidecar row was validated to contain exactly these fields in
this order:

1. schema_version
2. authority_commit
3. row_id
4. pair_id
5. intervention_type
6. intended_changed_axes
7. generator_source_fields
8. operator_cells

Every schema_version was validated as:

GEN4_OPERATOR_CELL_GENERATOR_STRUCTURE_V1

Every authority_commit was validated as:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

The four operator mappings were validated against frozen generator semantics.

## 7. Determinism evidence

A second materialization was performed only to a temporary validation path.

The temporary bytes were compared against the canonical sidecar.

Validation result:

BYTE_IDENTICAL = YES

DETERMINISM_VALIDATION = PASS

The temporary file was deleted after comparison.

No second canonical artifact remains.

## 8. Execution validation disposition

The completed execution established:

IMPLEMENTATION_CORRECTNESS = PREVIOUSLY_ESTABLISHED

EXECUTION_SUCCESS = YES

ARTIFACT_PROVENANCE_VALID = YES

SOURCE_IMMUTABILITY = YES

DETERMINISTIC_SERIALIZATION = YES

TRAINING_EVALUATION = NOT_PERFORMED

SCIENTIFIC_CONCLUSION = NOT_ESTABLISHED

## 9. Scientific boundary

This result does not establish:

- predictive usefulness of operator cells;
- causal relevance;
- representation-transition localization;
- mechanistic importance;
- performance improvement;
- feature promotion;
- any training or evaluation claim.

The artifact records frozen generator-declared structural identity only.

## 10. Authorized report delta after freeze

After this authority candidate itself is frozen and pushed, exactly one
result/provenance report may be created:

reports/reason_router_gen4_operator_cell_generator_structure_sidecar_materialization_result_provenance_report_candidate.md

That report may record only the already-validated identities and dispositions
listed in this authority.

It may not:

- rerun canonical materialization;
- modify the sidecar;
- modify the source dataset;
- add new scientific interpretation;
- train or evaluate a model;
- run inference;
- execute a tokenizer;
- use Kaggle.

## 11. Commit boundary

This authority candidate does not authorize staging or committing the sidecar
or future result/provenance report.

After the result/provenance report is created and statically reviewed, perform
a separate manual commit review over only the explicitly authorized result
artifacts.

Do not use git add .

## 12. Stop condition

Stop after this authority candidate is created and reviewed.

Do not create the result/provenance report until this authority is frozen and
pushed.

Do not rerun the canonical materializer.

Do not modify the validated sidecar.

Do not train.

Do not evaluate.

Do not run model inference.

Do not execute tokenizers.

Do not run Kaggle.