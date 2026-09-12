# ContraMamba Gen4 Operator-Cell Generator-Structure Sidecar Materialization Result/Provenance Report - Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_OPERATOR_CELL_GENERATOR_STRUCTURE_SIDECAR_MATERIALIZATION_RESULT_PROVENANCE
- Result/provenance recording authority: 3128889f4f74f52bae3992f8d3877c74ed442b3c
- Materialization execution authority: aaefde1ad6a795b68a5e0e53b3e3d05ec2fd81d5
- Materializer implementation freeze: ac75e91ac281c984a2de0540491120847adff0c5
- Generator semantic source authority: 91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea
- Training performed: NO
- Evaluation performed: NO
- Model inference performed: NO
- Tokenizer execution performed: NO
- Kaggle used: NO
- Scientific conclusion established: NO

This report records only the already-completed and already-validated
Gen4 generator-structure sidecar materialization result and provenance.

It introduces no new scientific interpretation.

## 2. Authority chain

Generator semantic source authority:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

Gen4 materialization specification authority:

7a1bc41178c2db2ae5c6ee62ce93ed2d36db51f2

Gen4 implementation authority:

3f9e65c1c6dfe5abfd6c1eb7969037471bb76b33

Frozen materializer implementation:

ac75e91ac281c984a2de0540491120847adff0c5

Canonical sidecar materialization execution authority:

aaefde1ad6a795b68a5e0e53b3e3d05ec2fd81d5

Result/provenance recording authority:

3128889f4f74f52bae3992f8d3877c74ed442b3c

## 3. Canonical source identity

Exact source path:

reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl

Physical SHA256:

eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3

Total rows:

3600

Pre-execution target population validation established:

- entity_swap: 300
- role_swap: 300
- title_name_swap: 300
- predicate_swap: 300
- total target rows: 1200
- unique target row IDs: 1200
- duplicate target row IDs: 0
- unique represented pairs: 300

The same source physical SHA256 was observed before and after materialization.

SOURCE_IMMUTABILITY = YES

## 4. Canonical sidecar artifact identity

Exact artifact path:

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

## 5. Validated output population

Validated intervention counts:

- entity_swap: 300
- role_swap: 300
- title_name_swap: 300
- predicate_swap: 300

Total:

1200

No non-target intervention was accepted.

## 6. Validated schema

Every accepted sidecar row contained exactly these top-level fields in this
order:

1. schema_version
2. authority_commit
3. row_id
4. pair_id
5. intervention_type
6. intended_changed_axes
7. generator_source_fields
8. operator_cells

Required schema_version:

GEN4_OPERATOR_CELL_GENERATOR_STRUCTURE_V1

Required authority_commit:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

SCHEMA_VALID = YES

## 7. Validated generator-structure mappings

The validated mapping for entity_swap was:

- intended_changed_axes: ["name"]
- generator_source_fields: {"name":"alternate_name"}
- operator_cells: ["entity_swap:name"]

The validated mapping for role_swap was:

- intended_changed_axes: ["role"]
- generator_source_fields: {"role":"alternate_role"}
- operator_cells: ["role_swap:role"]

The validated mapping for title_name_swap was:

- intended_changed_axes: ["title","name"]
- generator_source_fields: {"title":"alternate_title","name":"alternate_name"}
- operator_cells: ["title_name_swap:title","title_name_swap:name"]

The validated mapping for predicate_swap was:

- intended_changed_axes: ["predicate"]
- generator_source_fields: {"predicate":"alternate_predicate"}
- operator_cells: ["predicate_swap:predicate"]

GENERATOR_MAPPING_VALID = YES

These mappings are generator-declared structural identities.

They were not reconstructed from rendered text.

## 8. Forbidden semantic provenance disposition

The sidecar did not use the following as semantic-cell provenance:

- claim
- evidence
- final_label
- frame_compatible_label
- predicate_covered_label
- sufficiency_label
- polarity_label
- primary_failure_type
- model predictions
- logits
- probabilities
- evaluator outputs
- training outcomes
- evaluation outcomes
- Stage182a observed_changed_axes
- Stage182a rendered-text-derived changed-axis booleans
- rendered-text matching
- tokenizer execution

The materializer consumed source-row identity fields for targeting and joining
and used the frozen generator mapping for structural semantics.

## 9. Determinism validation

After canonical materialization, the same frozen materializer and same
canonical source were used to materialize a temporary validation output.

The temporary output was compared byte-for-byte against the canonical sidecar.

Result:

BYTE_IDENTICAL = YES

DETERMINISM_VALIDATION = PASS

The temporary validation artifact was deleted after comparison.

No second canonical sidecar remains.

## 10. Execution disposition

IMPLEMENTATION_CORRECTNESS = PREVIOUSLY_ESTABLISHED

EXECUTION_SUCCESS = YES

ARTIFACT_PROVENANCE_VALID = YES

SOURCE_IMMUTABILITY = YES

DETERMINISTIC_SERIALIZATION = YES

TRAINING_EVALUATION = NOT_PERFORMED

SCIENTIFIC_CONCLUSION = NOT_ESTABLISHED

## 11. Artifact/provenance conclusion

The canonical Gen4 operator-cell generator-structure sidecar was successfully
materialized under the frozen execution authority.

Its artifact identity is:

e371aa7b2768ee1aa97b9a4e5479e246b2317a52011e00a75d9681bb92b2d913

The artifact contains exactly 1200 validated target rows from 300 source pairs.

Artifact/provenance validation is complete for this materialization result.

This conclusion concerns artifact identity and provenance only.

## 12. Scientific boundary

This report does not establish that:

- operator-cell identity predicts model behavior;
- any operator cell is causally active;
- any operator cell localizes a representation transition;
- any operator cell is mechanistically important;
- the sidecar improves model performance;
- any feature should be promoted;
- any model should be trained or evaluated.

No scientific feature-effect claim follows from successful materialization.

## 13. Repository-state boundary

At report creation time:

- the validated canonical sidecar remains untracked;
- this result/provenance report is created separately;
- no sidecar modification is authorized;
- no canonical materialization rerun is authorized;
- no training or evaluation is authorized.

The sidecar and this report require a separate manual commit review before any
staging, commit, or push.

## 14. Final disposition

GEN4_GENERATOR_STRUCTURE_MATERIALIZATION_EXECUTION = PASS

GEN4_GENERATOR_STRUCTURE_ARTIFACT_PROVENANCE = PASS

CANONICAL_SIDECAR_SHA256 = e371aa7b2768ee1aa97b9a4e5479e246b2317a52011e00a75d9681bb92b2d913

CANONICAL_SIDECAR_ROWS = 1200

CANONICAL_SOURCE_SHA256 = eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3

SCIENTIFIC_CONCLUSION = NOT_ESTABLISHED

TRAINING_EVALUATION = NOT_PERFORMED

## 15. Stop condition

Stop after this report is created and statically reviewed.

Do not rerun the canonical materializer.

Do not modify the validated sidecar.

Do not train.

Do not evaluate.

Do not run model inference.

Do not execute tokenizers.

Do not run Kaggle.

Do not stage, commit, or push until a separate manual result-artifact review.