# ContraMamba Gen4 Operator-Cell Generator-Structure Sidecar Materialization Execution Authority Specification — Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_OPERATOR_CELL_GENERATOR_STRUCTURE_SIDECAR_MATERIALIZATION_EXECUTION
- Implementation freeze commit: ac75e91ac281c984a2de0540491120847adff0c5
- Generator semantic source authority: 91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea
- Parent materialization authority: 7a1bc41178c2db2ae5c6ee62ce93ed2d36db51f2
- Implementation authority: 3f9e65c1c6dfe5abfd6c1eb7969037471bb76b33
- When this candidate is frozen, canonical sidecar materialization authorized: YES
- Training authorized: NO
- Evaluation authorized: NO
- Model inference authorized: NO
- Tokenizer execution authorized: NO
- Kaggle authorized: NO
- Automatic Commit/Push: NO

This authority permits one deterministic CPU-local materialization of the
frozen Gen4 generator-structure sidecar from the exact current-lineage
canonical source dataset defined below.

It does not authorize feature-effect testing, model execution, training,
evaluation, or scientific promotion.

## 2. Frozen implementation

The only authorized materializer is:

scripts/materialize_reason_router_gen4_operator_cell_generator_structure.py

Implementation identity:

ac75e91ac281c984a2de0540491120847adff0c5

The focused implementation tests at that commit passed:

20 passed

No alternative implementation is authorized for this materialization.

## 3. Canonical current-lineage source dataset

The source dataset is exactly:

reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl

Required physical SHA256:

eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3

Required total row count:

3600

This dataset identity is the current P4-L / P3-W7 lineage source identity.

Historical controlled_v5 datasets under data/ are not authorized substitutes.

Any source path, byte hash, or row-count mismatch blocks execution.

## 4. Pre-execution static identity evidence

The read-only source audit at implementation commit
ac75e91ac281c984a2de0540491120847adff0c5 established:

entity_swap = 300

role_swap = 300

title_name_swap = 300

predicate_swap = 300

Target total = 1200

Target unique row IDs = 1200

Target unique pairs = 300

Target duplicate row IDs = 0

The complete source contains 12 intervention families with 300 rows each.

No sidecar was generated during this audit.

## 5. Exact authorized target population

Only rows with intervention_type equal to one of the following may be emitted:

- entity_swap
- role_swap
- title_name_swap
- predicate_swap

Expected output counts are exactly:

entity_swap = 300

role_swap = 300

title_name_swap = 300

predicate_swap = 300

Total output rows = 1200

Unique output row IDs = 1200

Unique pair IDs represented = 300

Any deviation blocks artifact acceptance.

## 6. Generator semantic mapping

Semantic-cell identity remains fixed by generator authority:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

Exact mapping:

entity_swap:
  intended_changed_axes = ["name"]
  generator_source_fields = {"name":"alternate_name"}
  operator_cells = ["entity_swap:name"]

role_swap:
  intended_changed_axes = ["role"]
  generator_source_fields = {"role":"alternate_role"}
  operator_cells = ["role_swap:role"]

title_name_swap:
  intended_changed_axes = ["title","name"]
  generator_source_fields = {"title":"alternate_title","name":"alternate_name"}
  operator_cells = ["title_name_swap:title","title_name_swap:name"]

predicate_swap:
  intended_changed_axes = ["predicate"]
  generator_source_fields = {"predicate":"alternate_predicate"}
  operator_cells = ["predicate_swap:predicate"]

No rendered-text semantic reconstruction is authorized.

## 7. Authorized output artifact

The exact output path is:

reports/reason_router_gen4_operator_cell_generator_structure_materialization_ac75e91ac281c984a2de0540491120847adff0c5/gen4_operator_cell_generator_structure_sidecar.jsonl

The output directory must not contain any unrelated artifact.

The sidecar must remain separate from the canonical source dataset.

The source dataset must remain byte-identical.

## 8. Required output schema

Every sidecar row must contain exactly these top-level fields in this order:

1. schema_version
2. authority_commit
3. row_id
4. pair_id
5. intervention_type
6. intended_changed_axes
7. generator_source_fields
8. operator_cells

schema_version must equal:

GEN4_OPERATOR_CELL_GENERATOR_STRUCTURE_V1

authority_commit must equal:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

No claim, evidence, label, prediction, evaluator result, or observed semantic
state may appear in the output.

## 9. Explicitly forbidden provenance

The materialization and post-validation must not derive semantic-cell identity
from:

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
- Stage182a changed-axis booleans
- rendered-text matching
- tokenizer execution

Stage182a observed semantic reconstruction remains forbidden.

## 10. Exact execution semantics

After this execution authority candidate is independently reviewed, frozen,
committed, and pushed, one local CPU materialization may be performed.

The materializer must run from the frozen implementation commit.

The exact source path and output path from this document must be used.

The source physical SHA256 and row count must be verified immediately before
materialization.

No canonical source bytes may be modified.

No GPU is required.

Kaggle is not required.

## 11. Required post-materialization validation

Artifact acceptance requires all of the following:

- output exists at the exact authorized path;
- output row count is exactly 1200;
- entity_swap count is exactly 300;
- role_swap count is exactly 300;
- title_name_swap count is exactly 300;
- predicate_swap count is exactly 300;
- unique row_id count is exactly 1200;
- duplicate row_id count is zero;
- unique pair_id count is exactly 300;
- every schema_version is exact;
- every authority_commit is exact;
- every row contains exactly the authorized eight fields in fixed order;
- no non-target intervention appears;
- every operator mapping exactly matches frozen generator semantics;
- source dataset physical SHA256 remains unchanged after execution;
- deterministic rerun bytes are identical when validated through a temporary output;
- output physical SHA256 is recorded after validation.

A successful materializer exit alone is insufficient.

## 12. Determinism check

Post-validation must perform a second materialization to a temporary path outside
the committed artifact namespace.

The temporary result must be byte-identical to the authorized output.

The temporary file must be deleted after comparison.

This is validation of deterministic serialization only.

It does not constitute a second scientific artifact.

## 13. Execution/result separation

The following claims must remain separate:

1. implementation correctness;
2. materialization execution success;
3. sidecar artifact/provenance validity;
4. scientific interpretation.

This authority can establish only items 2 and 3 after successful execution and
validation.

It cannot establish predictive value, causal relevance, mechanistic localization,
or model-performance effects.

## 14. Expected execution delta

Successful execution is expected to create only:

reports/reason_router_gen4_operator_cell_generator_structure_materialization_ac75e91ac281c984a2de0540491120847adff0c5/gen4_operator_cell_generator_structure_sidecar.jsonl

No source dataset modification is authorized.

No implementation modification is authorized.

No test modification is authorized.

No other generated artifact is authorized during the execution itself.

A result/provenance report may be authorized separately after validated execution.

## 15. Fail-closed conditions

Execution must stop if:

- HEAD is not the frozen execution-authority commit eventually containing this exact candidate;
- the implementation file differs from the implementation frozen at ac75e91ac281c984a2de0540491120847adff0c5;
- the canonical source path is missing;
- source physical SHA256 differs from eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3;
- source row count differs from 3600;
- target counts differ from 300 per operator before execution;
- target total differs from 1200;
- duplicate target row IDs exist;
- the authorized output already exists unexpectedly;
- materialization attempts to modify source data;
- output validation fails;
- deterministic byte comparison fails.

Any such condition requires stopping without scientific interpretation.

## 16. Commit and push boundary

This candidate does not itself authorize commit or push of a generated sidecar.

After validated materialization:

- inspect artifact identity and physical SHA256;
- record provenance/result evidence under separate authority if required;
- review repository state;
- explicitly stage only authorized result artifacts;
- manually commit and push.

Do not use git add .

## 17. Scientific boundary

The sidecar represents only pre-existing generator-declared structural identity.

It does not establish:

- that operator-cell identity predicts model behavior;
- that a cell is causally active;
- that a cell localizes a representation transition;
- that any operator is scientifically superior;
- that any feature should be promoted;
- that any model should be trained or evaluated.

Those questions require later feature-design and scientific execution authority.

## 18. Current decision

CURRENT_LINEAGE_SOURCE_IDENTITY = ESTABLISHED

GEN4_TARGET_POPULATION_IDENTITY = ESTABLISHED

IMPLEMENTATION_CORRECTNESS = ESTABLISHED

CANONICAL_SIDECAR_MATERIALIZATION = READY_AFTER_AUTHORITY_FREEZE

TRAINING_EVALUATION = FORBIDDEN

KAGGLE = NOT_REQUIRED

## 19. Stop condition

Stop after this execution authority candidate is reviewed.

Do not run the materializer before this exact authority is frozen and pushed.

Do not generate the canonical sidecar yet.

Do not train.

Do not evaluate.

Do not run model inference.

Do not execute tokenizers.

Do not run Kaggle.