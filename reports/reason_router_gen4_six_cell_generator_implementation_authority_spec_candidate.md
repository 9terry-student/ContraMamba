# ContraMamba Gen4 Six-Cell Generator Implementation Authority Specification - Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_SIX_CELL_GENERATOR_IMPLEMENTATION_AUTHORITY
- Implementation specification authority: 40d735d83018c7fc9b086226e95d1e46b141b0ab
- Source-structure feasibility authority: c90c1eda64560875232a3c5c1c33ef21886b2e60
- Contrast specification authority: 0a0da5354782e542520fb5bba146ab1a599d17ef
- Frozen generator semantic authority: 91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea
- Training/Evaluation allowed: NO
- Canonical materialization allowed: NO
- Kaggle allowed: NO
- Commit/Push: NO

This document is the bounded implementation authority for the Gen4 six-cell
masked-slot-substitution materializer.

It authorizes code construction and focused synthetic validation only after this
authority itself is reviewed, committed, and frozen.

## 2. Goal

Implement the already-frozen Gen4 six-cell generator contract exactly.

The implementation must encode:

- one held-constant mechanism;
- six canonical contrast cells;
- generator-declared semantic masks;
- same-renderer sham behavior;
- deterministic structural provenance;
- deterministic serialization;
- fail-closed validation.

This authority does not authorize scientific execution.

## 3. Exact authorized implementation delta

Exactly two implementation files are authorized:

scripts/materialize_reason_router_gen4_six_cell_contrast.py

tests/test_materialize_reason_router_gen4_six_cell_contrast.py

Therefore:

AUTHORIZED_IMPLEMENTATION_FILE_COUNT = 2

No third implementation file is authorized.

## 4. Explicitly protected files

The following must not be modified:

scripts/build_controlled_v5.py

All existing model code.

All existing loss code.

All tokenizer code.

All existing canonical controlled datasets.

The frozen Gen4 operator-cell sidecar.

The frozen contrast specification.

The frozen feasibility report.

The frozen implementation specification.

Therefore:

MODIFY_BUILD_CONTROLLED_V5 = FORBIDDEN

EXISTING_CANONICAL_ARTIFACT_MODIFICATION = FORBIDDEN

## 5. Governing implementation contract

The implementation must conform exactly to:

reports/reason_router_gen4_six_cell_generator_implementation_specification_candidate.md

frozen at:

40d735d83018c7fc9b086226e95d1e46b141b0ab

The authority does not permit reinterpretation of that specification.

If implementation convenience conflicts with the frozen specification:

IMPLEMENTATION_MUST_STOP = YES

## 6. Mechanism contract

Exact mechanism_id:

masked_slot_substitution_v1

Exact schema_version:

GEN4_SIX_CELL_MASKED_SLOT_SUBSTITUTION_V1

Exact canonical cell order:

1. C0_SHAM
2. C1_TITLE
3. C2_NAME
4. C3_ROLE
5. C4_PREDICATE
6. C5_TITLE_NAME

Exact axis order:

1. title
2. name
3. role
4. predicate

No additional cell or axis is authorized.

## 7. Source contract

Canonical structured source semantics remain frozen to:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

with source blob:

baee23a9f71333125f4a8735c2c92d20cab7eb4f

The implementation may consume generator-side structured facts.

It must not infer semantic identity from:

- rendered claim;
- rendered evidence;
- historical JSONL text;
- labels;
- model output;
- tokenizer output.

Therefore:

STRUCTURED_FACT_SOURCE_ONLY = REQUIRED

RENDERED_TEXT_SEMANTIC_RECONSTRUCTION = FORBIDDEN

## 8. Implementation behavior authorized

The implementation file may define:

- frozen constants;
- canonical cell definitions;
- structured-fact validation;
- deterministic row-id construction;
- six-cell block construction;
- structural provenance construction;
- deterministic JSONL serialization;
- canonical source-identity verification helpers;
- a thin CLI;
- fail-closed validation.

The implementation may import or load the frozen generator-side structured-fact
functions required by the specification.

It may not mutate that frozen generator source.

## 9. Test behavior authorized

The focused test file may create synthetic structured facts.

Synthetic tests may:

- construct one or more in-memory source facts;
- materialize six-cell blocks in memory;
- write temporary JSONL files under pytest temporary directories;
- read those temporary files back;
- intentionally provide malformed synthetic source facts;
- assert deterministic failure behavior.

Synthetic tests must not require canonical 300-pair generation.

Therefore:

FOCUSED_TESTS_SYNTHETIC_ONLY = REQUIRED

CANONICAL_300_PAIR_TEST_EXECUTION = FORBIDDEN

## 10. Required implementation coverage

Focused validation must cover at least:

- exact schema_version;
- exact mechanism_id;
- exact six cell IDs;
- exact cell ordering;
- exact masks;
- exact intended_changed_axes;
- exact generator_source_fields;
- deterministic row IDs;
- claim invariance;
- same-renderer empty-mask sham;
- title-only substitution;
- name-only substitution;
- role-only substitution;
- predicate-only substitution;
- title+name substitution;
- object/time/location invariance;
- within-pair alternate-value freezing;
- complete six-row block;
- duplicate source-pair rejection;
- duplicate generated-row rejection;
- missing required source-field rejection;
- non-string required source-field rejection;
- empty required source-field rejection;
- original/alternate equality rejection;
- structural label-field exclusion;
- source order preservation;
- deterministic serialization;
- rendered text not defining cell identity.

## 11. Canonical execution prohibition

This authority does not authorize running the materializer against:

fact_templates_for_count(300)

for the purpose of creating the canonical six-cell artifact.

It does not authorize writing the expected 1800-row artifact.

It does not authorize selecting a final canonical output path.

Therefore:

CANONICAL_SOURCE_PAIR_COUNT_EXPECTATION = 300

EXPECTED_CANONICAL_CONTRAST_ROW_COUNT = 1800

CANONICAL_300_PAIR_MATERIALIZATION = NOT_AUTHORIZED

CANONICAL_1800_ROW_ARTIFACT_CREATION = NOT_AUTHORIZED

The counts above are specification expectations, not execution permission.

## 12. No outcome-bearing execution

The implementation must not:

- attach labels;
- derive labels;
- inspect predictions;
- inspect logits;
- inspect probabilities;
- run evaluators;
- run a model;
- run a tokenizer;
- train;
- evaluate.

Therefore:

STRUCTURAL_ARTIFACT_LABEL_FIELDS = FORBIDDEN

MODEL_EXECUTION = NOT_AUTHORIZED

TOKENIZER_EXECUTION = NOT_AUTHORIZED

TRAINING_EVALUATION = NOT_AUTHORIZED

## 13. Exact validation authorized

After implementation, the following local CPU validation is authorized:

python -m py_compile scripts/materialize_reason_router_gen4_six_cell_contrast.py tests/test_materialize_reason_router_gen4_six_cell_contrast.py

and:

python -m pytest -q tests/test_materialize_reason_router_gen4_six_cell_contrast.py

No broader test suite is required by this authority.

No canonical dataset execution is part of validation.

## 14. Validation interpretation

Passing syntax compilation and focused pytest establishes only:

IMPLEMENTATION_CORRECTNESS_CANDIDATE = PASS

It does not establish:

- canonical execution success;
- canonical artifact identity;
- canonical artifact provenance;
- scientific outcome;
- predictive effect;
- causal effect;
- mechanistic effect;
- feature usefulness.

## 15. Working-tree boundary

Implementation work must begin from the authority freeze commit.

Only the exact two authorized implementation files may be created or modified.

Unrelated existing untracked files must remain untouched.

Do not clean or reset the worktree.

Do not stage unrelated files.

Do not use:

git add .

## 16. Commit and push boundary

This authority does not authorize automated or implicit commit/push.

After implementation and focused validation:

- stop;
- inspect exact working-tree delta;
- verify exact two-file scope;
- verify hashes;
- perform manual commit review.

Therefore:

IMPLEMENTATION_COMMIT_PUSH = NOT_AUTHORIZED_BY_THIS_DOCUMENT

## 17. Canonical materialization boundary

Even after implementation correctness is validated and the implementation is
frozen, canonical six-cell generation remains blocked.

A later separate authority must explicitly freeze:

- implementation commit identity;
- exact implementation file hashes;
- exact canonical output path;
- frozen generator authority identity;
- expected 300 source pairs;
- expected 1800 rows;
- provenance checks;
- deterministic rerun checks.

Therefore the later execution object is:

GEN4_SIX_CELL_CANONICAL_MATERIALIZATION_EXECUTION_AUTHORITY

## 18. Stop conditions

Implementation must stop immediately if:

- HEAD does not match the frozen implementation authority start point;
- the frozen implementation specification cannot be verified;
- scripts/build_controlled_v5.py has changed;
- implementation requires modifying a third file;
- canonical generation appears necessary for focused validation;
- rendered-text parsing appears necessary for semantic identity;
- label semantics appear necessary for structural generation;
- focused tests cannot validate the contract synthetically.

No scope expansion is authorized.

## 19. Required implementation report

After implementation and focused validation, the returned evidence must include:

- exact HEAD used;
- exact changed-file list;
- syntax compilation result;
- focused pytest result;
- implementation file SHA256;
- test file SHA256;
- confirmation that scripts/build_controlled_v5.py remained unchanged;
- confirmation that no canonical six-cell artifact was created;
- confirmation that no model/tokenizer/training/evaluation execution occurred.

## 20. Current decision

IMPLEMENTATION_SPECIFICATION = FROZEN

GEN4_SIX_CELL_GENERATOR_IMPLEMENTATION_AUTHORITY = CANDIDATE

AUTHORIZED_IMPLEMENTATION_FILE_COUNT = 2

FOCUSED_TESTS_SYNTHETIC_ONLY = REQUIRED

CANONICAL_300_PAIR_TEST_EXECUTION = FORBIDDEN

CANONICAL_300_PAIR_MATERIALIZATION = NOT_AUTHORIZED

CANONICAL_1800_ROW_ARTIFACT_CREATION = NOT_AUTHORIZED

STRUCTURAL_ARTIFACT_LABEL_FIELDS = FORBIDDEN

MODEL_EXECUTION = NOT_AUTHORIZED

TOKENIZER_EXECUTION = NOT_AUTHORIZED

TRAINING_EVALUATION = NOT_AUTHORIZED

IMPLEMENTATION_COMMIT_PUSH = NOT_AUTHORIZED_BY_THIS_DOCUMENT

## 21. Next boundary

If this authority candidate is reviewed and frozen:

IMPLEMENTATION = AUTHORIZED_FOR_EXACT_TWO_FILES

The authorized implementation delta will then be exactly:

scripts/materialize_reason_router_gen4_six_cell_contrast.py

tests/test_materialize_reason_router_gen4_six_cell_contrast.py

with focused synthetic validation only.

Canonical materialization remains blocked.

## 22. Stop condition for this authority phase

Stop after this implementation authority candidate is created and reviewed.

Do not create either implementation file yet.

Do not run focused pytest yet.

Do not generate canonical contrast rows.

Do not create an 1800-row artifact.

Do not train.

Do not evaluate.

Do not run model inference.

Do not execute tokenizers.

Do not run Kaggle.