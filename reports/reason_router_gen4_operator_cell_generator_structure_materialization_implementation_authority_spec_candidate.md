# ContraMamba Gen4 Operator-Cell Generator-Structure Materialization Implementation Authority Specification — Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_OPERATOR_CELL_GENERATOR_STRUCTURE_MATERIALIZATION_IMPLEMENTATION
- Parent frozen authority: 7a1bc41178c2db2ae5c6ee62ce93ed2d36db51f2
- Generator semantic source authority: 91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea
- When this candidate is frozen, implementation authorized: YES
- Scientific artifact materialization authorized: NO
- Training authorized: NO
- Evaluation authorized: NO
- Kaggle authorized: NO
- Commit/Push of implementation: NO until separate manual review

This authority permits only the bounded implementation and unit validation needed
to encode the already-frozen generator-side operator-cell mapping.

It does not authorize generation of a scientific sidecar from the canonical
controlled dataset.

## 2. Parent authority

The parent authority is:

7a1bc41178c2db2ae5c6ee62ce93ed2d36db51f2

That authority established:

DIRECT_FEATURE_DESIGN = BLOCKED

STAGE182A_OBSERVED_AXIS_REUSE = FORBIDDEN

MINIMAL_GENERATOR_STRUCTURE_MATERIALIZATION = REQUIRED

Those decisions remain unchanged.

## 3. Scientific source-of-truth

Generator semantic structure is frozen at:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea:scripts/build_controlled_v5.py

The authorized operator-to-structure mapping is exactly:

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

No runtime semantic inference is authorized.

## 4. Exact implementation scope

Exactly two implementation files are authorized:

scripts/materialize_reason_router_gen4_operator_cell_generator_structure.py

tests/test_materialize_reason_router_gen4_operator_cell_generator_structure.py

No other implementation file may be created or modified.

The parent authority report must not be modified.

The frozen generator must not be modified.

Canonical controlled datasets must not be modified.

Existing Stage182a code or artifacts must not be modified.

## 5. Required implementation behavior

The materializer implementation must provide deterministic logic that can later,
under a separate execution authority, read a canonical controlled JSONL dataset
and emit a separate Gen4 generator-structure sidecar.

The implementation may consult only these input-row fields for materialization:

- id
- pair_id
- intervention_type

The implementation may parse a JSON object containing other fields because the
canonical source rows contain them, but it MUST NOT inspect, branch on, compare,
copy, derive from, or otherwise semantically consult those additional fields.

Target rows are only rows whose intervention_type is one of:

- entity_swap
- role_swap
- title_name_swap
- predicate_swap

Non-target intervention rows must not be emitted into the sidecar.

Their semantic content must not be inspected.

## 6. Required output schema

Each future materialized row must contain exactly these fields in this fixed
top-level order:

1. schema_version
2. authority_commit
3. row_id
4. pair_id
5. intervention_type
6. intended_changed_axes
7. generator_source_fields
8. operator_cells

schema_version must be a fixed implementation constant:

GEN4_OPERATOR_CELL_GENERATOR_STRUCTURE_V1

authority_commit must be exactly:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

row_id must equal the source row id.

pair_id must equal the source row pair_id.

intervention_type must equal the source row intervention_type.

## 7. Canonical semantic-axis order

The canonical semantic-axis order is:

1. title
2. name
3. role
4. predicate

Any multi-axis list or mapping must follow this order.

Therefore title_name_swap must serialize as:

intended_changed_axes = ["title","name"]

generator_source_fields:
  title -> alternate_title
  name -> alternate_name

operator_cells:
  title_name_swap:title
  title_name_swap:name

Alphabetical reordering of that mapping is not authorized if it changes this
canonical axis order.

## 8. Determinism requirements

The implementation must produce deterministic output for identical identity
inputs.

Target source-row order must be preserved.

Serialization behavior must be explicitly fixed by the implementation.

The implementation must not depend on:

- filesystem enumeration order
- dictionary set iteration
- randomized hashing
- model state
- tokenizer state
- network state
- current date or time
- environment-dependent semantic inference

The implementation must not execute or import the historical generator merely to
rediscover the mapping.

The frozen mapping is already established by authority and should be encoded
directly with its provenance constant.

## 9. Forbidden semantic inputs

The implementation MUST NOT use any of the following to construct or validate
semantic cell identity:

- claim
- evidence
- final_label
- frame_compatible_label
- predicate_covered_label
- sufficiency_label
- polarity_label
- primary_failure_type
- model outputs
- model predictions
- logits
- probabilities
- evaluator outputs
- error classifications
- training outcomes
- evaluation outcomes
- Stage182a observed_changed_axes
- Stage182a entity_changed
- Stage182a title_name_changed
- Stage182a role_changed
- Stage182a predicate_changed
- any semantic value reconstructed from rendered text
- tokenizer-derived semantic reconstruction

No text matching against claim or evidence is permitted.

## 10. Fail-closed requirements

The implementation must fail closed for malformed target rows if:

- id is absent;
- id is empty or not a string;
- pair_id is absent;
- pair_id is empty or not a string;
- intervention_type is absent;
- intervention_type is not a string;
- a target operator lacks an authorized frozen mapping;
- duplicate target row ids are encountered;
- deterministic serialization cannot be guaranteed;
- an output row would contain a field outside the authorized schema.

The implementation must also fail closed if an attempt is made to configure a
different generator semantic authority commit.

The source authority commit is not a user-selectable scientific parameter.

## 11. Non-target rows

The canonical controlled dataset contains intervention types outside the four
Gen4 target operators.

Those rows are allowed to exist in the input dataset.

They must be skipped without semantic interpretation.

Their claim, evidence, labels, or other semantic fields must not be inspected.

The presence of a valid non-target intervention is not itself an error.

## 12. Implementation interface

The implementation should separate pure mapping logic from file I/O so the
scientific boundary can be tested without producing a repository artifact.

At minimum, the module should expose testable logic equivalent to:

- identify whether an intervention is a target operator;
- map an authorized intervention to its frozen structural specification;
- transform a source identity row into the exact sidecar row;
- transform an ordered sequence of source rows while preserving order;
- serialize deterministic JSONL.

Exact Python function names are implementation details and are not scientific
authority.

A CLI may be implemented for later use, but invoking it on the canonical
scientific dataset is NOT authorized in this phase.

## 13. Unit-test requirements

The focused test module must verify at least:

1. exact mapping for entity_swap;
2. exact mapping for role_swap;
3. exact two-cell mapping and canonical order for title_name_swap;
4. exact mapping for predicate_swap;
5. exact authority_commit value;
6. exact schema_version value;
7. exact output field set and field order;
8. preservation of source target-row order;
9. exclusion of non-target interventions;
10. duplicate target row-id rejection;
11. malformed target identity rejection;
12. deterministic repeated serialization;
13. forbidden semantic fields cannot alter output.

For the forbidden-field test, two synthetic rows with identical:

- id
- pair_id
- intervention_type

but contradictory or arbitrary values in fields such as claim, evidence and
labels must produce byte-identical materialized output.

This test is intended to demonstrate outcome-blind and text-blind behavior.

Tests must use synthetic or temporary inputs.

Tests must not materialize the canonical scientific dataset.

## 14. Validation authority

Only implementation validation is authorized.

Required validation commands are:

python -m py_compile scripts/materialize_reason_router_gen4_operator_cell_generator_structure.py tests/test_materialize_reason_router_gen4_operator_cell_generator_structure.py

python -m pytest tests/test_materialize_reason_router_gen4_operator_cell_generator_structure.py -q

These commands may exercise synthetic or temporary test data only.

Passing these validations establishes code correctness only.

It does not establish scientific execution success, artifact validity, or a
scientific conclusion.

## 15. Explicitly unauthorized actions

This authority does NOT permit:

- running the materializer against data/controlled_v5_seed.jsonl;
- running it against data/controlled_v5_v1.jsonl;
- running it against data/controlled_v5_v2.jsonl;
- running it against data/controlled_v5_v3.jsonl;
- running it against data/controlled_v5_v3_without_time_swap.jsonl;
- generating a repository Gen4 sidecar artifact;
- modifying canonical controlled data;
- using Stage182a observed fields as input;
- training;
- evaluation;
- hyperparameter search;
- model inference;
- tokenizer execution;
- Kaggle execution;
- feature-effect analysis;
- scientific promotion.

A later execution authority is required before canonical-data materialization.

## 16. Expected delta

After this authority itself is frozen, the complete authorized implementation
delta is exactly:

A  scripts/materialize_reason_router_gen4_operator_cell_generator_structure.py
A  tests/test_materialize_reason_router_gen4_operator_cell_generator_structure.py

No report modification is part of the implementation delta.

No generated sidecar is part of the implementation delta.

## 17. Commit and push boundary

Implementation commit/push is not automatically authorized by implementation
test PASS.

After implementation validation, staged scope must be reviewed separately.

Only the two authorized implementation files may be staged.

The user performs explicit staging, commit, and push.

Do not use git add .

## 18. Stop conditions

Stop implementation immediately if:

- the parent authority cannot be resolved to
  7a1bc41178c2db2ae5c6ee62ce93ed2d36db51f2;
- the frozen generator semantic source cannot be resolved to
  91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea;
- implementing the mapping requires claim/evidence inspection;
- implementing the mapping requires labels or outcome information;
- implementation requires modifying an existing scientific artifact;
- implementation requires more than the two authorized files;
- tests require canonical scientific artifact generation.

Any such condition requires returning to authority review.

## 19. Completion boundary

This implementation phase is complete only when:

- the materializer code exists;
- the focused tests exist;
- py_compile passes;
- the focused pytest module passes;
- no canonical scientific sidecar has been generated;
- no training or evaluation has occurred;
- the implementation delta remains exactly two files.

At that point stop for manual commit review.

A separate frozen execution authority is required before materializing the
canonical Gen4 generator-structure sidecar.