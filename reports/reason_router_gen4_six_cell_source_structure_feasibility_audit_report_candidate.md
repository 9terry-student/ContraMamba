# ContraMamba Gen4 Six-Cell Source-Structure Feasibility Audit Report - Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_SIX_CELL_SOURCE_STRUCTURE_FEASIBILITY_AUDIT
- Contrast specification authority: 0a0da5354782e542520fb5bba146ab1a599d17ef
- Frozen generator semantic authority: 91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea
- Frozen Gen4 sidecar SHA256: e371aa7b2768ee1aa97b9a4e5479e246b2317a52011e00a75d9681bb92b2d913
- Dataset generation performed: NO
- Contrast-row generation performed: NO
- Generator implementation performed: NO
- Model feature implementation performed: NO
- Training performed: NO
- Evaluation performed: NO
- Model inference performed: NO
- Tokenizer execution performed: NO
- Kaggle used: NO

This report records the completed read-only source-structure feasibility audit
authorized by the frozen Gen4 identifiability-breaking contrast specification.

It does not authorize implementation or execution.

## 2. Scientific question

The audit asked whether the frozen generator-side structured facts contain
sufficient provenance-valid source fields to construct the minimum six-cell
within-mechanism contrast without reconstructing semantic state from rendered
text.

The required cells are:

- C0_SHAM
- C1_TITLE
- C2_NAME
- C3_ROLE
- C4_PREDICATE
- C5_TITLE_NAME

The audit did not generate any of these rows.

## 3. Frozen generator identity

Generator semantic authority:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

Generator source:

scripts/build_controlled_v5.py

Frozen/current Git blob identity observed during audit:

baee23a9f71333125f4a8735c2c92d20cab7eb4f

Therefore:

FROZEN_GENERATOR_SOURCE_IDENTITY = PASS

## 4. Structured source population

The generator-side source audit evaluated:

STRUCTURED_FACT_COUNT = 300

STATIC_FACT_TEMPLATE_COUNT = 30

GENERATED_FACT_TEMPLATE_COUNT = 270

UNIQUE_STRUCTURED_PAIR_IDS = 300

The frozen Gen4 sidecar contains:

FROZEN_SIDECAR_PAIR_IDS = 300

The universes matched exactly:

PAIR_UNIVERSE_EXACT_MATCH = YES

No sidecar-only pair was missing from the generator structured facts.

No extra generator pair existed outside the frozen sidecar pair universe.

## 5. Required original source fields

Every one of the 300 structured source records directly contains a non-empty
string value for:

- title
- name
- role
- predicate

Therefore:

ORIGINAL_TITLE_AVAILABLE_ALL_300 = YES

ORIGINAL_NAME_AVAILABLE_ALL_300 = YES

ORIGINAL_ROLE_AVAILABLE_ALL_300 = YES

ORIGINAL_PREDICATE_AVAILABLE_ALL_300 = YES

These fields are generator-side structured values.

They were not inferred from rendered text.

## 6. Required alternate source fields

Every one of the 300 structured source records directly contains a non-empty
string value for:

- alternate_title
- alternate_name
- alternate_role
- alternate_predicate

Therefore:

ALTERNATE_TITLE_AVAILABLE_ALL_300 = YES

ALTERNATE_NAME_AVAILABLE_ALL_300 = YES

ALTERNATE_ROLE_AVAILABLE_ALL_300 = YES

ALTERNATE_PREDICATE_AVAILABLE_ALL_300 = YES

These are also generator-side structured values.

## 7. Original-versus-alternate distinctness

The audit verified across all 300 source pairs:

TITLE_ALTERNATE_DISTINCT_ALL_300 = YES

NAME_ALTERNATE_DISTINCT_ALL_300 = YES

ROLE_ALTERNATE_DISTINCT_ALL_300 = YES

PREDICATE_ALTERNATE_DISTINCT_ALL_300 = YES

Therefore no required contrast axis is structurally blocked by an
original/alternate equality in the current 300-pair source universe.

## 8. Text-blind provenance result

The audit did not inspect rendered claim or evidence strings to determine
semantic structure.

It did not inspect labels or model outcomes.

Therefore:

SOURCE_STRUCTURE_TEXT_INFERENCE_REQUIRED = NO

RENDERED_TEXT_INSPECTED = NO

LABELS_OR_OUTCOMES_INSPECTED = NO

The required semantic source identity exists before rendering and before model
outcome inspection.

## 9. Six-cell source feasibility

Given the directly available original and alternate fields, the source structure
can support the following masks without semantic reconstruction:

C0_SHAM:

(0,0,0,0)

C1_TITLE:

(1,0,0,0)

C2_NAME:

(0,1,0,0)

C3_ROLE:

(0,0,1,0)

C4_PREDICATE:

(0,0,0,1)

C5_TITLE_NAME:

(1,1,0,0)

The completed audit established:

TITLE_ONLY_SOURCE_STRUCTURALLY_FEASIBLE = YES

NAME_ONLY_SOURCE_STRUCTURALLY_FEASIBLE = YES

ROLE_ONLY_SOURCE_STRUCTURALLY_FEASIBLE = YES

PREDICATE_ONLY_SOURCE_STRUCTURALLY_FEASIBLE = YES

TITLE_NAME_JOINT_SOURCE_STRUCTURALLY_FEASIBLE = YES

EMPTY_MASK_SHAM_SOURCE_STRUCTURALLY_FEASIBLE = YES

Therefore:

COMPLETE_SIX_CELL_SOURCE_STRUCTURE_FEASIBLE = YES

## 10. Within-pair alternate-value control

All required alternate fields coexist in the same structured fact record for
each source pair.

Therefore a future implementation can, in principle, freeze one set of
alternate values per source pair and select among them only through the
axis mask.

The audit established:

WITHIN_PAIR_ALTERNATE_VALUES_CAN_BE_FIXED = YES

This is a source-structure feasibility conclusion only.

A later implementation must still enforce this property explicitly.

## 11. Sham feasibility boundary

An empty axis mask is structurally feasible because the same base structured
fact provides all original fields without requiring a semantic substitution.

Therefore:

EMPTY_MASK_SHAM_SOURCE_STRUCTURALLY_FEASIBLE = YES

This does not establish implementation equivalence with the historical
intervention_type = "none".

Therefore:

HISTORICAL_NONE_AS_SAME_MECHANISM_SHAM = NOT_ESTABLISHED

A future implementation must create or prove a same-mechanism empty-mask path
rather than silently reusing historical "none".

## 12. What this audit did not establish

This audit did not establish:

- correctness of a masked_slot_substitution implementation;
- correctness of rendered six-cell evidence;
- equality of non-target rendering behavior across masks;
- absence of unintended text changes after rendering;
- label semantics for new contrast rows;
- outcome behavior;
- predictive value;
- causal relevance;
- mechanistic importance;
- performance improvement.

It also did not create a new dataset.

## 13. Execution boundary

During the audit:

CONTROLLED_RECORD_DATASET_GENERATED = NO

SIX_CELL_ROWS_GENERATED = NO

DATASET_GENERATION = NOT_PERFORMED

FEATURE_IMPLEMENTATION = NOT_PERFORMED

TRAINING_EVALUATION = NOT_PERFORMED

MODEL_TOKENIZER_EXECUTION = NOT_PERFORMED

No tracked or staged repository delta was produced.

## 14. Feasibility conclusion

The minimum six-cell contrast specified at:

0a0da5354782e542520fb5bba146ab1a599d17ef

is structurally feasible for the complete frozen 300-pair source universe.

The feasibility does not require:

- rendered-text semantic reconstruction;
- label inspection;
- model output inspection;
- tokenizer execution;
- regeneration of the historical controlled dataset.

Therefore:

GEN4_SIX_CELL_SOURCE_STRUCTURE_FEASIBILITY_AUDIT = PASS

PAIR_UNIVERSE_EXACT_MATCH = YES

REQUIRED_ORIGINAL_FIELDS_AVAILABLE_ALL_300 = YES

REQUIRED_ALTERNATE_FIELDS_AVAILABLE_ALL_300 = YES

ALL_REQUIRED_ORIGINAL_ALTERNATE_PAIRS_DISTINCT = YES

WITHIN_PAIR_ALTERNATE_VALUES_CAN_BE_FIXED = YES

COMPLETE_SIX_CELL_SOURCE_STRUCTURE_FEASIBLE = YES

SOURCE_STRUCTURE_TEXT_INFERENCE_REQUIRED = NO

## 15. Scientific interpretation

The upstream identifiability-breaking design is no longer blocked by missing
generator-side source structure.

The remaining boundary is implementation design.

This feasibility result does not itself authorize implementation.

Therefore:

SOURCE_STRUCTURE_FEASIBILITY_BLOCKER = CLEARED

GENERATOR_IMPLEMENTATION = NOT_AUTHORIZED

CONTRAST_DATASET_GENERATION = NOT_AUTHORIZED

MODEL_FEATURE_IMPLEMENTATION = NOT_AUTHORIZED

TRAINING_EVALUATION = NOT_AUTHORIZED

## 16. Required next object

The next appropriate object is:

GEN4_SIX_CELL_GENERATOR_IMPLEMENTATION_SPECIFICATION

Its purpose is to define a minimal deterministic implementation that:

- consumes only generator-side structured facts;
- applies one held-constant mechanism;
- varies only the canonical axis mask;
- preserves within-pair alternate values;
- emits complete six-cell blocks;
- records explicit structural provenance;
- fails closed on incomplete source structure;
- does not infer semantic identity from rendered text.

That specification must remain separate from implementation authority and
execution authority.

## 17. Implementation-design requirements

A future implementation specification must define at least:

- exact mechanism_id;
- exact six contrast_cell_id values;
- exact canonical mask ordering;
- deterministic row identity;
- exact use of original and alternate fields;
- same-mechanism empty-mask sham behavior;
- rendering invariants;
- provenance schema;
- complete-block validation;
- duplicate handling;
- pair ordering;
- serialization rules;
- source-authority verification;
- fail-closed behavior.

It must also define how labels are excluded or handled without leaking outcome
semantics into structural identity.

No implementation is authorized by this report.

## 18. Current decision

UPSTREAM_CONTRAST_SPECIFICATION = FROZEN

GEN4_SIX_CELL_SOURCE_STRUCTURE_FEASIBILITY_AUDIT = PASS

FROZEN_GENERATOR_SOURCE_IDENTITY = PASS

STRUCTURED_FACT_COUNT = 300

PAIR_UNIVERSE_EXACT_MATCH = YES

REQUIRED_ORIGINAL_FIELDS_AVAILABLE_ALL_300 = YES

REQUIRED_ALTERNATE_FIELDS_AVAILABLE_ALL_300 = YES

ALL_REQUIRED_ORIGINAL_ALTERNATE_PAIRS_DISTINCT = YES

WITHIN_PAIR_ALTERNATE_VALUES_CAN_BE_FIXED = YES

COMPLETE_SIX_CELL_SOURCE_STRUCTURE_FEASIBLE = YES

HISTORICAL_NONE_AS_SAME_MECHANISM_SHAM = NOT_ESTABLISHED

SOURCE_STRUCTURE_FEASIBILITY_BLOCKER = CLEARED

GENERATOR_IMPLEMENTATION = NOT_AUTHORIZED

CONTRAST_DATASET_GENERATION = NOT_AUTHORIZED

FEATURE_IMPLEMENTATION = NOT_AUTHORIZED

TRAINING_EVALUATION = NOT_AUTHORIZED

NEXT_SCIENTIFIC_OBJECT = GEN4_SIX_CELL_GENERATOR_IMPLEMENTATION_SPECIFICATION

## 19. Stop condition

Stop after this feasibility report candidate is created and reviewed.

Do not implement masked_slot_substitution.

Do not generate any six-cell contrast rows.

Do not modify scripts/build_controlled_v5.py.

Do not modify the existing Gen4 sidecar.

Do not train.

Do not evaluate.

Do not run model inference.

Do not execute tokenizers.

Do not run Kaggle.

A later frozen authority must explicitly authorize the
GEN4_SIX_CELL_GENERATOR_IMPLEMENTATION_SPECIFICATION boundary.