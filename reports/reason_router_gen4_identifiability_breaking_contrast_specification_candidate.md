# ContraMamba Gen4 Identifiability-Breaking Contrast Specification - Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_IDENTIFIABILITY_BREAKING_CONTRAST_SPECIFICATION
- Upstream identifiability authority: 7fb89e1a0b8ed6bc73e9f57fe22a1a8a42329d0e
- Specification type: static scientific design
- Dataset generation authorized: NO
- Generator implementation authorized: NO
- Model feature implementation authorized: NO
- Training authorized: NO
- Evaluation authorized: NO
- Model inference authorized: NO
- Tokenizer execution authorized: NO
- Kaggle authorized: NO
- Automatic Commit/Push: NO

This specification defines the minimum structural contrast required to test
semantic-axis effects without reducing them to intervention identity.

It does not authorize generating the proposed data.

## 2. Upstream blocker

Frozen upstream evidence established:

GEN4_GENERATOR_STRUCTURE_MATERIALIZATION = CLOSED

GEN4_FEATURE_IDENTIFIABILITY_STATIC_AUDIT = PASS

INTERVENTION_TO_AXIS_PATTERN_INJECTIVE = YES

INTERVENTION_TO_OPERATOR_CELL_PATTERN_INJECTIVE = YES

TITLE_NAME_OPERATOR_CELLS_ALWAYS_COACTIVE = YES

NEW_INFORMATION_BEYOND_INTERVENTION_TYPE = NO_ON_OBSERVED_SUPPORT

SEPARATE_TITLE_NAME_OPERATOR_CELL_EFFECTS_IDENTIFIABLE = NO

DIRECT_FEATURE_PROMOTION_FROM_STRUCTURE_ALONE = BLOCKED

Therefore a new scientific design must introduce structural variation that
exists within a held-constant intervention mechanism.

## 3. Design principle

The critical confound is:

intervention mechanism <-> semantic-axis pattern

on the current observed support.

The primary identifiability-breaking rule is therefore:

SEMANTIC_AXIS_PATTERN_MUST_VARY_WITHIN_MECHANISM = YES

A mechanism that has only one fixed axis pattern cannot separately identify
a mechanism effect and an axis effect.

Merely adding a second fixed mechanism that happens to produce the same axis
pattern is not by itself sufficient if unrestricted mechanism fixed effects
remain in the scientific model.

Therefore:

DUPLICATE_AXIS_PATTERN_ACROSS_FIXED_MECHANISMS_ALONE = INSUFFICIENT

The minimum core design instead holds the transformation mechanism constant
while changing only a generator-declared axis mask.

## 4. Proposed held-constant mechanism

The future design object is conceptually named:

masked_slot_substitution

This is a design label only.

No generator implementation is authorized by this specification.

Its scientific requirement is that all contrast cells execute through the
same transformation mechanism and differ only in the declared semantic-axis
mask.

The mechanism must use generator-declared structure, not rendered-text
reconstruction.

## 5. Canonical axis order

Axis order remains:

1. title
2. name
3. role
4. predicate

Binary masks in this specification follow exactly that order.

## 6. Minimum six-cell structural core

For every eligible base source pair, the minimum complete contrast block is:

C0_SHAM

axis mask:

(0, 0, 0, 0)

No semantic slot is substituted.

This must be a sham invocation of the same future mechanism rather than reuse
of an unrelated historical "none" intervention code path.

C1_TITLE

axis mask:

(1, 0, 0, 0)

Only title is substituted.

C2_NAME

axis mask:

(0, 1, 0, 0)

Only name is substituted.

C3_ROLE

axis mask:

(0, 0, 1, 0)

Only role is substituted.

C4_PREDICATE

axis mask:

(0, 0, 0, 1)

Only predicate is substituted.

C5_TITLE_NAME

axis mask:

(1, 1, 0, 0)

Title and name are substituted jointly.

No cell may differ by an unrecorded intervention mechanism.

## 7. Why six cells are required

The primary design columns are:

1. intercept
2. title
3. name
4. role
5. predicate
6. title x name

The six required cell rows are:

C0_SHAM:
(1, 0, 0, 0, 0, 0)

C1_TITLE:
(1, 1, 0, 0, 0, 0)

C2_NAME:
(1, 0, 1, 0, 0, 0)

C3_ROLE:
(1, 0, 0, 1, 0, 0)

C4_PREDICATE:
(1, 0, 0, 0, 1, 0)

C5_TITLE_NAME:
(1, 1, 1, 0, 0, 1)

This design matrix has:

rows = 6

columns = 6

rank = 6

Therefore:

MINIMUM_CORE_DESIGN_FULL_RANK = YES

The six-cell design identifies, within one held-constant mechanism:

- title main effect;
- name main effect;
- role main effect;
- predicate main effect;
- title-by-name interaction.

No additive assumption is required to separate title and name from their joint
condition.

## 8. Title versus name separation

The current support contains only joint title-and-name activation for
title_name_swap.

The proposed core adds all four cells needed for the title/name factorial:

(0,0)

(1,0)

(0,1)

(1,1)

where the two coordinates are title and name.

Therefore the future design can represent the pair-level contrasts:

delta_title(p) =
    Y(p, title) - Y(p, sham)

delta_name(p) =
    Y(p, name) - Y(p, sham)

interaction_title_name(p) =
    Y(p, title+name)
    - Y(p, title)
    - Y(p, name)
    + Y(p, sham)

Y is only a symbolic future outcome.

No outcome variable is authorized or selected by this specification.

## 9. Role and predicate separation

Within the same held-constant mechanism:

delta_role(p) =
    Y(p, role) - Y(p, sham)

delta_predicate(p) =
    Y(p, predicate) - Y(p, sham)

These become within-mechanism contrasts rather than comparisons between
different intervention families.

That distinction is the identifiability-breaking operation.

## 10. Pair blocking

Every future base source pair admitted to the contrast dataset must contribute
the complete six-cell block.

Partial blocks are not acceptable for the canonical design.

Therefore:

COMPLETE_SIX_CELL_PAIR_BLOCK = REQUIRED

All six cells for a source pair must share:

- the same base source identity;
- the same underlying generator record;
- the same alternate slot values where applicable;
- the same mechanism implementation;
- the same provenance authority.

Only the declared axis mask may select which alternate fields are applied.

## 11. Alternate-value control

For one base source pair, alternate values must be frozen before cell
materialization.

For example, if the base record defines:

alternate_title
alternate_name
alternate_role
alternate_predicate

then all six cells for that pair must reference the same corresponding
alternate values.

A different alternate value may not be sampled independently for each cell.

Otherwise axis-mask effects would be confounded with alternate-value sampling.

Therefore:

WITHIN_PAIR_ALTERNATE_VALUES_FIXED = REQUIRED

## 12. Sham condition

The sham condition is scientifically necessary.

It provides the same mechanism code path with an empty axis mask.

It must not silently substitute the historical intervention_type "none" unless
a later authority proves that both execution paths are identical for the
scientific contrast being tested.

Therefore:

HISTORICAL_NONE_AS_SHAM = NOT_ASSUMED_EQUIVALENT

SAME_MECHANISM_EMPTY_MASK_SHAM = REQUIRED

## 13. Structural provenance

Future cell identity must be generator-declared.

At minimum a future contrast artifact must be able to record:

- source_pair_id;
- mechanism_id;
- axis_mask;
- intended_changed_axes;
- generator_source_fields;
- contrast_cell_id;
- generator authority identity.

The exact storage schema remains a later implementation-specification decision.

No rendered-text semantic reconstruction may define a contrast cell.

## 14. Forbidden contrast provenance

The following must not define or repair structural cell identity:

- rendered claim text;
- rendered evidence text;
- labels;
- primary failure type;
- model predictions;
- logits;
- probabilities;
- evaluator outputs;
- error cohorts;
- training outcomes;
- evaluation outcomes;
- Stage182a observed axis reconstruction;
- tokenizer-derived semantic reconstruction;
- manual post-hoc language interpretation.

Cell identity must exist before outcome inspection.

## 15. Structural acceptance gates

Before any future outcome-bearing experiment, the generated structural design
must establish all of the following:

- every admitted source pair has all six cells;
- no pair has duplicate cell identities;
- mechanism_id is identical across the six cells;
- axis masks match the six canonical masks exactly;
- alternate values are fixed within source pair;
- semantic identity comes from generator declarations;
- the design matrix rank is exactly 6;
- title-only exists;
- name-only exists;
- joint title+name exists;
- same-mechanism sham exists.

If any condition fails:

OUTCOME_TESTING = BLOCKED

## 16. Primary future estimands

If a later authority permits an outcome Y, the primary estimands are the
pair-blocked within-mechanism contrasts:

Delta_title
Delta_name
Delta_role
Delta_predicate

and:

Interaction_title_name

No intervention-family comparison is required to identify these estimands in
the minimum core.

## 17. Scientific nulls

A later outcome authority must instantiate the outcome and statistical test,
but the structural nulls are fixed here.

Axis-null family:

H0_axis:
Within the held-constant mechanism, changing the semantic-axis mask produces
no systematic outcome change relative to sham.

Title/name interaction null:

H0_title_name_interaction:
The joint title+name response is additive relative to the two single-axis
responses and sham.

Symbolically:

Interaction_title_name = 0

Title-versus-name equality null:

H0_title_equals_name:

Delta_title - Delta_name = 0

This specification does not choose significance thresholds, metrics, or
acceptance criteria.

## 18. Falsification discipline

A future claim that semantic-axis structure has an effect beyond intervention
identity is falsifiable only using within-mechanism mask variation.

Evidence consisting only of the old four intervention families is insufficient.

A future axis-effect claim fails if its predeclared within-mechanism contrast
does not satisfy the later frozen outcome criterion.

A future title/name separation claim fails if title-only and name-only cells
are unavailable, invalid, or not provenance-matched.

A future interaction claim fails if any of the four required title/name
factorial cells is unavailable.

## 19. Cross-mechanism generalization is a separate tier

The six-cell design establishes identifiability within one held-constant
mechanism.

It does not by itself establish mechanism-invariant semantic effects.

A stronger later claim that an axis effect generalizes across intervention
mechanisms requires replication of the same axis-mask contrasts under at least
one additional independently defined mechanism.

Therefore:

WITHIN_MECHANISM_AXIS_IDENTIFIABILITY = TARGET_OF_MINIMUM_CORE

MECHANISM_INVARIANT_AXIS_GENERALIZATION = NOT_ESTABLISHED_BY_MINIMUM_CORE

A second fully crossed mechanism may be authorized only after the minimum-core
question is scientifically justified.

## 20. Split boundary for future outcome testing

If later training or evaluation is authorized, all cells derived from one base
source pair must remain in the same split.

Therefore:

PAIR_GROUPED_SPLIT = REQUIRED

No mask variant from one source pair may appear in train while another mask
variant from the same source pair appears in dev/test.

This requirement prevents paired-contrast leakage.

No actual split is authorized here.

## 21. Relation to existing Gen4 sidecar

The existing frozen 1200-row sidecar remains valid provenance evidence.

It is not modified or superseded.

Its existing four intervention families may provide historical structural
context, but they do not replace the six-cell within-mechanism core.

Existing title_name_swap joint activation does not eliminate the need for a
same-mechanism title+name cell in the future contrast design.

Therefore:

EXISTING_GEN4_SIDECAR = PRESERVED

EXISTING_FOUR_OPERATOR_SUPPORT = INSUFFICIENT_FOR_AXIS_EFFECT_IDENTIFICATION

## 22. Minimality decision

The selected next design is intentionally narrow.

It does not require a full power-set over four semantic axes.

It requires only:

- one same-mechanism sham;
- four single-axis cells;
- one title+name joint cell.

This is sufficient to identify the four axis main effects and the specific
title/name interaction implicated by the current blocker.

No role interactions, predicate interactions, or higher-order interactions are
included in the minimum core.

Therefore:

FULL_4_AXIS_FACTORIAL = NOT_REQUIRED

MINIMUM_SIX_CELL_CORE = REQUIRED

## 23. Implementation boundary

This specification does not authorize:

- modifying scripts/build_controlled_v5.py;
- creating a new generator;
- adding masked_slot_substitution code;
- generating contrast rows;
- creating a new dataset;
- creating a new sidecar;
- modifying model inputs;
- modifying losses;
- training;
- evaluation;
- model inference;
- tokenizer execution;
- Kaggle execution.

Any implementation requires a separate frozen implementation authority.

Any materialization requires a separate execution authority.

## 24. Required next verification before implementation authority

Before implementation is authorized, a later static implementation-design
review must establish that the frozen generator source records contain the
required source fields needed to construct all six cells without semantic
reconstruction from rendered text.

In particular it must statically verify availability and provenance of:

- original title;
- original name;
- original role;
- original predicate;
- alternate_title;
- alternate_name;
- alternate_role;
- alternate_predicate;

for the intended source population.

This review must be source-structural only.

No tokenizer, model, label outcome, or rendered-text inference is permitted.

## 25. Current decision

UPSTREAM_IDENTIFIABILITY_BLOCKER = FROZEN

SEMANTIC_AXIS_PATTERN_MUST_VARY_WITHIN_MECHANISM = YES

DUPLICATE_AXIS_PATTERN_ACROSS_FIXED_MECHANISMS_ALONE = INSUFFICIENT

MINIMUM_CORE_CELL_COUNT = 6

MINIMUM_CORE_DESIGN_RANK = 6

COMPLETE_SIX_CELL_PAIR_BLOCK = REQUIRED

SAME_MECHANISM_EMPTY_MASK_SHAM = REQUIRED

WITHIN_PAIR_ALTERNATE_VALUES_FIXED = REQUIRED

TITLE_ONLY_CELL = REQUIRED

NAME_ONLY_CELL = REQUIRED

TITLE_NAME_JOINT_CELL = REQUIRED

PAIR_GROUPED_SPLIT = REQUIRED_FOR_FUTURE_OUTCOME_TESTING

WITHIN_MECHANISM_AXIS_IDENTIFIABILITY = TARGET_OF_MINIMUM_CORE

MECHANISM_INVARIANT_AXIS_GENERALIZATION = DEFERRED

FULL_4_AXIS_FACTORIAL = NOT_REQUIRED

DATASET_GENERATION = NOT_AUTHORIZED

FEATURE_IMPLEMENTATION = NOT_AUTHORIZED

TRAINING_EVALUATION = NOT_AUTHORIZED

## 26. Next boundary

If this specification is reviewed and frozen, the next authorized activity is
a read-only source-structure feasibility audit.

That audit may inspect the frozen generator source and generator-side structural
records only to determine whether the six required cells can be constructed
without rendered-text semantic reconstruction.

It may not generate any contrast example.

Therefore the next object after this specification freeze is:

GEN4_SIX_CELL_SOURCE_STRUCTURE_FEASIBILITY_AUDIT

## 27. Stop condition

Stop after this contrast specification candidate is created and reviewed.

Do not implement masked_slot_substitution.

Do not generate new controlled examples.

Do not modify the existing Gen4 sidecar.

Do not train.

Do not evaluate.

Do not run model inference.

Do not execute tokenizers.

Do not run Kaggle.

A later frozen authority must explicitly authorize the read-only
GEN4_SIX_CELL_SOURCE_STRUCTURE_FEASIBILITY_AUDIT.