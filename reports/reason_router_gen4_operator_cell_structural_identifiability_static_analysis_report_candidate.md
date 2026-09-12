# ContraMamba Gen4 Operator-Cell Structural Identifiability Static Analysis Report - Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_OPERATOR_CELL_STRUCTURAL_IDENTIFIABILITY_STATIC_ANALYSIS
- Repository evidence commit: a761ac211b33bad66106fb7a51b94ceb198c4713
- Canonical sidecar SHA256: e371aa7b2768ee1aa97b9a4e5479e246b2317a52011e00a75d9681bb92b2d913
- Training performed: NO
- Evaluation performed: NO
- Model inference performed: NO
- Tokenizer execution performed: NO
- Kaggle used: NO
- Dataset generation performed: NO
- Feature implementation authorized: NO
- Scientific feature promotion authorized: NO

This report records a read-only structural identifiability analysis over the
frozen Gen4 generator-structure sidecar.

It does not test model behavior.

It does not establish predictive, causal, mechanistic, or performance effects.

## 2. Evidence authority

The canonical generator-structure materialization artifacts were frozen at:

a761ac211b33bad66106fb7a51b94ceb198c4713

Canonical sidecar:

reports/reason_router_gen4_operator_cell_generator_structure_materialization_ac75e91ac281c984a2de0540491120847adff0c5/gen4_operator_cell_generator_structure_sidecar.jsonl

Physical SHA256:

e371aa7b2768ee1aa97b9a4e5479e246b2317a52011e00a75d9681bb92b2d913

Rows:

1200

Unique source pairs represented:

300

The sidecar itself is provenance-valid generator-declared structure.

This report asks a different question:

Does the observed structural support contain independent variation sufficient
to identify operator-cell or semantic-axis effects separately from
intervention identity?

## 3. Scope

Only the following frozen structural fields are relevant:

- intervention_type
- intended_changed_axes
- operator_cells

No outcome, label, prediction, model state, rendered-text semantic
reconstruction, or tokenizer-derived information is used.

The four supported interventions are:

- entity_swap
- role_swap
- title_name_swap
- predicate_swap

Each contributes exactly 300 rows.

## 4. Canonical semantic-axis basis

Axis order:

1. title
2. name
3. role
4. predicate

Observed axis activation patterns are exactly:

entity_swap:

(0, 1, 0, 0)

role_swap:

(0, 0, 1, 0)

title_name_swap:

(1, 1, 0, 0)

predicate_swap:

(0, 0, 0, 1)

Number of distinct intervention types:

4

Number of distinct observed axis patterns:

4

Therefore:

INTERVENTION_TO_AXIS_PATTERN_INJECTIVE = YES

AXIS_PATTERN_RECOVERS_INTERVENTION = YES

On the observed support, the semantic-axis pattern contains no intervention
identity ambiguity.

## 5. Canonical operator-cell basis

Operator-cell order:

1. entity_swap:name
2. role_swap:role
3. title_name_swap:title
4. title_name_swap:name
5. predicate_swap:predicate

Observed operator-cell patterns are exactly:

entity_swap:

(1, 0, 0, 0, 0)

role_swap:

(0, 1, 0, 0, 0)

title_name_swap:

(0, 0, 1, 1, 0)

predicate_swap:

(0, 0, 0, 0, 1)

Number of distinct observed operator-cell patterns:

4

Therefore:

INTERVENTION_TO_OPERATOR_CELL_PATTERN_INJECTIVE = YES

OPERATOR_CELL_PATTERN_RECOVERS_INTERVENTION = YES

On the observed support, operator-cell activation is also a deterministic code
for intervention identity.

## 6. Rank analysis

The four-row semantic-axis design matrix has:

rows = 4

columns = 4

rank = 4

Therefore:

AXIS_ONLY_4D_DESIGN_SATURATED_ON_4_OPERATOR_SUPPORT = YES

Adding an intercept produces:

columns = 5

rank = 4

Therefore:

AXIS_WITH_INTERCEPT_HAS_REDUNDANT_DOF = YES

The operator-cell design matrix has:

rows = 4

columns = 5

rank = 4

The fifth apparent operator-cell dimension does not create a fifth independent
structural degree of freedom on the observed support.

## 7. title_name_swap identifiability

Across all observed title_name_swap rows:

title_name_swap:title is active whenever title_name_swap:name is active.

title_name_swap:name is active whenever title_name_swap:title is active.

No row contains:

- title active without name; or
- name active without title.

Therefore:

TITLE_NAME_OPERATOR_CELLS_ALWAYS_COACTIVE = YES

SEPARATE_TITLE_NAME_OPERATOR_CELL_EFFECTS_IDENTIFIABLE = NO

Any attempt to estimate separate title and name effects from this support alone
would be structurally underidentified.

The problem exists before observing any model outcome.

## 8. Information-content conclusion

The mapping from intervention_type to both structural representations is
deterministic and injective on the observed four-operator support.

Therefore the current sidecar provides:

- valid semantic provenance;
- explicit structural naming;
- deterministic row-level structural identity;
- a safe join key for later analysis.

It does not, on the observed support, provide an independently varying feature
signal beyond intervention identity.

Therefore:

NEW_INFORMATION_BEYOND_INTERVENTION_TYPE = NO_ON_OBSERVED_SUPPORT

This statement concerns identifiability and information support, not predictive
performance.

No model outcome was inspected.

## 9. Reparameterization risk

If the current axis or operator-cell vectors are inserted directly into a model
or regression while the available support remains unchanged, a measured effect
may merely reflect a reparameterization of intervention identity.

In particular:

- entity_swap maps uniquely to name-only activation;
- role_swap maps uniquely to role-only activation;
- predicate_swap maps uniquely to predicate-only activation;
- title_name_swap maps uniquely to joint title-and-name activation.

Consequently, an apparent "axis effect" cannot currently be distinguished from
the intervention mechanism that uniquely carries that axis pattern.

Therefore:

DIRECT_FEATURE_PROMOTION_FROM_STRUCTURE_ALONE = BLOCKED

DIRECT_MODEL_FEATURE_IMPLEMENTATION = BLOCKED

## 10. Necessary condition for a stronger Gen4 scientific test

A future design that claims an axis-level or operator-cell-level effect distinct
from intervention identity must break the current deterministic confounding.

At minimum, future structural support must make operator identity and semantic
axis activation non-bijective.

Examples of structurally informative contrast types include:

- more than one intervention mechanism producing the same semantic-axis pattern;
- one intervention family varying which semantic axes are activated;
- controlled constructions that independently activate title and name;
- repeated semantic-axis activations under different intervention mechanisms.

These are design requirements only.

This report does not authorize generating such examples.

## 11. Necessary condition for title versus name separation

To estimate separate title and name effects, future support must contain
independent title/name variation.

At minimum the design requires observations corresponding to:

- title active, name inactive;
- title inactive, name active.

Joint title-and-name activation alone is insufficient.

The current sidecar contains neither independent condition.

Therefore separate title and name scientific claims are blocked under the
current support.

## 12. Falsifiability boundary

A future Gen4 feature hypothesis is scientifically meaningful only if its
effect can be distinguished from intervention identity.

A valid future design must specify in advance:

- the structural contrast that breaks intervention/axis confounding;
- which feature effect becomes identifiable under that contrast;
- the null hypothesis;
- the falsification criterion;
- the required provenance;
- the allowed outcome evidence;
- the split or holdout rule, if outcome testing is later authorized.

None of those outcome-bearing steps are authorized by this report.

## 13. Scientific interpretation

The current materialization solved the provenance problem.

It did not solve the identifiability problem.

These are separate questions.

The current evidence supports:

GENERATOR_STRUCTURE_PROVENANCE = VALID

OPERATOR_CELL_IDENTITY = VALID

STRUCTURAL_IDENTIFIABILITY_BEYOND_INTERVENTION = NOT_ESTABLISHED

TITLE_NAME_CELL_SEPARATION = NOT_IDENTIFIABLE

DIRECT_FEATURE_PROMOTION = BLOCKED

## 14. Next scientific object

The next appropriate scientific object is:

GEN4_IDENTIFIABILITY_BREAKING_CONTRAST_SPECIFICATION

Its purpose is to define the minimum controlled structural variation needed to
test whether semantic-axis or operator-cell structure carries an effect that is
not reducible to intervention identity.

That future specification must remain separate from implementation and execution
authority.

It must not silently authorize:

- new dataset generation;
- model changes;
- training;
- evaluation;
- inference;
- tokenizer execution;
- Kaggle execution.

## 15. Final disposition

GEN4_GENERATOR_STRUCTURE_MATERIALIZATION = CLOSED

GEN4_GENERATOR_STRUCTURE_PROVENANCE = PASS

GEN4_FEATURE_IDENTIFIABILITY_STATIC_AUDIT = PASS

INTERVENTION_TO_AXIS_PATTERN_INJECTIVE = YES

INTERVENTION_TO_OPERATOR_CELL_PATTERN_INJECTIVE = YES

AXIS_MATRIX_RANK = 4

AXIS_PLUS_INTERCEPT_RANK = 4

OPERATOR_CELL_MATRIX_RANK = 4

TITLE_NAME_OPERATOR_CELLS_ALWAYS_COACTIVE = YES

NEW_INFORMATION_BEYOND_INTERVENTION_TYPE = NO_ON_OBSERVED_SUPPORT

SEPARATE_TITLE_NAME_OPERATOR_CELL_EFFECTS_IDENTIFIABLE = NO

DIRECT_FEATURE_PROMOTION_FROM_STRUCTURE_ALONE = BLOCKED

NEXT_SCIENTIFIC_OBJECT = GEN4_IDENTIFIABILITY_BREAKING_CONTRAST_SPECIFICATION

TRAINING_EVALUATION = NOT_PERFORMED

MODEL_TOKENIZER_EXECUTION = NOT_PERFORMED

SCIENTIFIC_OUTCOME_CLAIM = NOT_ESTABLISHED

## 16. Stop condition

Stop after this static analysis report is created and reviewed.

Do not implement Gen4 model features.

Do not generate new controlled examples.

Do not alter the canonical sidecar.

Do not train.

Do not evaluate.

Do not run model inference.

Do not execute tokenizers.

Do not run Kaggle.

A later frozen authority must explicitly authorize the
GEN4_IDENTIFIABILITY_BREAKING_CONTRAST_SPECIFICATION boundary.