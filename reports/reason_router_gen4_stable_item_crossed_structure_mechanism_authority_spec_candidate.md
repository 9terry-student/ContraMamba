# Generation-4 Stable-Item Crossed-Structure Mechanism
# Static Audit Authority Specification

STATUS = CANDIDATE

PHASE =
GEN4_STABLE_ITEM_CROSSED_STRUCTURE_MECHANISM_STATIC_AUDIT

PARENT_VALIDATED_EVIDENCE =
857a64cd7521d5ccb30d331a57b0b0f73aa7f39d

PARENT_LOO_EXECUTION_AUTHORITY =
183b4a92be08e4a94ca19b3d43ccfdf1e29b57f1

FROZEN_PRIMARY_POPULATION =
reports/reason_router_gen4_authorization_boundary_susceptibility_static_audit_candidate/primary_population.jsonl

FROZEN_PRIMARY_POPULATION_SHA256 =
eb5aaacf38be224461801cab5e0629e46a9d375d7efaeadb72b2dbf3c5d56f3e

TRAINING_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_LOADING_ALLOWED =
NO

NEW_INFERENCE_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

GPU_ALLOWED =
NO


## 1. Established Generation-4 evidence

The validated Generation-4 evidence currently establishes all of the following.

First:

same-seed A0 authorization-boundary proximity strongly predicts historical D1
SUPPORT susceptibility across examples.

Second:

within the same stable_id, seed-to-seed changes in A0 boundary margin do NOT
track seed-to-seed D1 outcome.

Frozen within-item result:

N_DISCORDANT =
37

WITHIN_ITEM_CONCORDANCE =
0.4594594594594595

MEDIAN_DELTA_I =
0.02173464000225067

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY =
NOT_SUPPORTED

Third:

when the target seed's own A0 margin is excluded, the same stable_id's A0
margin measured only in OTHER seeds strongly predicts held-out D1
susceptibility.

Validated held-out AUCs:

seed180 =
0.9594117647058824

seed181 =
0.9730194191441829

seed182 =
0.9864643150123051

Therefore:

STABLE_ITEM_ASSOCIATED_VULNERABILITY =
SUPPORTED

and:

BOUNDARY_CAUSALITY_ESTABLISHED =
NO


## 2. Outcome-blind structural inventory

Before this authority was written, an outcome-blind structural inventory read
only:

- seed
- stable_id

from the frozen population.

It did NOT access:

- d_a0_ne_minus_support
- susceptibility_score
- y_d1_support
- d1_pred_label

The inventory established:

ROWS =
1590

UNIQUE_STABLE_IDS =
538

UNIQUE_BASE_IDS =
60

UNIQUE_OPERATORS =
9

Operators:

- entity_swap
- event_swap
- evidence_deletion
- evidence_truncation
- irrelevant_evidence
- location_swap
- predicate_swap
- role_swap
- title_name_swap

Observed base x operator cells:

538

Full possible grid cells:

540

Missing cells:

2

Complete bases containing all nine operators:

58

The two missing cells are:

generated_fact_139__role_swap

museum_purchase__role_swap

Thus the stable_id design is sufficiently crossed to prespecify base,
operator, additive, and cell-specific components without selecting a favorable
operator after outcome access.


## 3. Scientific question

The stable item-associated vulnerability may arise from at least three
structurally distinct sources:

1. a BASE-ITEM component shared across perturbation operators;
2. a PERTURBATION-OPERATOR component shared across base items;
3. a BASE x OPERATOR cell-specific component not explained by additive base
   and operator structure.

This audit asks which of these components carries predictive information about
held-out D1 SUPPORT susceptibility.

It is a structural predictive mechanism audit.

It is not a causal intervention.


## 4. Stable-id factorization

Every stable_id must be parsed exactly once by:

stable_id.rsplit("__", 1)

yielding:

base_id

and:

operator

No alternative parsing, manual regrouping, semantic relabeling, or operator
merging is permitted after outcome access.


## 5. Target-seed evaluation units

For each target seed:

s in {180, 181, 182}

begin with the same leave-one-seed-out eligible cells defined by the validated
Generation-4 LOO audit.

For cell:

c = (base b, operator o)

define:

m(c,s) =
mean A0 d_a0_ne_minus_support for the same stable_id using only available
frozen seeds t != s.

The target seed's own A0 margin is excluded.

No D1 outcome from any seed may be used to construct m(c,s).


## 6. Leave-cell-out BASE component

For target cell c=(b,o) and target seed s define:

BASE_MARGIN(c,s) =

mean of m((b,o'),s)

over eligible cells satisfying:

o' != o

The target cell itself is excluded.

Minimum BASE context:

BASE_CONTEXT_COUNT >= 6

If this gate is not met for a target cell, that cell is ineligible for all
primary component comparisons in that target seed.


## 7. Leave-cell-out OPERATOR component

For target cell c=(b,o) and target seed s define:

OPERATOR_MARGIN(c,s) =

mean of m((b',o),s)

over eligible cells satisfying:

b' != b

The target cell itself is excluded.

Minimum OPERATOR context:

OPERATOR_CONTEXT_COUNT >= 30

If this gate is not met for a target cell, that cell is ineligible for all
primary component comparisons in that target seed.


## 8. Leave-cell-out GLOBAL component

For target cell c and target seed s define:

GLOBAL_MARGIN(c,s) =

mean m(c',s)

over every eligible cell:

c' != c

Minimum global context:

GLOBAL_CONTEXT_COUNT >= 400


## 9. ADDITIVE component

For target cell c=(b,o):

ADDITIVE_MARGIN(c,s) =

BASE_MARGIN(c,s)
+
OPERATOR_MARGIN(c,s)
-
GLOBAL_MARGIN(c,s)

This is a prespecified crossed two-factor additive prediction of the target
cell's stable A0 boundary geometry.

No fitting to D1 outcomes is permitted.


## 10. CELL-SPECIFIC residual component

Define:

CELL_RESIDUAL_MARGIN(c,s) =

m(c,s)
-
ADDITIVE_MARGIN(c,s)

A negative residual means that the stable_id is closer to the SUPPORT boundary
than would be expected from its base and operator components alone.


## 11. Four prespecified susceptibility scores

For each target cell and target seed:

BASE_SCORE =
-BASE_MARGIN

OPERATOR_SCORE =
-OPERATOR_MARGIN

ADDITIVE_SCORE =
-ADDITIVE_MARGIN

RESIDUAL_SCORE =
-CELL_RESIDUAL_MARGIN

Higher values always represent the prespecified direction of greater
susceptibility.

No sign may be reversed after outcome access.


## 12. Held-out outcome

The outcome remains:

Y(c,s) =
target-seed y_d1_support

where:

Y = 1

iff historical D1 prediction in target seed s is SUPPORT.

No other D1 outcome may enter any predictor.


## 13. Common primary evaluation population

Within each target seed, all four primary predictors MUST be evaluated on the
same cells.

A cell enters the common primary population only if all are true:

1. validated LOO eligibility passes;
2. BASE_CONTEXT_COUNT >= 6;
3. OPERATOR_CONTEXT_COUNT >= 30;
4. GLOBAL_CONTEXT_COUNT >= 400;
5. all four scores are finite.

This prevents different predictors from receiving favorable populations.


## 14. Primary endpoints

There are exactly twelve primary endpoints:

BASE_AUC_180
BASE_AUC_181
BASE_AUC_182

OPERATOR_AUC_180
OPERATOR_AUC_181
OPERATOR_AUC_182

ADDITIVE_AUC_180
ADDITIVE_AUC_181
ADDITIVE_AUC_182

RESIDUAL_AUC_180
RESIDUAL_AUC_181
RESIDUAL_AUC_182

Each endpoint is a tie-adjusted Mann-Whitney / ROC AUC.


## 15. Outcome-count gate

For every target seed's COMMON primary population require:

positive Y count >= 10

and:

negative Y count >= 30

If any target seed fails:

GEN4_CROSSED_STRUCTURE_MECHANISM =
BLOCKED_NOT_TESTED


## 16. Primary statistical tests

For each of the twelve primary endpoints independently test:

H0:
AUC <= 0.5

versus:

H1:
AUC > 0.5

using a one-sided within-target-seed label permutation.

For each permutation:

- predictor values remain fixed;
- the common primary population remains fixed;
- positive count remains fixed.

Use:

100000 permutations per endpoint.

The exact RNG contract must be frozen in a committed execution manifest before
any primary AUC or D1 association is computed.


## 17. Multiplicity

All twelve primary tests form ONE multiplicity family.

Use:

Holm step-down FWER

with:

alpha = 0.05

No separate correction by component is permitted.


## 18. Minimum substantive effect

For a component to be scientifically supported, each of its three seed-specific
AUC values must satisfy:

AUC >= 0.70

Statistical significance alone is insufficient.


## 19. Component support definitions

BASE_COMPONENT =
SUPPORTED

only if all three BASE AUCs:

- exceed 0.5;
- are >= 0.70;
- have Holm-adjusted p < 0.05.

Otherwise:

BASE_COMPONENT =
NOT_SUPPORTED


OPERATOR_COMPONENT =
SUPPORTED

only if all three OPERATOR AUCs:

- exceed 0.5;
- are >= 0.70;
- have Holm-adjusted p < 0.05.

Otherwise:

OPERATOR_COMPONENT =
NOT_SUPPORTED


ADDITIVE_COMPONENT =
SUPPORTED

only if all three ADDITIVE AUCs:

- exceed 0.5;
- are >= 0.70;
- have Holm-adjusted p < 0.05.

Otherwise:

ADDITIVE_COMPONENT =
NOT_SUPPORTED


CELL_RESIDUAL_COMPONENT =
SUPPORTED

only if all three RESIDUAL AUCs:

- exceed 0.5;
- are >= 0.70;
- have Holm-adjusted p < 0.05.

Otherwise:

CELL_RESIDUAL_COMPONENT =
NOT_SUPPORTED


## 20. Structural interpretation map

The result is interpreted only through the complete four-component support
pattern.

If BASE is supported:

base-item identity carries vulnerability information shared across perturbation
operators.

If OPERATOR is supported:

perturbation-operator identity carries vulnerability information shared across
base items.

If ADDITIVE is supported:

the crossed base + operator structure predicts vulnerability without requiring
a target-cell-specific margin.

If CELL_RESIDUAL is supported:

stable_id-specific deviation beyond additive base/operator structure carries
substantial additional vulnerability information.

Multiple components may be supported simultaneously.

No requirement is imposed that one component be declared uniquely dominant.


## 21. Prohibited post-hoc dominance claims

The following are NOT authorized:

- choosing the component with the numerically largest AUC and calling it the
  unique mechanism;
- inventing a post-hoc AUC difference threshold;
- dropping a supported component because another is stronger;
- declaring one operator causal because its marginal mean is extreme;
- choosing only favorable seeds;
- excluding seed182;
- restricting to FROZEN51;
- restricting to previous Gen3 residual IDs;
- selecting a swap subgroup after seeing outcomes;
- merging or splitting operator categories;
- changing stable_id parsing;
- tuning BASE_CONTEXT_COUNT;
- tuning OPERATOR_CONTEXT_COUNT;
- fitting D1 outcomes to learn factor weights.


## 22. Prespecified secondary descriptive analyses

Only after the twelve primary endpoints and four support verdicts are frozen,
report:

1. per-target-seed common primary population coverage;
2. distribution of BASE_CONTEXT_COUNT;
3. distribution of OPERATOR_CONTEXT_COUNT;
4. variance of m(c,s);
5. variance of ADDITIVE_MARGIN(c,s);
6. variance of CELL_RESIDUAL_MARGIN(c,s);
7. Pearson correlation between m(c,s) and ADDITIVE_MARGIN(c,s);
8. Pearson correlation between m(c,s) and CELL_RESIDUAL_MARGIN(c,s).

These are descriptive only and cannot alter component support verdicts.


## 23. Interpretation boundary

Even if one or more components are SUPPORTED, this audit establishes
predictive structural localization only.

It does NOT establish:

- that base identity causes susceptibility;
- that perturbation operator causes susceptibility;
- that the additive decomposition is a neural mechanism;
- that the residual corresponds to one circuit or parameter group;
- that changing one component would change D1 outcome.

Therefore:

SCIENTIFIC_CLAIM_BOUNDARY =
STRUCTURAL_PREDICTIVE_NOT_CAUSAL


## 24. Positive research consequences

If BASE_COMPONENT is supported, a later independently authorized phase may
study base-item structural properties.

If OPERATOR_COMPONENT is supported, a later independently authorized phase may
study operator-defined semantic perturbation structure.

If ADDITIVE_COMPONENT is supported, a later phase may study whether the stable
vulnerability is largely explained by crossed base and operator factors.

If CELL_RESIDUAL_COMPONENT is supported, a later phase may study
base-by-operator interaction structure.

None of those later studies is authorized by this document.


## 25. Negative research consequences

If a component is NOT_SUPPORTED:

do not rescue it by:

- changing its sign;
- changing its context average;
- changing seed membership;
- lowering AUC threshold;
- selecting favorable operators or bases.

If all four components are NOT_SUPPORTED:

the current stable-id crossed-structure hypothesis is rejected and a new
independently motivated item representation is required.


## 26. Execution sequencing

After this authority is committed, the next authorized operation is:

GEN4_CROSSED_STRUCTURE_EXECUTION_MANIFEST_FREEZE

That manifest must freeze:

- authority commit;
- population SHA256;
- stable_id parsing;
- LOO margin construction;
- context thresholds;
- common-population rule;
- predictor formulas;
- predictor processing order;
- target-seed processing order;
- 100000 permutations per endpoint;
- deterministic RNG seed;
- twelve-endpoint Holm family;
- AUC >= 0.70 threshold.

No primary D1 association may be computed before the manifest is committed.


## 27. Final authority state

PRIMARY_COMPONENTS =
BASE_OPERATOR_ADDITIVE_CELL_RESIDUAL

PRIMARY_ENDPOINT_COUNT =
12

COMMON_EVALUATION_POPULATION =
YES

BASE_CONTEXT_MINIMUM =
6

OPERATOR_CONTEXT_MINIMUM =
30

GLOBAL_CONTEXT_MINIMUM =
400

MINIMUM_AUC_EACH_SEED =
0.70

PERMUTATIONS_EACH_ENDPOINT =
100000

MULTIPLICITY =
HOLM_FWER_0.05_ACROSS_12_ENDPOINTS

TRAINING_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

END_OF_GEN4_STABLE_ITEM_CROSSED_STRUCTURE_MECHANISM_AUTHORITY
