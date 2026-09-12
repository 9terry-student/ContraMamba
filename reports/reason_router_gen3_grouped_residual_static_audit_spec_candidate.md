# Generation-3 Grouped Residual Static Audit Specification

STATUS = CANDIDATE

PHASE = GEN3_GROUPED_RESIDUAL_STATIC_AUDIT

PARENT_VALIDATED_EVIDENCE =
c97fd33dd8aa9c116f45071ee4545093ebba8f1c

TRAINING_ALLOWED = NO

EVALUATION_EXECUTION_ALLOWED = NO

KAGGLE_ALLOWED = NO

CHECKPOINT_INFERENCE_ALLOWED = NO

NEW_MODEL_EXECUTION = NO


## 1. Purpose

The completed Generation-3 grouped-factorial matrix supports:

PRIMARY_GROUPED_INTERPRETATION =
MIXED_GROUP_AND_GLOBAL_COMPONENT

CROSS_GROUP_CUMULATIVE_INTERACTION = SUPPORTED

REPRODUCIBLE_GROUP_INTERACTION = U+D

MACRO_GROUP_SUFFICIENT = NOT_SUPPORTED

GLOBAL_ONLY_CUMULATIVE_THRESHOLD = REJECTED

The grouped matrix recurrently recovers two of the six historically recurrent
FROZEN51 stable IDs.

Four historically recurrent FROZEN51 IDs remain unrecovered at the frozen
cross-seed recurrence threshold under every tested proper grouped subset.

This audit must determine, using existing validated prediction artifacts only,
whether those four residual IDs exhibit a coherent near-threshold geometry
under proper subsets or remain specifically associated with the historical
all-group D1 condition.


## 2. Frozen residual population

The audit population is exactly:

- clinic_expansion__predicate_swap
- generated_fact_045__role_swap
- generated_fact_193__predicate_swap
- generated_fact_258__title_name_swap

No other stable IDs may be promoted into the primary residual population.


## 3. Allowed evidence

Read-only use is allowed for the already validated prediction exports from:

- matched canonical A0;
- historical D1 / GLOBAL-HALF;
- grouped U;
- grouped Q;
- grouped D;
- grouped U+Q;
- grouped U+D;
- grouped Q+D.

The grouped validated-evidence analysis committed at the parent authority may
be used as an index and consistency reference.

No new training, evaluation execution, inference, checkpoint loading, lambda
change, seed addition, edge regrouping, or intervention generation is allowed.


## 4. Primary per-occurrence measurements

For each frozen residual stable-ID occurrence and each available proper grouped
subset, record relative to matched A0:

- final prediction;
- q_AUTHORIZED;
- entitlement probability;
- SUPPORT-minus-NOT_ENTITLED final margin;
- frame probability;
- predicate coverage probability;
- sufficiency probability;
- q_FRAME;
- q_PREDICATE;
- q_SUFFICIENCY;
- polarity margin.

Also record the same D1 values.


## 5. Boundary-distance analysis

The principal scalar is:

SUPPORT_MINUS_NE_MARGIN =
support_ne_margin_active

For each occurrence and proper subset X, compute:

DELTA_MARGIN_X =
MARGIN_X - MARGIN_A0

and:

DISTANCE_TO_SUPPORT_BOUNDARY_X =
0 - MARGIN_X

when MARGIN_X remains negative.

For D1 compute the same quantities.

The audit must not replace row-level analysis with aggregate means.


## 6. D1-approach ratio

For occurrences where D1 increases the SUPPORT-minus-NOT_ENTITLED margin
relative to A0, compute:

D1_APPROACH_RATIO_X =
(MARGIN_X - MARGIN_A0) /
(MARGIN_D1 - MARGIN_A0)

This quantity is descriptive only.

It must not be interpreted as causal contribution, parameter ownership, or
fractional mechanism attribution.


## 7. Authorization-side concordance

For each proper subset and D1, record whether:

- q_AUTHORIZED increases relative to A0;
- entitlement probability increases relative to A0;
- SUPPORT-minus-NOT_ENTITLED margin increases relative to A0;
- the final boundary crosses from negative to positive.

Uniform polarity-margin movement is not required.

The frozen historical signature remains authorization-side increase together
with SUPPORT-vs-NOT_ENTITLED final-boundary movement.


## 8. Cross-seed requirement

Stable IDs must remain seed-resolved.

For each of the four frozen residual IDs report:

- historical D1 occurrence seeds;
- each proper subset occurrence seeds;
- boundary-distance ranking by seed;
- whether the same proper subset is the closest proper subset in at least two
  frozen seeds where the historical residual exists.

No aggregate-only localization is allowed.


## 9. Prespecified interpretation classes

The audit may support only the following bounded descriptive classes.

PROPER_SUBSET_NEAR_THRESHOLD:

At least one proper grouped subset reproducibly approaches the D1
authorization/final-boundary geometry for the same residual stable ID across
at least two relevant frozen seeds, while remaining below the final error
boundary in at least one of those seeds.

DISTRIBUTED_NEAR_THRESHOLD:

No single proper subset reproducibly dominates, but several proper subsets
move the same recurrent residual toward the D1 authorization/final-boundary
geometry across seeds.

D1_SPECIFIC_RESIDUAL_GEOMETRY:

The historical D1 condition produces a recurrent authorization/final-boundary
transition that is not reproducibly approximated by any tested proper subset.

HETEROGENEOUS_RESIDUAL:

The four residual stable IDs do not support one common bounded geometry class.


## 10. Falsification

PROPER_SUBSET_NEAR_THRESHOLD is rejected for an ID if no proper subset shows a
reproducible cross-seed approach toward the D1 final-boundary geometry.

DISTRIBUTED_NEAR_THRESHOLD is rejected if proper-subset movements are weak,
inconsistent, or fail to reproduce the authorization-side direction.

D1_SPECIFIC_RESIDUAL_GEOMETRY is rejected for an ID if a proper subset
reproducibly approximates the D1 boundary/authorization geometry.

A global common residual mechanism must not be claimed if the four IDs split
across different bounded classes.


## 11. Claim boundary

This audit cannot establish:

- causal necessity;
- causal sufficiency;
- unique edge causation;
- unique group causation;
- parameter-level ownership;
- gradient orthogonality;
- a native Mamba state mechanism;
- optimal lambda;
- an untested three-way or higher-order interaction;
- production readiness.

In particular, a D1-specific residual geometry does not by itself establish
that all ten attenuated edges are mechanistically necessary.


## 12. Required output

Produce one validated-evidence static-audit report containing:

- exact source identities;
- exact four-ID residual population;
- seed-resolved row-level geometry;
- proper-subset boundary-distance tables;
- D1-approach ratios where defined;
- cross-seed recurrence/consistency;
- one bounded interpretation per residual ID;
- an overall residual verdict;
- explicit exclusions;
- the decision whether any new execution is scientifically justified.

No execution authority is granted by this specification.

END_OF_GEN3_GROUPED_RESIDUAL_STATIC_AUDIT_SPEC
