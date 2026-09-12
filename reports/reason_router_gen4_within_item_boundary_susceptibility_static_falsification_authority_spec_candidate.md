# Generation-4 Within-Item Boundary Susceptibility
# Static Falsification Authority Specification

STATUS = CANDIDATE

PHASE =
GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY_STATIC_FALSIFICATION

PARENT_VALIDATED_STATIC_EVIDENCE =
5388460bf0ac2ed1f8489e35c6942f6044da3df0

PARENT_DESIGN_AUTHORITY =
1fec5e47bcae970b5ec6a23e3724087fd3f10029

PARENT_EXECUTION_MANIFEST =
96ccb359c579515293c24f7544d128b0591343d3

FROZEN_PRIMARY_POPULATION_ARTIFACT =
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


## 1. Scientific purpose

The previous Generation-4 static audit established that baseline A0
SUPPORT / NOT_ENTITLED boundary proximity strongly predicts historical D1
supportward failure across frozen seeds 180, 181, and 182.

That result remains predictive, not causal.

A major remaining alternative explanation is stable item-level confounding:

some examples may simply be intrinsically harder, closer to the decision
boundary, and more likely to fail under D1.

The next falsification therefore conditions on stable_id.

The question is:

When the SAME stable_id has different D1 outcomes across frozen training
seeds, is its A0 boundary margin smaller in the seed where D1 flips to
SUPPORT?


## 2. Frozen input

This audit may read only the previously frozen Generation-4 primary
population artifact:

primary_population.jsonl

SHA256:

eb5aaacf38be224461801cab5e0629e46a9d375d7efaeadb72b2dbf3c5d56f3e

No raw checkpoint, dataset, model output generation, or new prediction source
is required.

The relevant frozen row fields are:

- seed
- stable_id
- d_a0_ne_minus_support
- y_d1_support

No field may be redefined.


## 3. Unit of analysis

The PRIMARY unit of analysis is:

stable_id

not:

(seed, stable_id)

Each stable_id is permitted to contribute exactly one primary within-item
effect value.


## 4. Eligibility

For each stable_id:

1. retain only rows already present in the frozen Gen4 primary population;
2. require observations from at least two of seeds 180, 181, 182;
3. require at least one row with y_d1_support = 1;
4. require at least one row with y_d1_support = 0.

Such a stable_id is:

WITHIN_ITEM_DISCORDANT_ELIGIBLE

Stable IDs with the same D1 outcome in every observed seed are not part of the
primary discordant analysis.

They must still be counted and reported descriptively.

No stable_id may be included or excluded based on its d_A0 value.


## 5. Primary within-item effect

For each discordant eligible stable_id i define:

mean_flip_i =
mean d_A0 across observed seeds where y_d1_support = 1

mean_nonflip_i =
mean d_A0 across observed seeds where y_d1_support = 0

and:

delta_i =
mean_flip_i - mean_nonflip_i

The prespecified susceptibility hypothesis predicts:

delta_i < 0

because the same item should be closer to the SUPPORT boundary in seeds where
D1 flips.


## 6. Primary endpoint

Let:

N_discordant =
number of discordant eligible stable IDs

N_nonzero =
number of discordant eligible stable IDs with delta_i != 0

K_negative =
number with delta_i < 0

The primary effect-size endpoint is:

WITHIN_ITEM_CONCORDANCE =
K_negative / N_nonzero

Exact delta_i = 0 ties are excluded from the binomial denominator and reported
separately.

No magnitude threshold, trimming, winsorization, normalization, or learned
weighting is permitted.


## 7. Primary statistical test

Under the null:

P(delta_i < 0) = 0.5

Use an exact one-sided binomial sign test:

H0:
P(delta_i < 0) <= 0.5

H1:
P(delta_i < 0) > 0.5

No asymptotic approximation is permitted for the primary p-value.

There is exactly one primary inferential endpoint, so no multiplicity
correction is required for the primary test.


## 8. Prospective feasibility gate

The primary hypothesis is testable only if:

N_discordant >= 15

and:

N_nonzero >= 15

If either condition fails:

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY =
BLOCKED_NOT_TESTED

No threshold may be relaxed after inspecting the count.


## 9. Minimum effect size

Statistical significance alone is insufficient.

The frozen minimum effect is:

WITHIN_ITEM_CONCORDANCE >= 0.70

This threshold may not be lowered after outcome access.


## 10. Support rule

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY =
SUPPORTED

only if all are true:

1. N_discordant >= 15;
2. N_nonzero >= 15;
3. WITHIN_ITEM_CONCORDANCE >= 0.70;
4. exact one-sided sign-test p < 0.05;
5. median(delta_i) < 0;
6. frozen primary-population SHA256 passes;
7. each stable_id contributes exactly one primary delta_i.


## 11. Partial and negative verdicts

If feasibility passes and:

WITHIN_ITEM_CONCORDANCE > 0.5

but either the 0.70 minimum effect or p < 0.05 requirement fails:

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY =
DIRECTIONAL_ONLY_NOT_SUPPORTED

If feasibility passes and:

WITHIN_ITEM_CONCORDANCE <= 0.5

then:

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY =
NOT_SUPPORTED


## 12. Prespecified secondary seed-pair audit

After the primary verdict is fixed, report descriptive pairwise strata:

180_vs_181
180_vs_182
181_vs_182

For each pair, among stable IDs observed in both seeds with discordant D1
outcomes, report:

- pairwise discordant count;
- count where the flip seed has smaller d_A0;
- pairwise concordance;
- median pairwise margin difference.

These pairwise strata are secondary diagnostics only.

They may not override the stable_id-level primary verdict.

No favorable seed pair may be selected as the primary result.


## 13. Secondary magnitude description

After the primary verdict is fixed, report:

- median delta_i;
- mean delta_i;
- first quartile delta_i;
- third quartile delta_i;
- minimum delta_i;
- maximum delta_i.

These are descriptive only.

No post-hoc magnitude cutoff is authorized.


## 14. Anti-cherry-picking constraints

After outcome access, the following are prohibited:

- changing the discordance definition;
- restricting to recurrent Gen3 residual IDs;
- restricting to FROZEN51;
- restricting to one intervention family;
- removing seed182;
- selecting only favorable seed pairs;
- weighting stable IDs by number of observed seeds;
- allowing one stable_id to contribute multiple primary pairwise observations;
- switching from d_A0 logits to probability confidence;
- using absolute margin;
- tuning a margin threshold;
- lowering N_discordant = 15;
- lowering concordance = 0.70;
- changing the primary test;
- adding training seeds;
- launching new Gen3 experiments to rescue the result.


## 15. Interpretation boundary

If SUPPORTED, the permitted claim is:

The association between baseline A0 authorization-boundary proximity and D1
supportward susceptibility persists within stable_id across frozen training
seeds, reducing stable item identity as a sufficient explanation of the
previous Gen4 predictive result.

Even if supported, this does NOT establish:

- that boundary proximity causally produces D1 failure;
- that manipulating the boundary would prevent D1 failure;
- that D1 acts through one internal parameter path;
- that one edge or group is causal;
- that the same mechanism explains every example.

Therefore:

SCIENTIFIC_CLAIM_BOUNDARY =
ITEM_CONTROLLED_PREDICTIVE_NOT_CAUSAL


## 16. Consequence of a positive result

If SUPPORTED, the next scientifically justified phase is:

GEN4_BOUNDARY_CAUSAL_INTERVENTION_DESIGN

That phase must independently freeze a manipulation that changes baseline
authorization-boundary position while minimizing changes to the underlying
example and D1 mechanism.

This authority does NOT authorize that intervention.


## 17. Consequence of a negative result

If NOT_SUPPORTED or DIRECTIONAL_ONLY_NOT_SUPPORTED:

- do not retune d_A0;
- do not switch confidence metrics;
- do not remove unfavorable seeds;
- do not reinterpret the prior cross-sectional AUC as causal evidence.

A different mechanism requires a new independently motivated authority.


## 18. Authorized next operation after freeze

After this authority is committed, exactly one operation is authorized:

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY_READ_ONLY_STATIC_FALSIFICATION

It may:

- read the frozen primary_population.jsonl;
- authenticate its SHA256;
- construct the prespecified discordant stable_id population;
- compute delta_i;
- execute the exact sign test;
- compute frozen secondary diagnostics;
- write static evidence artifacts.

It may not:

- train;
- perform model inference;
- load checkpoints;
- use Kaggle;
- use GPU;
- alter Gen3 evidence;
- alter the prior Gen4 static evidence.


## 19. Final authority state

PRIMARY_UNIT =
STABLE_ID

PRIMARY_EFFECT =
MEAN_FLIP_D_A0_MINUS_MEAN_NONFLIP_D_A0

PRIMARY_DIRECTION =
NEGATIVE

PRIMARY_ENDPOINT =
WITHIN_ITEM_CONCORDANCE

PRIMARY_TEST =
EXACT_ONE_SIDED_BINOMIAL_SIGN_TEST

MINIMUM_DISCORDANT_IDS =
15

MINIMUM_NONZERO_IDS =
15

MINIMUM_CONCORDANCE =
0.70

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

END_OF_GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY_STATIC_FALSIFICATION_AUTHORITY
