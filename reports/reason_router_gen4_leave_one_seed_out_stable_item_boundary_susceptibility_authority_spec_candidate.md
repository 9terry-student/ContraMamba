# Generation-4 Leave-One-Seed-Out Stable-Item Boundary Susceptibility
# Static Audit Authority Specification

STATUS = CANDIDATE

PHASE =
GEN4_LEAVE_ONE_SEED_OUT_STABLE_ITEM_BOUNDARY_SUSCEPTIBILITY_STATIC_AUDIT

PARENT_WITHIN_ITEM_FALSIFICATION_EVIDENCE =
c81eba59cb11a0ed845371ea4ef37dabc05ec9ad

PARENT_CROSS_SECTIONAL_EVIDENCE =
5388460bf0ac2ed1f8489e35c6942f6044da3df0

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


## 1. Scientific motivation

The Generation-4 cross-sectional audit established strong prediction of D1
supportward failure from same-seed A0 SUPPORT / NOT_ENTITLED boundary
proximity.

The subsequent within-item falsification did NOT support the hypothesis that
seed-to-seed changes in the same item's A0 boundary proximity track its
seed-to-seed D1 outcome.

Observed within-item result:

N_DISCORDANT =
37

K_NEGATIVE =
17

K_POSITIVE =
20

WITHIN_ITEM_CONCORDANCE =
0.4594594594594595

MEDIAN_DELTA_I =
0.02173464000225067

EXACT_ONE_SIDED_SIGN_TEST_P =
0.7443121092073852

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY =
NOT_SUPPORTED

Therefore direct causal promotion of same-seed boundary proximity is not
authorized.

A remaining scientifically distinct possibility is that the strong
cross-sectional signal reflects a stable item-associated boundary component.

This audit tests that possibility without using the target seed's own A0
margin as its predictor.


## 2. Primary hypothesis

STABLE_ITEM_BOUNDARY_COMPONENT_HYPOTHESIS =

An item's A0 authorization-boundary position measured in OTHER frozen training
seeds predicts whether that same item undergoes a D1 SUPPORT flip in a held-out
target seed.

The hypothesis concerns stable cross-seed item-associated geometry.

It does not assert that the boundary itself is causal.


## 3. Frozen input

This audit may read only the frozen Generation-4 primary population artifact:

primary_population.jsonl

SHA256:

eb5aaacf38be224461801cab5e0629e46a9d375d7efaeadb72b2dbf3c5d56f3e

Relevant frozen fields are exactly:

- seed
- stable_id
- d_a0_ne_minus_support
- y_d1_support

No raw prediction source, checkpoint, dataset, or model forward is authorized.


## 4. Held-out unit

For each target seed:

s in {180, 181, 182}

the evaluation unit is an already-frozen row:

(s, stable_id)

The target row is eligible only when the same stable_id has at least one
OTHER-seed row in the frozen primary population.


## 5. Leave-one-seed-out predictor

For eligible target row (i, s), define:

d_LOO(i,s) =
mean d_a0_ne_minus_support(i,t)

over all available frozen rows for the same stable_id i satisfying:

t != s

The susceptibility score is:

score_LOO(i,s) =
-d_LOO(i,s)

Smaller other-seed boundary margin therefore corresponds to larger predicted
susceptibility.

CRITICAL LEAKAGE RULE:

The target seed's own d_a0_ne_minus_support MUST NOT enter d_LOO(i,s).


## 6. Held-out outcome

For target row (i,s):

Y(i,s) =
y_d1_support(i,s)

where:

Y = 1

means the frozen D1 prediction in target seed s is SUPPORT.

Other-seed D1 outcomes MUST NOT enter the predictor.


## 7. Primary endpoints

For each target seed independently compute ROC AUC using:

score_LOO(i,s)

to predict:

Y(i,s)

The three prespecified endpoints are:

LOO_AUC_180
LOO_AUC_181
LOO_AUC_182

No pooled AUC may replace the three seed-specific endpoints.


## 8. Minimum outcome-count gate

For every target seed, the eligible held-out population must contain at least:

10 positive Y = 1 rows

and:

30 negative Y = 0 rows

If any seed fails this requirement:

GEN4_STABLE_ITEM_BOUNDARY_COMPONENT =
BLOCKED_NOT_TESTED


## 9. Primary statistical test

For each target seed independently test:

H0:
LOO_AUC_s <= 0.5

against:

H1:
LOO_AUC_s > 0.5

using a one-sided label-permutation test within that target seed.

The permutation preserves:

- exact eligible held-out rows;
- exact LOO predictor values;
- exact number of positive outcomes.

Use:

100000

permutations per target seed.

The permutation RNG seed MUST be frozen in a committed execution manifest
before any LOO AUC or outcome association is computed.

The execution-manifest RNG seed shall be derived deterministically as:

int(first_8_hex_of_this_authority_commit, 16)

Correct the three primary p-values using:

HOLM_FWER_ALPHA_0.05


## 10. Minimum effect

For each target seed:

LOO_AUC_s >= 0.70

is the frozen minimum substantive effect.

This threshold may not be lowered after outcome inspection.


## 11. Primary support rule

GEN4_STABLE_ITEM_BOUNDARY_COMPONENT =
SUPPORTED

only if all are true:

1. every target seed passes the minimum outcome-count gate;
2. every LOO_AUC_s > 0.5;
3. every LOO_AUC_s >= 0.70;
4. every Holm-adjusted p < 0.05;
5. exact frozen population SHA256 passes;
6. target-seed d_A0 is absent from every LOO predictor;
7. other-seed D1 outcomes are absent from every predictor.


## 12. Partial and negative verdicts

If all three AUC values are > 0.5 but one or more seeds fail either:

LOO_AUC >= 0.70

or:

HOLM_ADJUSTED_P < 0.05

then:

GEN4_STABLE_ITEM_BOUNDARY_COMPONENT =
DIRECTIONAL_ONLY_NOT_SUPPORTED

If any target seed has:

LOO_AUC <= 0.5

then:

GEN4_STABLE_ITEM_BOUNDARY_COMPONENT =
NOT_SUPPORTED


## 13. Secondary predictor-stability description

Only after the primary verdict is fixed, the audit may report, for each seed
pair:

180_vs_181
180_vs_182
181_vs_182

among stable IDs present in both populations:

- common stable_id count;
- Pearson correlation of d_A0;
- Spearman rank correlation of d_A0.

These are descriptive only.

They may not alter the primary verdict.


## 14. Secondary coverage description

For each target seed report:

- original frozen target-seed primary population count;
- LOO-eligible count;
- excluded count due solely to absence of any other-seed A0 margin;
- positive and negative counts after LOO eligibility.

Coverage is descriptive except for the frozen minimum outcome-count gate.


## 15. Anti-leakage and anti-cherry-picking constraints

After outcome access, the following are prohibited:

- including target-seed d_A0 in the predictor;
- using target-seed probability confidence instead;
- using other-seed D1 outcomes as predictor inputs;
- outcome-weighted averaging across other seeds;
- choosing only the closer other-seed margin;
- choosing only the farther other-seed margin;
- selecting favorable target seeds;
- dropping seed182;
- restricting to FROZEN51;
- restricting to recurrent residual IDs;
- restricting to swap family after seeing results;
- changing the LOO averaging rule;
- changing the AUC threshold;
- lowering the outcome-count gate;
- adding training seeds;
- fitting a learned classifier;
- threshold tuning;
- returning to Gen3 combinatorial search.


## 16. Interpretation of a positive result

If SUPPORTED, the permitted claim is:

A stable cross-seed item-associated A0 boundary component predicts D1
supportward susceptibility in held-out training seeds, even when the target
seed's own A0 margin is excluded from the predictor.

This would support the interpretation that the previous cross-sectional
boundary signal primarily reflects stable item-associated vulnerability rather
than seed-specific boundary fluctuation.

It would NOT establish:

- that boundary position causes D1 failure;
- what latent item property generates that stable geometry;
- that manipulating the margin will alter D1 outcome;
- any unique edge, group, or gradient-path mechanism.


## 17. Interpretation of a negative result

If NOT_SUPPORTED or DIRECTIONAL_ONLY_NOT_SUPPORTED:

the boundary-susceptibility research line is not authorized to rescue itself
by redefining confidence, margin, seeds, or subgroups.

The prior cross-sectional AUC remains a valid association but is insufficient
to establish a stable cross-seed vulnerability signal.

A different independently motivated mechanism is then required.


## 18. Consequence of a positive result

If SUPPORTED, the next justified research phase is:

GEN4_STABLE_ITEM_VULNERABILITY_MECHANISM_DESIGN

That phase should seek independently motivated item-level structural features
that could explain the stable vulnerability.

This authority does not authorize that subsequent mechanism audit.


## 19. Authorized next operation after freeze

After this authority is committed, the next authorized operation is:

GEN4_STABLE_ITEM_BOUNDARY_COMPONENT_EXECUTION_MANIFEST_FREEZE

The manifest must freeze:

- authority commit;
- frozen population SHA256;
- exact LOO construction;
- target seeds;
- permutation count;
- deterministic RNG seed;
- AUC threshold;
- count gates;
- Holm alpha.

Scientific outcome association MUST NOT be computed before that manifest is
committed.


## 20. Final authority state

PRIMARY_PREDICTOR =
LEAVE_ONE_SEED_OUT_MEAN_A0_NE_MINUS_SUPPORT_MARGIN

PRIMARY_OUTCOME =
HELD_OUT_SEED_D1_SUPPORT_FLIP

PRIMARY_ENDPOINTS =
LOO_AUC_180_LOO_AUC_181_LOO_AUC_182

MINIMUM_AUC_EACH_SEED =
0.70

PERMUTATIONS_EACH_SEED =
100000

MULTIPLICITY =
HOLM_FWER_0.05

TRAINING_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

END_OF_GEN4_LEAVE_ONE_SEED_OUT_STABLE_ITEM_BOUNDARY_SUSCEPTIBILITY_AUTHORITY
