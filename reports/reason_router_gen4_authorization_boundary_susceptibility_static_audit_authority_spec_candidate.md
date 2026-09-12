# Generation-4 Authorization Boundary Susceptibility
# Static-Audit Authority Specification

STATUS = CANDIDATE

PHASE =
GEN4_AUTHORIZATION_BOUNDARY_SUSCEPTIBILITY_STATIC_AUDIT

PARENT_GEN3_VALIDATED_EVIDENCE =
ab80d9fbab103c73aaa459b10c5caeaa2f69dfd4

PARENT_GROUPED_VALIDATED_EVIDENCE =
c97fd33dd8aa9c116f45071ee4545093ebba8f1c

FROZEN51_VALIDATED_EVIDENCE =
792b7a7e9bdb7fa957fa3ecda6ee9c6e37ce0f96

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

NEW_GEN3_COMBINATORIAL_EXECUTION_ALLOWED =
NO


## 1. Scientific motivation

Generation-3 did not identify one stable edge, one stable macro-group, or one
D1-specific higher-order geometry that explains the residual failure
population.

The final Generation-3 grouped residual audit found:

PROPER_SUBSET_NEAR_THRESHOLD =
2_OF_4

DISTRIBUTED_NEAR_THRESHOLD =
2_OF_4

D1_SPECIFIC_RESIDUAL_GEOMETRY =
0_OF_4

OVERALL_RESIDUAL_VERDICT =
HETEROGENEOUS_RESIDUAL

NEW_HIGHER_ORDER_EXECUTION_JUSTIFIED_BY_THIS_AUDIT =
NO

The dominant proper subset can vary across seed and example while several
already-tested subsets nevertheless move examples toward the final decision
boundary.

Generation-4 therefore changes the scientific question.

It does not ask:

WHICH_EDGE_OR_GROUP_IS_THE_SINGLE_CAUSE

and does not continue residual combinatorial search.

It asks whether susceptibility to the known D1 supportward failure is already
visible in the baseline A0 decision geometry before D1 is applied.


## 2. Primary Gen4 hypothesis

AUTHORIZATION_BOUNDARY_SUSCEPTIBILITY_HYPOTHESIS =

Among examples that canonical A0 classifies correctly as NOT_ENTITLED,
examples that lie closer to the A0 SUPPORT / NOT_ENTITLED decision boundary
are more susceptible to a D1-induced SUPPORT prediction.

The hypothesis concerns pre-existing decision-boundary susceptibility.

It does not assert that boundary proximity is the causal mechanism that
creates D1 perturbations.


## 3. Unit of analysis

The primary analysis unit is:

(training_seed, stable_id)

for the three frozen training seeds:

180
181
182

using the exact historical clean-dev population already admitted by
Generation-3 provenance validation.

No additional seed is permitted.


## 4. Primary population

For each frozen seed independently, begin from the exact clean-dev prediction
population and retain only rows satisfying:

1. canonical A0 prediction is correct;
2. gold label is NOT_ENTITLED;
3. canonical A0 predicted label is NOT_ENTITLED;
4. both canonical A0 and historical D1 final logits are available under the
   same external class order;
5. stable_id joins exactly between canonical A0 and historical D1.

No row may be selected using Generation-3 edge, pairwise, grouped, FROZEN51,
or residual membership.

Those prior labels are not inputs to primary population construction.


## 5. Primary predictor

For each admitted A0 row define:

d_A0 =
z_A0(NOT_ENTITLED) - z_A0(SUPPORT)

where z denotes the frozen final decision logit.

Because the primary population requires A0 prediction = NOT_ENTITLED,
positive d_A0 means the row is on the NOT_ENTITLED side of the
SUPPORT / NOT_ENTITLED boundary.

Smaller d_A0 means closer baseline boundary proximity.

The susceptibility score used for discrimination is:

s_A0 =
-d_A0

so a larger score means more susceptible under the prespecified hypothesis.

No alternative confidence statistic, probability transform, entropy,
top-two margin, normalized margin, temperature scaling, or learned score may
replace this primary predictor after outcome inspection.


## 6. Primary outcome

For each admitted row define:

Y_D1 = 1

iff the matched historical D1 prediction is:

SUPPORT

Otherwise:

Y_D1 = 0

The primary outcome is therefore a supportward authorization failure under D1.

The primary analysis does not redefine failure using aggregate accuracy,
FROZEN51 membership, single-edge overlap, grouped overlap, or residual class.


## 7. Primary endpoint

For each seed s independently compute:

AUC_s =
P(
    d_A0(D1_SUPPORT_FLIP)
    <
    d_A0(D1_NON_SUPPORT)
)

with standard half-credit for ties.

Equivalently, AUC_s is the ROC AUC obtained using susceptibility score:

s_A0 = -d_A0

to discriminate Y_D1.

The three prespecified primary endpoints are therefore:

AUC_180
AUC_181
AUC_182


## 8. Primary statistical test

For each seed independently test:

H0:
AUC_s <= 0.5

against:

H1:
AUC_s > 0.5

using a one-sided permutation test that permutes Y_D1 labels within that
seed while preserving:

- the exact admitted A0 population;
- the number of Y_D1 = 1 rows;
- every d_A0 value.

Use at least:

100000

deterministic permutations per seed unless exact enumeration is smaller.

The permutation RNG seed must be frozen before execution and must be the same
for reruns of this audit.

Correct the three primary p-values using:

HOLM_FWER_ALPHA_0.05


## 9. Minimum effect size

Statistical significance alone is not sufficient.

For each seed:

AUC_s >= 0.60

is the prespecified minimum nontrivial discrimination effect.

This threshold is frozen before the new boundary audit is executed.

It may not be reduced after results are observed.


## 10. Primary support rule

GEN4_BOUNDARY_SUSCEPTIBILITY =
SUPPORTED

only if all of the following hold:

1. all three seeds have AUC_s > 0.5;
2. all three seeds have AUC_s >= 0.60;
3. all three Holm-adjusted primary p-values are < 0.05;
4. each seed contains at least 10 Y_D1 = 1 primary rows;
5. each seed contains at least 30 Y_D1 = 0 primary rows;
6. exact source provenance and joins pass.

This is intentionally a reproducibility requirement across all three frozen
training seeds.


## 11. Partial and negative outcomes

If:

all three AUC_s > 0.5

but one or more seeds fail either:

AUC_s >= 0.60

or:

HOLM_ADJUSTED_P < 0.05

then:

GEN4_BOUNDARY_SUSCEPTIBILITY =
DIRECTIONAL_ONLY_NOT_SUPPORTED

If any seed has:

AUC_s <= 0.5

then:

GEN4_BOUNDARY_SUSCEPTIBILITY =
NOT_SUPPORTED

If any seed fails the minimum outcome-count requirement, provenance binding,
schema validation, or exact A0/D1 join:

GEN4_BOUNDARY_SUSCEPTIBILITY =
BLOCKED_NOT_TESTED


## 12. Secondary descriptive analysis

Only after the primary endpoints are frozen and computed, the audit may
report a fixed five-bin calibration description.

Within each seed, rank admitted rows by increasing d_A0 and partition them
into exactly five equal-frequency bins as deterministically as possible.

BIN_1 =
closest 20 percent to the SUPPORT / NOT_ENTITLED boundary

BIN_5 =
farthest 20 percent

Report for each bin:

- row count;
- D1 SUPPORT-flip count;
- D1 SUPPORT-flip rate;
- median d_A0.

This five-bin analysis is descriptive only.

It may not override the primary AUC/Holm verdict.


## 13. Prespecified Gen3 linkage analysis

After the primary verdict is fixed, the audit may descriptively annotate
whether D1 SUPPORT-flip rows belong to previously frozen:

- FROZEN51;
- recurrent FROZEN51;
- final four-ID grouped residual;
- PROPER_SUBSET_NEAR_THRESHOLD;
- DISTRIBUTED_NEAR_THRESHOLD.

These annotations are secondary interpretation only.

They must not alter:

- primary population;
- primary predictor;
- primary outcome;
- primary p-values;
- support criterion.


## 14. Anti-cherry-picking constraints

The following are prohibited after outcome access:

- switching from logits to probabilities;
- switching to generic top-two confidence;
- choosing a different class pair;
- restricting to FROZEN51 to improve separation;
- restricting to recurrent residual IDs;
- removing difficult intervention families;
- selecting only one or two favorable seeds;
- adding seeds;
- changing the AUC >= 0.60 criterion;
- changing Holm alpha;
- changing the five-bin definition;
- fitting a nonlinear classifier;
- fitting a learned threshold;
- tuning a margin cutoff;
- choosing an edge/group conditional score;
- adding interaction terms;
- using grouped condition outcomes as primary predictors;
- running new Gen3 arms to rescue a negative result.


## 15. Interpretation boundary

If supported, the permitted claim is:

Baseline A0 SUPPORT / NOT_ENTITLED boundary proximity is a reproducible
predictor of D1 supportward susceptibility across the three frozen seeds.

The following stronger claims remain unauthorized:

- boundary proximity causes D1 failure;
- D1 acts only through decision-boundary geometry;
- one parameter group is causal;
- one gradient path is causal;
- U+D is universally causal;
- all residual errors share one mechanism;
- the router is calibrated;
- the architecture is production ready.

A positive static audit may justify a later independently frozen causal
Generation-4 intervention design.

It does not itself authorize that intervention.


## 16. Failure consequence

If the primary hypothesis is NOT_SUPPORTED or
DIRECTIONAL_ONLY_NOT_SUPPORTED:

- do not tune the margin definition;
- do not lower the AUC threshold;
- do not change the primary class;
- do not launch a higher-order Gen3 search;
- do not add seeds to rescue the hypothesis.

A different Generation-4 mechanism would require a new independently
motivated hypothesis and new authority.


## 17. Provenance requirements before audit

Before any new boundary statistic is computed, the audit must establish:

1. exact historical A0 prediction artifact identity for seeds 180/181/182;
2. exact historical D1 prediction artifact identity for seeds 180/181/182;
3. SHA256 binding to previously validated Generation-3 evidence;
4. exactly 720 clean-dev stable IDs per source where historically expected;
5. unique stable IDs;
6. exact external class order;
7. valid final-logit schema;
8. exact A0/D1 stable-id join per seed;
9. no checkpoint loading;
10. no model forward.

If the exact six source artifacts cannot be authenticated:

AUDIT_RESULT =
BLOCKED_PROVENANCE


## 18. Authorized next operation after freeze

After this authority is frozen, the next authorized operation is exactly:

GEN4_BOUNDARY_SUSCEPTIBILITY_READ_ONLY_STATIC_AUDIT

That operation may:

- read the six already-existing A0/D1 prediction exports;
- authenticate their provenance;
- compute the prespecified primary and secondary statistics;
- write static analysis artifacts.

It may not:

- train;
- evaluate a model by forward pass;
- load checkpoints;
- change model code;
- change dataset split;
- execute Kaggle;
- use GPU;
- run new experimental arms.


## 19. Final authority state

GEN4_SCIENTIFIC_QUESTION_FROZEN =
YES_ONCE_COMMITTED

GEN4_PRIMARY_PREDICTOR =
A0_NOT_ENTITLED_MINUS_SUPPORT_LOGIT_GAP

GEN4_PRIMARY_OUTCOME =
D1_SUPPORT_PREDICTION_AMONG_A0_CORRECT_NOT_ENTITLED

GEN4_PRIMARY_ENDPOINTS =
AUC_180_AUC_181_AUC_182

GEN4_MULTIPLICITY_CONTROL =
HOLM_FWER_0.05

GEN4_MINIMUM_EFFECT =
AUC_AT_LEAST_0.60_EACH_SEED

GEN4_EXECUTION_TYPE =
READ_ONLY_STATIC_AUDIT_ONLY

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

NEW_GEN3_COMBINATORIAL_EXECUTION_ALLOWED =
NO

END_OF_GEN4_AUTHORIZATION_BOUNDARY_SUSCEPTIBILITY_STATIC_AUDIT_AUTHORITY
