# Generation-3 pairwise interaction selection static audit

```text
VERDICT = PASS_CANDIDATE
PHASE = GEN3_PAIRWISE_SELECTION_STATIC_AUDIT
AUTHORITY_INPUT = 28d7fa2fcc0d286a0cd16723c302a45f214b2901
SCIENTIFIC_EXECUTION = NOT_AUTHORIZED
TRAINING_EVALUATION_INFERENCE_CHECKPOINT_LOADING = NOT_PERFORMED
IMPLEMENTATION_CHANGE = NONE
PAIR_SELECTION_BY_NUMERIC_BEST_ARM_RANKING = PROHIBITED
```

## 1. Purpose and boundary

Generation-3 first-pass evidence is frozen at commit
`28d7fa2fcc0d286a0cd16723c302a45f214b2901`.

Its primary interpretation is `DISTRIBUTED_OWNERSHIP_DEPENDENCE`, with an
unresolved cumulative/non-additive component: GLOBAL-HALF/D1 is materially
worse than every single-edge arm, and 51/111 D1 A0-correct breaks are absent
from every single-edge condition.

This static audit asks only:

> Which bounded pairwise motif families are scientifically defensible for a
> future interaction study without selecting pairs by post-hoc numerical arm
> ranking?

This record does not authorize pairwise execution, Kaggle work, training,
evaluation, checkpoint loading, lambda changes, implementation changes,
commit, or push by itself.

## 2. Frozen scientific inputs

The pairwise design, if later authorized, must preserve the completed Gen3
coordinate exactly:

- router: `explicit_product`
- reason loss weight: `0`
- split seed: `8192`
- training seeds: `180, 181, 182`
- frozen Mamba encoder
- same dataset and split identities
- same primary reason order
- same diagnostic-only secondary reasons
- same final three-way semantics
- `frame_downstream_gradient_mode=joint`
- `gradient_ownership_mode=edge_specific`
- `lambda_probe=.5`
- no lambda sweep
- no adaptive ownership
- no conditional-first-blocker arm
- no K-series evidence consumption

For a future pairwise arm, exactly two named edges would be `.5` and the
remaining eight would be `1.0`.

## 3. Non-ranking selection rule

The pairwise candidate set is not derived from the numerical ordering of Gen3
single-edge task scores.

The selection rule uses two inputs that are independent of “pick the worst
two arms”:

1. the frozen pre-Gen3 D1 residual localization, which placed the unresolved
   residual on downstream authorization/polarity/final-boundary observables
   rather than on one earlier structural target; and
2. the frozen Gen3 topology, which identifies convergence at Q and D and the
   direct Q->D serial boundary.

Therefore the bounded pairwise study is restricted to downstream convergence
and Q->D serial motifs. Upstream-only pair families are excluded from the
first pairwise pass.

## 4. Admissible motif families

### Family Q — convergence into Q

Edges:

- G4 `F_TO_Q`
- G5 `P_TO_Q`
- G6 `S_TO_Q`

Complete within-family pair set:

1. `G4+G5`
2. `G4+G6`
3. `G5+G6`

Scientific question: does simultaneous attenuation of two semantic inputs to
the polarity/authorization representation create damage not present under
either constituent single-edge intervention?

### Family D — convergence into final decision D

Edges:

- G7 `F_TO_D`
- G8 `P_TO_D`
- G9 `S_TO_D`
- G10 `Q_TO_D`

Complete within-family pair set:

4. `G7+G8`
5. `G7+G9`
6. `G7+G10`
7. `G8+G9`
8. `G8+G10`
9. `G9+G10`

Scientific question: does simultaneous attenuation of two direct semantic
ingresses to the final decision reproduce global-only breakage that neither
single edge produces alone?

### Family QD — serial Q-to-D motifs

Pairs:

10. `G4+G10`
11. `G5+G10`
12. `G6+G10`

Scientific question: does weakening one semantic ingress into Q together with
Q's downstream ingress into D produce a serial non-additive effect?

These 12 pairs are the complete first-pass candidate matrix under this audit.
No pair is included because it had the lowest or highest Gen3 single-edge
metric.

## 5. Explicit exclusions

The following are not admitted to the first pairwise candidate matrix:

- all 45 pair combinations
- pair selection by the two numerically worst single-edge arms
- lambda values other than `.5`
- three-edge or higher-order attenuation
- adaptive or learned edge lambdas
- upstream-only exploratory pairs
- residual-stable-ID-specific pair selection
- seed-specific pair selection
- post-hoc pair replacement after seeing one pairwise result
- K-series-derived pairs

The excluded pairs may become a separate future question only after this
bounded downstream matrix is resolved.

## 6. Required future pairwise measurements

If execution is later authorized, each pair must be analyzed against the
matched historical A0, D1, and both constituent single-edge arms.

Required measurements are:

1. aggregate accuracy and macro-F1;
2. NE / REFUTE / SUPPORT F1;
3. exact A0-correct break and A0-wrong repair sets;
4. overlap with the D1 A0-correct break set;
5. overlap with the frozen 51 D1 break occurrences that were absent from all
   single-edge arms;
6. new D1-overlap rows created by the pair beyond the union of its two
   constituent single-edge break sets;
7. cross-seed recurrence of those new rows;
8. provenance/configuration reconciliation.

No scientific conclusion may rest on only one seed, one class, one stable ID,
or aggregate score ranking.

## 7. Predefined descriptive interpretations

A future analysis may use the following bounded categories.

### `PAIRWISE_NONADDITIVE_SIGNAL`

A pair produces reproducible D1-direction breakage not present in the union of
its two constituent single-edge break sets, with a consistent task-quality or
class-specific degradation direction across seeds.

This is evidence for a controlled pairwise non-additive interaction under the
defined intervention. It is not parameter-level mechanistic proof.

### `PAIRWISE_CUMULATIVE_ONLY`

The pair is more harmful than either constituent single-edge arm but its break
set is largely explained by the union of constituent single-edge breaks.

### `PAIRWISE_NULL_OR_TOLERANT`

The pair adds little interpretable damage relative to its constituent
single-edge arms.

### `PAIRWISE_ANTAGONISTIC_OR_REPAIRING`

The pair attenuates damage or repairs rows relative to one or both
constituents. Such a result must not be relabeled as “optimal ownership.”

No category establishes native Mamba state ownership, a unique parameter path,
an optimal lambda, or production readiness.

## 8. Implementation and verification gate for any future execution

No new model implementation is expected. Before pairwise execution can be
authorized, a separate verification must establish on the frozen Gen3
implementation commit lineage that:

1. the ten-entry edge map accepts exactly two `.5` values and eight `1.0`
   values without fallback;
2. forward values remain unchanged by the pairwise gradient attenuation;
3. both named recipient ingresses are scaled and no third edge is attenuated;
4. owner-local gradients remain preserved;
5. run provenance serializes the exact full ten-edge map and pair arm ID;
6. legacy JOINT and GLOBAL-HALF behavior remain untouched.

If existing tests already prove these properties generically, the verifier may
cite those exact tests. Otherwise only the narrow missing verification may be
added; no refactor is authorized.

## 9. Decision

```text
GEN3_FIRST_PASS = FROZEN
PAIRWISE_DESIGN_JUSTIFICATION = PRESENT
PAIRWISE_SELECTION_RULE = TOPOLOGY_AND_PRE_GEN3_DOWNSTREAM_LOCALIZATION
PAIRWISE_CANDIDATE_COUNT = 12
PAIRWISE_EXECUTION = NOT_AUTHORIZED
NEXT_PHASE = PAIRWISE_IMPLEMENTATION_CAPABILITY_READ_ONLY_VERIFICATION
```

The immediate next step after freezing this audit is a read-only verification
of whether the current frozen Gen3 implementation already supports the exact
two-edge `.5` configuration and provenance contract. No scientific pairwise
run should begin before that verification and a separate execution authority.
