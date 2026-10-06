# Gen5 A-init Temporal Mechanism Static Synthesis Report

## Status

`PASS_STATIC_TEMPORAL_MECHANISM_SYNTHESIS`

This is a CPU/static synthesis of already frozen evidence. It performs no
model forward, GPU work, training, optimizer construction/step, backward, or
new evaluation. It introduces no new seeds, rows, thresholds, controls, or
critical times.

## Evidence identity

- synthesis HEAD: `acbd90f78a7aaa659c93548fc7fac6ba9a869a53`
- recovered execution HEAD: `d9b790b62f8db9f875cfa13d7ba155ac75137a2b`
- recovery summary SHA256: `05cd49181ec3874fdcee16c78ce9b7e1ba5f7d907718bf0e7c56d17f9865bfeb`
- behavioral-coordinate SHA256: `41518c0dbac345b972bf61920fe98681541f6393d6770494acb0df1cce149101`
- shared-vulnerability static SHA256: `cb51c6158419860baa4727d18ad78dd4c313de10f0d5115717fe62e34b8c720a`
- behavioral-onset summary SHA256: `dd5cd23bc8d03bfcaf4db849e31fe1886e4b46a864430465d90e09222fe4a997`
- frozen 120-row identity: authenticated exactly
- t20 valid-token geometry authentication: PASS
- t20 finite-effect/replay authentication: PASS
- full local-projector endpoint authentication: FAIL / claim blocked

## 1. First-update algebra and B1 origin

Already frozen first-update evidence established the structural boundary:

`B0 = 0 -> grad_A0 = 0 -> A1 is decay-only -> B1 is the first gradient-driven write`

The recovered four-state forward counterfactual now closes the functional side:

- `A_DECAY_ONLY` versus `T0` logit max abs:
  `0`
- `B_UPDATE_ONLY` versus `FULL_T1` prediction disagreement:
  `0`
- `B_UPDATE_ONLY` versus `FULL_T1` logit max abs:
  `1.430511474609375e-06`

Bounded conclusion:

> With B0 exactly zero, the A1 decay-only state is functionally identical to
> T0. Holding A at A0 while replacing B0 by B1 reproduces FULL_T1 predictions
> exactly and logits to approximately 1.43e-6 max absolute error. The first
> post-update behavioral birth is therefore already carried by the B1-driven
> write.

This does **not** make B1 independent of A0. The frozen first-update mechanism
shows that the B0 gradient, and hence B1, is A0-dependent.

## 2. t1 geometry, continuous decision coordinates, and behavior co-birth

At t1 the temporal-mechanism pairwise definition gives:

- A-axis prediction-disagreement aggregate: `144`
- R-axis prediction-disagreement aggregate: `0`
- two-margin A normalized residual: `0.0365629211956`
- two-margin R normalized residual: `7.7918783138e-05`
- A/R two-margin residual ratio: `469.244x`
- vulnerable-120 A/R pairwise margin-distance ratio:
  `354.268x`

Thus first-update operator/state birth, continuous decision-space separation,
and discrete A-axis behavioral divergence all appear at the first post-update
boundary. A-init dominance is not an argmax-only artifact.

### Aggregate-definition note

The older behavioral-onset artifact reports t1 A-axis disagreement `72`,
whereas the temporal-mechanism pairwise aggregate is `144`. These are different
aggregation definitions. The correction commit fixes the temporal runtime gate
to the current all-unordered-pairs definition; historical artifacts are not
rewritten.

## 3. Shared-120 transient excursion structure

The shared vulnerable population is exactly the same 120 rows in all nine
factor cells, producing 1,080 row-cell episodes.

Every episode follows the same discrete corridor:

`SUPPORT correct -> REFUTE transient decisive-error -> NOT_ENTITLED`

Frozen facts:

- gold SUPPORT rows: `120 / 120`
- exact corridor rows: `120 / 120`
- exact corridor event instances:
  `1080 / 1080`
- re-entry episodes: `0`
- persistent decisive-wrong episodes at t20:
  `0`

A-init owns onset timing:

- A fraction of rowwise onset SS: `0.996708721887`
- R fraction: `0.00109709270433`
- A×R fraction: `0.00219418540867`
- A/R timing SS ratio: `908.5`
- mean onset difference, same A / different R:
  `0.00740740740741` steps
- mean onset difference, same R / different A:
  `1.26296296296` steps

Peak active transient timing also shifts by A-init:

- A6201: t=12, active=354
- A6202: t=10, active=345
- A6203: t=11, active=360

This supports a shared vulnerability identity with A-controlled temporal
placement, not separate seed-specific vulnerable subsets.

## 4. Continuous margin path through the excursion and t10/t11 peak region

For the prospectively frozen 120 rows across all nine cells, the following are
descriptive means over the 1,080 row-cell episodes. No post-hoc threshold or
row subset is introduced.

| t | mean m_refute | mean m_support | SUPPORT count | NOT_ENTITLED count | REFUTE count |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.58289991021 | 0.283617582048 | 1080 | 0 | 0 |
| 1 | 2.55676248316 | 0.295350033252 | 1080 | 0 | 0 |
| 4 | 2.25584683617 | 0.430184956861 | 1077 | 0 | 3 |
| 10 | 0.0519270361022 | 0.721812738285 | 108 | 12 | 960 |
| 11 | -0.24552982593 | 0.541869984181 | 21 | 81 | 978 |
| 16 | -0.885851401091 | -0.568292048132 | 0 | 1077 | 3 |
| 17 | -0.925488342124 | -0.690875082408 | 0 | 1080 | 0 |
| 20 | -0.98345487471 | -0.881513566717 | 0 | 1080 | 0 |

The frozen behavioral-onset scan's own disagreement metric peaks at:

- t=10, A-init disagreement count=868

Its accuracy spread peaks at:

- t=11, spread=0.186904761904762

Continuous two-margin separation remains A-dominated in this peak region:

- t10 A residual: `0.398730280496`
- t10 R residual: `0.00284476869942`
- t10 A/R ratio: `140.163x`
- t11 A residual: `0.34818521368`
- t11 R residual: `0.00281900994958`
- t11 A/R ratio: `123.513x`

This supplies the continuous decision-coordinate bridge through the already
frozen SUPPORT -> REFUTE -> NOT_ENTITLED excursion.

## 5. t16 -> t17 argmax-only reconvergence

The frozen A-axis prediction-disagreement aggregate changes from
`6`
at t16 to
`0`
at t17.

Yet at t17:

- two-margin A normalized residual: `0.0618699436478`
- vulnerable-120 A pairwise margin mean L2:
  `0.141074352258`
- vulnerable-120 R pairwise margin mean L2:
  `0.00237139611434`
- vulnerable-120 A/R ratio:
  `59.49x`
- t17 two-margin A residual / t1 A residual:
  `1.69215x`

Therefore the permanent prediction reconvergence at t17 is:

`ARGMAX_ONLY_RECONVERGENCE`

The cells enter a common decision region before their continuous task
coordinates or internal A-init representations become equivalent.

## 6. Reconvergence is much faster in task space than in internal geometry

Across the exact t16 -> t17 boundary:

- two-margin A residual:
  `0.0884677324135 -> 0.0618699436478`
  (`30.065%` drop)
- raw_write A residual:
  `0.535523146767 -> 0.512088690638`
  (`4.376%` drop)
- layer22_out_proj A residual:
  `0.199701622663 -> 0.187527993551`
  (`6.096%` drop)

The exact argmax reconvergence step therefore does not correspond to an
internal representation collapse.

## 7. t17 -> t20 differential convergence and persistent state difference

After predictions have already reconverged:

- two-margin A residual:
  `0.0618699436478 -> 0.0227506430119`
  (`63.228%` drop)
- raw_write A residual:
  `0.512088690638 -> 0.459227098173`
  (`10.323%` drop)
- layer22_out_proj A residual:
  `0.187527993551 -> 0.155009034787`
  (`17.341%` drop)

The task coordinates converge substantially faster than the internal
A-init-dependent state geometry.

For comparison, raw_write -> layer22_out_proj normalized-residual survival is:

- t1: `0.845423`
- t20: `0.337543`

Training therefore changes the downstream expression/gain of the A-init state
difference without erasing the underlying representational separation.

## 8. Quotient / representational non-identifiability interpretation

The combined frozen evidence supports the following bounded interpretation:

`large persistent A-init-dependent internal difference`
`+ shrinking task-coordinate difference`
`+ common final argmax`
`= task-relevant representational non-identifiability`

This is consistent with the already frozen endpoint task-reachable quotient
and low-gain-complement evidence: different internal coordinates can belong to
the same task-equivalent behavioral class because much of their difference is
weakly expressed downstream.

This is an **interpretation supported by the measured quotient-like
compression**, not a claim that a complete mathematical equivalence relation
over all hidden states has been proven.

## 9. Claim boundary and milestone closure

The recovered execution does **not** fully authenticate the local
projector-energy gate at t1/t17 because the original task-visible accumulator
included padded coordinates.

Therefore do **not** claim:

`FULLY_AUTHENTICATED_EARLIEST_INTERNAL_TASK_VISIBLE_STAGE_AT_T1 = raw_write`

The correction at `acbd90f78a7aaa659c93548fc7fac6ba9a869a53` fixes future valid-token semantics but does
not retroactively replace the executed projector statistics.

Current closure:

- M3b / t1 within-update origin:
  **CLOSED — B1-driven write under the frozen four-state counterfactual**
- transient excursion structure:
  **CLOSED descriptively — shared 120-row corridor with A-controlled timing**
- M4 / reconvergence:
  **CLOSED — ARGMAX_ONLY_RECONVERGENCE**
- M5 / finite functional birth:
  **CLOSED at parameter-state finite-counterfactual level**
- M6 / temporal propagation:
  **CLOSED for valid-token geometry and continuous task-coordinate bridge**
- earliest internal task-visible projector gate at t1/t17:
  **NOT FULLY AUTHENTICATED; no claim**
- M7 / factor-swap causal closure:
  **NOT YET DONE**

## Integrated temporal mechanism

`A_INIT_COORDINATE_CHOICE`
`-> A0_DEPENDENT_STEP0_B_GRADIENT`
`-> B1_WRITE_BIRTH`
`-> t1_CONTINUOUS_AND_BEHAVIORAL_CO_BIRTH`
`-> SHARED_120_TRANSIENT_EXCURSION_WITH_A_CONTROLLED_TIMING`
`-> PROGRESSIVE_DOWNSTREAM_ATTENUATION`
`-> ARGMAX_ONLY_RECONVERGENCE`
`-> PERSISTENT_TASK_EQUIVALENT_INTERNAL_DIFFERENCE`

## Next scientific action

Do not rerun the eight-hour temporal scan and do not return to projector
recovery before the next causal test.

The next scientific execution is M7: a bounded finite factor-swap
counterfactual over the frozen t1 factors. It should test whether behavior
follows recipient A0 coordinates, donor B1, or only their matched pair.

M7 must use:

- the frozen 3x3 A-init × training-RNG grid;
- already frozen t1 tensors only;
- no new seeds;
- no training;
- no optimizer construction/step;
- no backward;
- no parameter updates;
- finite forward counterfactuals only.

A successful M7 is required before upgrading the current statement from
"A-init/factor structure predicts the trajectory" to a claim that a specific
learned factor causally controls excursion/reconvergence behavior.
