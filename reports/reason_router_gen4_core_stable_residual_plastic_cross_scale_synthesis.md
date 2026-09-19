# ContraMamba Cross-Scale Mechanistic Synthesis
## CORE-STABLE / RESIDUAL-PLASTIC through Mamba-1.4B

### Status

`STATIC_DESCRIPTIVE_CROSS_SCALE_SYNTHESIS`

This report synthesizes the frozen ContraMamba native-Mamba evidence through the
prospective Mamba-1.4B test of the pre-specified:

`CORE-STABLE / RESIDUAL-PLASTIC`

hypothesis.

It introduces:

- no new model execution;
- no training or backward pass;
- no new response population;
- no new p-value;
- no alternative-tail test;
- no layer, token, epsilon, or plane rescue;
- no reopening of discovery selection;
- no reinterpretation of the failed historical 370M joint residual-recurrence criterion.

The Mamba-1.4B residual section is descriptive only.

Frozen synthesis HEAD:

`1ccd58eb24dabe028be0c85f65a24f2745400e2b`

---

## 1. Frozen evidence chain

### 1.1 Prospective hypothesis design

Design:

`reports/reason_router_gen4_core_stable_residual_plastic_cross_scale_hypothesis_design.md`

The design defined two deliberately different claims.

**Core-stable** means recurrence of a scale-local dominant causal role under the same
scientific procedure. It does not require the same principal-plane rank across scales.

For each scale:

`k*_s = unique argmax_k mean_i[Q_restored(s,k,i) - Q_neutralized(s,k,i)]`

with response-blind control:

`c*_s = argmax_{k != k*_s} lambda^+_s,k`.

The fresh confirmatory endpoint is:

`D_CORE_s = Q_restored(k*_s) - Q_control(c*_s)`.

The prospective 1.4B core hypothesis required:

`mean(D_CORE_1.4B) > 0`

and exactly one one-sided Student t-test at alpha `0.05`.

**Residual-plastic** is a comparative descriptive model. It allows the non-dominant
residual to change in:

- signed individual effects;
- coefficient-mass distribution;
- generator-family coupling;
- dominant residual rank;
- and aggregate sign.

A residual result cannot rescue a failed core test.

### 1.2 Mamba-1.4B frozen execution chain

Geometry freeze:

`f97b597fb4da08a8d360ed07e48c6727e15ae0be`

Discovery evidence freeze:

`e7804bcab88a50dd57c2edd86af88daeb90a56d5`

Core confirmation evidence freeze:

`693c12fef2194cbc48d17ed8fc563cc5065dcae0`

Residual characterization implementation:

`08fbb5f2c00d31ac50a60a7199aa2e2dd8cfa1ec`

Residual characterization evidence freeze:

`1ccd58eb24dabe028be0c85f65a24f2745400e2b`

The three 1.4B response populations were disjoint:

- discovery: `xg1_fact_3901..4200`, `N=300`;
- core confirmation: `xg1_fact_4201..4500`, `N=300`;
- residual characterization: `xg1_fact_4501..4800`, `N=300`.

---

## 2. Core-stable result

### 2.1 Scale-local rank identity changes and is not the invariant

The frozen selected ranks are:

- Mamba-370M: `k* = P3`;
- Mamba-1.4B: `k* = P5`.

The response-blind controls are:

- Mamba-370M: `P5`;
- Mamba-1.4B: `P4`.

This is not evidence against the core-stable hypothesis.

The hypothesis design explicitly states that the integer plane rank is local to each
independently reconstructed backbone and that semantic one-to-one identity across
same-numbered planes is not assumed.

The invariant under test is the **scale-local dominant causal role**, not the rank
number.

### 2.2 Mamba-370M dominant component

The frozen 370M discovery selected:

`P3`

against response-blind control:

`P5`.

On the fresh 370M confirmation cohort:

- `mean(D_DOM) = +3.974290010502882e-08`;
- `sd = 2.6418775104758776e-08`;
- `t(299) = 26.055985544025067`;
- raw one-sided `p = 3.192928180754558e-79`;
- Holm-adjusted `p = 6.385856361509116e-79`;
- Cohen's `dz = 1.5043430267843867`;
- positive-pair fraction `= 0.94`.

The dominant-component endpoint therefore recurred positively at 370M.

However, the historical 370M protocol also contained a positive aggregate-residual
hypothesis. That residual endpoint failed, so the historical preregistered joint
cross-backbone qualitative recurrence criterion remained:

`NOT ESTABLISHED`.

Nothing in this report changes that historical result.

### 2.3 Prospective Mamba-1.4B core test

The fresh 1.4B discovery selected:

`P5`

and geometry-only response-blind control:

`P4`.

On the disjoint confirmation cohort:

`D_CORE = Q_restored(P5) - Q_control(P4)`.

Frozen result:

- `mean(D_CORE) = +9.282848764823318e-09`;
- `sd = 9.892592353853887e-09`;
- `t(299) = 16.25293464497027`;
- one-sided primary `p = 2.624876428685113e-43`;
- Cohen's `dz = 0.938363619239498`;
- positive-pair fraction `= 0.7833333333333333`.

Exactly one primary p-value was executed.

The pre-specified 1.4B criterion required:

- `mean(D_CORE) > 0`; and
- one-sided `p < 0.05`.

Both conditions were satisfied.

Therefore:

`MAMBA14B_CORE_CONFIRMATION_SUPPORTED`

### 2.4 Core synthesis

Across the tested scale extension, the dominant plane rank changes, but a fresh
scale-local dominant causal role remains detectable under the frozen discovery and
matched-control confirmation procedure.

The appropriate statement is therefore:

> The evidence supports recurrence of a scale-local dominant causal role through
> Mamba-1.4B. The recurring object is the causal role under the frozen procedure,
> not a fixed principal-plane rank identity.

This is the precise sense in which the current evidence supports the
**CORE-STABLE** component.

---

## 3. Residual organization at 370M

The frozen 370M residual set relative to dominant `P3` was:

`R_370M = P1 ⊕ P2 ⊕ P4 ⊕ P5`.

Fresh residual decomposition means:

| local rank | mean residual effect |
|---|---:|
| `P1` | `+5.757457855574916e-09` |
| `P2` | `-1.4613450843693005e-08` |
| `P4` | `+2.8695081507774797e-08` |
| `P5` | `-6.287583022028684e-08` |

Aggregate quantities:

- `mean(S_RES) = -3.9308513194261373e-08`;
- `mean(S_SUM) = -4.303674170063013e-08`;
- `mean(I_RES) = +3.728228506368763e-09`;
- `corr(S_RES, S_SUM) = 0.9630714681512498`;
- `abs(mean(I_RES))/abs(mean(S_RES)) = 0.09484531984061521`.

Thus the negative 370M residual was predominantly an approximately additive signed
mixture rather than a large aggregate-only interaction.

The rank-aligned normalized signed profile was:

| local rank | `v_k` |
|---|---:|
| `P1` | `+0.051432590908350874` |
| `P2` | `-0.13054505267028216` |
| `P4` | `+0.25633924299456146` |
| `P5` | `-0.56168311342680555` |

The frozen 370M native residual coefficient-energy shares were:

| local rank | energy share `w_k` |
|---|---:|
| `P1` | `0.06398415677148447` |
| `P2` | `0.1475047279832685` |
| `P4` | `0.09374188922906401` |
| `P5` | `0.6947692260161830` |

Coefficient concentration:

`HHI_C(370M) = 0.5173434363105359`.

P5 was the largest-energy residual plane on all `300/300` pairs and the largest
absolute residual-effect contributor on `224/300 = 0.7466666666666667` pairs.

The prior frozen synthesis localized the negative P2/P5 effects primarily to negative
XG2-side susceptibility reorganization, with P4 providing a strong positive opposing
component.

---

## 4. Residual organization at 1.4B

The fresh 1.4B residual set relative to dominant `P5` was:

`R_1.4B = P1 ⊕ P2 ⊕ P3 ⊕ P4`.

This population was disjoint from both discovery and confirmation and did not access
core-confirmation inference.

No p-values were assigned to residual observables.

### 4.1 Signed residual effects

Fresh means:

| local rank | mean `S_k` | positive fraction |
|---|---:|---:|
| `P1` | `-2.4655218718445654e-09` | `0.37` |
| `P2` | `-2.5429833881164543e-08` | `0.00` |
| `P3` | `+1.6489021094661662e-09` | `0.6333333333333333` |
| `P4` | `-5.0151715170837445e-09` | `0.21` |

Aggregate quantities:

- `mean(S_RES) = -2.7596804037748638e-08`;
- `mean(S_SUM) = -3.1261625160626691e-08`;
- `mean(I_RES) = +3.6648211228780461e-09`;
- `sd(I_RES) = 3.6383018058550883e-09`;
- `corr(S_RES, S_SUM) = 0.9945431073595347`;
- `abs(mean(I_RES))/abs(mean(S_RES)) = 0.13279875154619622`.

The 1.4B aggregate residual is therefore also well tracked by the same-pair sum of
individual residual effects.

As at 370M, the aggregate residual is not primarily an aggregate-only nonlinear
interaction artifact.

The 1.4B normalized signed profile is:

| local rank | `v_k` |
|---|---:|
| `P1` | `-0.07134150986019625` |
| `P2` | `-0.7358291018602760` |
| `P3` | `+0.04771207566411521` |
| `P4` | `-0.14511731261541272` |

P2 is the largest absolute individual residual contributor on:

`271/300 = 0.9033333333333333`

pairs.

P3 is largest on the remaining:

`29/300 = 0.09666666666666666`.

P1 and P4 are never the largest absolute residual contributor on this cohort.

### 4.2 Native coefficient-mass distribution

Mean branch-local coefficient energies:

| local rank | mean `C_k` | share `w_k` |
|---|---:|---:|
| `P1` | `13.492112458105508` | `0.05006148924938505` |
| `P2` | `154.82357404982073` | `0.5744614649420721` |
| `P3` | `85.61775940772279` | `0.3176783884254689` |
| `P4` | `15.577362899756803` | `0.05779865738307401` |

Coefficient concentration:

`HHI_C(1.4B) = 0.43677237067714686`.

Largest-energy counts:

- `P2`: `262/300 = 0.8733333333333333`;
- `P3`: `38/300 = 0.12666666666666668`;
- `P1`: `0/300`;
- `P4`: `0/300`.

Thus 1.4B residual coefficient mass is concentrated mainly in P2 and secondarily P3,
rather than in a single P5-indexed mode as at 370M.

### 4.3 Energy-effect relationship

Pair-level correlations between native coefficient energy and signed residual effect:

| local rank | `corr(C_k,S_k)` | `corr(C_k,abs(S_k))` |
|---|---:|---:|
| `P1` | `+0.38049834374610464` | `-0.18606144092128077` |
| `P2` | `-0.13655636297906878` | `+0.13655636297906878` |
| `P3` | `+0.58586996257299784` | `-0.21210211194740744` |
| `P4` | `-0.5397055960615309` | `+0.5389876467127146` |

The rank with the largest coefficient mass is usually also the rank with the largest
absolute effect, but coefficient energy does not determine causal sign or effect
magnitude uniformly.

The residual mechanism therefore again requires both a state-mass description and a
functional-coupling description.

### 4.4 XG2/XG4 family localization

By construction:

`S_k = DeltaE_XG2,k - DeltaE_XG4,k`.

Fresh 1.4B means:

| local rank | `DeltaE_XG2` | `DeltaE_XG4` | `S_k` |
|---|---:|---:|---:|
| `P1` | `+5.031173265460296e-10` | `+2.968639198390595e-09` | `-2.465521871844565e-09` |
| `P2` | `-2.188056625987858e-09` | `+2.3241777255176685e-08` | `-2.5429833881164543e-08` |
| `P3` | `+1.196502065424381e-09` | `-4.524000440417857e-10` | `+1.6489021094661664e-09` |
| `P4` | `-4.632695616195329e-09` | `+3.824759008884157e-10` | `-5.015171517083745e-09` |

The strongest negative 1.4B residual mode, P2, differs functionally from the main
negative 370M residual modes.

At 370M, the negative P2/P5 residual effects were localized primarily to a negative
XG2-side response.

At 1.4B, the large negative P2 effect is instead dominated by the positive
`DeltaE_XG4` term in the subtraction:

`DeltaE_XG2 - DeltaE_XG4`.

P4 is negative mainly through a negative XG2-side contribution.

This is direct descriptive evidence that generator-family coupling has reorganized
again rather than preserving the 370M residual mechanism unchanged.

---

## 5. 370M -> 1.4B residual comparison

### 5.1 Signed profile reorganizes

370M rank-aligned normalized signed profile:

`[P1:+0.0514, P2:-0.1305, P4:+0.2563, P5:-0.5617]`

1.4B rank-aligned normalized signed profile:

`[P1:-0.0713, P2:-0.7358, P3:+0.0477, P4:-0.1451]`

Because rank labels are scale-local, this comparison is descriptive rather than
semantic matching.

Nevertheless, the organization clearly does not preserve the 370M pattern:

- 370M is dominated by a strong negative P5 contribution opposed by positive P4;
- 1.4B is dominated by a strong negative P2 contribution;
- 370M P4 is strongly positive;
- 1.4B P4 is negative;
- the positive residual contribution at 1.4B is the much smaller P3 term.

### 5.2 Coefficient-mass distribution reorganizes

370M residual mass profile:

`[P1:0.0640, P2:0.1475, P4:0.0937, P5:0.6948]`

1.4B residual mass profile:

`[P1:0.0501, P2:0.5745, P3:0.3177, P4:0.0578]`

The largest-energy local rank changes from:

`370M: P5`

to:

`1.4B: P2`.

Concentration decreases:

`HHI_C: 0.5173434363 -> 0.4367723707`.

Thus the 1.4B residual is still strongly structured, but coefficient mass is less
single-rank concentrated and is redistributed mainly across P2 and P3.

### 5.3 Additivity is retained while composition changes

370M:

- `corr(S_RES,S_SUM) = 0.9630714681512498`;
- `abs(mean(I_RES))/abs(mean(S_RES)) = 0.09484531984061521`.

1.4B:

- `corr(S_RES,S_SUM) = 0.9945431073595347`;
- `abs(mean(I_RES))/abs(mean(S_RES)) = 0.13279875154619622`.

At both scales the aggregate residual is predominantly explained by the signed
constituent effects rather than by a dominant aggregate-only interaction.

What changes substantially is **which local residual ranks carry the mass and signed
causal effect, and how their XG2/XG4 couplings generate those effects**.

---

## 6. Relation to the earlier 130M evidence

The earlier 130M frozen residual evidence, summarized before the 1.4B hypothesis was
formulated, showed:

- positive raw native-neutralization attenuation for all four tested residual ranks;
- residual coefficient-energy mass dominated by local `P1`;
- `P1` residual energy share approximately `0.7776`.

At 370M:

- residual mass moved to local `P5`;
- `P2` and `P5` became negative;
- P4 remained strongly positive;
- the aggregate residual became negative.

At 1.4B:

- residual mass moved primarily to local `P2`, secondarily `P3`;
- `P2` became the dominant negative residual effect;
- P4 was also negative;
- the largest P2 negative effect was generated by a different XG2/XG4 balance than
  the major negative 370M modes.

The robust cross-scale observation is therefore not persistence of any same-numbered
plane.

It is:

> the detailed rank-aligned residual realization changes substantially across the
> tested scales while remaining structured and causally measurable.

---

## 7. Pre-specified interpretation outcome

The prospective design defined:

### Outcome A

**Core passes, residual reorganizes again.**

Pre-specified interpretation:

`CORE-STABLE / RESIDUAL-PLASTIC` is strengthened across a third tested scale.

The frozen evidence matches Outcome A:

1. 1.4B discovery produced a unique scale-local dominant candidate;
2. the pre-specified fresh 1.4B core confirmation passed;
3. the fresh residual characterization remained structured;
4. its signed profile changed relative to 370M;
5. its coefficient-mass distribution changed relative to 370M;
6. its dominant residual rank changed;
7. its generator-family coupling changed;
8. none of these residual observations were used to select or rescue the core result.

Therefore the appropriate synthesis is:

`OUTCOME_A_OBSERVED`

with interpretation:

`CORE-STABLE / RESIDUAL-PLASTIC STRENGTHENED THROUGH MAMBA-1.4B`

This statement uses the exact meaning defined prospectively:

- **core-stable**: recurrence of a scale-local dominant causal role under the same
  frozen procedure;
- **residual-plastic**: scale-dependent reorganization of the surrounding residual
  realization.

It does not assert a universal scaling law.

---

## 8. Mechanistic synthesis

The current three-scale evidence is most consistent with the following narrow model.

1. A low-dimensional dominant causal role recurs under independent reconstruction.
2. The principal-plane rank realizing that role is not fixed across scale.
3. The non-dominant residual remains structured rather than disappearing.
4. Residual aggregate behavior is largely explained by a signed mixture of local
   constituent effects at both 370M and 1.4B.
5. The local rank carrying the majority of residual coefficient mass changes with
   scale.
6. Signed residual effects reorganize with that mass redistribution.
7. Mass alone is insufficient to determine effect; functional XG2/XG4 coupling also
   changes.
8. The generator-family mechanism of the strongest negative residual mode at 1.4B is
   different from the main negative residual modes at 370M.
9. The residual realization is therefore better described as a joint
   **state-distribution + functional-coupling reorganization** than as a fixed
   rank-specific mechanism.

A concise claim is:

> Across the tested native-Mamba scales, a scale-local dominant causal role recurs
> under the frozen procedure, while the surrounding residual mechanism remains
> structured but reorganizes in rank-aligned signed effects, coefficient-mass
> distribution, and generator-family coupling.

---

## 9. Interpretation boundaries

This synthesis does **not** establish:

- semantic identity of same-numbered principal planes across scales;
- a universal dominant plane rank;
- that `P3`, `P5`, or any other rank is universally special;
- a universal monotonic scaling law;
- monotonic residual concentration with parameter count;
- that residual aggregate sign must be negative or positive;
- exact additivity or absence of interaction;
- statistical significance for the descriptive residual comparisons;
- statistical significance of the cross-scale residual reorganization itself;
- architecture-independent universality;
- transformer generalization;
- behavioral improvement or benchmark superiority;
- causal claims outside the frozen local intervention and endpoint;
- rescue of the historical failed 370M joint residual-recurrence criterion.

The 370M historical joint criterion remains failed as originally recorded.

The prospective 1.4B core test is a separate later hypothesis and passed on its own
pre-specified fresh confirmation population.

---

## 10. Final status

Mamba-1.4B code correctness:

`PASS`

Mamba-1.4B discovery execution / artifact provenance:

`PASS`

Mamba-1.4B core confirmation execution / artifact provenance:

`PASS`

Mamba-1.4B prospective core criterion:

`SUPPORTED`

Mamba-1.4B residual characterization execution / artifact provenance:

`PASS`

Mamba-1.4B residual analysis:

`DESCRIPTIVE — P_VALUE_COUNT=0`

Cross-scale static analysis:

`PASS — NO NEW P-VALUES`

Historical 370M failed joint criterion rescued:

`NO`

Pre-specified 1.4B outcome:

`OUTCOME A — CORE PASSES, RESIDUAL REORGANIZES AGAIN`

Final narrow synthesis:

`CORE-STABLE / RESIDUAL-PLASTIC STRENGTHENED THROUGH MAMBA-1.4B`
