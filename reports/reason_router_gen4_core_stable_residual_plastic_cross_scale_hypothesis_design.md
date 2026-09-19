# ContraMamba Cross-Scale Hypothesis Design
## Core-Stable / Residual-Plastic Native-Mamba Mechanism

### Status

`PROSPECTIVE_HYPOTHESIS_DESIGN_ONLY`

This document defines the next cross-scale scientific hypothesis after the completed
130M and 370M evidence chain.

It does **not** authorize training, GPU execution, evaluation, or a 1.4B scientific run.
It introduces no new p-values and does not reinterpret the failed preregistered 370M
joint residual-recurrence criterion.

The hypothesis was formulated after observing the 130M and 370M results and must
therefore be tested, if pursued, on a new held-out scale and fresh response cohorts.

Current synthesis anchor:

`bde30b33b489b4abea7a976d443345626d0e57f8`

---

## 1. Motivation from frozen evidence

The completed 130M evidence supports a low-dimensional dominant causal component
under the frozen local susceptibility mechanism, including matched-control necessity
and restoration sufficiency.

The independently reconstructed 370M experiment then found:

- a strong positive dominant-component confirmation result;
- failure of the preregistered positive aggregate-residual recurrence endpoint;
- a fresh residual decomposition in which the residual was predominantly an
  approximately additive signed mixture;
- a dominant negative P5-indexed contribution opposed by a positive P4-indexed
  contribution and an additional negative P2-indexed contribution;
- strong native residual coefficient-mass concentration in the P5-indexed rank;
- signed coupling reorganization, especially on the XG2 side.

The 370M result therefore argues against treating exact residual-plane rank identities,
signs, or mass shares as cross-scale invariants.

It does not argue against recurrence of a dominant causal role.

---

## 2. Hypothesis

### 2.1 Core-stable component

For each independently reconstructed Mamba scale `s`, let the frozen geometry produce
five paired principal planes:

`P_s,1 ... P_s,5`.

The integer rank label is local to that scale.

No semantic identity is assumed between:

`P_130M,k`, `P_370M,k`, and `P_1.4B,k`.

For a fresh discovery population at scale `s`, define the same dominant-component
discovery response used by the 370M protocol:

`G_s,k(i) = Q_restored(s,k,i) - Q_neutralized(s,k,i)`.

Define:

`g_bar_s,k = mean_i G_s,k(i)`.

The scale-local dominant candidate is:

`k*_s = unique argmax_k g_bar_s,k`.

The response-blind matched control is selected only from frozen geometry, not from
XG1 responses:

`c*_s = argmax_{k != k*_s} lambda^+_s,k`.

On a disjoint fresh confirmation population define:

`D_CORE_s(i) = Q_restored(s,k*_s,i) - Q_control(s,c*_s,i)`.

The **core-stable hypothesis** predicts that a scale-local dominant causal role remains
detectable after independent reconstruction:

`E[D_CORE_s] > 0`.

The hypothesis concerns the recurrence of a causal role under a fixed scientific
procedure. It does not require:

`k*_130M = k*_370M = k*_1.4B`.

### 2.2 Residual-plastic component

At each scale define the residual subspace relative to the scale-local dominant
candidate:

`R_s = direct sum of all P_s,k for k != k*_s`.

The residual-plastic statement is deliberately weaker than a sign-recurrence claim.

It states that the residual mechanism is structured and causally measurable, but its
rank-aligned:

- signed individual effects;
- coefficient-mass distribution;
- XG2/XG4 family coupling;
- dominant residual rank;
- and aggregate sign

are **not assumed to be invariant across scale**.

Therefore a positive, negative, or near-zero aggregate residual at 1.4B is not, by
itself, a success or failure criterion for the core hypothesis.

Residual behavior is characterized prospectively and descriptively on a separate
fresh population.

---

## 3. Falsification logic

### 3.1 Core hypothesis failure

The prospective 1.4B core test fails if either:

- the frozen discovery procedure does not produce a unique scale-local dominant
  candidate; or
- the pre-specified fresh confirmation test does not reject

`H0: E[D_CORE_1.4B] <= 0`

in favor of

`H1: E[D_CORE_1.4B] > 0`

at the pre-specified alpha.

If this occurs, the claim that the dominant causal role is stable through the tested
1.4B scale is not supported.

No residual result may rescue a failed core test.

### 3.2 What does not falsify the core hypothesis

The following do not, by themselves, falsify the core hypothesis:

- a different selected principal-plane rank;
- a different residual sign pattern;
- a different largest residual-energy rank;
- a negative aggregate residual;
- a positive aggregate residual;
- stronger or weaker residual concentration.

These are residual-organization outcomes, not the core criterion.

### 3.3 Residual plasticity is a comparative model, not an omnibus success gate

"Residual-plastic" does not mean that the residual must change at every larger scale.

A 1.4B residual closely resembling 370M would instead suggest a possible higher-scale
stable regime.

A 1.4B residual that reorganizes again would strengthen the interpretation that
residual realization remains scale-dependent.

The residual analysis must therefore report the observed organization without forcing
either pattern into a binary success label.

---

## 4. Candidate held-out scale

Candidate backbone:

`state-spaces/mamba-1.4b-hf`

Current public configuration has:

- `48` Mamba layers;
- hidden size `2048`;
- expansion factor `2`;
- corresponding intermediate size `4096`;
- state size `16`;
- vocabulary size approximately `50k`.

The exact Hugging Face revision, snapshot file identities, tokenizer identities, and
runtime package identities are **not frozen by this document**. They must be pinned
before any implementation or execution.

The scientific attraction of this scale is that 370M and 1.4B both have 48 layers,
while the 1.4B hidden width is doubled relative to 370M. This makes 370M -> 1.4B a
cleaner width/capacity extension than the earlier 130M -> 370M transition, which
changed both depth and width.

---

## 5. Structural rules that may transfer

The following scientific rules may be inherited because they define the measurement
procedure rather than a learned response:

- `K = 5` paired principal planes;
- XG2/XG4 geometry reconstructed independently for the new backbone;
- same XG2/XG4 structural generator semantics;
- `epsilon = 0.025`;
- same within-layer strong-mask rule:
  `k^2 > mean(k^2)`;
- same local layer-offset semantics;
- same target-token alignment semantics;
- same Q construction from XG2 versus XG4 directional susceptibility;
- no response-guided layer, token, epsilon, or plane rescue.

Because 1.4B has 48 layers, the inherited relative layer mapping from the original
130M layer-17 location yields the same homologous 48-layer location used by 370M:

- source block: `33`
- target residual layer: `34`
- intervention layer: `35`

The strong-channel count is not transferred. It is an emergent result of applying the
same mask rule to the independently reconstructed 1.4B state.

---

## 6. Objects that must not transfer from 130M or 370M

The prospective 1.4B experiment must not copy:

- principal-plane vectors;
- plane rank identity;
- selected dominant rank;
- response-blind control rank;
- strong-channel indices;
- XG2/XG4 basis vectors;
- residual sign pattern;
- residual coefficient-energy shares;
- P5 dominance;
- P4/P5 opposition;
- discovery response values;
- confirmation response values;
- interaction estimates;
- or any outcome-derived threshold.

In particular:

`P3` is not preselected at 1.4B.

`P5` is not predesignated as a residual control or dominant residual mode at 1.4B.

---

## 7. Prospective 1.4B response populations

If the 1.4B study is authorized after feasibility verification, use fresh XG1 ranges
that have not appeared in the completed 130M/370M response chain.

Proposed prospective partition:

`XG1 3901..4200` — dominant-component discovery

`XG1 4201..4500` — core confirmation

`XG1 4501..4800` — residual mechanistic characterization

Each population contains exactly:

`N = 300` source pairs.

The ranges must be structurally built and frozen before their corresponding scientific
responses are observed.

The confirmation range must remain inaccessible to discovery logic.

The residual-characterization range must not affect the dominant-component selection
or core confirmation inference.

---

## 8. Prospective core confirmation

### 8.1 Discovery

On `3901..4200`:

- evaluate all five independently reconstructed 1.4B planes;
- compute `g_bar_1.4B,k`;
- require a unique argmax;
- freeze `k*_1.4B`;
- select `c*_1.4B` from geometry only using the inherited response-blind rule;
- do not access confirmation responses.

No positivity gate is used in discovery.

### 8.2 Confirmation

On `4201..4500`:

Primary endpoint:

`D_CORE(i) = Q_restored(k*) - Q_control(c*)`.

Primary hypotheses:

`H0: mean(D_CORE) <= 0`

`H1: mean(D_CORE) > 0`.

Primary test:

- one-sample Student t-test;
- one-sided `greater`;
- `N = 300`;
- alpha `0.05`;
- exactly one primary core p-value.

The 1.4B core hypothesis is supported only if:

- `mean(D_CORE) > 0`; and
- the one-sided primary test rejects at alpha `0.05`.

There is no residual-positive gate in the primary family.

There is no post-hoc alternative-tail rescue.

---

## 9. Prospective residual characterization

The residual characterization is scientifically secondary to the core test and uses
the disjoint fresh range `4501..4800`.

For every non-dominant plane `k != k*`, record:

`S_k(i) = Q_native(i) - Q_k-neutralized(i)`.

Also record:

`S_RES(i) = Q_native(i) - Q_all-residual-neutralized(i)`.

Define:

`S_SUM(i) = sum_{k != k*} S_k(i)`.

Define downstream nonadditivity:

`I_RES(i) = S_RES(i) - S_SUM(i)`.

The characterization must also record, in the same run when possible, native
branch-local coefficient energy:

`C_k(i) = mean_branch[a_k(i)^2 + b_k(i)^2]`.

Define scale-local residual energy share:

`w_k = mean(C_k) / sum_j mean(C_j)`.

For XG2/XG4 family localization record:

`DeltaE_XG2,k = E_XG2(native) - E_XG2(k-neutralized)`

`DeltaE_XG4,k = E_XG4(native) - E_XG4(k-neutralized)`

with:

`S_k = DeltaE_XG2,k - DeltaE_XG4,k`.

### Required descriptive outputs

The residual artifact must report:

- signed mean effect vector;
- fraction positive/negative per residual plane;
- pairwise residual effect-correlation matrix;
- largest absolute contributor count/fraction;
- native coefficient-energy vector;
- residual energy-share vector;
- largest-energy rank count/fraction;
- energy versus signed-effect correlation;
- energy versus absolute-effect correlation;
- `corr(S_RES, S_SUM)`;
- `mean(I_RES)`;
- `sd(I_RES)`;
- `abs(mean(I_RES)) / abs(mean(S_RES))`, when the denominator is nonzero;
- XG2 and XG4 family contributions by plane;
- normalized signed residual profile:
  `v_k = mean(S_k) / sum_j abs(mean(S_j))`;
- coefficient concentration:
  `HHI_C = sum_k w_k^2`.

These are descriptive observables only in the initial 1.4B study.

No p-values are assigned to residual-plane ranks in this stage.

---

## 10. Cross-scale comparison objects

After artifact validation, compare scales using quantities that do not assume direct
vector-space identity across different hidden dimensions.

### Core

Compare:

- whether a unique scale-local dominant candidate exists;
- confirmatory `mean(D_CORE)`;
- standardized core effect size;
- positive-pair fraction.

Do not compare plane rank numbers as semantic identities.

### Residual signed profile

For each scale use the normalized signed profile:

`v_s = [mean(S_k)] / sum_j abs(mean(S_j))`

in local rank order, while explicitly labeling it rank-aligned rather than semantic.

### Residual mass profile

Use:

`w_s = [mean(C_k)] / sum_j mean(C_j)`.

Report:

- entropy or HHI concentration;
- largest-energy rank;
- largest-energy share.

### Additivity

Use:

- `corr(S_RES, S_SUM)`;
- `mean(I_RES)`;
- interaction-to-aggregate mean ratio.

These describe whether aggregate residual behavior is predominantly explained by
signed constituent effects or by strong aggregate-only downstream interaction.

### Generator-family coupling

Use per-plane:

`(DeltaE_XG2,k, DeltaE_XG4,k)`

to identify whether signed reorganization is driven mainly by XG2-side, XG4-side, or
joint response changes.

Two-dimensional cosine similarity may be reported only as a coarse functional
direction diagnostic. It must not be interpreted as semantic plane matching when
different planes are nearly collinear in that reduced representation.

---

## 11. Pre-specified interpretation table

### Outcome A — core passes, residual reorganizes again

Interpretation:

`CORE-STABLE / RESIDUAL-PLASTIC` is strengthened across a third tested scale.

This would support a model in which the dominant causal role is more scale-stable than
the detailed residual realization.

### Outcome B — core passes, residual resembles 370M

Interpretation:

The core-stable component is strengthened.

The residual evidence would suggest that 370M and 1.4B may occupy a shared higher-scale
regime rather than showing unrestricted continuing plasticity.

This is not a failure.

### Outcome C — core fails

Interpretation:

The dominant-core recurrence claim does not extend prospectively through 1.4B under
the frozen procedure.

Residual results cannot rescue this failure.

### Outcome D — no unique discovery candidate

Interpretation:

The operational low-dimensional dominant-core procedure fails at 1.4B before
confirmation.

No response-guided tie breaking or alternative selection rule is permitted.

---

## 12. Training and execution boundary

A scientifically comparable 1.4B study requires the same downstream task semantics as
the completed 130M/370M chain.

However, this document does not freeze an executable 1.4B training command.

Before training authorization, a separate static/runtime feasibility step must verify:

- exact 1.4B snapshot and tokenizer identities;
- exact model configuration;
- compatibility with the current ContraMamba Mamba instrumentation;
- CUDA fast-path availability;
- compact/full checkpoint storage requirements;
- optimizer and activation memory feasibility on the available two-T4 environment;
- whether the already frozen G3-GROUP-D-HALF training semantics can be reproduced
  without changing the scientific objective.

A runtime limitation may justify an implementation strategy change.

It may not justify changing labels, split semantics, loss semantics, selection criteria,
or the scientific endpoint.

---

## 13. Scope boundary

This hypothesis concerns the tested native-Mamba mechanism only.

It does not claim:

- a universal scaling law;
- semantic identity of principal-plane ranks across backbones;
- monotonic residual concentration with parameter count;
- that P5 is universally special;
- that opponent residual organization is mandatory at every scale;
- that the residual becomes less important with scale;
- behavioral improvement;
- benchmark superiority;
- transformer generalization;
- or architecture-independent universality.

The prospective third-scale test is intended to distinguish a scale-stable dominant
causal role from scale-dependent realization of the surrounding residual mechanism.

---

## Final prospective hypothesis

The narrow hypothesis to test at a new Mamba scale is:

> Independent reconstruction at a larger Mamba scale will recover a scale-local
> dominant component whose causal restoration effect remains positive on a fresh
> confirmation population, while the detailed signed and mass distribution of the
> non-dominant residual is allowed to reorganize and is characterized independently
> rather than required to preserve a rank-specific sign pattern.

Short label:

`CORE-STABLE / RESIDUAL-PLASTIC`
