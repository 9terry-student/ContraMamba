# ContraMamba Gen4 — Precursor v3 Numerically Stable Dynamic Causal Susceptibility Stage A Design

## Status

`PROSPECTIVE_PRECURSOR_V3_NUMERICALLY_STABLE_DCS_DESIGN_CANDIDATE`

This is a new prospective design created after the Precursor-v2 backend-equivalence prerequisite failed before any N=800 scientific execution.

Precursor-v2 scientific status remains:

`NOT_EXECUTED`

The observed failure was numerical/backend equivalence of the derived central-difference quantities, not a scientific null result.

No Precursor-v2 Stage A generation outcome, supported/unsupported group assignment, Z value, or p-value was observed.

## 1. Scientific question

Before a future forced-decisive supported versus unsupported commitment is emitted, does the frozen block-35 P3 local causal susceptibility, relative to the frozen response-blind P5 control plane, differ on prefix-only states?

The scientific question is unchanged from Precursor-v2.

## 2. Frozen elements inherited unchanged

The following remain exactly frozen:

- model: `state-spaces/mamba-370m-hf`
- revision: `589179554943157be31701edd8b4558889276674`
- intervention block: `35`
- readout block: `47`
- selected plane: `P3`
- response-blind control plane: `P5`
- strong-channel dimension: `650`
- offsets: `t*-4, t*-3, t*-2, t*-1`
- forced-decisive grammar and exact tokenization
- gold-aligned decisive margin readout
- response-free N=800 cohort already materialized and frozen
- 400 Refuted / 400 Supported cohort balance
- native supported/unsupported outcome definition
- equal-weight four-offset endpoint aggregation
- one two-sided Welch independent-samples t-test
- alpha `0.05`
- minimum 30 rows per outcome group
- no multiplicity correction because the confirmatory family contains one test
- no training, learned probe, layer scan, coordinate scan, plane reselection, offset selection, or subgroup rescue

The frozen N=800 cohort may be reused because no Precursor-v2 scientific response was produced or inspected and cohort selection was response-free.

## 3. Numerical-equivalence failure motivating redesign

At `epsilon = 0.025`, the bounded historical-row CPU-slow versus CUDA-fast gate showed:

- all primitive branch-token logits within frozen `atol=1e-4, rtol=1e-4`;
- all primitive gold-aligned decisive margins within the same tolerance;
- all applied perturbation L2 values within tolerance;
- derived P3 central-difference components exceeded tolerance;
- consequently `chi_P3` and `D_t` exceeded tolerance.

This is consistent with numerical amplification by the central-difference denominator:

`2 * epsilon = 0.05`.

No scientific Stage A response was accessed.

## 4. Single redesign change

Precursor-v3 changes exactly one scientific measurement constant:

`epsilon = 0.05`

instead of:

`epsilon = 0.025`.

Reason:

The previous epsilon placed the derived central-difference observable too close to the backend numerical floor for the already-frozen CPU/CUDA tolerance.

The new value is exactly one doubling of the previous perturbation magnitude.

This choice is made from implementation-validation evidence only, before any N=800 scientific response is observed.

There is no epsilon sweep.

## 5. Dynamic causal susceptibility

For plane `Pk` in `{P3, P5}`, frozen orthonormal basis direction `s` in `{plus, minus}`, and prefix time `t`:

`r_k,s,t = [F_t(h_t + 0.05*u_k,s) - F_t(h_t - 0.05*u_k,s)] / 0.10`

Then:

`chi_k,t = sqrt(r_k,plus,t^2 + r_k,minus,t^2)`

and:

`D_t = chi_P3,t - chi_P5,t`

Per item:

`Z_i = mean(D_i,t*-4, D_i,t*-3, D_i,t*-2, D_i,t*-1)`

No alternative endpoint is permitted.

## 6. Mandatory bounded backend-equivalence gate

Before any N=800 scientific execution, run exactly one bounded historical-row CPU-slow versus CUDA-fast equivalence gate.

The gate must:

- use historical precursor data only;
- not access the frozen N=800 Precursor-v3 cohort;
- use offset `t*-4` only;
- execute exactly 8 CPU probe forwards and 8 CUDA probe forwards;
- compare the same primitive and derived quantities as the Precursor-v2 gate;
- use exactly `atol = 1e-4`, `rtol = 1e-4`;
- compare branch-token logits;
- compare gold-aligned decisive margins;
- compare applied perturbation L2;
- compare both P3 derivatives;
- compare both P5 derivatives;
- compare `chi_P3`;
- compare `chi_P5`;
- compare `D_t`;
- perform zero statistical tests;
- add zero p-values;
- produce no scientific conclusion.

No tolerance relaxation is permitted.

## 7. Hard stop rule

If the exact Precursor-v3 equivalence gate at `epsilon = 0.05` fails any frozen comparison:

`PRECURSOR_V3_DCS_BACKEND_EQUIVALENCE_NOT_ESTABLISHED`

and:

`PRECURSOR_DCS_FINITE_DIFFERENCE_BRANCH_CLOSED`

No further epsilon increase, epsilon sweep, alternative tolerance, alternative historical row, alternative layer, alternative plane, or alternative offset may rescue this branch.

If the gate passes:

`PASS_PRECURSOR_V3_DCS_ONE_ROW_CPU_CUDA_EQUIVALENCE`

then and only then may the frozen N=800 scientific Stage A execute.

## 8. Scientific execution accounting

The scientific measurement structure is unchanged:

- native forced-decisive generation: 12 full-prefix forwards per row;
- 2 planes × 2 basis directions × 2 signs = 8 probe forwards per offset;
- 4 offsets = 32 probe forwards per row;
- total = 44 scientific full-model forwards per row;
- N=800 total = 35,200 scientific full-model forwards.

No backward pass.

## 9. Primary inference

Primary groups:

- future unsupported forced-decisive commitments;
- future supported forced-decisive commitments.

Primary endpoint:

`Z_i`.

Primary test:

two-sided Welch independent-samples t-test.

Support criterion:

- all 800 rows valid;
- at least 30 rows in each outcome group;
- single primary p-value `< 0.05`.

Supported claim:

`DYNAMIC_CAUSAL_SUSCEPTIBILITY_TEMPORAL_PRECEDENCE_OBSERVED`

Otherwise:

`DYNAMIC_CAUSAL_SUSCEPTIBILITY_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`

A backend-equivalence failure is neither of these scientific outcomes.

## 10. Prohibited moves

Do not:

- inspect N=800 generation outcomes before equivalence PASS;
- change epsilon again after this design freeze;
- sweep epsilon;
- relax equivalence tolerance;
- replace the historical equivalence row after seeing a failure;
- select a different plane;
- select a different layer;
- select a different offset;
- change P5;
- change the Welch test;
- add subgroup inference;
- reinterpret Precursor-v2 equivalence failure as a scientific null;
- reuse a failed run identity.

## 11. Immediate implementation scope

The next implementation delta is intentionally limited to the bounded
Precursor-v3 equivalence gate.

Before equivalence PASS:

1. do not implement or execute the N=800 scientific runner;
2. reuse the frozen Precursor-v2 DCS measurement code only as a substrate;
3. override epsilon only inside the Precursor-v3 equivalence path;
4. freeze `epsilon = 0.05`;
5. retain the historical row, P3/P5, block 35/47, offset `t*-4`,
   branch readout, and `atol=1e-4, rtol=1e-4`;
6. add targeted static tests proving the scoped epsilon change and proving
   that the frozen Precursor-v2 module returns to `epsilon = 0.025`;
7. commit/push the design, equivalence implementation, and tests together;
8. execute exactly one bounded historical-row CPU/CUDA equivalence gate.

Only after equivalence PASS may a Precursor-v3 N=800 scientific runner be
implemented. If equivalence fails, no scientific runner is created and the
finite-difference DCS branch closes.

## 12. Boundary

This redesign is an adaptive numerical-method redesign after implementation validation failure.

It is not a response-adaptive scientific redesign because no Precursor-v2 Stage A scientific response was produced or inspected.

No claim from Precursor-v2 is changed.

No scientific evidence is created by this design document.
