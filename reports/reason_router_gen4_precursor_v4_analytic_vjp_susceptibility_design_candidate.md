# ContraMamba Gen4 — Precursor v4 Analytic Local Differential Susceptibility Feasibility Design

## Status

`PROSPECTIVE_PRECURSOR_V4_ANALYTIC_LOCAL_DIFFERENTIAL_SUSCEPTIBILITY_DESIGN_CANDIDATE`

Precursor-v2 and Precursor-v3 scientific Stage A were never executed.

Both were stopped by the mandatory pre-execution CPU-slow versus CUDA-fast
backend-equivalence gate for the finite-difference-derived susceptibility.
The finite-difference DCS estimator branch is closed. No additional epsilon,
epsilon sweep, or tolerance relaxation is permitted.

Precursor-v4 replaces only the numerically conditioned central-difference
estimator with the corresponding analytic local directional derivative.

## 1. Scientific question

Before a future forced-decisive supported versus unsupported commitment is
emitted, does the frozen P3 local differential susceptibility of the frozen
gold-aligned decisive margin, relative to the frozen response-blind P5 plane,
differ on prefix-only states?

This is a prospective measurement redesign before any N=800 scientific
response has been generated or inspected.

## 2. Frozen scientific objects

Unchanged from the prior DCS design:

- model: `state-spaces/mamba-370m-hf`
- revision: `589179554943157be31701edd8b4558889276674`
- local intervention site: block-35 `in_proj` content stream
- strong-channel dimension: `650`
- selected plane: `P3`
- response-blind control plane: `P5`
- late decisive readout: frozen causal-LM output after block 47/final norm/head
- gold-aligned decisive margin `F_t`
- offsets: `t*-4, t*-3, t*-2, t*-1`
- frozen forced-decisive grammar
- frozen response-free N=800 cohort
- equal-weight four-offset item endpoint
- future supported versus unsupported grouping
- exactly one two-sided Welch independent-samples t-test
- alpha `0.05`
- minimum 30 valid rows per outcome group
- no layer, plane, coordinate, offset, or subgroup selection

## 3. Estimator change

Finite-difference v2/v3 used:

`[F(h + epsilon*u) - F(h - epsilon*u)] / (2*epsilon)`

Precursor-v4 uses the analytic local derivative at the unperturbed state:

`g_t = grad_h F_t(h_t)`

For each frozen basis direction `u`:

`r_u,t = g_t^T u`

No epsilon exists in the Precursor-v4 primary measurement protocol.

The derivative target `h_t` is exactly the block-35 `in_proj` strong-content
slice that the frozen DCS implementation previously perturbed. The full
block-35 `in_proj` activation is detached and reintroduced as a local
autograd leaf at the hook boundary, so the derivative is explicitly local:
upstream computation is held fixed and only downstream sensitivity is
differentiated.

All model parameters remain frozen with `requires_grad=False`; no parameter
gradient or weight update is permitted.

## 4. Plane susceptibility

For P3:

`r_P3,+,t = g_t^T u_P3,+`

`r_P3,-,t = g_t^T u_P3,-`

`chi_P3,t = sqrt(r_P3,+,t^2 + r_P3,-,t^2)`

For P5:

`r_P5,+,t = g_t^T u_P5,+`

`r_P5,-,t = g_t^T u_P5,-`

`chi_P5,t = sqrt(r_P5,+,t^2 + r_P5,-,t^2)`

Specificity-adjusted local differential susceptibility:

`D_t = chi_P3,t - chi_P5,t`

Future item-level Stage A endpoint:

`Z_i = mean(D_i,t*-4, D_i,t*-3, D_i,t*-2, D_i,t*-1)`

No alternative aggregation is permitted.

## 5. Terminology boundary

Stage A terminology:

`LOCAL_DIFFERENTIAL_SUSCEPTIBILITY`

The analytic gradient alone is not called a causal effect.

The P3/P5 geometry originates from prior causal-intervention work, but a
broader causal-precursor claim requires a separately valid intervention stage.

## 6. Mandatory historical one-row feasibility/equivalence gate

Before implementing or executing the N=800 scientific runner, run exactly one
bounded response-free historical-row gate at offset `t*-4`.

Selection remains:

- old historical precursor population only;
- first decisive-gold row;
- no generation-response-based selection;
- generated prefix length `5`;
- no access to the fresh N=800 scientific cohort.

Each backend performs exactly:

- one unperturbed full-model forward;
- one local autograd VJP/backward from `F_t` to the block-35 `in_proj` leaf.

Backends:

- CPU: frozen Transformers sequential/slow Mamba path;
- GPU: exact frozen fast CUDA kernel path.

The gate compares:

- `F_t`;
- `g^T u_P3,+`;
- `g^T u_P3,-`;
- `g^T u_P5,+`;
- `g^T u_P5,-`;
- `chi_P3`;
- `chi_P5`;
- `D_t`.

Tolerance remains exactly:

- `atol = 1e-4`
- `rtol = 1e-4`

No tolerance relaxation is permitted.

The four directional projections and all derived norms are computed after
moving the backend-specific strong-channel gradient to canonical CPU float64.
This keeps the gate focused on model/backend gradient reproducibility rather
than adding a second device-dependent reduction kernel.

## 7. Gate accounting

Historical gate only:

- CPU full-model forwards: `1`
- CPU local VJP/backward calls: `1`
- GPU full-model forwards: `1`
- GPU local VJP/backward calls: `1`
- total equivalence model forwards: `2`
- total equivalence VJP/backward calls: `2`
- scientific model forwards: `0`
- statistical tests: `0`
- p-values: `0`
- training: `0`
- parameter updates: `0`

## 8. Pass/fail rule

If every frozen comparison passes:

`PASS_PRECURSOR_V4_ANALYTIC_VJP_CPU_CUDA_EQUIVALENCE`

Only then may the N=800 analytic Stage A runner be implemented.

If any comparison fails:

`PRECURSOR_V4_ANALYTIC_VJP_BACKEND_EQUIVALENCE_NOT_ESTABLISHED`

and the local differential-susceptibility precursor branch closes.

No epsilon, finite-difference, tolerance, row, layer, plane, coordinate, or
offset rescue follows.

An inability of the exact CUDA kernel path to provide the required local VJP
also counts as feasibility failure for this design unless a clearly identified
implementation defect can be corrected without changing the scientific
observable.

## 9. Future Stage A inference, contingent on gate PASS

The frozen response-free N=800 cohort may be reused because no v2/v3/v4
scientific response has yet been generated or inspected.

Primary endpoint:

`Z_i`

Primary comparison:

- unsupported decisive future commitments;
- supported decisive future commitments.

Exactly one two-sided Welch independent-samples t-test, alpha `0.05`.

If either group has fewer than 30 valid rows:

`PRECURSOR_V4_STAGE_A_NOT_ESTIMABLE`

If the single test is significant:

`LOCAL_DIFFERENTIAL_SUSCEPTIBILITY_TEMPORAL_PRECEDENCE_OBSERVED`

Otherwise:

`LOCAL_DIFFERENTIAL_SUSCEPTIBILITY_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`

## 10. Prohibited moves

Do not:

- reuse finite-difference DCS as the v4 primary estimator;
- introduce any epsilon into the v4 primary protocol;
- relax `atol` or `rtol`;
- change the historical equivalence row after observing the gate;
- inspect N=800 generation outcomes before gate PASS;
- implement the N=800 runner before gate PASS;
- change P3/P5;
- change block 35 or the decisive readout;
- scan offsets;
- train a probe or classifier;
- call Stage A gradient evidence a causal effect.

## 11. Immediate scope

The next implementation delta contains only:

1. this design;
2. one historical-row analytic/VJP CPU-CUDA gate;
3. targeted static/unit tests.

No N=800 scientific runner is part of this implementation delta.
