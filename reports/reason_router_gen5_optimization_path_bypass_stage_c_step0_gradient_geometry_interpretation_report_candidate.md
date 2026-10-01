# Gen5 Optimization-Path Bypass Stage C
## Step-0 Gradient Geometry Interpretation

### Status

Validated Stage C imported evidence:

`PASS`

Run:

`gen5-stagec-step0-gradient-geometry-b4439f0-r1`

Execution HEAD:

`b4439f05a31e6f554e0a99b990ce9222d50afc61`

Implementation freeze:

`93dfe8984bcb7210833cf8f6d8d021acdf139c9f`

Stage B evidence freeze:

`b49c1339231a1c5dbb7f870ddc81159109b08c1e`

Source Phase3A execution:

`d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`

---

## 1. Execution validity

The Stage C matrix executed exactly:

- 9 cells
- 9 forwards
- 9 backwards
- 2 independent Tesla T4 workers
- no DDP
- no optimizer construction
- no optimizer step
- no gradient clipping
- no training
- no task evaluation
- no confirmatory 9601..9900 access
- no scientific p-values

The two-GPU topology was:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

The imported artifact hashes matched the execution-side recorded hashes exactly.

Summary SHA256:

`7dcd4c556a5546f375ec40cdd106325e1d018521332067d095148657bbb7f8a6`

Gradient tensor SHA256:

`1e1f683b562e0106b1aacd1e2d6ec95dd78f3a20db0beed9c4edb1b32f0d84e3`

Provenance SHA256:

`da3a53798541bbabab1504f76e09aeb31bbeb6a2e60eebd0bad2034714b36482`

Step-0 Phase3A loss replay was exact in all nine cells:

`MAX_STEP0_LOSS_ABS_DELTA=0`

Because B was exactly zero initialized, the task-loss-driven step-0 A gradient
was exactly zero in all cells:

`MAX_A_GRADIENT_ABS=0`

---

## 2. Rank validity

All nine step-0 B gradients had effective rank 2.

All nine final learned B matrices had effective rank 2.

Therefore the rank-2 principal-angle interpretation is valid in every cell.

No rank-collapse exception occurred.

---

## 3. R22 projected gradient energy

Overall mean R22-projected step-0 gradient energy fraction:

`7.99510771547e-05`

Rank-2 random-subspace affinity reference:

`8.13802083333e-05`

The weighted R22 gradient-energy fraction is therefore approximately at the
ambient random-subspace scale.

Pressure-level means were:

- P0: `7.88137815764e-05`
- PR: `7.88559632630e-05`
- PC: `8.21834866247e-05`

There is no substantial pressure-induced shift toward R22.

---

## 4. Step-0 gradient versus R22

Overall mean subspace affinity:

`0.000591578889876`

This is numerically above the rank-2 random-subspace affinity reference, but
remains extremely small in absolute terms.

Across cells, R22 principal angles were approximately:

- smallest observed first angle: `87.124799°`
- largest observed second angle: `89.976650°`

Thus the step-0 task-loss gradient plane is nearly orthogonal to R22 in every
cell.

The central interpretation must therefore be based on the absolute geometry,
not solely on enrichment over a tiny high-dimensional random baseline.

---

## 5. Step-0 gradient versus C22

Overall mean C22-projected gradient energy fraction:

`0.000107416577487`

Overall mean gradient/C22 subspace affinity:

`0.000389014444865`

Principal angles are again near 90 degrees.

The initial task-loss gradient therefore does not preferentially occupy the C22
control plane either.

---

## 6. Step-0 gradient versus final learned B

Overall mean subspace affinity between:

`span(∇B_step0)`

and:

`span(B_final)`

was:

`0.0316953048769`

This is far above the high-dimensional rank-2 random reference, while remaining
modest in absolute subspace overlap.

Observed principal angles were approximately:

- first angle: `75.93°` to `79.96°`
- second angle: `81.44°` to `82.58°`

Therefore there is reproducible partial geometric continuity from the initial
loss-driven gradient into the final learned B plane.

However, the final learned plane is not already fixed at step 0.

Optimization continues to rotate and refine the learned bypass geometry.

---

## 7. Pressure invariance

Mean gradient-to-final-B affinity:

- P0: `0.0316873194123`
- PR: `0.0317035793904`
- PC: `0.0316950158278`

Mean gradient-to-R22 affinity:

- P0: `0.000586438049118`
- PR: `0.000579456346401`
- PC: `0.000608842274109`

The pressure manipulation did not materially redirect the initial optimization
geometry toward R22.

This is consistent with the prior Phase3A failure to induce R22 contention and
with Stage A's near pressure-invariance of the final correction geometry.

---

## 8. Scientific interpretation

Stage C supports the bounded conclusion:

`GEN5_STAGE_C_INITIAL_TASK_LOSS_GRADIENT_BYPASSES_R22_WITH_PARTIAL_CONTINUITY_TO_FINAL_LEARNED_B`

The observed pattern is not the expected signature of an
R22-first-then-reroute trajectory.

The initial task-loss-driven factorized update geometry is already nearly
orthogonal to the previously causally validated R22 subspace.

At the same time, the step-0 gradient has substantially more overlap with the
eventual learned B plane than expected from an arbitrary high-dimensional
rank-2 pair.

Therefore the data support:

1. initial loss-gradient/native-causal misalignment;
2. a bypass direction already present at the first loss-driven update;
3. continued optimization-time rotation within the non-R22 solution family.

---

## 9. Relation to Stages A and B

Stage A established that final learned correction geometry has very low R22
occupancy and is strongly seed-dependent but nearly pressure-invariant.

Stage B established that almost all learned task benefit is functionally carried
by the R22-orthogonal component.

Stage C now establishes that the task-loss gradient does not first request R22
and then abandon it.

Instead, the loss-driven optimization path begins outside R22.

Together these stages support the distinction:

`native causal importance != optimization privilege`

for this frozen Gen5 intervention setting.

---

## 10. Hypothesis implications

H1, loss-gradient misalignment, is directly supported in the bounded Stage C
setting.

H4, distributed or multiple bypass solutions, remains compatible with the
combined Stage A-C evidence and is strengthened by the strong seed dependence
of the learned solution family.

H2, downstream functional substitutability, is not yet established.

H3, objective shortcut structure, is not yet isolated.

H5, downstream Jacobian degeneracy, is not yet established.

H6, native-computation-specific causal importance rather than
task-objective-specific optimization privilege, remains consistent with the
current evidence but is not independently established by Stage C alone.

---

## 11. Next discriminating stage

The next stage is Stage D downstream functional-equivalence analysis.

It must distinguish:

1. subspace functional equivalence:
   compare the downstream images of R22 and the learned bypass plane;

2. actual perturbation equivalence:
   compare downstream effects of the realized learned perturbation and its R22
   and R22-orthogonal components.

The Stage D analysis must not collapse these two questions into one metric.

A natural first object is the downstream Jacobian action on orthonormal bases:

- `J Q_R`
- `J Q_B`

with basis-invariant comparison of their downstream image subspaces.

Separately, actual perturbations should compare:

- `δh = B A x`
- `P_R δh`
- `(I - P_R) δh`

Stage D should use existing frozen checkpoints and frozen evaluation domains
before considering any prospective new training.
