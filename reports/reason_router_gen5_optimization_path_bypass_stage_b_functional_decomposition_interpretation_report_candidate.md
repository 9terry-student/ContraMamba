# Gen5 Optimization-Path Bypass Stage B
## Functional Decomposition Interpretation

### Status

Stage B forward-only execution and imported-artifact validation:

`PASS`

Execution run:

`gen5-stageb-functional-decomposition-f70c0b2-r2`

Execution HEAD:

`f70c0b265b2bba8c2a6728f3f96f8c831c805346`

Implementation freeze:

`0107c853b8b943e9171b2df68a869a7d6127f6dc`

Execution authority:

`f70c0b265b2bba8c2a6728f3f96f8c831c805346`

Population:

`FROZEN_PHASE3A_DEV_MATCHED_PRESSURE`

No confirmatory 9601..9900 examples were loaded.

No backward, optimizer step, training, or scientific p-value was executed.

---

## 1. Question

Stage A established that the learned Phase3A corrections occupy very little of
the previously causally validated R22 output-state subspace.

Stage B asks whether that small R22 component nevertheless carries substantial
functional task benefit.

For every frozen Phase3A checkpoint, the learned output-write factor B was
decomposed as:

- FULL = B
- R22_ONLY = P_R B
- R22_REMOVED = (I - P_R) B
- ZERO = 0

A was held fixed.

Thus:

R22_ONLY A + R22_REMOVED A = FULL A

up to the validated numerical reconstruction tolerance.

---

## 2. Artifact validity

Imported files:

- functional_decomposition_rows.jsonl
- functional_decomposition_summary.json
- run_provenance.json

Validated row count:

`30240`

Validated condition evaluations:

`36`

Validated matrix:

`3 seeds × 3 pressures × 4 conditions`

Rows per condition:

`840`

Summary SHA256:

`9ad03ea26345a528255b1f66d2144d5185708b59333f77af13a05414fefa0580`

Rows SHA256:

`0bb9a11f963b8291ef78c29b29fa13b2fe12b193e344d2febef2a5d5567bfeac`

Maximum decomposition reconstruction error:

`1.862645149230957e-09`

Maximum residual R22 projection after R22 removal:

`2.444721758365631e-08`

All frozen provenance, cardinality, decomposition, parent-identity, and
scientific-firewall checks passed.

---

## 3. Main result

Across all nine seed-pressure checkpoints, removing the R22 component from the
learned correction preserved essentially all of the task-loss improvement.

Mean across the nine cells:

- FULL gain versus ZERO:
  `0.502665698528`
- R22_REMOVED gain versus ZERO:
  `0.502598557207`
- R22_ONLY gain versus ZERO:
  `0.000044213401`

Mean retained FULL gain after R22 removal:

`0.999866381543`

or approximately:

`99.9866%`

Cell-level R22_REMOVED retained-gain fractions ranged from approximately:

`99.9847%` to `99.9895%`

The mean loss penalty from removing R22 relative to FULL was only:

`6.714132097e-05`

By contrast, the isolated R22 component retained essentially none of the learned
loss improvement. Its retained-gain fraction fluctuated around zero and was
negative for seed 6202.

---

## 4. Accuracy pattern

The accuracy result is categorical and consistent across all nine cells.

For every seed-pressure checkpoint:

- FULL accuracy:
  `0.714285714286`
- R22_REMOVED accuracy:
  `0.714285714286`

For R22_ONLY and ZERO:

- P0:
  `0.269047619048`
- PR:
  `0.267857142857`
- PC:
  `0.267857142857`

Thus R22 removal caused no observed accuracy loss on the frozen Stage B dev
domain, while the R22-only correction behaved like the zero-correction
baseline.

---

## 5. Pressure-level descriptive means

P0:

- mean FULL gain = `0.499985973040`
- mean R22_ONLY gain = `0.000053962072`
- mean R22_REMOVED gain = `0.499919354916`
- mean R22_REMOVED minus FULL loss = `0.000066618125`

PR:

- mean FULL gain = `0.502935965856`
- mean R22_ONLY gain = `0.000050544739`
- mean R22_REMOVED gain = `0.502869506677`
- mean R22_REMOVED minus FULL loss = `0.000066459179`

PC:

- mean FULL gain = `0.505075156689`
- mean R22_ONLY gain = `0.000028133392`
- mean R22_REMOVED gain = `0.505006810029`
- mean R22_REMOVED minus FULL loss = `0.000068346659`

The qualitative decomposition pattern is therefore stable across P0, PR, and
PC.

---

## 6. Scientific interpretation

Stage B supports the following bounded conclusion:

`GEN5_STAGE_B_LEARNED_TASK_BENEFIT_IS_PRIMARILY_CARRIED_OUTSIDE_R22_ON_FROZEN_PHASE3A_DEV`

The Phase3A learned correction does not merely have low geometric occupancy of
R22.

Its task-improving functional effect is also carried almost entirely by the
R22-orthogonal output-write component.

The R22-only component is insufficient to reproduce the learned task benefit,
while removing R22 preserves essentially the full benefit.

This is direct functional evidence for a learned bypass around the previously
validated native causal R22 realization.

---

## 7. Relation to Stage A

Stage A showed:

- pressure manipulation barely changes A, B, or BA within seed;
- seed changes the learned solution substantially;
- learned B has very low overlap with R22;
- the learned corrections form a structured but seed-variable family.

Stage B adds the missing functional fact:

the non-R22 component is not merely geometrically dominant; it carries the
learned task improvement.

Together, Stages A and B support a multiple-solution / bypass interpretation
more strongly than a hidden-high-leverage-R22 interpretation of the final
learned correction.

---

## 8. What Stage B does not establish

Stage B does not establish that:

- R22 is unimportant to the frozen native Mamba computation;
- R22 causal necessity or restorability was false;
- R22 and the learned bypass plane are generally downstream-functionally
  equivalent;
- the optimizer never initially demanded R22;
- the step-0 task-loss gradient is orthogonal to R22;
- trajectory rerouting did or did not occur;
- the downstream Jacobian is degenerate;
- the task objective explicitly prefers a particular bypass mechanism.

The prior causal necessity/restoration evidence and the present learned
functional decomposition concern different objects and remain compatible.

Causal importance does not imply optimization privilege.

---

## 9. Next discriminating experiment

The next authorized scientific question should be Stage C gradient geometry.

Because B is exactly zero initialized, the first task-loss-driven factorized
correction geometry is governed by:

`∇_B L`

rather than by `∇_A L`.

Stage C should therefore measure at step 0, for each frozen training seed and
pressure:

- projected gradient energy into R22;
- projected gradient energy into C22;
- effective rank of ∇B;
- principal angles between R22 and span(∇B);
- principal angles between C22 and span(∇B);
- principal angles between span(∇B_step0) and span(B_final).

The primary discrimination is:

1. If step-0 ∇B is already far from R22 and aligned with B_final, then the
   optimizer did not initially demand R22.

2. If step-0 ∇B is near R22 but B_final is far from R22, then the optimization
   trajectory rerouted away from the native causal realization.

Stage D downstream functional-equivalence analysis remains separate and should
not be inferred from Stage B alone.
