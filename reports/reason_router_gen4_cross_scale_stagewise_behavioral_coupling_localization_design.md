# ContraMamba Cross-Scale Stagewise Behavioral-Coupling Localization
## Prospective mechanistic design: Mamba-370M vs Mamba-1.4B

### Status

`PROSPECTIVE_MECHANISTIC_LOCALIZATION_DESIGN_FREEZE`

Repository anchor before implementation:

`c337b8c894b63efc6f4ae9e8c014eaaef3362df3`

This experiment is inserted before AVeriTeC model execution.

The AVeriTeC shared 130M/370M token gate at `c337b8c` remains frozen and is not
modified by this experiment.

This design authorizes no training and no model selection. It defines a new
mechanistic observation using the already frozen behavioral-bridge population and
already frozen scale-local causal objects.

---

## 1. Motivation

The completed cross-scale behavioral bridge established:

- Mamba-370M: behavioral bridge supported;
- Mamba-1.4B: behavioral bridge not supported;
- cross-scale behavioral recurrence through 1.4B: not established.

A zero-new-forward static decomposition then showed:

### Mamba-370M

`A_sel = M_native - M_neutralized = +0.0002482416232426961`

`D_BEH = M_native - M_control = +0.0011206856990853946`

therefore:

`B_ctrl = M_control - M_neutralized = A_sel - D_BEH = -0.0008724440758426985`

### Mamba-1.4B

`A_sel = -0.00032495816548665365`

`D_BEH = -0.002012885312239329`

therefore:

`B_ctrl = +0.0016879271467526752`

The final downstream ordering is therefore inverted:

- 370M: selected component favorable, matched control unfavorable;
- 1.4B: selected component slightly unfavorable, matched control favorable.

The static analysis cannot establish where this inversion arises.

The present experiment asks exactly that mechanistic question.

---

## 2. Scientific question

> After the frozen layer-35 intervention, at what subsequent computational stage does
> the 370M-positive / 1.4B-negative behavioral ordering become readable by the frozen
> downstream decision path?

This is not a rescue experiment.

It does not ask whether another plane, layer, token, coefficient, control, endpoint,
or subset could make Mamba-1.4B pass.

---

## 3. Scales

Exactly two scales participate:

1. `mamba370m`;
2. `mamba14b`.

### Mamba-370M

- frozen checkpoint unchanged;
- selected plane: `P3`;
- response-blind control: `P5`;
- intervention layer: `35`.

### Mamba-1.4B

- frozen checkpoint unchanged;
- selected plane: `P5`;
- response-blind control: `P4`;
- intervention layer: `35`.

No selection is reopened.

Plane-number equality or inequality is not interpreted as semantic cross-scale
identity.

---

## 4. Why Mamba-130M is contextual only

Mamba-130M already has a strong positive behavioral bridge and provides useful
context:

- `A_sel > 0`;
- `B_ctrl < 0`;
- `D_BEH > 0`;
- large positive `C2_NAME` effect.

However its completed behavioral cohort is `xg1_fact_2701..3000`, whereas the directly
paired 370M/1.4B behavioral experiment uses `xg1_fact_4801..5100`.

Adding a new 130M run now would introduce a new scale/cohort question and is not
necessary to localize the already observed paired 370M/1.4B divergence.

Therefore:

`NEW_130M_FORWARD_COUNT = 0`

The 130M result remains contextual evidence only.

---

## 5. Population

Reuse exactly the already frozen shared behavioral population:

`xg1_fact_4801..xg1_fact_5100`

with:

`N = 300 source pairs`

and exactly the two behavioral cells:

- `C0_SHAM`;
- `C2_NAME`.

No new population is generated.

No row selection is changed.

---

## 6. Conditions

Exactly three conditions per row:

1. `native`;
2. `dominant_neutralized`;
3. `dominant_control`.

`dominant_restored` is omitted because the completed behavioral bridge already
established exact restoration as the native round trip.

For every stage `k`, define correct-class margin readouts:

- `M_native(k)`;
- `M_neutralized(k)`;
- `M_control(k)`.

Then define:

`A_sel(k) = M_native(k) - M_neutralized(k)`

`B_ctrl(k) = M_control(k) - M_neutralized(k)`

`D(k) = M_native(k) - M_control(k)`

with the algebraic identity:

`D(k) = A_sel(k) - B_ctrl(k)`.

No inferential test is attached to these stagewise quantities.

---

## 7. Full-model forward budget

Per scale:

`300 pairs × 2 cells × 3 conditions = 1800 full Mamba forwards`

Across two scales:

`3600 full Mamba forwards`

These are observation forwards only.

No backward pass.

No training.

No gradient-based probe.

---

## 8. Stage capture

Both frozen backbones contain 48 Mamba blocks and use intervention block `35`
(zero-based block index).

Capture the full residual-stream tensor at:

1. `pre_block_35` — the residual entering intervention block 35;
2. `post_block_35`;
3. `post_block_36`;
4. ...;
5. `post_block_47`;
6. `post_final_norm`.

The first stage is a required zero-difference sanity anchor because all three
conditions are identical before the intervention inside block 35.

The stage capture hooks are observational only and must not replace module outputs.

---

## 9. Frozen downstream lens

A hidden-state tensor by itself has no P3/P5/P4 scalar ordering.

Therefore every captured residual state is read through the existing frozen
ContraMamba downstream path.

The historical model already supports:

`encoder_hidden_states=<captured state>`

which skips the Mamba backbone and executes only the frozen downstream
frame/predicate/sufficiency/polarity/decision path.

For every stage and condition record:

- `frame_prob`;
- `predicate_coverage_prob`;
- `sufficiency_prob`;
- `positive_energy`;
- `negative_energy`;
- `q_authorized`;
- `entitlement_prob`;
- final three-way logits;
- correct-class logit margin.

This is called a:

`stagewise downstream lens`

It is not a claim that the model makes a real decision at every intermediate block.

It asks whether the frozen final downstream readout can already read a given
intermediate residual representation in the final positive or negative orientation.

---

## 10. Final-state equivalence gate

`post_final_norm` passed through the downstream-only replay must reproduce the actual
full-model downstream output for the same condition within a strict numerical
tolerance.

Additionally, the actual final outputs for:

- native;
- dominant neutralized;
- dominant control;

must reproduce the already frozen `4801..5100` behavioral rows.

If instrumentation changes the final behavioral result beyond tolerance, the
localization run is invalid.

---

## 11. Hidden propagation diagnostics

Without adding forwards, compare captured hidden states between conditions.

At every stage record:

### selected-component propagation

`H_native - H_neutralized`

### control-component propagation

`H_control - H_neutralized`

### behavioral contrast propagation

`H_native - H_control`

For each difference record:

- full attended-sequence L2 norm;
- target-token L2 norm;
- attended suffix L2 norm from the intervention token onward;
- maximum absolute difference before the intervention token.

The pre-intervention prefix must remain invariant under the causal intervention.

These are descriptive propagation diagnostics only.

---

## 12. Primary localization quantities

No p-values are added.

For each scale and stage report the source-pair distribution of:

- `A_sel(k)`;
- `B_ctrl(k)`;
- `D(k)`.

Report:

- mean;
- median;
- fraction positive;
- fraction negative.

Also report C0 and C2 separately.

For each primitive pathway report condition differences:

- frame;
- predicate coverage;
- sufficiency;
- q-authorized;
- positive energy;
- negative energy.

---

## 13. Descriptive localization rules

For Mamba-1.4B identify:

- first stage at which mean `D(k) < 0` and remains negative at every later stage;
- first stage at which mean `A_sel(k) < 0` and remains negative;
- first stage at which mean `B_ctrl(k) > 0` and remains positive.

For Mamba-370M analogously identify persistent:

- positive `D`;
- positive `A_sel`;
- negative `B_ctrl`.

Across scales identify the first stage at which:

`mean D_370(k) > 0` and `mean D_1.4B(k) < 0`

and that opposition persists through the final stage.

Perform the same descriptive check for `C2_NAME`.

These are deterministic descriptive labels, not inferential endpoints.

---

## 14. Interpretation

Possible outcomes include:

### A. Reversal already present at `post_block_35`

Interpretation boundary:

The divergence is readable immediately after the intervention block. This experiment
cannot then assign the inversion specifically to later blocks 36--47.

The mismatch lies no later than the first post-intervention residual representation
as read by the frozen downstream path.

### B. Reversal first becomes persistent at block `k > 35`

Interpretation:

The remaining recurrent backbone transport reorganizes the intervention into a
representation that the same frozen downstream path reads in the opposite orientation.

### C. Reversal appears only at `post_final_norm`

Interpretation:

Final normalization is implicated in the readout transition.

### D. Primitive-pathway transition

If the margin reversal coincides with a persistent change in frame, predicate,
sufficiency, or q-authorized condition differences, that pathway becomes a localized
candidate for the downstream coupling reorganization.

This is localization, not proof of a unique causal mediator.

---

## 15. What this experiment does not do

It does not:

- rerun discovery;
- change P3/P5/P4 selection;
- change intervention layer;
- change token anchor;
- change coefficient magnitude;
- change target cells;
- change the behavioral endpoint;
- add a significance test;
- rescue Mamba-1.4B;
- use AVeriTeC responses;
- train a probe;
- perform backward propagation.

---

## 16. Execution boundary

Implementation and tests must be frozen at a specific commit before Kaggle execution.

The raw runner may not import SciPy or execute any t-test.

The analyzer may not execute any inferential p-value.

A successful run establishes only valid stagewise observation.

Scientific interpretation occurs after validated artifact import.

### Final design label

`PASS_READY_FOR_STAGEWISE_COUPLING_LOCALIZATION_IMPLEMENTATION`
