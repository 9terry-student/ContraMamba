# ContraMamba Gen4 Low-Displacement Behavioral Robustness Prospective Plan

## 0. Status

This file is a prospective experiment plan, not a new authority document.

It does not reopen discovery, selected/control planes, intervention layer,
token position, tokenizer semantics, checkpoints, model family, behavioral
endpoint, or cohort selection after outcome inspection.

No Codex-generated implementation is part of this plan.

Plan creation base:

`a0c6fdb9c5d816e96eb037b46a15c2e3e26829a2`

## 1. Motivation

Study C established that the frozen Experiment-1 behavioral interventions are
moderate relative to total activation norm but large relative to native local
neighborhood spacing.

The unresolved reviewer question is:

> Does the observed scale-dependent behavioral effect persist when the exact
> frozen correction is applied at substantially smaller magnitude?

This experiment is a robustness / intervention-faithfulness follow-up.
It is not an optimization study.

The known historical behavioral sign pattern is:

- Mamba-370M: positive selected-vs-control behavioral contrast.
- Mamba-1.4B: negative selected-vs-control behavioral contrast.

The experiment asks whether that already-known scale-dependent sign pattern
persists at quarter-strength intervention.

## 2. Scale scope

Primary scope is exactly:

- Mamba-370M
- Mamba-1.4B

Mamba-130M is intentionally not placed into the same primary family.

Reason:

- the current intervention-faithfulness vulnerability was measured directly for
  the frozen 370M/1.4B behavioral correction in Study C;
- 370M and 1.4B share the same behavioral-bridge task, cells, local intervention
  semantics, and Study-C displacement audit;
- the historical 130M small-epsilon study measures a different causal/spectral
  endpoint and should not be silently treated as the same behavioral endpoint.

The existing 130M small-epsilon evidence may be cited as contextual prior
robustness only. A 130M behavioral/readout completion remains a separate
predeclared program item.

## 3. Frozen scale-local geometry

Use the existing frozen scale-local geometry without re-selection.

### Mamba-370M

- model: `state-spaces/mamba-370m-hf`
- frozen selected plane: `P3`
- frozen response-blind control plane: `P5`
- source block: `33`
- target residual layer: `34`
- intervention layer: `35`
- anchor: `A_IDENTITY`
- target offset: `+2`
- content-half intervention semantics unchanged.

### Mamba-1.4B

- model: `state-spaces/mamba-1.4b-hf`
- frozen selected plane: `P5`
- frozen response-blind control plane: `P4`
- source block: `33`
- target residual layer: `34`
- intervention layer: `35`
- anchor: `A_IDENTITY`
- target offset: `+2`
- content-half intervention semantics unchanged.

Use the exact frozen model/checkpoint/tokenizer identities already used by the
370M/1.4B behavioral bridge and Study B/C.

No plane, layer, token, control, or checkpoint re-selection is allowed.

## 4. Fresh population

Materialize exactly one new disjoint XG1 cohort:

- first pair: `xg1_fact_5401`
- last pair: `xg1_fact_5700`
- source pairs: `N = 300`
- target cells:
  - `C0_SHAM`
  - `C2_NAME`

This cohort is disjoint from:

- Study B/C: `4801..5100`
- Study A adjacent response: `5101..5400`

The population must be generated deterministically from the existing frozen XG1
generator semantics.

A new structural holdout builder may extend the existing prior inventory through
`5400`, but it must not alter any previously frozen holdout.

Before GPU execution, validate:

- exact pair sequence `5401..5700`;
- exact 300-pair count;
- no prior pair overlap;
- no prior claim/evidence overlap under the existing inventory checks;
- exact C0/C2 semantic contract;
- no outcome-derived fields;
- exact tokenizer/anchor eligibility for both model scales.

If the fixed cohort fails a technical tokenizer/anchor requirement, the run is
blocked. Do not delete rows, substitute another cohort, or select a favorable
subset after inspecting outcomes.

## 5. Frozen behavioral endpoint

Preserve the exact Experiment-1 behavioral quantity.

For scale `s`, source pair `q`, cell `c`, and correction magnitude `alpha`:

`M_control(alpha)` is the correct-class logit margin after applying the scaled
dominant-control correction.

`M_restored` is the frozen restored/native condition. Its condition correction
is algebraically zero.

At the pair level, average the two cells exactly as in Experiment 1:

`M_control(s,q,alpha) = mean_c M_control(s,q,c,alpha)`

`M_restored(s,q) = mean_c M_restored(s,q,c)`

Define:

`D_BEH(s,q,alpha) = M_restored(s,q) - M_control(s,q,alpha)`

No new behavioral endpoint is introduced.

## 6. Exact correction scaling

For each native local activation, reconstruct the frozen selected and control
components exactly as in the existing behavioral bridge.

Let:

`C_sel = a * u_sel,+ + b * u_sel,-`

where `a,b` are the frozen native selected-plane coefficients.

Construct the coefficient-matched frozen control component:

`C_ctrl = a * u_ctrl,+ + b * u_ctrl,-`

The historical dominant-control correction is:

`delta_control = -C_sel + C_ctrl`

The low-displacement control correction is:

`delta_control(alpha) = alpha * delta_control`

Use exactly two new behavioral magnitudes:

- `alpha = 0.5`
- `alpha = 0.25`

No other new alpha value is allowed.

Do not perform:

- alpha sweep;
- scale-specific alpha selection;
- adaptive magnitude reduction;
- best-alpha selection;
- outcome-conditioned rerun.

## 7. Historical alpha=1 boundary

Do not execute a new alpha=1 behavioral condition on the fresh cohort.

Historical alpha=1 behavioral results remain contextual reference only because
they were measured on the frozen Experiment-1 cohort.

However, for geometric displacement only, the fresh run may capture the unscaled
`delta_control` and native activation once. A later CPU static audit may then
construct:

- `h_native + 1.0 * delta_control`
- `h_native + 0.5 * delta_control`
- `h_native + 0.25 * delta_control`

without any additional model forward.

Thus alpha=1 may be used as a same-fresh-cohort **geometric reference**, not as a
new behavioral endpoint.

## 8. Behavioral execution budget

For each scale × pair × cell:

1. one restored/native forward;
2. one `alpha=0.5` control forward;
3. one `alpha=0.25` control forward.

Therefore:

- pairs: `300`
- cells: `2`
- scales: `2`
- forwards per scale/pair/cell: `3`
- total scientific full-model forwards: `3600`

No backward pass.
No training.
No parameter update.
No gradient measurement.
No alpha=1 behavioral forward.

The implementation may reuse a native/restored forward across the two alpha
contrasts only if the numerical condition is exactly identical.

## 9. Primary confirmatory family

The historical scale-specific signs are already known, so this is a robustness
follow-up conditioned on those signs, not an independent behavioral discovery.

Use alpha `0.25` as the only inferential low-displacement magnitude.

### H_370

Endpoint:

`D_370(q) = D_BEH(mamba370m,q,0.25)`

Test:

- null: `E[D_370] <= 0`
- alternative: `E[D_370] > 0`
- one-sample Student t-test
- `N = 300`

Mandatory sign gate:

`mean(D_370) > 0`

### H_14

Endpoint:

`D_14(q) = D_BEH(mamba14b,q,0.25)`

Test:

- null: `E[D_14] >= 0`
- alternative: `E[D_14] < 0`
- one-sample Student t-test
- `N = 300`

Mandatory sign gate:

`mean(D_14) < 0`

### Multiplicity

Exactly two primary p-values.

Use Holm family-wise correction across:

1. `H_370`
2. `H_14`

Family alpha:

`0.05`

No other p-value is permitted in this study.

## 10. Primary conclusion labels

If both Holm-adjusted tests pass and both mandatory sign gates pass:

`LOW_DISPLACEMENT_SCALE_SIGN_PATTERN_PRESERVED_AT_ALPHA_0_25`

If exactly one scale passes its adjusted test and sign gate:

`LOW_DISPLACEMENT_SCALE_SIGN_PATTERN_PARTIALLY_PRESERVED`

Otherwise:

`LOW_DISPLACEMENT_SCALE_SIGN_PATTERN_NOT_ESTABLISHED`

A failed result is retained as-is. No rescue experiment follows from the same
study.

## 11. Alpha=0.5 role

Alpha `0.5` is secondary descriptive evidence only.

For each scale report:

- mean `D_BEH(alpha=0.5)`
- population SD
- min/q25/median/q75/max
- positive fraction

For 1.4B also report the negative fraction.

Do not compute a p-value at alpha `0.5`.

Do not promote alpha `0.5` to primary if alpha `0.25` fails.

## 12. Cross-scale descriptive context

At each alpha, report descriptively:

`R(alpha) = mean(D_BEH_370M(alpha)) - mean(D_BEH_1.4B(alpha))`

and the pairwise matched difference distribution:

`R_q(alpha) = D_BEH_370M(q,alpha) - D_BEH_1.4B(q,alpha)`

Report mean/SD/quantiles and positive fraction.

No cross-scale p-value is added.

The experiment is not a scaling law and does not define a transition threshold.

## 13. Displacement capture

During the same raw execution, capture for every scale × pair × cell:

- exact native full local activation;
- exact native strong-subspace activation used by Study C;
- exact unscaled `delta_control` in the actual applied dtype;
- exact frozen strong indices;
- scale/cell/pair ordering metadata.

No additional full-model forward is required for this capture.

Do not compute `R_NN` or inferential conclusions during the GPU raw run.

## 14. CPU-only displacement audit

After raw behavioral and state evidence is frozen, compute on CPU for:

- `alpha = 1.0` geometric reference;
- `alpha = 0.5`;
- `alpha = 0.25`.

For every scale × cell × alpha, compute exactly the Study-C definitions.

### Relative displacement

`R_rel_full = ||alpha * delta|| / max(||h_native_full||, 1e-12)`

`R_rel_strong = ||alpha * delta_strong|| / max(||h_native_strong||, 1e-12)`

### Native-neighborhood-normalized displacement

Within each scale × cell, using the same 300 native states and excluding self:

`d_native(q) = min_{r != q} ||h_native(q) - h_native(r)||`

`d_int(q,alpha) = min_{r != q} ||h_native(q) + alpha*delta(q) - h_native(r)||`

`R_NN(q,alpha) = d_int(q,alpha) / max(d_native(q), 1e-12)`

Compute full and strong variants.

Report descriptively:

- mean
- population SD
- min
- q25
- median
- q75
- max
- fraction `R_NN > 1`
- fraction `R_NN > 2`
- nearest-native identity-change rate
- zero-native-norm counts
- nonfinite counts

No p-value.

Do not define a binary on-manifold/off-manifold threshold.

## 15. Expected geometric sanity

Because the correction is deterministically scaled:

- `R_rel(alpha=0.5)` must equal `0.5 * R_rel(alpha=1.0)` up to numeric tolerance.
- `R_rel(alpha=0.25)` must equal `0.25 * R_rel(alpha=1.0)` up to numeric tolerance.

`R_NN` is not assumed to scale linearly.

Failure of the exact `R_rel` scaling identity is a technical blocker.

## 16. Raw information boundary

The GPU raw runner may read only:

- structural fresh population;
- frozen model/checkpoint/tokenizer identity;
- frozen scale-local selected/control geometry;
- frozen correction construction semantics.

During raw execution it must not read:

- historical Experiment-1 behavioral means or p-values;
- Study-B readout-alignment result values;
- Study-B pair-level explanatory merge values;
- Study-C displacement result values;
- Study-A response result values.

The runner outputs no p-value and no scientific conclusion.

## 17. Required raw artifacts

Behavioral raw directory must contain at minimum:

1. `low_displacement_behavioral_items.jsonl`
2. `raw_behavioral_summary.json`
3. `artifact_manifest.json`
4. `SHA256SUMS.txt`

State-capture raw directory must contain at minimum:

1. `low_displacement_states.npz`
2. `capture_index.jsonl`
3. `artifact_manifest.json`
4. `SHA256SUMS.txt`

The behavioral and state-capture artifact boundaries remain separate even if
created by one GPU run.

## 18. Required CPU analysis artifacts

After raw freeze, create a behavioral static-analysis directory containing:

1. `primary_analysis.json`
2. `pair_level_contrasts.jsonl`
3. `artifact_manifest.json`
4. `SHA256SUMS.txt`

Create a separate displacement-audit directory containing:

1. `low_displacement_audit.json`
2. `artifact_manifest.json`
3. `SHA256SUMS.txt`

The behavioral analysis must record exactly two primary p-values.
The displacement audit must record zero p-values.

## 19. Technical gate

Before scientific GPU raw execution, run a non-scientific technical gate that
verifies only:

- both exact model snapshots load;
- exact checkpoints load;
- both GPUs are visible when required;
- correction reconstruction is finite;
- alpha scaling produces finite correction;
- alpha=0.5 and alpha=0.25 applied deltas match the algebraic scale exactly;
- strong indices are valid;
- one test row per scale can execute the three required forwards;
- raw artifact serialization supports the actual activation dtype.

The gate must retain no behavioral margin, Q value, p-value, or displacement
metric.

## 20. Claim boundary

If the alpha=0.25 sign pattern is preserved and displacement is materially lower,
the supported interpretation is:

> The previously observed scale-dependent behavioral contrast persists under a
> substantially smaller version of the exact frozen intervention.

This directly weakens the objection that the sign pattern exists only at the
original large local-neighborhood displacement.

Do not claim:

- on-manifold intervention;
- infinitesimal causal identification;
- zero OOD concern;
- complete local linearity;
- universal robustness to intervention strength;
- a scaling law;
- a phase transition.

If alpha=0.25 fails, the correct interpretation is:

> The scale-dependent behavioral effect is sensitive to finite intervention
> magnitude and should not be generalized to substantially smaller
> perturbations.

No rescue is permitted.

## 21. 130M contextual boundary

The existing 130M small-epsilon robustness evidence may be discussed separately
as showing robustness of the historical 130M spectral/causal endpoint under
smaller epsilon.

It is not pooled with the two-scale behavioral primary family above.

Do not claim that the old 130M epsilon study and the new behavioral alpha study
measure the identical endpoint.

The planned 130M readout-alignment completion remains a separate next program
item after this low-displacement study.

## 22. Stop rule

Run the fixed fresh cohort once.

Do not change after viewing outcomes:

- pair range;
- cells;
- scales;
- alpha values;
- selected/control planes;
- checkpoint;
- intervention layer/token;
- primary alpha;
- primary tail directions;
- Holm family;
- displacement definitions.

After behavioral analysis and displacement audit are frozen, close this study
regardless of result and proceed to the next predeclared program item.
