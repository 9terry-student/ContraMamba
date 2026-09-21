# ContraMamba Gen4 Factor-2 Calibration Static Provenance / Code Audit Report

## 0. Status

- Status: STATIC EVIDENCE REPORT
- Repository evidence HEAD: `d4cb446f4b9f242ab2d96ae41d3c1350572e2814`
- Branch: `gen4-mamba370m-core-replication`
- Training performed in this audit: NO
- Model forward performed in this audit: NO
- Backward performed in this audit: NO
- Tokenizer execution performed in this audit: NO
- Kaggle / GPU used in this audit: NO
- New p-values added: 0
- Row filtering / rescue: NO
- New experiment executed: NO

This report records a read-only static provenance/code audit of the approximately
two-fold calibration observed between the native directional readout `Delta_L`
and finite behavioral response `D_BEH(alpha)`.

It also freezes the design of the next single bounded prospective study.
That prospective section is a plan only and does not itself authorize or claim
execution evidence.

## 1. Audit question

The observed low-displacement descriptive merge showed approximately

`D_BEH(0.25) ~= 0.5 * Delta_L`

rather than

`D_BEH(0.25) ~= 0.25 * Delta_L`.

Combined with the earlier full-strength Study-B pair-level evidence, the
empirical pattern is approximately

`D_BEH(alpha) ~= 2 * alpha * Delta_L`

over the already observed alpha range.

The first audit question is therefore:

> Is the factor near 2 forced by a definition, sign convention, selected/control
> correction construction, margin convention, cell aggregation rule, logit
> serialization rule, duplicated hook application, or other implementation
> convention?

The second question is:

> If no such convention exists, how stable is the factor-2 calibration across
> alpha, scale, and cohort in already frozen evidence?

No new forward execution is needed to answer either question.

## 2. Frozen evidence used

### 2.1 Low-displacement native readout

Raw evidence directory:

`reports/reason_router_gen4_mamba370m14b_low_displacement_native_readout_raw_v1`

Frozen raw evidence commit:

`e8f8877cb9a2e63e25c8f437edb627e820538f22`

Population:

`xg1_fact_5401..xg1_fact_5700`

Scales:

- Mamba-370M
- Mamba-1.4B

Cells:

- `C0_SHAM`
- `C2_NAME`

Raw rows:

`1200 = 2 scales * 300 pairs * 2 cells`

Native readout item SHA256:

`0e69d42c09e445514cab799bc3646ef10500f35598cc1a10ce81502c2e2bb9cc`

### 2.2 Low-displacement behavioral evidence

Frozen behavioral pair analysis:

`reports/reason_router_gen4_mamba370m14b_low_displacement_behavioral_analysis_v1`

Population:

`xg1_fact_5401..xg1_fact_5700`

Behavioral pair-level contrast SHA256:

`712cce9c854f6d5a3b350c04bf11e0e06b19557c0a9e244dd3a93e195972c79d`

Already executed behavioral magnitudes:

- `alpha = 0.5`
- `alpha = 0.25`

No alpha=1 behavioral forward was executed on this fresh low-displacement cohort.

### 2.3 Frozen low-displacement readout-behavior merge

Directory:

`reports/reason_router_gen4_mamba370m14b_low_displacement_native_readout_behavior_pair_merge_v1`

Freeze commit:

`d4cb446f4b9f242ab2d96ae41d3c1350572e2814`

This artifact added no p-value and retained:

`scientific_conclusion = None`

### 2.4 Historical full-strength pair-level comparison

Directory:

`reports/reason_router_gen4_mamba370m14b_readout_behavior_pair_merge_v1`

Population:

`xg1_fact_4801..xg1_fact_5100`

This is a separate cohort and provides historical alpha=1 behavioral context.

## 3. Exact native readout definition

The readout code installs an autograd leaf at the exact local intervention
boundary:

- module: intervention `mixer.in_proj`
- token: frozen target token
- activation half: first / content half
- all upstream autograd history cut at that activation
- native activation value preserved exactly

The task margin is:

`M = correct_logit - active_wrong_logit`

where `active_wrong_logit` is the larger of the two wrong-class logits at the
native point.

Let:

`g = dM / dh`

at this exact local activation.

The native selected component is:

`C_sel = a * u_sel,+ + b * u_sel,-`

using coefficients `(a,b)` obtained by projecting the native strong activation
onto the frozen selected plane.

The coefficient-matched control component is:

`C_ctrl = a * u_ctrl,+ + b * u_ctrl,-`.

The stored directional quantities are:

`L_selected = g^T C_sel`

`L_control = g^T C_ctrl`

and therefore:

`Delta_L = L_selected - L_control`

or exactly:

`Delta_L = g^T (C_sel - C_ctrl)`.

At pair level, the readout uses the same aggregation convention as prior Study B:

`Delta_L_pair = mean_cell(Delta_L_row)`

over exactly `C0_SHAM` and `C2_NAME`.

There is no factor 2 in this definition.

## 4. Exact behavioral correction definition

The dominant-control correction is implemented as:

`delta = -C_sel + C_ctrl`

or equivalently:

`delta = C_ctrl - C_sel`.

For low-displacement execution:

`delta(alpha) = alpha * delta`.

The hook applies this correction once:

`h_alpha = h + alpha * delta`.

The applied correction is added only to:

- the same intervention module;
- the same target token;
- the same first/content activation half;
- the frozen strong indices.

The gate half, non-strong indices, earlier tokens, and later tokens are required
to remain unchanged.

The low-displacement raw artifact records exact alpha scaling. Across the frozen
control rows:

`max scaling_identity_max_abs_residual = 0`

and:

`max applied_correction_max_abs_residual <= 4.76837158203125e-7`.

The correction is therefore not duplicated by the low-displacement hook.

## 5. Behavioral endpoint definition

For each scale, pair, cell, and alpha:

`M_restored`

is the correct-class logit margin under `restored_native`.

The restored/native condition has algebraically zero correction.

`M_control(alpha)`

is the correct-class logit margin after applying `alpha * delta`.

At pair level:

`M_restored(pair) = mean_cell M_restored(cell)`

`M_control(pair,alpha) = mean_cell M_control(cell,alpha)`

and:

`D_BEH(pair,alpha) = M_restored(pair) - M_control(pair,alpha)`.

There is no subtraction of two independently perturbed arms and no factor 2 in
this endpoint.

## 6. First-order identity implied by the code

Because:

`delta = C_ctrl - C_sel`

and:

`Delta_L = g^T(C_sel - C_ctrl) = -g^T delta`,

a first-order expansion around the native point gives:

`M(h + alpha*delta) = M(h) + alpha*g^T delta + O(alpha^2)`.

Therefore:

`D_BEH(alpha)`
`= M(h) - M(h + alpha*delta)`
`= -alpha*g^T delta + O(alpha^2)`
`= alpha*Delta_L + O(alpha^2)`.

Thus the implementation's local first-order convention predicts a unit
calibration in:

`D_BEH(alpha) / (alpha * Delta_L)`.

The observed factor near 2 is not algebraically implied by the readout or
behavioral definitions.

## 7. Margin and logit convention audit

The readout margin uses the model output tensor:

`output["logits"]`.

Behavioral serialization writes:

`final_logits = output["logits"]`

without temperature scaling, summation, averaging, or other transformation.

The behavioral correct-class margin is then recomputed from these same three
logits as:

`correct_logit - max(wrong_logits)`.

Therefore readout and behavioral endpoints use the same logit scale.

No factor 2 is introduced by serialization.

### 7.1 Native point identity

Across all `1200` scale-cell native/restored rows:

- behavioral `restored_native` and readout native margin differ by at most
  `1.1920928955078125e-7`;
- native replay residual in the behavioral runner is exactly `0`.

The gradient and finite-response measurements therefore refer to the same
numerical native function point up to ordinary float precision.

### 7.2 Active-wrong-class switching

A possible concern is that the readout differentiates the native active wrong
class while the finite behavioral margin re-evaluates `max(wrong)` after
intervention.

Frozen artifact audit:

At `alpha = 0.25`:

- Mamba-370M wrong-class switches: `0 / 600`
- Mamba-1.4B wrong-class switches: `0 / 600`

At `alpha = 0.5`:

- Mamba-370M wrong-class switches: `1 / 600`
- Mamba-1.4B wrong-class switches: `1 / 600`

Recomputing finite responses against the native fixed wrong class leaves the
factor-2 calibration essentially unchanged.

Therefore active-wrong switching does not explain the observed factor.

## 8. Static calibration audit on the fresh low-displacement cohort

Define the prospective-style calibration statistic:

`K(alpha) = origin-slope[D_BEH(alpha) on alpha * Delta_L]`.

Equivalently, if `s(alpha)` is the origin slope of `D_BEH(alpha)` on
`Delta_L`, then:

`K(alpha) = s(alpha) / alpha`.

### 8.1 Mamba-370M

At `alpha = 0.25`:

- origin slope `D` on `Delta_L`: `0.5044018870314899`
- `K(0.25)`: `2.0176075481259597`
- Pearson: `0.9996813118085738`
- RMSE under `alpha * Delta_L`: `0.0004252771609340755`
- RMSE under `2 * alpha * Delta_L`: `0.000021987255993627566`
- RMSE improvement from 1x to 2x calibration: `19.341984332075405 x`

At `alpha = 0.5`:

- origin slope `D` on `Delta_L`: `1.0171790885690783`
- `K(0.5)`: `2.0343581771381567`
- Pearson: `0.9991635407827589`
- RMSE under `alpha * Delta_L`: `0.0008662825895921723`
- RMSE under `2 * alpha * Delta_L`: `0.00007474902864007168`
- RMSE improvement from 1x to 2x calibration: `11.589215343030865 x`

### 8.2 Mamba-1.4B

At `alpha = 0.25`:

- origin slope `D` on `Delta_L`: `0.49716096872366233`
- `K(0.25)`: `1.9886438748946493`
- Pearson: `0.9997547798314159`
- RMSE under `alpha * Delta_L`: `0.0007091785704955762`
- RMSE under `2 * alpha * Delta_L`: `0.00003671231082787928`
- RMSE improvement from 1x to 2x calibration: `19.31718691913632 x`

At `alpha = 0.5`:

- origin slope `D` on `Delta_L`: `0.9851441064710171`
- `K(0.5)`: `1.9702882129420343`
- Pearson: `0.9989851287680873`
- RMSE under `alpha * Delta_L`: `0.001397724283307284`
- RMSE under `2 * alpha * Delta_L`: `0.00015051491999530493`
- RMSE improvement from 1x to 2x calibration: `9.286283933518908 x`

## 9. Within-cohort alpha scaling

The same fresh cohort provides a direct static check of response scaling from
`alpha=0.25` to `alpha=0.5`.

### Mamba-370M

Origin slope:

`D(0.5) on D(0.25) = 2.017191255495757`

Pearson:

`0.9997681187906967`

RMSE of:

`D(0.5) - 2*D(0.25)`

is:

`0.00003934636750432476`.

The secant derivative over `.25 -> .5`,

`[D(.5)-D(.25)] / .25`,

regressed through the origin on `Delta_L`, has slope:

`2.0511088061503533`

with Pearson:

`0.9981992389080243`.

### Mamba-1.4B

Origin slope:

`D(0.5) on D(0.25) = 1.982763314471707`

Pearson:

`0.9996948277754938`

RMSE of:

`D(0.5) - 2*D(0.25)`

is:

`0.0000809948413989404`.

The `.25 -> .5` secant derivative regressed on `Delta_L` has slope:

`1.9519325509894179`

with Pearson:

`0.9976024010428372`.

Thus the behavioral response is nearly proportional to alpha over the observed
low-displacement interval, while its proportionality to the native derivative
is approximately two-fold.

## 10. Cross-cohort historical alpha=1 context

The earlier full-strength pair-level cohort is disjoint:

`xg1_fact_4801..xg1_fact_5100`.

Static recalculation gives:

### Mamba-370M, historical alpha=1

- origin slope `D` on `Delta_L`: `2.073244285367443`
- `K(1)`: `2.073244285367443`
- Pearson: `0.9982297751731574`
- RMSE under `Delta_L`: `0.00182680063858512`
- RMSE under `2*Delta_L`: `0.00024339257816893505`

### Mamba-1.4B, historical alpha=1

- origin slope `D` on `Delta_L`: `1.9353088523190094`
- `K(1)`: `1.9353088523190094`
- Pearson: `0.9955534403379731`
- RMSE under `Delta_L`: `0.0026990193088575645`
- RMSE under `2*Delta_L`: `0.0006118338756883042`

The alpha=1 values come from a different cohort, so they are contextual
cross-cohort replication rather than a same-cohort alpha curve.

Nevertheless, the approximate factor-2 calibration appears at:

- two model scales;
- two disjoint XG1 cohorts;
- alpha `0.25`;
- alpha `0.5`;
- historical alpha `1`.

## 11. Displacement context

The low-displacement Study-C audit showed that `alpha=0.25` is reduced relative
to total activation norm but is not infinitesimal relative to native
nearest-neighbor spacing.

For example, at alpha `0.25`:

Mamba-370M:

- full-space `R_NN` means are approximately `3.55`;
- strong-space `R_NN` means are approximately `6.0`.

Mamba-1.4B:

- full-space `R_NN` means are approximately `4.8`;
- strong-space `R_NN` means are approximately `9.3`.

Therefore the existing alpha range should not be described as an established
infinitesimal local regime.

## 12. Static audit disposition

The following candidate explanations are ruled out by code/provenance audit:

- a hidden factor 2 in `Delta_L` definition: NO;
- selected/control sign reversal producing a factor 2: NO;
- pair-level two-cell summation instead of averaging: NO;
- duplicated intervention application: NO;
- alpha applied twice or omitted: NO;
- gate/content-half mismatch: NO;
- readout and behavior evaluated at different intervention boundaries: NO;
- logit serialization / temperature scaling factor: NO;
- restored condition containing a nonzero corrective arm: NO;
- active-wrong-class switching as the systematic cause: NO;
- behavioral response access contaminating the native raw readout: NO.

The audited definitions imply:

`D_BEH(alpha) = alpha * Delta_L + O(alpha^2)`

as `alpha -> 0`.

The frozen evidence instead shows, over the already measured finite range:

`D_BEH(alpha) ~= 2 * alpha * Delta_L`.

Therefore:

`FACTOR_2_DEFINITIONAL_CONVENTION = NOT_FOUND`

`FACTOR_2_IMPLEMENTATION_DUPLICATION = NOT_FOUND`

`FACTOR_2_MARGIN_SWITCH_ARTIFACT = NOT_SUPPORTED`

`FACTOR_2_FINITE_RANGE_EMPIRICAL_CALIBRATION = OBSERVED`

This audit does not establish the unique functional mechanism that produces the
factor.

In particular, it does not establish that ordinary quadratic curvature alone is
the cause.

## 13. Scientific interpretation boundary

A smooth differentiable function with the audited gradient definition must
satisfy:

`lim_(alpha->0) D_BEH(alpha)/(alpha*Delta_L) = 1`

for nonzero `Delta_L` under the same fixed local branch.

The current evidence instead looks approximately plateau-like near `2` over the
measured `.25-.5` fresh-cohort interval and near `2` again in historical
full-strength evidence.

This creates a mechanistic question that is distinct from the original reviewer
robustness question:

> At what displacement scale does the response leave the approximately 2x
> finite-gain regime and recover the 1x native derivative regime?

The existing artifacts cannot answer that question because no behavioral
forward below alpha `0.25` exists.

No stronger mechanism claim is made here.

A cautious description of the existing result is:

> Across two scales and disjoint cohorts, finite behavioral responses exhibit a
> highly stable approximately two-fold proportional calibration relative to the
> native first-order directional readout over the measured intervention range.

The phrase "local derivative law" should not be used for the current finite
displacements.

## 14. Next bounded prospective study

The next study is a single prospective small-alpha curve.

It is motivated by the unexpected static-audit finding above, not by a failed
primary behavioral result.

### 14.1 Scope

Models:

- Mamba-370M
- Mamba-1.4B

Fresh population:

`xg1_fact_5701..xg1_fact_6000`

Pair count:

`N = 300`

Cells:

- `C0_SHAM`
- `C2_NAME`

The cohort must be disjoint from:

- Study B/C: `4801..5100`
- adjacent-response cohort: `5101..5400`
- low-displacement robustness cohort: `5401..5700`

Use the already frozen scale-local geometry without selection or tuning.

Mamba-370M:

- selected plane: `P3`
- control plane: `P5`

Mamba-1.4B:

- selected plane: `P5`
- control plane: `P4`

Use the same frozen checkpoint, layer, anchor, target offset, content-half
intervention, tokenizer, and correction construction already validated for the
current 370M/1.4B program.

### 14.2 Native quantity

For every scale, pair, and cell compute before inspecting behavioral responses:

`Delta_L = g^T(C_sel - C_ctrl)`.

Pair aggregation remains:

`Delta_L_pair = mean_cell Delta_L_row`.

The native readout and behavioral raw execution boundaries must remain
separated so that readout computation does not access behavioral response.

### 14.3 Intervention

Use the same:

`delta = C_ctrl - C_sel`.

Apply:

`h_alpha = h + alpha * delta`.

Prospectively freeze exactly four positive magnitudes:

- `alpha = 0.25`
- `alpha = 0.125`
- `alpha = 0.0625`
- `alpha = 0.03125`

No extra alpha value may be added after outcome inspection.

No scale-specific alpha choice is allowed.

No adaptive rerun is allowed.

### 14.4 Behavioral endpoint

For each scale and pair:

`D_BEH(alpha) = M_restored - M_control(alpha)`

with the same two-cell mean convention.

No alternate endpoint is introduced.

### 14.5 Primary descriptive quantity

For each scale and alpha define:

`K(alpha) = origin-slope[D_BEH(alpha) on alpha * Delta_L]`.

This is the primary curve to be reported.

The central figure is:

`alpha` versus `K(alpha)`.

The study is intended to locate whether the observed finite-gain calibration
moves toward the mathematically required local derivative limit as alpha
decreases.

No result-dependent transition threshold is selected after execution.

### 14.6 Secondary quantities

For each scale and alpha report:

- Pearson correlation between `Delta_L` and `D_BEH(alpha)`;
- Spearman correlation, coefficient only;
- sign agreement;
- RMSE under `alpha * Delta_L`;
- RMSE under `2*alpha * Delta_L`;
- mean and population SD of `D_BEH(alpha)`;
- pair-level residual distribution.

No p-value is required for this mechanistic calibration curve unless separately
predeclared before execution.

### 14.7 Geometric context

Using the same native states and applied corrections, retain enough raw state
evidence to compute CPU-only displacement metrics at all four alpha values.

Reuse the Study-C definitions:

- `R_rel_full`
- `R_rel_strong`
- `R_NN_full`
- `R_NN_strong`
- nearest-native identity-change rate.

The scientific comparison of interest is whether movement of `K(alpha)` toward
`1` accompanies entry into a more native-local displacement regime.

No binary on-manifold threshold is introduced.

### 14.8 No negative-alpha arm in the first study

Do not add `-alpha` to this first bounded prospective execution.

A symmetric `+/- alpha` study is scientifically attractive because its central
difference can cancel leading even-order curvature, but it is deliberately
reserved as a later branch.

It may be considered only after the positive small-alpha curve is frozen and
interpreted.

### 14.9 Stop rule

Run this alpha grid once.

After the single fresh cohort is executed and frozen:

- do not add another alpha;
- do not choose a favorable subset;
- do not extend the grid because the curve is ambiguous;
- do not rerun a scale because its calibration is less clean;
- do not add the negative-alpha arm within the same study.

The completed four-alpha curve is retained regardless of outcome.

## 15. Possible outcome interpretations, prospectively separated

These are qualitative interpretations, not preselected acceptance thresholds.

### Pattern A: recovery toward 1

If `K(alpha)` moves materially toward `1` as alpha decreases, the evidence would
support a transition from an amplified finite-displacement regime toward the
native first-order regime.

The interpretation would be:

> The native derivative is progressively recovered as the intervention
> approaches the native state, while larger finite displacements exhibit an
> amplified proportional response.

### Pattern B: approximately 2 down to alpha=0.03125

If the curve remains approximately flat near `2` through the smallest magnitude,
ordinary "the current alpha was simply too large" is no longer a sufficient
description.

The next mechanistic question would then concern either still-smaller numerical
scales or a symmetric `+/- alpha` derivative audit.

This first study must stop before pursuing either branch.

### Pattern C: unstable / scale-specific curve

If the curve is not monotone, not stable across scales, or loses strong
readout-response alignment, the factor-2 phenomenon should be treated as a
finite-range empirical regularity rather than a universal calibration law.

No rescue sweep follows.

## 16. Final disposition

Static audit:

`FACTOR_2_DEFINITIONAL_CONVENTION = NOT_FOUND`

`FACTOR_2_IMPLEMENTATION_DUPLICATION = NOT_FOUND`

`FACTOR_2_MARGIN_SWITCH_ARTIFACT = NOT_SUPPORTED`

`LOWDISP_K_370M_ALPHA_0_25 = 2.0176075481259597`

`LOWDISP_K_370M_ALPHA_0_5 = 2.0343581771381567`

`LOWDISP_K_14B_ALPHA_0_25 = 1.9886438748946493`

`LOWDISP_K_14B_ALPHA_0_5 = 1.9702882129420343`

`HISTORICAL_K_370M_ALPHA_1 = 2.073244285367443`

`HISTORICAL_K_14B_ALPHA_1 = 1.9353088523190094`

`FACTOR_2_FINITE_RANGE_EMPIRICAL_CALIBRATION = OBSERVED`

Unique mechanistic cause:

`NOT_ESTABLISHED`

Next study:

`FRESH_POPULATION = xg1_fact_5701..xg1_fact_6000`

`ALPHAS = [0.25, 0.125, 0.0625, 0.03125]`

`PRIMARY_CURVE = K(alpha)`

`NEGATIVE_ALPHA_ARM = NOT_IN_FIRST_STUDY`

`SELECTION_TUNING = NONE`

`STOP_AFTER_ONE_GRID = YES`

## 17. Stop condition

After this report is frozen:

- do not run another analysis on the old cohorts to select alpha values;
- do not reopen selected/control geometry;
- do not add a new model scale;
- do not add negative alpha to the first prospective run;
- do not execute GPU work until the fresh `5701..6000` structural/token
  preflight and bounded implementation are ready at a specific commit.

The next routine repository task is the minimal implementation / validation
needed for this single prospective study.
