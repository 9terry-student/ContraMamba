# ContraMamba Gen4 Mamba-1 Vanilla-LM Functional Control Plan

## Status

`PROSPECTIVE_VANILLA_MAMBA_FUNCTIONAL_CONTROL`

This document freezes the functional-control design before any vanilla-LM
readout value is observed.

No vanilla-LM scientific execution has occurred at the time of this freeze.

## 1. Scientific question

Previously frozen evidence establishes two distinct facts:

1. pretrained Mamba kernel statistics and response-blind native geometry
   reorganize across model scale;
2. a ContraMamba task-margin probe shows scale-dependent reorganization of
   native-component magnitude / functional angular alignment, including
   Delta-L mean-sign reversal.

The unresolved question is whether the functional effect is specific to the
trained ContraMamba downstream/router, or whether the same frozen Mamba native
directions show corresponding scale-dependent functional alignment under the
pretrained Mamba language-model objective itself.

This experiment changes the functional gradient source while preserving the
native Mamba measurement site and frozen geometry.

## 2. Models

Use the exact pretrained HF Mamba causal-LM checkpoints already pinned by the
frozen scale artifacts.

### Mamba-370M

- repository: `state-spaces/mamba-370m-hf`
- revision: `589179554943157be31701edd8b4558889276674`
- intervention layer: 35
- source block: 33
- target residual layer: 34
- strong-channel count: 650

### Mamba-790M

- repository: `state-spaces/mamba-790m-hf`
- revision: `9822dd4b76af2bd9099b6ce2f19efd8329189a7e`
- intervention layer: 35
- source block: 33
- target residual layer: 34
- strong-channel count: 975

### Mamba-1.4B

- repository: `state-spaces/mamba-1.4b-hf`
- revision: `6e46eae61c27280517feef46f536d16b91076f08`
- intervention layer: 35
- source block: 33
- target residual layer: 34
- strong-channel count: 829

### Mamba-2.8B

- repository: `state-spaces/mamba-2.8b-hf`
- revision: `96c48e0292b63f5346b6d30061af2551f7101e26`
- intervention layer: 47
- source block: 45
- target residual layer: 46
- strong-channel count: 1003

The model must be loaded directly as `MambaForCausalLM` or the equivalent
`AutoModelForCausalLM`.

No ContraMamba downstream checkpoint may be loaded into the vanilla-LM model.

No additional head is trained.

## 3. Frozen populations

The vanilla control reuses the exact rows of the corresponding frozen
ContraMamba readout whenever available.

### 370M

- population: `xg1_fact_4801..xg1_fact_5100`
- pair count: 300
- cells: `C0_SHAM`, `C2_NAME`
- item count: 600

### 790M

- population: `xg1_fact_7501..xg1_fact_7800`
- pair count: 300
- cells: `C0_SHAM`, `C2_NAME`
- item count: 600

### 1.4B

- population: `xg1_fact_4801..xg1_fact_5100`
- pair count: 300
- cells: `C0_SHAM`, `C2_NAME`
- item count: 600

### 2.8B

- population: `xg1_fact_6601..xg1_fact_6900`
- pair count: 300
- cells: `C0_SHAM`, `C2_NAME`
- item count: 600

No row filtering, cohort replacement, or response-guided reselection is allowed.

If any frozen row cannot satisfy the vanilla-LM token-validity contract below,
the scale execution blocks rather than dropping that row.

## 4. Measurement site

For every row, reuse the same event anchor and target definition as the frozen
ContraMamba readout:

`target_token_index = absolute_anchor_token_index + TARGET_OFFSET`

with:

`TARGET_OFFSET = 2`.

At the frozen intervention-layer Mamba mixer `in_proj`:

1. detach the native output;
2. extract the native non-gate branch at the exact target token;
3. clone that vector as the only differentiable leaf;
4. preserve the native gate branch exactly;
5. replace the original non-gate branch by the equal-valued leaf;
6. require bitwise forward-value equality before continuing.

No model parameter receives `.grad`.

## 5. Vanilla functional objective

Let:

`t = target_token_index`

and:

`y = input_ids[0, t + 1]`.

Require:

- `0 <= t < sequence_length - 1`;
- the target token position is active;
- position `t + 1` is active in the attention mask.

No row may be removed if these requirements fail; the run blocks.

Run the exact pretrained Mamba causal-LM forward and obtain the vocabulary
logits at position `t`.

Define the scalar functional objective:

`J_LM = log_softmax(lm_logits[0, t])[y]`.

This is the teacher-forced log probability assigned by vanilla pretrained
Mamba to the observed next token.

Define:

`g_LM = d J_LM / d leaf`.

Only this local leaf gradient is computed.

No ContraMamba logits, class labels, active-wrong class, downstream/router
state, or G3 edge-gradient ownership is used in this objective.

## 6. Frozen native components

The strong mask and frozen principal-plane bases remain exactly those already
frozen for each scale.

For every row, reconstruct the native selected coefficients `(a,b)` from the
frozen selected plane and native strong activation exactly as in the existing
Delta-L readout.

Matched control components reuse the same `(a,b)` coefficients.

Selected/control component norms must match to frozen numerical tolerance.

No plane is selected using the vanilla-LM response.

## 7. Contrast A: TASK_MATCHED

The scale-specific task-matched contrasts are fixed before execution:

- 370M: `P3 - P5`
- 790M: `P2 - P5`
- 1.4B: `P5 - P4`
- 2.8B: `P3 - P5`

For each row:

`L_selected_LM = dot(g_LM, selected_component)`

`L_control_LM = dot(g_LM, matched_control_component)`

`Delta_L_LM_TASK_MATCHED = L_selected_LM - L_control_LM`.

There is no G3 ownership factor in the vanilla-LM endpoint.

## 8. Contrast B: COMMON_P3_P5

As a prospectively frozen sensitivity analysis, the same row gradient also
produces:

`Delta_L_LM_COMMON_P3_P5 = L_P3_LM - L_P5_LM`

for all four scales.

This requires no additional model forward or backward pass.

The common-label contrast is secondary.

Identical plane numbers across separately reconstructed backbones are not
assumed to imply semantic plane homology.

## 9. Frozen ContraMamba comparator

For `TASK_MATCHED`, compare the vanilla-LM endpoint against the already frozen
ContraMamba readout on the same scale/population.

The frozen ContraMamba mean-sign vector for the four-scale battery is:

`370M:+, 790M:+, 1.4B:-, 2.8B:-`.

Because ContraMamba G3 ownership introduces a positive x0.5 gradient scale,
magnitude comparisons use the frozen forward-equivalent ContraMamba Delta-L
where applicable.

Sign, rank, and Pearson/Spearman correlation are invariant to this positive
ownership conversion.

## 10. Prospectively frozen descriptive outputs

For each scale and each contrast report:

- N;
- arithmetic pair aggregation over `C0_SHAM` and `C2_NAME`;
- mean;
- sample standard deviation;
- median;
- Q25 and Q75;
- min and max;
- positive / negative / zero fraction;
- C0_SHAM descriptive values;
- C2_NAME descriptive values;
- mean cosine gap;
- `corr(component_norm, cosine_gap)`;
- `NORM_GAP`;
- gradient norm descriptive values.

For TASK_MATCHED additionally report, against the frozen ContraMamba pair
endpoint on the same rows:

- Pearson correlation;
- Spearman correlation;
- pair-sign agreement fraction.

These are descriptive only.

## 11. Pre-frozen pattern diagnostics

No p-value is attached.

Define:

`TASK_MATCHED_SIGN_VECTOR_LM`

as the four mean signs in order:

`370M, 790M, 1.4B, 2.8B`.

Define:

`FULL_TASK_MATCHED_SIGN_CONCORDANCE`

as true only when this exact vector equals the frozen ContraMamba vector:

`+, +, -, -`.

Also report separately:

`POSITIVE_SCALE_SIGN_PRESERVATION`

for 370M and 790M,

and:

`NEGATIVE_SCALE_SIGN_PRESERVATION`

for 1.4B and 2.8B.

These diagnostics are fixed before execution and do not authorize outcome-based
reruns, plane reselection, cohort replacement, or alternative objectives.

## 12. Interpretation logic

If the vanilla-LM TASK_MATCHED readout preserves the frozen ContraMamba sign
structure and corresponding state-magnitude/alignment anti-coupling across
scale, that would provide evidence that the functional organization is not
specific to the trained ContraMamba downstream/router.

If the vanilla-LM readout does not preserve it, the native Mamba-side geometry
findings remain valid, but the ContraMamba Delta-L reversal must be interpreted
as task-functional coupling rather than a vanilla-Mamba functional reversal.

Intermediate or mixed outcomes must be reported as such.

No outcome licenses a post-hoc change to the objective or contrast definitions.

## 13. Execution accounting

Planned scientific work:

- four scales;
- 600 rows per scale;
- one native causal-LM forward per row;
- one local-leaf backward per row;
- 2400 forwards total;
- 2400 local backwards total;
- zero intervention-condition forwards;
- zero training steps;
- zero parameter-gradient accumulation;
- zero p-values.

The COMMON_P3_P5 sensitivity reuses each row's existing gradient and therefore
adds no model forward or backward.

## 14. Scientific boundary

This control is designed to distinguish:

`ContraMamba-downstream-specific functional coupling`

from:

`functional alignment already present under the pretrained Mamba LM objective`.

It does not by itself establish:

- universality across datasets or objectives;
- a population-level Mamba scaling law;
- semantic homology of same-numbered planes across backbones;
- a continuous model-size threshold;
- causality from parameter count;
- architectural universality beyond Mamba-1.

## Provenance

Native-Mamba / functional-coupling static bridge freeze:

`67140036d2478738b806fb51de5cf6e8351bfc10`

`VANILLA_MAMBA_LM_FUNCTIONAL_CONTROL_PLAN = FROZEN_CANDIDATE`
