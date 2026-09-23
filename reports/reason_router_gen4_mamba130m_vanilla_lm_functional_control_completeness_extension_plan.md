# ContraMamba Gen4 Mamba-130M Vanilla-LM Functional Control Completeness Extension

## Status

`PROSPECTIVE_POST_PRIMARY_SCALE_COMPLETENESS_EXTENSION`

This document freezes an additional Mamba-130M vanilla-LM functional-control
measurement after the original four-scale vanilla-LM control
(370M, 790M, 1.4B, 2.8B) has already been executed and analyzed.

The omission of 130M from the original four-scale battery is treated as a
scale-coverage completeness gap.

The existing four-scale raw artifacts and frozen analysis remain immutable.

This extension may be shown alongside the original four scales in a five-scale
descriptive table or figure, but it must not be described as part of the
original pre-outcome four-scale prospective battery.

No result from the already-observed four-scale vanilla-LM control may alter any
130M model, population, geometry, contrast, endpoint, aggregation rule, or
analysis defined below.

## 1. Scientific purpose

Measure whether the already-frozen Mamba-130M native P3-versus-P5 geometry is
functionally aligned under the pretrained vanilla-Mamba next-token LM objective,
using the same local-leaf causal-LM directional-readout definition as the
completed four-scale vanilla-LM functional control.

The purpose is scale completeness, not rescue of the original four-scale result.

## 2. Immutable prior result boundary

Already frozen and not reopened:

- original vanilla-LM control plan:
  `1fe9a198a15c9cea0e5451d918cd949bc21bf7e0`
- four-scale raw freeze:
  `f78c56418902dc208adaef6c9a188378f9a7fa45`
- four-scale static analysis freeze:
  `88a6d6c469d44071a070b494485efe56db4faa58`
- original four-scale TASK_MATCHED vanilla-LM sign vector:
  `370M:-, 790M:-, 1.4B:+, 2.8B:+`
- frozen four-scale ContraMamba comparator sign vector:
  `370M:+, 790M:+, 1.4B:-, 2.8B:-`

The 130M result cannot alter:

- `FULL_TASK_MATCHED_SIGN_CONCORDANCE=False`;
- the original four-scale sign vector;
- the original four-scale interpretation;
- any already-frozen four-scale pair correlation;
- any existing row inclusion or contrast definition.

## 3. Model identity

Use exactly:

- scale: `mamba130m`
- repository: `state-spaces/mamba-130m-hf`
- revision: `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`
- model class: `MambaForCausalLM` or equivalent direct causal-LM loader
- hidden size: `768`
- intermediate size: `1536`
- layer count: `24`
- state size: `16`
- convolution kernel: `4`

Load the pretrained causal-LM checkpoint directly.

Prohibited:

- loading the ContraMamba downstream checkpoint into the model;
- loading the ContraMamba classifier/router head into the model;
- training or fitting any additional head;
- parameter updates.

The historical state-spaces Mamba LM-head contract remains the same as in the
completed four-scale control:

`lm_head.weight = backbone.embedding.weight`

The implementation must retain the same hard-gated historical direct parameter
alias reconstruction and provenance checks used by the completed vanilla-LM
runner.

The pinned HF snapshot contains a single `model.safetensors` weight file.

Frozen pretrained weight identity:

- `model.safetensors` SHA256:
  `1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea`

Execution provisioning must authenticate the pinned snapshot before scientific
execution.

## 4. Frozen 130M population

Reuse exactly the existing frozen 130M readout population:

- family: `XG1`
- pair range: `xg1_fact_2701..xg1_fact_3000`
- pair count: `300`
- cells per pair:
  - `C0_SHAM`
  - `C2_NAME`
- item count: `600`

No row filtering, replacement, subgroup selection, or response-guided
reselection is allowed.

If any row fails the vanilla-LM token-validity contract, execution blocks rather
than deleting or replacing the row.

## 5. Frozen 130M geometry

Reuse exactly the already-frozen 130M geometry:

- selected plane: `P3`
- response-blind matched control: `P5`
- intervention layer: `17`
- anchor: `A_IDENTITY`
- target offset: `+2`
- strong-channel dimension: `395`
- frozen geometry JSON SHA256:
  `e6e9db909eb7d2c6bbdb493a4efeca8c18e4d474cf99a943be0f3f7b9dee1012`
- frozen geometry tensor SHA256:
  `de3ae6a450c2ba0a85b4f53919e3765e6e7dfb6dba3676535554c437b1647a1c`
- geometry source checkpoint SHA256:
  `afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f`

The geometry artifacts are used only as frozen native-state measurement objects.
The associated ContraMamba downstream checkpoint must not be loaded into the
vanilla causal-LM model.

No plane refit, plane reselection, coefficient fitting, or geometry
reconstruction from a different checkpoint is allowed.

## 6. Measurement site

For every frozen row:

`target_token_index = absolute_anchor_token_index + 2`

At Mamba layer 17 `mixer.in_proj`:

1. detach the native output;
2. extract the native non-gate/content branch at the target token;
3. clone that vector as the only differentiable leaf;
4. preserve the native gate branch exactly;
5. replace the original content branch by the equal-valued leaf;
6. require exact forward-value equality before continuing.

No model parameter receives `.grad`.

## 7. Vanilla LM objective

Let:

`t = target_token_index`

and:

`y = input_ids[0, t + 1]`.

Require the same token-validity contract as the completed four-scale control:

- `0 <= t < sequence_length - 1`;
- target position active;
- `t + 1` active.

Define:

`J_LM = log_softmax(lm_logits[0, t])[y]`

and:

`g_LM = d J_LM / d leaf`.

Only this local leaf gradient is computed.

No ContraMamba logits, labels, active-wrong class, downstream/router state, or
G3 gradient ownership is used in the vanilla-LM objective.

## 8. Frozen component construction

Use the frozen P3 native coordinates `(a,b)` and coefficient-matched P5 control
exactly as in the existing 130M readout geometry.

Define:

`C_P3 = a*u_P3,+ + b*u_P3,-`

`C_P5 = a*u_P5,+ + b*u_P5,-`

Selected/control component norms must match to the existing frozen numerical
tolerance.

No response from this experiment may alter the component construction.

## 9. Primary extension endpoint

For every row:

`L_P3_LM = dot(g_LM, C_P3)`

`L_P5_LM = dot(g_LM, C_P5)`

`Delta_L_LM_130M = L_P3_LM - L_P5_LM`.

For 130M:

- `TASK_MATCHED = P3 - P5`
- `COMMON_P3_P5 = P3 - P5`

Therefore TASK_MATCHED and COMMON_P3_P5 are identical by construction and do
not require duplicate forward/backward execution.

There is no G3 ownership multiplier in the vanilla-LM endpoint.

## 10. Pair aggregation

For each source pair:

`Delta_L_LM_pair = 0.5 * (Delta_L_LM_C0_SHAM + Delta_L_LM_C2_NAME)`.

The descriptive pair sample is exactly `N=300`.

No inferential p-value is added.

## 11. Frozen ContraMamba comparator

Use the already-frozen 130M downstream pair endpoint on the same
`xg1_fact_2701..3000` rows.

Historical 130M stored readout values are `Delta_L_owned` under
`G3-GROUP-D-HALF`.

For magnitude comparison use the deterministic forward-equivalent conversion:

`Delta_L_forward = 2 * Delta_L_owned`.

The already-frozen aggregate comparator mean is positive:

- `mean Delta_L_owned = +0.001120366036532127`
- `mean Delta_L_forward = +0.002240732073064253`

No existing 130M inferential test is rerun.

## 12. Prospectively frozen outputs

Report the same descriptive quantities as the completed four-scale control.

For pair-level `Delta_L_LM_130M`:

- N;
- mean;
- sample SD;
- median;
- Q25 and Q75;
- min and max;
- positive / negative / zero fractions.

Also report:

- C0_SHAM descriptives;
- C2_NAME descriptives;
- mean cosine gap;
- `corr(component_norm, cosine_gap)`;
- `NORM_GAP`;
- gradient strong-norm descriptives;
- exact `Delta_L = ||g|| * ||component|| * cosine_gap` reconstruction audit.

Against the frozen forward-equivalent 130M ContraMamba pair endpoint report:

- Pearson correlation;
- Spearman correlation;
- pair-sign agreement fraction.

All are descriptive only.

## 13. Pre-frozen extension diagnostics

Define before execution:

`M130_TASK_MATCHED_SIGN_LM`

as the sign of the 130M pair-mean vanilla-LM endpoint.

Frozen ContraMamba 130M comparator sign:

`M130_CONTRAMAMBA_SIGN = +`.

Define:

`M130_SIGN_CONCORDANCE`

as true only if:

`M130_TASK_MATCHED_SIGN_LM == +`.

After this extension is frozen, a five-scale descriptive vector may be reported:

`130M, 370M, 790M, 1.4B, 2.8B`

using:

- 130M from this extension;
- 370M/790M/1.4B/2.8B from immutable prior artifacts.

This five-scale vector is explicitly `POST_PRIMARY_DESCRIPTIVE_COMPLETENESS`
and is not a replacement for the original prospective four-scale diagnostic.

No p-value, monotonicity test, threshold estimate, or scaling-law fit is
authorized.

## 14. Execution accounting

Scientific execution:

- one scale;
- 300 pairs;
- 600 rows;
- 600 native causal-LM forwards;
- 600 local-leaf backwards;
- zero intervention-condition forwards;
- zero parameter gradients;
- zero training steps;
- zero parameter updates;
- zero p-values.

Use the two available GPUs as fixed execution shards:

- GPU0: `xg1_fact_2701..2850`
- GPU1: `xg1_fact_2851..3000`

## 15. Implementation constraint

Prefer extending the already-frozen generic vanilla-LM runner rather than
creating a scientifically different endpoint.

The 130M adapter may use existing frozen 130M infrastructure only for:

- frozen structural rows;
- tokenizer/anchor construction;
- frozen P3/P5 geometry;
- frozen strong mask / geometry identity.

It must not use that infrastructure to load the downstream checkpoint into the
vanilla causal-LM model.

All existing 370M/790M/1.4B/2.8B runner semantics and raw artifacts remain
unchanged.

## 16. No-rescue rules

After any 130M vanilla-LM numeric response is generated, prohibit:

- row filtering or replacement;
- alternative population;
- alternative layer/token/anchor;
- alternative plane/control;
- alternative LM objective;
- gradient normalization replacing the frozen endpoint;
- checkpoint substitution;
- head training;
- alternative sign diagnostic;
- reinterpretation of the original four-scale primary diagnostic.

A technical implementation defect may be corrected only if it leaves the
scientific observable unchanged and is documented before rerun.

## 17. Interpretation boundary

This extension may complete the five sampled Mamba-1 scales in the paper.

It does not make the entire five-scale control prospectively pre-outcome.

Allowed paper statement:

> The original four-scale vanilla-LM functional control was prospectively
> frozen before observation; a subsequently frozen 130M scale-completeness
> extension used the identical local causal-LM readout definition and completed
> the five sampled model scales.

Do not claim:

- a universal Mamba scaling law;
- a continuous parameter-count threshold;
- semantic homology of same-numbered planes across independently reconstructed
  backbones;
- simple rowwise inversion between ContraMamba and vanilla-LM objectives;
- that the post-primary 130M extension retroactively changes the original
  four-scale prospective design.

`MAMBA130M_VANILLA_LM_COMPLETENESS_EXTENSION = FROZEN_CANDIDATE`
