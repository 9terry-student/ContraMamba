# Gen5 Correction Residual Propagation Localization Validated Evidence Report Candidate

## Status

VALIDATED_CORRECTION_RESIDUAL_PROPAGATION_LOCALIZATION_EVIDENCE_CANDIDATE

## Evidence identity

Localization authority commit:

`69473ac76629ef1e7c78b45c8c802a89e1e9695b`

Source task-reachable quotient evidence freeze:

`0e6191fd54e23388abcce1abd9e01a453d2dc73c`

Source quotient execution:

`c731270221c4e0e131fb68173f40bf0ad8a2bfdd`

Source functional-equivalence evidence freeze:

`5f079a66f7b0eb0caea30a8d5bc9a0fe757cc449`

Source functional-fingerprint execution:

`17a783c61e310d6298d4105233ea3f1b71ec8add`

Run:

`gen5-correction-residual-localization-69473ac-r1`

Pinned command SHA256:

`c6eb03582d45e520a2ea020c3e48667144b7a34797317fd3263d280b9f3704c9`

Imported ZIP SHA256:

`b8acbc3b07f1831b2706470e58b43e34251d41a6ef81703bcf54e48799d84c86`

Run log SHA256:

`18fdf0760156b6c7e2692c6957009fe83705e5d970714ff7e708ec919abfd3b6`

Run meta SHA256:

`3d692e0649d222484b987a02ca1c5e79a1ecd4ebaf1b412f0c47b1e4a0bd5dfd`

Collector status:

`PASS`

Import status:

`PASS`

Validated imported files:

`3`

## Execution boundary

Frozen Phase3A P0 dev contract:

- arm: `G5-C0`
- pressure: `P0`
- dev rows: `840`
- split seed: `16384`
- valid tokens: `60094`

Full-model frozen-dev pass count:

`1`

Offline correction replay cells:

`9`

No training, backward pass, optimizer construction, checkpoint mutation, or
confirmatory 9601-9900 access occurred.

## Replay authentication

The offline localization replay was checked against the frozen
`_streaming_correction_impl`.

Maximum absolute error:

`0`

Therefore the stagewise correction replay exactly matched the frozen backend for
the authenticated check case.

The replayed same-training-RNG / different-A raw-write residual was:

`0.459225933132`

The preceding quotient audit reported:

`0.459227092986`

The difference is within the predeclared replay cross-check tolerance and
reproduces the source endpoint.

## Same training RNG, different A-init residual chain

Mean normalized residual by stage:

- raw write: `0.459225933132`
- recurrent correction state: `0.235282854833`
- C readout before gate: `0.201189474462`
- gated correction scan: `0.139629099045`
- layer-22 out-projection contribution: `0.155009066845`
- final delta logits: `0.0119526876812`
- final centered delta logits: `0.0128792116035`

Mean cosine:

- raw write: `0.904811548823`
- layer-22 out-projection contribution: `0.990243702092`
- final centered delta logits: `0.999917998280`

## Same training RNG, different A-init step survival

Mean step survival ratios:

- raw write -> recurrent state: `0.512510823956`
- recurrent state -> C readout: `0.850645809015`
- C readout -> gated scan: `0.681808472346`
- gated scan -> layer-22 out projection: `1.10812144879`
- layer-22 out projection -> final delta logits: `0.0735487770792`
- final delta logits -> final centered delta logits: `1.07681714201`

The descriptively smallest step-survival ratio was:

`layer22_out_proj__to__final_delta_logits`

with mean survival:

`0.0735487770792`

This is a descriptive localization result, not a statistical significance claim.

## Same A-init, different training RNG controls

Mean normalized residuals:

- raw write: `0.013636773789`
- layer-22 out-projection contribution: `0.00769027717265`
- final centered delta logits: `0.000265643132588`

Thus the much larger residual chain is specific to changing A initialization
under the tested factorial comparison, while fixed-A variation remains small.

## Primary validated result

Different A initialization produces a substantial task-reachable correction
difference at the raw layer-22 write.

The frozen Mamba correction path suppresses this difference substantially but
does not eliminate it:

`0.459225933132 -> 0.155009066845`

by the layer-22 correction contribution.

The largest remaining reduction occurs after the layer-22 correction
contribution and before the final task logits:

`0.155009066845 -> 0.0119526876812`

Therefore the near-identical final task function cannot be attributed solely to
collapse inside the local layer-22 correction recurrence.

## Important interpretation boundary

The stagewise residual chain measures geometric discrepancy magnitude.

It does NOT establish that the residual disappearing downstream is itself
decision-relevant information.

A large hidden residual may lie mostly in directions to which the downstream
task map has low or zero local sensitivity.

Therefore the observed post-layer-22 reduction must not yet be interpreted as
the network actively removing semantically important information.

## Supported bounded conclusions

`GEN5_DIFFERENT_A_INIT_RESIDUAL_IS_SUBSTANTIALLY_SUPPRESSED_BUT_NOT_ELIMINATED_BY_THE_LOCAL_LAYER22_CORRECTION_PATH`

`GEN5_THE_LARGEST_OBSERVED_RESIDUAL_REDUCTION_OCCURS_BETWEEN_LAYER22_CORRECTION_OUTPUT_AND_FINAL_TASK_LOGITS_UNDER_THE_FROZEN_DEV_CONTRACT`

`GEN5_LOCAL_LAYER22_RECURRENCE_IS_A_REAL_BUT_INCOMPLETE_FILTER_OF_A_INIT_DEPENDENT_CORRECTION_DIFFERENCES`

`GEN5_STAGEWISE_GEOMETRIC_RESIDUAL_DECAY_ALONE_DOES_NOT_ESTABLISH_TASK_RELEVANCE_OF_THE_RESIDUAL`

## What is not established

The present evidence does not establish:

- that the remaining layer-22 residual is a downstream null direction;
- that it is merely low-gain rather than exactly null;
- that it contains a small task-sensitive component;
- that A-init-specific residuals are more null-aligned than matched control
  directions;
- a decomposition into task-visible and task-null latent components;
- causal controllability of a task-visible latent component;
- any precursor representation before layer 22;
- behavior outside the frozen Phase3A P0 dev contract;
- confirmatory-seed behavior;
- a universal Mamba mechanism.

## Next scientific action

The next primary experiment is NOT further blockwise norm-decay localization.

Instead perform a downstream task-sensitivity / null-alignment audit at the
layer-22 correction-output boundary.

For same-training-RNG / different-A pairs, define the actual layer-22 residual:

`d_A = h_i - h_j`

and compare its downstream centered-logit sensitivity against prospectively
defined controls.

Primary object:

`S(d) = ||J d|| / ||d||`

where `J` is the local Jacobian of the frozen downstream centered-logit map at a
predeclared reference point.

Required controls should include:

1. same-A / different-training-RNG residual directions;
2. norm-matched task-state-supported control directions;
3. norm-matched orientation controls that preserve relevant support/norm
   structure without using final-logit outcomes for selection.

The primary question is whether actual A-init residuals are preferentially
aligned with downstream-null or low-gain directions rather than merely
benefiting from the large generic kernel expected when a high-dimensional
latent state maps to two centered-logit degrees of freedom.

Only if such preferential null/low-gain alignment is established should the
research proceed to:

- causal latent intervention on the task-visible component;
- upstream localization of where that task-visible component first forms;
- precursor analysis for earlier layer/token/time-step signatures.

No new training, hyperparameter sweep, rank change, architecture change, or
confirmatory population access is warranted before the null-alignment audit.
