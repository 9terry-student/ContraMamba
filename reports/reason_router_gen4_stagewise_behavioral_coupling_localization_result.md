# ContraMamba Cross-Scale Stagewise Behavioral-Coupling Localization Result
## Mamba-370M vs Mamba-1.4B, XG1 4801..5100

### Status

`STAGEWISE_BEHAVIORAL_COUPLING_LOCALIZATION_VALIDATED`

Execution commit:

`4e88537383861d60fbd7cc0400afd17fa1e302b6`

Validated recovery run:

`g4k-stagewise-coupling-localization-370m14b-xg1-4801-5100-2gpu-4e88537-analysis-recovery1`

Recovery command SHA256:

`652c5f87affbe9c56e2b9e64d0355d7c58c97737c5a8b5f74f299662c943d711`

Imported handoff ZIP SHA256:

`e9888816658e135c19035eba2f804099cbd81f466bd2ce02b54ac5a0d04ebf36`

The failed original run is not treated as a successful collectible run.

Its raw stagewise outputs were validated and copied byte-identically into the
successful recovery run. The recovery added:

- model forwards: `0`;
- downstream replays: `0`;
- p-values: `0`;
- model selection: `0`;
- rescue attempts: `0`.

The completed scientific observation comprises:

- original raw scientific full-model forwards: `3600`;
- original downstream-only replays: `7200`;
- scales: Mamba-370M and Mamba-1.4B;
- shared population: `xg1_fact_4801..5100`;
- cells: `C0_SHAM`, `C2_NAME`;
- conditions: native, dominant-neutralized, dominant-control.

---

## 1. Final decomposition reproduced exactly

For each stage:

`A_sel = M_native - M_neutralized`

`B_ctrl = M_control - M_neutralized`

`D = M_native - M_control = A_sel - B_ctrl`

### Mamba-370M

At `post_final_norm`:

- `A_sel = +0.00024824162324269612`
- `B_ctrl = -0.00087244407584269844`
- `D = +0.0011206856990853946`

Thus the selected component is favorable and the matched control is unfavorable at
the final behavioral readout.

### Mamba-1.4B

At `post_final_norm`:

- `A_sel = -0.00032495816548665365`
- `B_ctrl = +0.0016879271467526754`
- `D = -0.0020128853122393289`

Thus the selected component is unfavorable and the matched control is favorable at
the final behavioral readout.

The previously observed downstream ordering inversion is reproduced exactly.

---

## 2. Stagewise localization

### Mamba-370M

First persistent stages:

- `C2_NAME D > 0`: `post_block_35`
- overall `D > 0`: `post_block_36`
- `B_ctrl < 0`: `post_block_42`
- `A_sel > 0`: `post_block_47`

Interpretation:

The entitlement-sensitive `C2_NAME` favorable contrast is already readable
immediately after the intervention block. The aggregate pair-average favorable
ordering becomes persistent one block later. The control's unfavorable contribution
becomes persistent in the late backbone, and the selected component's own favorable
contribution becomes persistent only at the final recurrent block.

### Mamba-1.4B

First persistent stages:

- `C2_NAME D < 0`: `post_block_35`
- overall `D < 0`: `post_block_47`
- `A_sel < 0`: `post_block_47`
- `B_ctrl > 0`: `post_final_norm`

Interpretation:

The entitlement-sensitive `C2_NAME` unfavorable contrast is already readable
immediately after the intervention block, but the aggregate behavioral reversal is
not persistent until the final recurrent block. The control component does not become
persistently favorable until the final normalization/readout stage.

---

## 3. Cross-scale opposition

First persistent cross-scale `C2_NAME` opposition:

`post_block_35`

That is, immediately after the intervention block, the downstream lens already reads:

- 370M: `C2_NAME D > 0`
- 1.4B: `C2_NAME D < 0`

and that opposition remains through the final stage.

First persistent aggregate pair-average opposition:

`post_block_47`

At `post_block_47`:

- 370M `D = +0.0015577687323093414`
- 1.4B `D = -0.00005929705997308095`

At `post_final_norm`:

- 370M `D = +0.0011206856990853946`
- 1.4B `D = -0.002012885312239329`

Therefore the major aggregate scale divergence is consolidated at the last recurrent
block and then strongly amplified in the final normalized downstream readout.

---

## 4. What this localizes

The previous static decomposition suggested:

`core-stable internal causality + scale-dependent downstream coupling`

The stagewise result sharpens that claim.

The evidence is most consistent with two distinct phenomena:

1. **Early entitlement-sensitive routing divergence.**
   The 370M/1.4B `C2_NAME` sign opposition is already present at
   `post_block_35`, immediately after the intervention block.

2. **Late aggregate decision-orientation consolidation.**
   The full pair-average `D` opposition becomes persistent only at
   `post_block_47`; the 1.4B control component becomes persistently favorable only
   after final normalization.

A concise descriptive interpretation is therefore:

`EARLY_C2_ROUTING_DIVERGENCE + LATE_AGGREGATE_READOUT_REORGANIZATION`

This is more precise than attributing the failure solely to the downstream head or
solely to the remaining recurrent backbone.

---

## 5. Interpretation boundaries

This experiment does **not** establish a unique causal mediator for the sign reversal.

The stagewise downstream lens asks how the fixed final downstream path reads
intermediate residual representations. It does not imply that the model literally
executes a complete decision at every intermediate block.

Therefore:

- `post_block_35` localization means the C2-specific scale divergence is already
  encoded in a form readable by the frozen downstream path at that stage;
- `post_block_47` localization means the aggregate behavioral opposition becomes
  stably readable by that stage;
- `post_final_norm` control positivity implicates final normalization/readout in the
  final control ordering, but does not prove that final normalization alone causes it.

No new plane, token, coefficient, layer, subset, or control was selected.

No significance test was added.

---

## 6. Recovery validity

The original execution completed both scale-local raw runners before the descriptive
analyzer failed.

The analyzer defect was purely structural:

- row JSON was written with canonical `sort_keys=True`;
- the analyzer incorrectly required dictionary iteration order of `stage_lens` to
  equal semantic `STAGE_ORDER`;
- the valid condition is instead equality of the stage-key set together with the
  separately stored explicit `stage_order`.

Recovery validated:

- raw scientific forward count: `3600`;
- raw downstream replay count: `7200`;
- raw p-value count: `0`;
- 12 raw files copied byte-identically;
- analyzer-only validation patch;
- scientific protocol unchanged;
- model code unchanged;
- recovery model forwards added: `0`;
- recovery downstream replays added: `0`;
- recovery p-values added: `0`.

The analyzer implementation in the repository should be corrected to use
order-independent stage-key validation so future analyses do not require the recovery
patch.

---

## 7. Consequence for the research program

The localization question is now closed sufficiently for the current program.

No additional 1.4B rescue run is justified.

The next experiment remains:

`AVeriTeC GOLD-EVIDENCE NATURAL-LANGUAGE EXTERNAL CAUSAL TRANSFER`

with the already frozen primary family:

- Mamba-130M;
- Mamba-370M;
- shared 462-item response-blind AVeriTeC cohort;
- Mamba-1.4B excluded from the first external-transfer family.

The mechanistic localization result should be carried forward as context:

> Internal causal recurrence across scale does not imply invariant downstream
> entitlement routing. The C2-sensitive divergence is readable immediately after the
> intervention block, while the aggregate behavioral inversion is consolidated late
> in the recurrent stack and final readout.
