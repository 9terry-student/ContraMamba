# ContraMamba Experiment 1 Static Feasibility Audit
## Mamba-370M + Mamba-1.4B Behavioral Bridge

### Status

`STATIC_FEASIBILITY_PASS_READY_FOR_IMPLEMENTATION`

This is a read-only static feasibility result under the post-synthesis research
program.

Program anchor:

`027a3ed88ab505a80b69e27deb39cb95803f27a8`

Program document:

`reports/reason_router_gen4_post_synthesis_research_program.md`

No training, model execution, CUDA execution, new response inspection, new p-value,
or behavioral inference was performed by this audit.

The next authorized scientific objective remains Experiment 1:

`370M + 1.4B behavioral bridge`

---

## 1. Question

Can the already frozen scale-local dominant causal mechanism affect the final
ContraMamba three-way decision, rather than only the internal recurrent
susceptibility endpoint `Q`?

The new behavioral bridge will test:

- Mamba-370M;
- Mamba-1.4B;

using their independently frozen local dominant and response-blind control planes.

This audit asks only whether the frozen checkpoints, downstream heads, geometry,
intervention path, labels, and fresh population machinery are sufficient to implement
that test without changing the scientific objective.

Result:

`YES`

---

## 2. Exact Mamba-370M checkpoint

Compact checkpoint:

`reports/reason_router_gen4_mamba370m_core_replication_checkpoint_compact/seed181/G3-GROUP-D-HALF/selected_downstream_checkpoint.pt`

Identity:

- bytes: `1915197`
- SHA256:
  `9d8e3db22af4636938679aac6a8a97dd45344937d434fab29eac2ddc41a52a72`
- training seed: `181`
- split seed: `8192`
- arm: `G3-GROUP-D-HALF`
- selected epoch: `19`
- training execution commit:
  `3e0e9a435068c552abf20f3a74e0c3eccca344a3`

Pinned backbone:

- repo: `state-spaces/mamba-370m-hf`
- revision:
  `589179554943157be31701edd8b4558889276674`

Downstream state:

- key count: `36`
- tensor bytes: `1879836`
- canonical SHA256:
  `eb16ae88a3fe42dd44ca102cdc122d09fe69eb8cc3d3f5ff761f285ab2af1730`

The compact reconstruction was previously verified to reproduce the full selected
checkpoint exactly.

---

## 3. Exact Mamba-1.4B checkpoint

Compact checkpoint:

`reports/reason_router_gen4_mamba14b_training_runs/g4k-mamba14b-train-g3d-seed181-dualt4-cache-ae6ab9a/selected_downstream_checkpoint.pt`

Identity:

- bytes: `2964219`
- SHA256:
  `915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a`
- training seed: `181`
- split seed: `8192`
- arm: `G3-GROUP-D-HALF`
- selected epoch: `20`
- training execution commit:
  `ae6ab9a1393b68e4959d6a1d70e9647f8e78f21d`

Pinned backbone:

- repo: `state-spaces/mamba-1.4b-hf`
- revision:
  `6e46eae61c27280517feef46f536d16b91076f08`

Downstream state:

- key count: `36`
- tensor bytes: `2928412`
- canonical SHA256:
  `b84b6af218f3d73e10a20e98f3a4118cf2e58d37e25c829a54b81c49eb9fd8ea`

The compact reconstruction was previously verified against the exact pretrained
backbone plus selected downstream state.

---

## 4. Shared downstream decision contract

Both 370M and 1.4B reconstruction paths call:

`reason_router_gen4_six_cell_tier2_inference_adapter.build_historical_model_from_backbone`

with the appropriate backbone-specific hidden size.

Both therefore reconstruct the same historical ContraMamba downstream architecture
around a different frozen Mamba backbone.

The adapter's external class order is exactly:

`REFUTE, NOT_ENTITLED, SUPPORT`

with IDs:

- `0 -> REFUTE`
- `1 -> NOT_ENTITLED`
- `2 -> SUPPORT`.

The 1.4B selected checkpoint metadata independently records the same final mapping.

The generic adapter:

`historical_forward(...)`

returns the frozen final model output and requires:

`logits.shape == (batch, 3)`.

Therefore the behavioral margin endpoint can be shared across both scales without
changing label semantics or head semantics.

Result:

`DOWNSTREAM_LOGIT_CONTRACT_COMPATIBLE`

---

## 5. Full-forward intervention compatibility

Both backbone-specific geometry reconstruction modules:

- load the exact pinned Mamba backbone;
- construct the same historical ContraMamba downstream model;
- insert the exact 36-key downstream selected state;
- strict-load the resulting full model state;
- expose the complete model rather than a backbone-only evaluator.

The frozen intervention layer mapping for both 48-layer backbones is:

- source block: `33`;
- target residual layer: `34`;
- intervention layer: `35`;
- target token offset: `+2`.

Both runtime paths expose the intervention mixer's:

`in_proj`

and the frozen strong-channel mask.

Therefore the same branch-local correction can be installed as a forward hook during a
full ContraMamba forward and allowed to propagate through the remaining backbone and
downstream task heads.

This is the exact capability required for a behavioral bridge.

Result:

`FULL_FORWARD_INTERVENTION_PATH_COMPATIBLE`

---

## 6. Frozen scale-local causal objects

### Mamba-370M

Frozen discovery:

- selected dominant candidate: `P3`;
- response-blind control: `P5`;
- selection unique: `true`.

The existing confirmation correction already implements the matched-control
construction:

`- selected_component + coefficient-matched control_component`.

### Mamba-1.4B

Frozen discovery:

- selected dominant candidate: `P5`;
- response-blind control: `P4`;
- selection unique: `true`.

The 1.4B confirmation correction implements the same matched-control construction.

No plane selection is needed for Experiment 1.

No behavioral response may reopen either selection.

---

## 7. Behavioral conditions are implementable without new geometry

For each scale, use exactly four behavioral conditions:

1. `native`
2. `dominant_neutralized`
3. `dominant_restored`
4. `dominant_control`

Semantics:

### native

No correction.

### dominant_neutralized

Remove the native selected-plane component:

`- selected_component`.

This is needed only for the secondary behavioral-necessity diagnostic.

### dominant_restored

Use the frozen exact restoration construction:

`- selected_component + selected_component`.

Its net correction is exactly zero.

This is intentionally equivalent to the native state at the intervention coordinate,
but retains the same explicit intervention semantics used by the causal confirmation
protocol.

### dominant_control

Remove the selected component and replace it with the response-blind matched-control
component using the selected plane's native `(a,b)` coefficients.

This is the primary comparator.

No new direction, plane, rank, or coefficient is learned from behavioral responses.

---

## 8. Behavioral rows and labels

The earlier frozen behavioral bridge established the structural semantics of exactly
two XG1 rows per source pair:

### `C0_SHAM`

The claim and evidence are identical.

Frozen behavioral label:

`SUPPORT -> class 2`

### `C2_NAME`

The claim is preserved and the evidence-side entity name is replaced by the
deterministically generated alternate name.

Frozen behavioral label:

`NOT_ENTITLED -> class 1`

These semantics are data-generator properties and do not depend on backbone scale.

The same cells are already used as target-plus / target-minus cells in the 370M and
1.4B causal measurement chain, and the same `A_IDENTITY` target-token construction is
available.

Result:

`BEHAVIORAL_ROW_AND_LABEL_CONTRACT_COMPATIBLE`

---

## 9. Primary endpoint

For behavioral row `r` with frozen correct class `y`, define:

`m_r = z_y - max_{c != y} z_c`.

For each source pair:

`M_condition = (m_C0_SHAM + m_C2_NAME) / 2`.

For each scale `s`:

`D_BEH,s = M_dominant_restored - M_dominant_control`.

Primary hypotheses for each scale:

`H0_s: E[D_BEH,s] <= 0`

versus:

`H1_s: E[D_BEH,s] > 0`.

The experiment contains exactly two new primary tests:

- Mamba-370M;
- Mamba-1.4B.

Family control:

- one-sided one-sample Student t-tests;
- `N = 300`;
- alpha family `0.05`;
- Holm correction across the two scale tests.

The historical earlier-scale behavioral bridge is context only and is not a third
p-value in this family.

---

## 10. XG1 global occupancy

Frozen structural XG1 cohorts occupy the following consecutive 300-pair blocks:

- `001..300`
- `301..600`
- `601..900`
- `901..1200`
- `1201..1500`
- `1501..1800`
- `1801..2100`
- `2101..2400`
- `2401..2700`
- `2701..3000`
- `3001..3300`
- `3301..3600`
- `3601..3900`
- `3901..4200`
- `4201..4500`
- `4501..4800`

Thus the exact structural inventory is contiguous through:

`xg1_fact_4800`.

The existing builder chain supports arbitrary deterministic continuation by global
one-based pair index and does not require model execution or response fields.

---

## 11. Fresh behavioral population decision

Use one **shared** fresh cohort for both scales:

`xg1_fact_4801..xg1_fact_5100`

with:

`N = 300`.

Reason for sharing the cohort:

- both scale tests are prospectively frozen before response access;
- using the same examples removes unnecessary between-cohort variation;
- scale-specific behavioral effects can be compared descriptively on matched source
  pairs;
- Holm family-wise control does not require the two scale tests to be independent;
- no response from one scale may be used to modify the other scale's protocol.

Both scale runners and both primary tests must be implemented and frozen before the
first behavioral response on `4801..5100` is inspected.

There is no scale-specific rescue cohort.

---

## 12. Forward budget

Per scale:

`300 pairs × 2 rows × 4 conditions = 2400 full-model forwards`.

Preferred deterministic two-GPU sharding:

- shard 0: pairs `4801..4950`, `1200` forwards;
- shard 1: pairs `4951..5100`, `1200` forwards.

Across both scales:

`4800 full-model forwards`.

This is scientific inference only.

No training or backward pass is required.

---

## 13. Implementation reuse

A common behavioral runner is feasible.

Shared logic:

- fresh population loader;
- C0/C2 label validation;
- tokenizer/anchor analysis;
- feature encoding;
- correct-class logit margin;
- four behavioral conditions;
- output row schema;
- shard budget / provenance checks;
- two-scale statistical analysis.

Scale adapter fields:

- HF repo/revision;
- compact checkpoint path/hash;
- geometry root;
- strong-mask width;
- selected plane;
- control plane;
- local plane tensors.

The existing 370M/1.4B geometry and confirmation modules already provide the required
checkpoint reconstruction and frozen local plane semantics.

Do not duplicate the historical 130M runner verbatim.

Implement one scale-parameterized behavioral runner to avoid semantic drift between
370M and 1.4B.

---

## 14. Static feasibility verdict

Checkpoint availability:

`PASS`

Exact checkpoint identity:

`PASS`

Downstream three-way head compatibility:

`PASS`

Class-order compatibility:

`PASS`

Full-model forward availability:

`PASS`

Layer-35 intervention hook availability:

`PASS`

Frozen selected/control geometry availability:

`PASS`

Behavioral C0/C2 label semantics:

`PASS`

Fresh deterministic XG1 continuation:

`PASS`

Fresh response-blind population available:

`PASS — 4801..5100`

Expected execution budget:

`2400 forwards per scale / 4800 total`

Training required:

`NO`

Backward pass required:

`NO`

GPU required for this static audit:

`NO`

### Final audit label

`PASS_READY_FOR_370M_14B_BEHAVIORAL_BRIDGE_IMPLEMENTATION`

---

## 15. Next action after this audit is frozen

Implement exactly:

1. structural builder for shared `xg1_fact_4801..5100`;
2. one scale-parameterized 370M/1.4B behavioral runner;
3. one two-scale analysis module implementing the frozen two-test Holm family;
4. tests for checkpoint identities, population blindness, label semantics,
   selected/control identities, forward budgets, and no rescue behavior.

Do not execute the new behavioral population until implementation and tests are frozen
at a specific commit.

No additional authority/specification document is needed unless implementation exposes
a genuine scientific ambiguity.
