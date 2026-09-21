# ContraMamba Gen4 Study-B Pair-Level Explanatory Merge Plan

## 0. Status and scope

This file is a prospective static-analysis plan, not a new execution authority.

It does not reopen any frozen selection, intervention, cohort, plane, layer, token,
epsilon, model, checkpoint, or scientific endpoint. It introduces no training,
no GPU execution, no forward/backward pass, and no new primary inferential test.

The analysis is already permitted by the frozen Study-B readout-alignment design:
after raw gradient evidence is frozen, the frozen Experiment-1 behavioral
`D_BEH` values may be merged with pair-level `Delta_L` for descriptive-only
explanatory analysis.

Current frozen analysis HEAD at plan creation:

`5fff4b9ee6856ec396f90a905246039b164a5b39`

## 1. Scientific question

The aggregate Study-B result established that the native local first-order
readout quantity changes sign across model scale:

- Mamba-370M: mean `Delta_L > 0`
- Mamba-1.4B: mean `Delta_L < 0`

and this sign pattern matches the historical Experiment-1 behavioral bridge.

The next question is narrower:

> Does pair-level local first-order readout variation explain pair-level
> behavioral variation within each scale?

This analysis is explanatory and descriptive. It is not an independent
confirmation of the already-known Experiment-1 behavioral result.

## 2. Frozen population

Use exactly the shared Experiment-1 / Study-B population:

- source pairs: `xg1_fact_4801..xg1_fact_5100`
- `N = 300`
- cells: `C0_SHAM`, `C2_NAME`
- scales:
  - `mamba370m`
  - `mamba14b`

No row filtering, cohort replacement, response-guided selection, or rescue is
allowed.

## 3. Frozen Study-B input

Use:

`reports/reason_router_gen4_mamba370m14b_readout_alignment_analysis_v1/readout_alignment_pair_values.jsonl`

Expected pair-level fields:

- `source_pair_id`
- `Delta_L_370M`
- `Delta_L_1.4B`
- `R`

The pair-level `Delta_L` values are already the frozen two-cell averages from
Study B.

Before analysis, validate the enclosing analysis artifact with:

`reports/reason_router_gen4_mamba370m14b_readout_alignment_analysis_v1/SHA256SUMS.txt`

No Study-B endpoint is recomputed or reselected.

## 4. Frozen Experiment-1 behavioral input

Use the frozen historical run:

`reports/reason_router_gen4_mamba370m14b_behavioral_bridge_runs/g4k-mamba370m14b-behavioral-bridge-xg1-4801-5100-2gpu-2b41f28-retry1`

For each scale, read exactly:

- `mamba370m/shard0/behavioral_rows.jsonl`
- `mamba370m/shard1/behavioral_rows.jsonl`
- `mamba14b/shard0/behavioral_rows.jsonl`
- `mamba14b/shard1/behavioral_rows.jsonl`

and validate each shard using its existing `SHA256SUMS.txt`.

The existing behavioral definition is preserved exactly.

For each source pair `q`, cell `c`, and scale `s`, read the frozen
`correct_class_logit_margin` for:

- `dominant_restored`
- `dominant_control`

First average each condition across the two frozen cells:

`M_restored(s,q) = mean_c M(s,q,c,dominant_restored)`

`M_control(s,q) = mean_c M(s,q,c,dominant_control)`

Then reconstruct:

`D_BEH(s,q) = M_restored(s,q) - M_control(s,q)`

No historical behavioral p-value is recomputed for this merge.

## 5. Exact merge

Join only on exact `source_pair_id`.

Required pair order:

`xg1_fact_4801, ..., xg1_fact_5100`

For every pair, produce:

- `source_pair_id`
- `Delta_L_370M`
- `D_BEH_370M`
- `residual_370M = D_BEH_370M - Delta_L_370M`
- `sign_agree_370M`
- `Delta_L_1.4B`
- `D_BEH_1.4B`
- `residual_1.4B = D_BEH_1.4B - Delta_L_1.4B`
- `sign_agree_1.4B`

Exact zero is treated as its own sign for sign-agreement accounting. Do not
drop zero rows.

## 6. Descriptive endpoints

For each scale separately, report exactly:

### 6.1 Pair-level association

- Pearson correlation:
  - `corr(Delta_L_s, D_BEH_s)`
- Spearman rank correlation:
  - descriptive only
  - no p-value retained or reported

### 6.2 First-order residual

Define:

`e_s,q = D_BEH_s,q - Delta_L_s,q`

Report:

- mean
- population SD
- min
- q25
- median
- q75
- max

### 6.3 Sign agreement

Report:

- fraction `sign(Delta_L_s,q) == sign(D_BEH_s,q)`
- count of:
  - both positive
  - both negative
  - `Delta_L > 0`, `D_BEH < 0`
  - `Delta_L < 0`, `D_BEH > 0`
  - any exact-zero category, if present

### 6.4 Optional cross-scale descriptive context

The existing Study-B and Experiment-1 aggregate means may be copied into the
output only as provenance/context. Do not add a new cross-scale hypothesis test.

## 7. Statistical boundary

This analysis adds:

- primary p-values: `0`
- secondary p-values: `0`

Do not retain or report the p-value returned by a Spearman implementation.

No permutation test, bootstrap significance test, regression significance test,
multiple-comparison procedure, subgroup analysis, or post-hoc threshold is
allowed.

## 8. Output artifact

Create exactly one static-analysis directory:

`reports/reason_router_gen4_mamba370m14b_readout_behavior_pair_merge_v1/`

with:

1. `pair_level_merge.jsonl`
2. `descriptive_analysis.json`
3. `artifact_manifest.json`
4. `SHA256SUMS.txt`

The manifest must record at minimum:

- result:
  `PASS_GEN4_READOUT_BEHAVIOR_PAIR_LEVEL_DESCRIPTIVE_MERGE`
- execution type:
  `cpu_static_analysis`
- primary p-value count: `0`
- secondary p-value count: `0`
- inference performed: `false`
- training executed: `false`
- forward executed: `false`
- backward executed: `false`
- row filtering performed: `false`
- rescue performed: `false`
- source pair count: `300`
- scales:
  - `mamba370m`
  - `mamba14b`

## 9. Validation requirements

The analyzer must block unless all of the following hold:

- Study-B analysis checksums validate.
- All four historical behavioral shard checksums validate.
- Study-B pair count is exactly 300.
- Behavioral pair count is exactly 300 per scale.
- Pair IDs are exactly `xg1_fact_4801..xg1_fact_5100`.
- No duplicate pair/cell/condition tuple exists.
- Both frozen cells exist for every pair and scale.
- Both required behavioral conditions exist for every pair/cell/scale.
- All `Delta_L` and `D_BEH` values are finite.
- Exact pair ordering is preserved.
- No input file is modified.

## 10. Interpretation boundary

If pair-level correlation is substantial, the supported interpretation is:

> The native local first-order readout quantity captures meaningful pair-level
> variation in the frozen behavioral intervention effect in addition to
> reproducing the aggregate scale-dependent sign pattern.

If pair-level correlation is weak, the supported interpretation is:

> The local first-order readout quantity captures the aggregate scale-dependent
> sign pattern, while substantial pair-specific behavioral variation remains
> unexplained by the first-order approximation.

Neither result changes the already-frozen Study-B primary conclusion.

Do not claim:

- exact behavioral prediction;
- complete causal mediation;
- item-level sufficiency;
- linearity of the full intervention response;
- independent confirmation of Experiment-1;
- a new significant association;
- that residual variation is noise.

## 11. Stop rule

Run this analysis once on the frozen inputs.

Do not change:

- population;
- cells;
- scale set;
- behavioral endpoint;
- `Delta_L` definition;
- correlation metric set;
- residual definition;
- sign-agreement definition

after viewing the result.

After the artifact is frozen, proceed to the next predeclared program item.
