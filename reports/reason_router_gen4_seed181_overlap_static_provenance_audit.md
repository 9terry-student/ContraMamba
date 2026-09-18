# Gen4 seed181 P3 overlap static provenance audit

## Verdict

`PASS_WITH_STAGE_A_METADATA_CAVEAT`

No evidence of seed180 geometry reuse or seed180-geometry contamination was found in the seed181 replication path.

The near-maximal seed180-PP3 / seed181-P3 overlap is therefore retained as a valid descriptive cross-checkpoint result.

This audit executed zero scientific model forwards, required no GPU, and modified no experimental artifact.

## Audited result

Frozen seed181 result commit:

`978b9d4cf68bc7b823ad7745dbab75ef81843c26`

Execution commit:

`8e96fd164ac0321add7d6dff8d7020872824b8ab`

Seed181 checkpoint SHA256:

`afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f`

## 1. Stage A checkpoint provenance

The execution runner was unchanged between the execution commit and the later result-freeze commit.

Before Stage A execution, the runner authenticated:

- checkpoint existence;
- exact byte size;
- exact checkpoint SHA256;
- registry binding for seed `181`, arm `G3-GROUP-D-HALF`.

Inside `geometry_extract_family`, the exact supplied checkpoint was loaded and the returned checkpoint SHA was required to equal the frozen seed181 SHA before geometry capture proceeded.

Stage A then performed fresh baseline model captures and constructed each `alignment_delta_h` from those captured states.

Result:

`PASS`

## 2. Seed180 raw-plan reuse exclusion

Frozen seed181 Stage A tensor SHA256 values:

- XG2: `058e576c16b75d7e0944508f5a0ef41e7eb9a6dfbe1995703892cd535bcd7902`
- XG4: `90b55f36f1c70a758d5c79b16a0420770c778e240617ed8709de2080999102c6`

Historical seed180 plan SHA256 values:

- XG2: `b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c`
- XG4: `792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f`

Both seed181 tensors differ from their historical seed180 counterparts.

Result:

`SEED180_PLAN_REUSE_EXCLUDED=PASS`

## 3. Seed181 basis reconstruction

The frozen seed181 XG2/XG4 `alignment_delta_h` tensors were independently reloaded and passed through the frozen basis reconstruction algorithm.

The independently reconstructed bases agreed with the saved seed181 geometry tensor to:

- XG2 basis max absolute residual:
  `7.771561172376096e-16`
- XG4 basis max absolute residual:
  `7.771561172376096e-16`

The independently reconstructed principal planes agreed with the saved seed181 principal planes to maximum absolute residual:

`1.1379786002407855e-15`

Result:

`SAVED_GEOMETRY_RECONSTRUCTION=PASS`

Therefore the saved seed181 bases and principal planes are reproducible directly from the frozen seed181 Stage A tensors.

## 4. Historical H180 usage boundary

The code path was verified statically.

The sequence is:

1. reconstruct seed181 XG2 basis from seed181 XG2 raw plans;
2. reconstruct seed181 XG4 basis from seed181 XG4 raw plans;
3. construct the five seed181 principal planes via `projector_modes`;
4. only afterward call `match_homolog`;
5. `match_homolog` loads frozen historical seed180 PP3 solely for projector-overlap matching.

`projector_modes` does not load or reference historical seed180 PP3.

Result:

- `H180_ABSENT_FROM_PRINCIPAL_GEOMETRY_CONSTRUCTION=PASS`
- `H180_ENTERED_ONLY_AT_HOMOLOG_MATCHING=PASS`

## 5. Homolog recomputation

Using the independently reconstructed seed181 planes, homolog matching again selected:

- homolog: `P3`
- control: `P5`

Recomputed overlap vector:

`[3.0908557781495854e-31, 8.594229158748837e-31, 1.9999999999999971, 3.475154386341597e-31, 1.1060499061576294e-31]`

The direct seed180-PP3 / seed181-P3 projector comparison gave:

`||P180 - P181||_F = 1.443265413401759e-16`

This is the preferred numerical diagnostic near exact subspace equality.

The overlap statistic is saturated near its rank-2 maximum of `2`; subtracting it from `2`, or applying `acos` to singular values extremely close to `1`, is numerically ill-conditioned and should not be used as a high-precision distance estimate.

Scientific description:

The seed180 PP3 and seed181 P3 rank-2 projectors agree to numerical precision under the frozen construction.

This is descriptive geometric evidence, not an additional causal test.

## 6. Stage D intervention provenance

The seed181 replication constructs intervention aliases directly from the seed181 geometry object:

- historical semantic `pp3_plus/minus` aliases map to selected seed181 `P3_plus/minus`;
- historical semantic `pp5_plus/minus` aliases map to selected seed181 `P5_plus/minus`.

Those aliases are passed into the restoration implementation, where the condition correction explicitly uses the supplied vectors.

All 300 frozen Stage D rows record:

- the exact seed181 checkpoint SHA;
- homolog plane `3`;
- control plane `5`.

Result:

- `STAGE_D_SEED181_CHECKPOINT_BINDING=PASS`
- `STAGE_D_SELECTED_SEED181_P3_P5_USAGE=PASS`

## Metadata caveat

Stage A JSONL rows record:

- `replication_seed=181`
- `replication_arm=G3-GROUP-D-HALF`

but do not duplicate the checkpoint SHA in every individual row.

The raw Stage A `.pt` tensors are tensor-only payloads and likewise do not contain a checkpoint-SHA metadata field.

Their checkpoint identity is instead bound through the execution-level checkpoint authentication, exact execution commit, deterministic generation path, frozen artifact checksums, and checkpoint metadata in the replication summary and principal-geometry artifact.

Therefore:

`STAGE_A_CHECKPOINT_SHA_EMBEDDED_IN_EVERY_ROW=False`

This is a self-description limitation of the frozen Stage A artifact schema, not evidence of checkpoint or geometry contamination. The frozen raw artifacts must not be rewritten merely to add metadata.

## Final audit conclusion

The static audit excludes reuse of the historical seed180 Stage A plans, independently reconstructs the saved seed181 bases and principal planes from the frozen seed181 Stage A tensors, verifies that historical seed180 PP3 enters only after seed181 principal geometry construction, and verifies that Stage D uses the selected seed181 P3/P5 vectors under the seed181 checkpoint.

Final status:

`STATIC_PROVENANCE_AUDIT=PASS_WITH_STAGE_A_METADATA_CAVEAT`

Accordingly, the near-identical seed180 PP3 and seed181 P3 projectors may be reported as a striking descriptive cross-checkpoint geometric invariance result, while preserving the stated Stage A metadata caveat.
