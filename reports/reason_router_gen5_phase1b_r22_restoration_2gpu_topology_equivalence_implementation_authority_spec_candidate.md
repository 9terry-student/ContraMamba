# ContraMamba Gen5 Phase 1B — R22 Restoration 2×T4 Topology Equivalence Gate Implementation Authority

## 0. Status

PHASE =
`GEN5_PHASE1B_R22_RESTORATION_2GPU_TOPOLOGY_EQUIVALENCE_GATE_IMPLEMENTATION`

STATUS =
`FROZEN_ON_COMMIT`

IMPLEMENTATION_ALLOWED_AFTER_FREEZE =
`YES_BOUNDED`

SCIENTIFIC_EXECUTION_ALLOWED =
`NO`

KAGGLE_ALLOWED =
`NO`

MODEL_FORWARD_ALLOWED =
`NO`

TRAINING_ALLOWED =
`NO`

BACKWARD_ALLOWED =
`NO`

README_UPDATE_REQUIRED =
`NO`

This authority permits only implementation and zero-forward verification of a
small non-confirmatory topology/equivalence gate for the already-frozen
two-T4 restoration runner.

It does not authorize any model execution.

---

## 1. Parent implementation

RESTORATION_IMPLEMENTATION_FREEZE_COMMIT =
`d8d87aa8891ae1f2bef16ed3f1b174e58f87b978`

FROZEN_RESTORATION_FILES =

- `scripts/reason_router_gen5_phase1b_r22_restoration_confirmation.py`
  - git blob `786f186ab89ffc44bb5373b70df7a1041165b7a5`
- `scripts/reason_router_gen5_phase1b_r22_restoration_fast_cuda_2gpu.py`
  - git blob `ed73904fdf9a555e11d30e3b9068ac12941225cd`
- `scripts/verify_reason_router_gen5_phase1b_r22_restoration_confirmation.py`
  - git blob `e50888814d0b489ffb37f0e97c89a102e4709f41`
- `tests/test_reason_router_gen5_phase1b_r22_restoration_confirmation.py`
  - git blob `55e38b2c31f76e85ab99e7aa2c27900a336f541d`

These files are frozen and must not be modified by this gate implementation.

---

## 2. Scientific boundary

The gate is backend/topology qualification only.

It must not:
- read the restoration-confirmation cohort;
- compute a confirmatory p-value;
- emit a scientific restoration conclusion;
- alter R22, C22, PP3, Q22, checkpoint, tokenizer, or restoration semantics;
- tune tolerances after observing gate outputs.

RESTORATION_CONFIRMATION_POPULATION_ACCESS =
`FORBIDDEN`

SCIENTIFIC_P_VALUE_COUNT =
`0`

SCIENTIFIC_CONCLUSION =
`NONE`

---

## 3. Gate population

Only the already-used non-confirmatory Phase 1B construction cohort may be read.

DATA_ROOT =
`data/reason_router_gen5_phase1b_xg1_construction_v1`

CONSTRUCTION_PAIR_RANGE =
`xg1_fact_7801..xg1_fact_8100`

GATE_PAIRS =
`xg1_fact_7801..xg1_fact_7804`

GATE_PAIR_COUNT =
`4`

The four gate pairs are frozen before gate outcomes are observed.

No alternative pair selection or replacement is permitted.

Construction static identities:

CHECKSUMS_SHA256 =
`c84902de093eab2ae27f157ed814041d2c26a4a5c272cf0ad7f807ccdbb40855`

STRUCTURED_SOURCE_FACTS_SHA256 =
`3f8eac771794e1d022bee9f669f315c3ac05feeaa9c2195095b8a892b4269ea1`

SIX_CELL_ROWS_SHA256 =
`5a82508e54cd6097aeee5afe10d7c416357302701d558cfdc8740382168feca4`

STRUCTURAL_MANIFEST_SHA256 =
`d68d812ca43a6c651d284f0aa2889f87adb7e030c599af37cb49deaa9d6d8105`

TOKENIZER_ANCHOR_MANIFEST_SHA256 =
`1dbdd3e072245072f197d87443cc8cf81cbbcd2e61437c99b26e0d82815e33b4`

TOKENIZER_ELIGIBILITY_SUMMARY_SHA256 =
`7e304fb1d6f2ecb9ed6fb87911623c47d0b6ae9c2eb9bf9b8f45f70ae1fcb06a`

---

## 4. Frozen model/runtime identities

REPRESENTATIVE_CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

NATIVE_BACKBONE_SIGNATURE_SHA256 =
`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

CANONICAL_MODEL_TOKENIZER_REVISION =
`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

QUALIFIED_BACKEND =
`FROZEN_MAMBA_SSM_KERNEL_CAPTURE_PLUS_LAYER22_STATE_REPLAY`

QUALIFIED_BACKEND_SOURCE_BLOB =
`421d30f00cf71690ed41c983ccf0540808e1de1c`

KERNELS_VERSION =
`0.10.2`

CUDA_RUNTIME =
`12.8`

DEVICE =
`Tesla T4`

COMPUTE_CAPABILITY =
`7.5`

MAMBA_BINARY_SHA256 =
`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

CAUSAL_CONV1D_BINARY_SHA256 =
`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

---

## 5. Reference topology

The reference arm must execute all four gate pairs sequentially in one process
with only physical GPU 0 visible as logical `cuda:0`.

REFERENCE_PHYSICAL_GPU =
`0`

REFERENCE_VISIBLE_GPU_COUNT =
`1`

REFERENCE_PAIRS =
`xg1_fact_7801..xg1_fact_7804`

REFERENCE_PAIR_COUNT =
`4`

The reference arm must reuse the exact frozen restoration `run_pair` semantics.

No scientific decision is computed.

---

## 6. Candidate two-GPU topology

The candidate arm uses exactly two independent processes.

Candidate worker 0:

PHYSICAL_GPU =
`0`

PAIRS =
`xg1_fact_7801..xg1_fact_7802`

PAIR_COUNT =
`2`

Candidate worker 1:

PHYSICAL_GPU =
`1`

PAIRS =
`xg1_fact_7803..xg1_fact_7804`

PAIR_COUNT =
`2`

Each worker must see exactly one logical `cuda:0` through
`CUDA_VISIBLE_DEVICES`.

No DDP, gradient synchronization, model sharing, or scientific cross-device
reduction is allowed.

Candidate outputs are merged into canonical order:

`7801, 7802, 7803, 7804`

---

## 7. Exact gate forward budget

Frozen restoration semantics use:

- donor: 40 forwards per pair;
- B: 40 forwards per pair;
- RR: 40 forwards per pair;
- RC: 40 forwards per pair.

Therefore:

FORWARDS_PER_PAIR =
`160`

REFERENCE_FORWARD_BUDGET =
`640`

CANDIDATE_GPU0_FORWARD_BUDGET =
`320`

CANDIDATE_GPU1_FORWARD_BUDGET =
`320`

CANDIDATE_TOTAL_FORWARD_BUDGET =
`640`

TOTAL_GATE_FORWARD_BUDGET =
`1280`

CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET =
`0`

Any mismatch blocks the gate.

---

## 8. Frozen equivalence comparison

Reference and candidate must use the same:
- checkpoint;
- tokenizer;
- PP3 planes;
- R22/C22;
- directions;
- epsilon;
- anchor coordinates;
- donor/B/RR/RC semantics;
- qualified Q22 backend.

Discrete identities must match exactly:
- pair ID;
- pair index;
- direction order;
- branch role;
- orientation;
- target token;
- condition labels;
- upstream condition labels;
- row-dropping flag;
- forward counts.

For floating scientific scalars and manipulation diagnostics that are present in
the frozen public restoration item surface:

FLOAT_ATOL =
`1e-9`

FLOAT_RTOL =
`1e-7`

For each compared scalar:

`abs(reference - candidate) <= FLOAT_ATOL + FLOAT_RTOL * max(abs(reference), abs(candidate), 1.0)`

The gate implementation must report the maximum absolute difference and the
maximum normalized bound usage ratio across all compared floating leaves.

Tensor SHA256 audit fields are not required to match across physical GPUs;
they remain per-run provenance diagnostics and may not replace scalar numerical
equivalence.

No tolerance may be changed after observing gate output.

---

## 9. Gate PASS rule

The future gate may report:

`PASS_GEN5_PHASE1B_RESTORATION_2GPU_TOPOLOGY_EQUIVALENCE`

only if all are true:

1. exact repository and frozen-source identities pass;
2. restoration-confirmation cohort is not read;
3. exactly two physical Tesla T4 devices are present;
4. each worker sees exactly one logical `cuda:0`;
5. reference executes exactly 640 forwards;
6. candidate GPU0 executes exactly 320 forwards;
7. candidate GPU1 executes exactly 320 forwards;
8. total gate budget is exactly 1280 forwards;
9. CPU scientific forwards are zero;
10. parameter signatures are unchanged within every process;
11. candidate merge order is exactly 7801..7804;
12. all exact discrete comparisons pass;
13. all compared floating leaves satisfy the frozen tolerance;
14. confirmatory p-value count is zero;
15. scientific conclusion is null.

Otherwise the gate fails closed.

A failed gate does not authorize full restoration execution.

---

## 10. Allowed gate artifacts

A future gate run may persist only:

- `topology_equivalence_items.jsonl`
- `topology_equivalence_summary.json`
- `artifact_manifest.json`
- `SHA256SUMS.txt`

The artifacts may contain only:
- pair identities;
- comparison metrics;
- exact discrete comparison results;
- float-difference summaries;
- per-arm/per-worker forward accounting;
- runtime/provenance identities.

Raw WRITE22 or POST_STATE22 vectors must not be persisted.

No confirmatory p-value may be persisted.

---

## 11. Allowed implementation files

Exactly these new files may be created:

`scripts/reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence.py`

`scripts/verify_reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence.py`

`tests/test_reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence.py`

Existing production files must not be modified.

README must not be modified.

---

## 12. Zero-forward implementation verification

Before implementation freeze, require:

1. `git diff --check`;
2. narrow pytest;
3. independent static verifier;
4. exact parent implementation blob identities;
5. exact construction-cohort identities;
6. exact gate pair list 7801..7804;
7. exact reference/candidate shard plan;
8. exact 640/320/320/1280 forward accounting;
9. exact frozen float tolerances;
10. no restoration-confirmation cohort access;
11. no p-value call;
12. model loaded = false;
13. checkpoint loaded = false;
14. model forward count = 0;
15. CUDA scientific execution = false.

Successful verification establishes only:

`PASS_READY_FOR_GEN5_PHASE1B_RESTORATION_2GPU_TOPOLOGY_GATE_EXECUTION_AUTHORITY`

It does not authorize the gate execution itself.

---

## 13. Next boundary

Only after:

1. this implementation authority is frozen;
2. the three gate files are implemented;
3. zero-forward verification passes;
4. gate implementation is frozen;

may a separate execution authority authorize exactly one 1,280-forward
non-confirmatory topology/equivalence gate.

Only after that gate is collected/imported and validated PASS may the project
authorize the full 300-pair restoration-confirmation run.

STATUS =
`FROZEN_ON_COMMIT`
