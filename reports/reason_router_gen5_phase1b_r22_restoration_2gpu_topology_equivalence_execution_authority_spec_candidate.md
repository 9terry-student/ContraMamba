# ContraMamba Gen5 Phase 1B — R22 Restoration 2×T4 Topology Equivalence Gate Execution Authority

## 0. Status

PHASE =
`GEN5_PHASE1B_R22_RESTORATION_2GPU_TOPOLOGY_EQUIVALENCE_GATE_EXECUTION`

STATUS =
`FROZEN_ON_COMMIT`

IMPLEMENTATION_ALLOWED =
`NO`

SCIENTIFIC_EXECUTION_ALLOWED_AFTER_FREEZE =
`YES_EXACTLY_ONE_NONCONFIRMATORY_GATE`

KAGGLE_ALLOWED_AFTER_FREEZE =
`YES_EXACTLY_FOR_THIS_GATE`

TRAINING_ALLOWED =
`NO`

BACKWARD_ALLOWED =
`NO`

TASK_EVALUATION_ALLOWED =
`NO`

README_UPDATE_REQUIRED =
`NO`

This authority permits exactly one non-confirmatory 2×T4 topology/equivalence
gate execution using the frozen gate implementation.

It does not authorize the full restoration-confirmation population.

---

## 1. Frozen gate implementation

GATE_IMPLEMENTATION_FREEZE_COMMIT =
`a54fcaa84159336bd79e6b01397a58dd4be1ac54`

GATE_RUNNER =
`scripts/reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence.py`

GATE_RUNNER_GIT_BLOB =
`575eec3b16ffab52479e1eb2aceafa1a35fbc797`

GATE_VERIFIER =
`scripts/verify_reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence.py`

GATE_VERIFIER_GIT_BLOB =
`ad6e36d7dc138eed3eda2b99bc80cf656d44cae8`

GATE_TEST =
`tests/test_reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence.py`

GATE_TEST_GIT_BLOB =
`641c06cb3de518000520599e32fa40b597a727f9`

No implementation file may be modified before or during this execution.

EXECUTION_HEAD =
`THIS_EXECUTION_AUTHORITY_FREEZE_COMMIT`

No descendant commit may substitute for the execution head without a new
execution authority.

---

## 2. Parent restoration implementation

RESTORATION_IMPLEMENTATION_FREEZE_COMMIT =
`d8d87aa8891ae1f2bef16ed3f1b174e58f87b978`

RESTORATION_RUNNER_GIT_BLOB =
`ed73904fdf9a555e11d30e3b9068ac12941225cd`

RESTORATION_SEMANTICS_GIT_BLOB =
`786f186ab89ffc44bb5373b70df7a1041165b7a5`

QUALIFIED_Q22_BACKEND_GIT_BLOB =
`421d30f00cf71690ed41c983ccf0540808e1de1c`

These frozen sources must remain byte-identical.

---

## 3. Scientific boundary

This is a backend/topology qualification run only.

SCIENTIFIC_CONFIRMATORY_TEST =
`FORBIDDEN`

SCIENTIFIC_P_VALUE_COUNT =
`0`

SCIENTIFIC_CONCLUSION =
`NONE`

RESTORATION_CONFIRMATION_POPULATION_ACCESS =
`FORBIDDEN`

The run may not read:

`data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1`

The run may not interpret RR/RC differences as scientific restoration evidence.

Gate PASS authorizes only a later restoration execution authority.

---

## 4. Gate population

Only the already-used non-confirmatory construction population is authorized.

DATA_ROOT =
`data/reason_router_gen5_phase1b_xg1_construction_v1`

GATE_PAIRS =
`xg1_fact_7801..xg1_fact_7804`

GATE_PAIR_COUNT =
`4`

No alternative pair, replacement pair, extra pair, or extension beyond 7804 is
authorized.

Frozen construction identities:

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

## 5. Frozen checkpoint/tokenizer identities

REPRESENTATIVE_CHECKPOINT =
`reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt`

REPRESENTATIVE_CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

NATIVE_BACKBONE_SIGNATURE_SHA256 =
`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

CANONICAL_MODEL_TOKENIZER_REVISION =
`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

MODEL_CONFIG_SHA256 =
`784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`

TOKENIZER_JSON_SHA256 =
`b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`

TOKENIZER_CONFIG_SHA256 =
`9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`

SPECIAL_TOKENS_MAP_SHA256 =
`57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

No substitution is permitted.

---

## 6. Frozen CUDA runtime

The gate requires exactly two physical Tesla T4 devices.

EXPECTED_RUNTIME =

- Python `3.12.13`
- NumPy `2.0.2`
- Torch `2.10.0+cu128`
- Transformers `5.0.0`
- tokenizers `0.22.2`
- kernels `0.10.2`
- CUDA runtime `12.8`
- physical GPU count `2`
- device 0 `Tesla T4`
- device 1 `Tesla T4`
- compute capability `(7,5)` for both GPUs

MAMBA_KERNEL_REVISION =
`c8ffc584c147878a6eb978ae0e8db4d116c93a8c`

CAUSAL_CONV1D_KERNEL_REVISION =
`f2651e776f66069cdcf842840db637583def1223`

MAMBA_BINARY_SHA256 =
`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

CAUSAL_CONV1D_BINARY_SHA256 =
`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

Any mismatch blocks execution.

GPU must remain OFF during bootstrap and external-asset provisioning where
possible, and ON only for the exact runtime check and gate execution.

---

## 7. Exact topology

Reference arm:

- physical GPU 0 only;
- logical device `cuda:0`;
- pairs `7801..7804`;
- sequential;
- exact 640 model forwards.

Candidate arm:

- independent process 0:
  - physical GPU 0 only;
  - logical `cuda:0`;
  - pairs `7801..7802`;
  - exact 320 model forwards.
- independent process 1:
  - physical GPU 1 only;
  - logical `cuda:0`;
  - pairs `7803..7804`;
  - exact 320 model forwards.

Candidate workers run concurrently only after the reference worker has completed.

No DDP, model sharing, gradient synchronization, or cross-device scientific
reduction is authorized.

---

## 8. Exact forward accounting

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

TOTAL_GATE_MODEL_FORWARD_BUDGET =
`1280`

CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET =
`0`

A partial, interrupted, failed, over-budget, or under-budget execution is not a
valid gate and must not be interpreted.

---

## 9. Frozen equivalence criterion

FLOAT_ATOL =
`1e-9`

FLOAT_RTOL =
`1e-7`

All exact discrete leaves must match exactly.

All compared finite floating leaves must satisfy:

`abs(reference - candidate) <= 1e-9 + 1e-7 * max(abs(reference), abs(candidate), 1.0)`

Tensor SHA256 audit fields may differ across physical GPUs and are excluded from
the numerical equivalence comparison exactly as implemented.

No tolerance change is authorized.

---

## 10. PASS rule

The gate result is:

`PASS_GEN5_PHASE1B_RESTORATION_2GPU_TOPOLOGY_EQUIVALENCE`

only if the frozen implementation itself reports that exact PASS result with:

- pair range `7801..7804`;
- reference forward count 640;
- candidate GPU0 forward count 320;
- candidate GPU1 forward count 320;
- total forward count 1280;
- CPU scientific forwards 0;
- confirmatory p-value count 0;
- scientific conclusion null;
- all discrete equivalence checks PASS;
- all floating equivalence checks PASS;
- both physical devices identified as Tesla T4 with capability 7.5;
- parameter signatures unchanged in all three worker processes.

Any gate failure blocks full restoration execution.

---

## 11. Artifact boundary

The exact gate output directory must contain only:

`topology_equivalence_items.jsonl`

`topology_equivalence_summary.json`

`artifact_manifest.json`

`SHA256SUMS.txt`

Raw WRITE22 and POST_STATE22 vectors must not be persisted.

After a successful run, `cm collect` and local `cm import` are required before
the gate may be considered provenance-valid.

Execution success without validated import is insufficient.

---

## 12. Run identity

RUN_NAME =
`gen5-phase1b-r22-restoration-2gpu-topology-equivalence`

OUTPUT_DIR =
`reports/reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence_<EXECUTION_HEAD_SHORT>`

The exact `<EXECUTION_HEAD_SHORT>` is the first seven hexadecimal characters of
the commit that freezes this execution authority.

The run name must not be reused after a failed wrapper/run attempt.

A retry requires a new run name suffix and still must execute the same exact
execution head and scientific command.

---

## 13. Execution command contract

After this authority is frozen and the exact execution HEAD is known, the
scientific command must invoke:

`python -u -m scripts.reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence`

with exactly these semantic arguments:

- `--expected-head <THIS_EXECUTION_AUTHORITY_FREEZE_COMMIT>`
- `--model-snapshot <EXACT_PROVISIONED_CANONICAL_MODEL_DIRECTORY>`
- `--tokenizer-snapshot <SAME_EXACT_PROVISIONED_CANONICAL_MODEL_DIRECTORY>`
- `--checkpoint <EXACT_PROVISIONED_REPRESENTATIVE_CHECKPOINT>`
- `--output-dir <FROZEN_OUTPUT_DIR>`

No `--worker` argument may be supplied by the top-level shell command.

The coordinator alone launches the three worker processes and controls
`CUDA_VISIBLE_DEVICES`.

The command must be stored by `cm run save` and executed by `cm run`.

---

## 14. Stop conditions

Stop rather than bypass if:

- execution HEAD differs from the authority-freeze commit;
- worktree is dirty;
- two Tesla T4 devices are not available;
- any frozen runtime identity mismatches;
- external model/tokenizer/checkpoint hashes mismatch;
- exact kernel binary hashes mismatch;
- gate output directory already exists;
- construction input hashes mismatch;
- restoration-confirmation data is accessed;
- any worker fails;
- any forward count mismatches;
- any equivalence comparison fails;
- any p-value is computed;
- scientific conclusion is non-null;
- collect/import provenance mismatches.

No fallback to one GPU is authorized by this authority.

---

## 15. Success boundary

A validated imported PASS establishes only:

`TWO_T4_RESTORATION_EXECUTION_TOPOLOGY_QUALIFIED`

It does not establish restoration.

It does not establish the Phase 1B bridge.

Only after the gate artifact is imported, validated, and frozen may a new
authority permit the full 300-pair restoration-confirmation execution:

`xg1_fact_8401..xg1_fact_8700`

with exactly 48,000 CUDA scientific model forwards and exactly one confirmatory
p-value.

STATUS =
`FROZEN_ON_COMMIT`
