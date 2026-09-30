# ContraMamba Gen5 Phase 2 Fresh Ownership Assay Execution Authority

## 0. Status

PHASE =
`GEN5_PHASE2_FRESH_OWNERSHIP_ASSAY_EXECUTION`

STATUS =
`FROZEN_ON_COMMIT`

SCIENTIFIC_EXECUTION_ALLOWED_AFTER_FREEZE =
`YES_EXACTLY_ONE_CONFIRMATORY_OWNERSHIP_RUN`

KAGGLE_ALLOWED_AFTER_FREEZE =
`YES_EXACTLY_FOR_THIS_RUN`

TRAINING_ALLOWED =
`NO`

BACKWARD_ALLOWED =
`NO`

TASK_EVALUATION_ALLOWED =
`NO`

IMPLEMENTATION_MODIFICATION_ALLOWED =
`NO`

CODEX_ALLOWED =
`NO`

This authority permits exactly one prospective confirmatory Gen5 Phase 2
fresh ownership assay using the frozen implementation, the already completed
nine-cell Phase 2 correction matrix, and the already frozen fresh XG1 cohort.

It does not authorize retraining, implementation changes, additional seeds,
cohort replacement, additional confirmatory tests, task evaluation, or rescue
experiments.

---

## 1. Frozen implementation

IMPLEMENTATION_AUTHORITY_COMMIT =
`037922770ef51d5ac825690458790e59ff774d63`

IMPLEMENTATION_FREEZE_COMMIT =
`6f202464b7c933611728ac861c6e7153510c7270`

The scientific execution head must descend from the implementation freeze and
the execution-authority commit.

The following files must remain byte-identical from the implementation freeze
through scientific execution:

- `scripts/reason_router_gen5_phase2_fresh_ownership_assay.py`
- `tests/test_reason_router_gen5_phase2_fresh_ownership_assay.py`

Dedicated implementation verification at the freeze boundary:

- dedicated pytest: `20 passed`
- static verification:
  `PASS_GEN5_PHASE2_FRESH_OWNERSHIP_ASSAY_IMPLEMENTATION_STATIC_VERIFICATION`
- XG1 pair count: `300`
- XG1 row count: `1800`
- authenticated training corrections: `9`
- scientific model forward count: `0`
- training executed: false
- backward executed: false
- task evaluation executed: false
- scientific p-value count: `0`
- scientific execution: false

---

## 2. Frozen scientific design

PHASE2_DESIGN_COMMIT =
`d9b84bca8871464807d2dccf6380a6e911d6dbf8`

Ownership dimension:

`STATE_UPDATE_AUTHORITY`

Owner:

`R22`

Matched control:

`C22`

Training arms:

- `G5-C0`
- `G5-C1`
- `G5-M1`

Training seeds:

- `5201`
- `5202`
- `5203`

Primary comparison:

`G5-M1_MINUS_G5-C1`

C0 is descriptive only and is excluded from the confirmatory statistical test.

No rank, layer, arm, seed, projector, correction, endpoint, or cohort search is
authorized.

---

## 3. Frozen completed training evidence

TRAINING_EXECUTION_COMMIT =
`acdf2ee8940070a6e60190a1c2fcbede8933e419`

TRAINING_RUN =
`gen5-phase2-dualt4-matrix-acdf2ee-r1`

Validated facts:

- nine cells completed;
- exactly 20 optimizer steps per cell;
- exactly 180 optimizer steps total;
- parent parameters unchanged;
- R22/C22 unchanged;
- fresh XG1 not loaded;
- task evaluation not executed;
- scientific p-value count zero;
- scientific conclusion null.

The known completed-run checksum-finalization defect is restricted to the stale
`matrix_provenance.json` entry in the historical `SHA256SUMS.txt`.

Frozen known stale entry SHA256:

`91727eafc46f2dc40e460e1932dbf88694bffd8e126759cbc34f86920a800aab`

Frozen actual final `matrix_provenance.json` SHA256:

`6fb216b75e084bbdac4fbcd6143de94f2233c3c0c2a879d6a1143277673dd603`

All other internal artifact hashes were independently validated.

The completed training artifact tree must not be repaired, rewritten, or
regenerated.

---

## 4. Frozen training-artifact tree identity

TRAINING_ARTIFACT_TREE_FILE_COUNT =
`35`

TRAINING_ARTIFACT_TREE_CANONICALIZATION =
`POSIX_RELATIVE_PATH_UTF8_LEXICOGRAPHIC_ASCENDING`

TRAINING_ARTIFACT_TREE_CANONICAL_ROW =
`relative_path<TAB>byte_size<TAB>sha256<LF>`

TRAINING_ARTIFACT_TREE_CANONICAL_BYTES =
`3589`

TRAINING_ARTIFACT_TREE_SHA256 =
`1d82a999b2c5f52cc77c4890de04ed8047eaa38b781626b1e3b0e62eef5f53f2`

INTERNAL_ROOT =
`gen5-phase2-dualt4-matrix-acdf2ee-r1`

The scientific input identity is the exact 35-file artifact tree, not a ZIP
container representation.

The previously recorded deterministic transport ZIP SHA256 is superseded as an
execution gate because byte-identical artifact trees produced different ZIP
container SHA256 values across Windows and Linux despite identical per-file
SHA256 and byte sizes.

This correction changes no training artifact, checkpoint, scientific endpoint,
cohort, correction tensor, or analysis rule.

Before scientific execution:

1. exactly 35 files must be present under the frozen internal root;
2. the canonical tree identity above must equal the frozen tree SHA256;
3. the runner must independently validate the complete training-artifact tree,
   including the known historical matrix_provenance checksum-finalization defect;
4. all nine correction checkpoints and their tensor/provenance chains must pass;
5. any tree/hash/artifact mismatch blocks execution.

An existing authenticated Kaggle copy of this exact 35-file tree may be used
directly. Repacking or re-extraction is not required.

No alternative training checkpoint bundle may be substituted.

---

## 5. Frozen parent and basis identities

Representative parent checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Native backbone signature SHA256:

`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

Frozen Hugging Face revision:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Model config SHA256:

`784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`

R22 SHA256:

`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22 SHA256:

`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

R22/C22 shape:

`[24576, 2]`

No parent, basis, or correction parameter may mutate during the assay.

---

## 6. Frozen fresh ownership cohort

DATA_ROOT =
`data/reason_router_gen5_phase2_xg1_ownership_assay_v1`

PAIR_FIRST =
`xg1_fact_8701`

PAIR_LAST =
`xg1_fact_9000`

PAIR_COUNT =
`300`

ROWS_PER_PAIR =
`6`

ROW_COUNT =
`1800`

Frozen static identities:

`structured_source_facts.jsonl`

SHA256 =
`f316d7e2a187e90451ff0743829d6c615c063969b66ca51f91e15158bcc1ee06`

`synthetic_reason_router_six_cell.jsonl`

SHA256 =
`529027ba79d3bd9fc4e300fbf317f0d84668152ed4fa0593c55af601e7934976`

`tokenizer_anchor_manifest.jsonl`

SHA256 =
`880586e043e50e960c05482f3edc6b953b358e50ed0eba5fc279c7785cc33c00`

`tokenizer_eligibility_summary.json`

SHA256 =
`ff64b60e3651cf78fe6cb9e9757446593471781cc5977d427eb4c1a6bffd0229`

No row dropping, reordering, replacement, extension, or filtering is authorized.

---

## 7. Frozen tokenizer semantics

Tokenizers version:

`0.22.2`

Maximum sequence length:

`128`

Claim budget:

`63`

Evidence budget:

`64`

Serialization:

`claim[:63]+EOS(0)+evidence[:64]`

Effective pad token id:

`0`

All 300 pairs / 1800 rows already passed frozen tokenizer-anchor eligibility.

---

## 8. Frozen ownership-assay semantics

Every item starts from:

`PP3_NEUTRALIZED_AT_LAYER17`

For every trained arm and seed, the learned Phase 2 correction remains active.

The native donor write is defined before the Phase 2 correction is added:

`a_native = R22^T w22_native`

Corrected background for arm A:

`w22_B^A = w22_B_native + DeltaW_eff^A`

Restoration conditions:

`w22_RR^A = w22_B^A + R22 a_native`

`w22_RC^A = w22_B^A + C22 a_native`

The same `a_native` must be used for RR and RC.

The same trained correction must remain identically active across B, RR, and RC.

The frozen Phase1B Q definition remains:

`Q = E_XG2_22 - E_XG4_22`

Per arm/seed/item:

- `Q_B`
- `Q_RR`
- `Q_RC`
- `S_R = Q_RR - Q_B`
- `S_C = Q_RC - Q_B`
- `I_A = Q_RR - Q_RC`

No alternative endpoint is authorized.

---

## 9. Frozen seed aggregation and confirmatory test

Training seeds are not independent item replicates.

For each fresh item i:

`Ibar_M1_i = mean_seed(I_M1_i)`

`Ibar_C1_i = mean_seed(I_C1_i)`

Primary ownership contrast:

`D_OWN_i = Ibar_M1_i - Ibar_C1_i`

Confirmatory sample size:

`300`

Exactly one scientific confirmatory p-value is authorized:

`one-sided one-sample Student t-test(D_OWN, alternative mean > 0)`

No seed-specific p-value is authorized.

No C0 p-value is authorized.

No task-performance p-value is authorized.

---

## 10. Frozen scientific decision rule

Ownership support requires all provenance/manipulation/runtime gates plus:

1. `mean(Ibar_M1) > 0`
2. seed-averaged `mean(Q_RR_M1) > 0`
3. seed-averaged `mean(S_R_M1) > 0`
4. `mean(D_OWN) > 0`
5. `p_one_sided_greater < 0.05`

Positive label:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_PRESERVES_CAUSAL_ROLE_INTEGRITY_OVER_MATCHED_C22_CONTROL`

Otherwise:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_ADVANTAGE_NOT_ESTABLISHED`

A valid negative result is terminal for this exact Phase 2 hypothesis.

No rescue by seed expansion, rank change, layer change, projector change,
architecture change, cohort change, or endpoint change is authorized.

---

## 11. Exact 2×T4 topology

Exactly two physical Tesla T4 GPUs are required.

GPU0 worker:

- pairs `xg1_fact_8701..xg1_fact_8850`
- 150 pairs
- `CUDA_VISIBLE_DEVICES=0`
- one logical device `cuda:0`

GPU1 worker:

- pairs `xg1_fact_8851..xg1_fact_9000`
- 150 pairs
- `CUDA_VISIBLE_DEVICES=1`
- one logical device `cuda:0`

No DDP.

No gradient synchronization.

No CPU scientific fallback.

No one-GPU fallback.

---

## 12. Frozen scientific forward accounting

The implementation reuses the Phase1B restoration forward topology and shares the
same frozen-parent captures analytically across the nine correction checkpoints.

FORWARDS_PER_PAIR =
`160`

GPU0_FORWARD_BUDGET =
`24000`

GPU1_FORWARD_BUDGET =
`24000`

FULL_CUDA_SCIENTIFIC_MODEL_FORWARD_BUDGET =
`48000`

CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET =
`0`

Any over-budget, under-budget, partial, interrupted, or reordered result is invalid.

---

## 13. Frozen runtime

Scientific execution requires the already qualified Kaggle CUDA runtime:

- Python `3.12.13`
- NumPy `2.0.2`
- PyTorch `2.10.0+cu128`
- Transformers `5.0.0`
- tokenizers `0.22.2`
- kernels `0.10.2`
- CUDA `12.8`
- exactly two Tesla T4 GPUs
- compute capability `(7,5)` on both GPUs
- fp32
- autocast false

Frozen kernel/binary gates inherited from the qualified Phase1B CUDA path must pass.

A runtime or binary mismatch blocks execution.

---

## 14. Execution-head binding

The scientific execution commit is the commit that freezes this execution authority.

The command must supply:

`--expected-head <EXECUTION_AUTHORITY_FREEZE_COMMIT>`

`--implementation-freeze-commit 6f202464b7c933611728ac861c6e7153510c7270`

`--execution-authority-commit <EXECUTION_AUTHORITY_FREEZE_COMMIT>`

The implementation validates:

- execution-authority commit ancestry;
- implementation-freeze ancestry;
- implementation byte identity since the freeze;
- execution-authority blob identity;
- clean worktree;
- frozen scientific inputs.

No descendant implementation modification may substitute for the frozen runner.

---

## 15. Scientific command contract

After bootstrap of the exact execution-authority commit, the top-level scientific
command must invoke:

`python -u -m scripts.reason_router_gen5_phase2_fresh_ownership_assay`

with:

- `--run-assay`
- exact `--expected-head`
- exact `--implementation-freeze-commit`
- exact `--execution-authority-commit`
- authenticated frozen model snapshot
- authenticated frozen tokenizer snapshot
- authenticated representative parent checkpoint
- authenticated extracted 35-file training-artifact root
- a new empty output directory

The exact shell command, concrete Kaggle paths, output directory, and run name will
be generated only after the authority commit is known and `cm kaggle` has
authenticated that exact commit.

The command must authenticate the frozen 35-file stable tree identity before
scientific invocation. An already preserved byte-identical Kaggle tree may be
used directly without repacking.

---

## 16. Output boundary

The scientific output bundle is restricted to:

- `ownership_items.jsonl`
- `ownership_summary.json`
- `worker0_manifest.json`
- `worker1_manifest.json`
- `artifact_manifest.json`
- `SHA256SUMS.txt`

Raw full recurrent-state vectors must not be persisted.

Raw full native-write vectors must not be persisted.

Raw full POST_STATE22 vectors must not be persisted.

Final checksums must be computed only after all mutable output bytes are finalized.

---

## 17. Collection and interpretation boundary

After successful execution:

1. `cm run save <run-name>` and `cm run <run-name>` provenance must be valid;
2. `cm collect <run-name>` is mandatory;
3. local `cm import <handoff.zip>` is mandatory;
4. imported artifact identities and checksums must be independently validated.

Execution success alone does not establish the scientific conclusion.

Scientific interpretation begins only after successful collect/import validation.

Failed runs are never collected.

---

## 18. Stop conditions

Stop rather than bypass if any of the following occurs:

- wrong branch or HEAD;
- dirty worktree;
- implementation drift;
- execution-authority mismatch;
- frozen training-artifact stable tree hash mismatch;
- training-artifact mismatch;
- fresh XG1 identity/count/order mismatch;
- tokenizer identity mismatch;
- parent checkpoint mismatch;
- model snapshot/config mismatch;
- R22/C22 mismatch;
- correction checkpoint/tensor mismatch;
- fewer or more than nine correction checkpoints;
- correction, parent, or basis mutation;
- runtime or exact CUDA kernel mismatch;
- anything other than exactly two Tesla T4 GPUs;
- worker failure;
- item drop or reorder;
- forward-budget mismatch;
- non-finite endpoint;
- more or fewer than one scientific confirmatory p-value;
- training or backward execution;
- task evaluation;
- output collision;
- artifact checksum failure.

No fallback, tuning, repair, rescue, or automatic retry is authorized inside the
scientific run.

A failed execution requires a new run identity; the failed run is not collected.

---

## 19. Success boundary

A completed valid run produces:

`PASS_GEN5_PHASE2_FRESH_OWNERSHIP_ASSAY`

with:

- exactly 300 fresh item-level observations;
- exactly nine frozen trained corrections evaluated;
- exactly 48,000 CUDA scientific model forwards;
- zero CPU scientific model forwards;
- exactly one confirmatory p-value;
- no training;
- no backward;
- no task evaluation;
- authenticated immutable parent, bases, and corrections.

After successful collect/import validation, proceed to:

`GEN5_PHASE2_SCIENTIFIC_INTERPRETATION`

No scientific conclusion is accepted before that validation boundary.
