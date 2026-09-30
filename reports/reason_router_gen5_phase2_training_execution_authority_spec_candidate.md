# ContraMamba Gen5 Phase 2 Training Execution Authority Specification Candidate

Status: CANDIDATE — becomes active only after this exact file is committed on `gen5-causal-role-state-ownership`.

## 1. Authority identity and phase

Authority name:

`GEN5_PHASE2_STATE_UPDATE_OWNERSHIP_TRAINING_EXECUTION_AUTHORITY`

Phase:

`TRAINING_EXECUTION_ENABLEMENT_AND_EXECUTION`

This authority is the next stage after the frozen Phase 2 design, static preparation,
WRITE22 implementation, accelerated-backend implementation, and representative-parent
bounded forward/backward verification.

This document is intentionally narrow. It authorizes only the exact training runner
opening, preflight, and 3-arm × 3-seed training matrix specified below.

It does **not** authorize the fresh XG1 ownership assay, inferential statistics,
scientific interpretation, rank/layer/projector search, or any rescue experiment.

Codex is not authorized for this stage.

## 2. Frozen ancestry

The following commits are frozen references:

- Phase 2 scientific design:
  `d9b84bca8871464807d2dccf6380a6e911d6dbf8`
- Phase 2 static training provenance and fresh assay cohort:
  `44e8853d08355641735b006459cd341ddb1dd631`
- Phase 2 differentiable WRITE22 implementation authority:
  `e2f8975d8271c0e95c92b9389dfc4717a221f7df`
- Initial differentiable WRITE22 implementation:
  `95f7d8b7392a9bad98fd450f1f23ea221bd4e242`
- Accelerated WRITE22 correction backend:
  `a69a0dd47853e3ec68f7c5aa21bb8d692c89a86a`

The execution-authority commit created from this document must descend from
`a69a0dd47853e3ec68f7c5aa21bb8d692c89a86a` with no unrelated code changes.

## 3. Frozen parent identity

Representative historical parent:

- model family: `state-spaces/mamba-130m-hf`
- historical evaluator/training seed: `180`
- historical arm: `G3-GROUP-D-HALF`
- representative checkpoint SHA256:
  `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`
- native backbone signature SHA256:
  `81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

Canonical local Mamba config snapshot:

- revision:
  `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`
- `config.json` SHA256:
  `784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`
- `config.json` bytes: `895`

The parent checkpoint must be authenticated before deserialization and strict-loaded.
No parent parameter may be reinitialized, fine-tuned, or otherwise mutated.

## 4. Frozen R22/C22 identity

Use only the Phase1B frozen bases:

- R22 SHA256:
  `a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`
- C22 SHA256:
  `c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`
- shape: `[24576, 2]`
- rank: `2`

The bases remain frozen buffers with no gradient and may not be reconstructed,
rotated, replaced, refit, or selected from alternatives.

## 5. Completed pre-execution implementation gates

### 5.1 Dedicated implementation tests

At accelerated-backend freeze preparation:

- dedicated pytest: `11 passed`
- bounded implementation verifier: PASS
- fresh XG1 loaded: false
- training executed: false
- scientific execution: false

Observed accelerated/reference agreement:

- forward max absolute residual: `0.0`
- A-gradient max absolute residual: `0.0`
- B-gradient max absolute residual:
  `3.637978807091713e-12`

The accelerated backend is therefore authorized as the training backend.

### 5.2 Representative-parent bounded verification

Verifier identity:

- external verifier filename:
  `gen5_phase2_representative_parent_gate_v2.py`
- verifier SHA256:
  `7819721ca02f40946525be848bd569396b4856e9cd39dfc7784fc3a67a0ea202`

Executed against clean HEAD:

`a69a0dd47853e3ec68f7c5aa21bb8d692c89a86a`

Bounded runtime:

- Python `3.13.2`
- PyTorch `2.10.0+cpu`
- Transformers `5.0.0`
- `use_cache=False`
- CPU sequential Mamba reference path

Observed PASS evidence:

- strict representative checkpoint load: PASS
- target layer: zero-based layer `22`
- zero-correction full-model equivalence: exact PASS
- logits max absolute residual: `0.0`
- Q max absolute residual: `0.0`
- entitlement max absolute residual: `0.0`
- correction-only backward: PASS
- A-gradient norm:
  `4.847323725698516e-05`
- B-gradient norm:
  `0.016385739669203758`
- parent gradient count: `0`
- basis gradient count: `0`
- parent immutability after bounded optimizer step: PASS
- R22 immutability after bounded optimizer step: PASS
- C22 immutability after bounded optimizer step: PASS
- fresh XG1 loaded: false
- task evaluation executed: false
- scientific p-value count: `0`
- training executed: false
- repository mutation: false

This gate establishes code/runtime viability only. It is not scientific evidence.

## 6. Authorized implementation delta before execution

The current training script is intentionally fail-closed. This authority permits
opening it for the exact execution defined here.

Only these tracked files may be modified during training-runner opening:

1. `scripts/train_reason_router_gen5_phase2_state_update_ownership.py`
2. `tests/test_reason_router_gen5_phase2_state_update_ownership.py`

No other tracked file may be modified.

In particular, the following are frozen and must not change:

- `src/contramamba/gen5_phase2_state_update_ownership.py`
- `scripts/verify_reason_router_gen5_phase2_state_update_ownership.py`
- `src/contramamba/modeling_v6b_minimal.py`
- `src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py`
- all Phase1B basis artifacts
- all Phase 2 static-preparation data/artifacts
- all fresh XG1 assay files

The runner-opening delta may only:

- bind this authority commit identity;
- authenticate the exact repository/parent/basis/data/runtime inputs;
- implement the frozen training data loading and exact split;
- implement the fixed 3 × 3 training matrix;
- load the already-frozen CUDA backend through existing repository machinery;
- install the frozen Phase 2 WRITE22 wrapper;
- execute the frozen optimizer/objective schedule;
- write the exact provenance/checkpoint/loss artifacts defined below;
- provide static and synthetic/non-scientific preflight modes.

It may not alter the correction mathematics or the frozen parent model.

## 7. Frozen training population and split

Training source dataset:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Authoritative historical Git-LF SHA256:

`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

Semantic SHA256:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

Integrity sidecar:

`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl`

Sidecar authoritative Git-LF SHA256:

`9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d`

Sidecar semantic SHA256:

`2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9`

Split contract:

- split seed: `8192`
- total rows: `3600`
- train rows: `2880`
- dev rows: `720`
- train pairs: `240`
- dev pairs: `60`
- ordered train-row SHA256:
  `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8`
- ordered dev-row SHA256:
  `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4`

The training runner must reconstruct this exact split and fail closed on any identity,
count, ordering, or semantic mismatch.

The dev split must not be used for checkpoint selection.

## 8. Frozen tokenization and sequence semantics

Training uses the historical training tokenizer/runtime semantics already recovered
and frozen by Phase 2 static preparation:

- Transformers `4.45.0`
- tokenizers `0.20.3`
- maximum sequence length: `128`

The runner must reproduce the frozen historical training encoding semantics from the
existing historical training implementation/provenance.

The fresh XG1 analysis tokenizer (`tokenizers 0.22.2`) is **not** a training tokenizer
and must not be substituted into training.

Fresh XG1 `xg1_fact_8701..xg1_fact_9000` must not be loaded during training.

## 9. Frozen trainable ownership

Exactly two trainable tensors are permitted:

- `correction.A_theta.weight`
- `correction.B_theta.weight`

Total trainable parameter count:

`50688`

All historical parent parameters, historical task heads, R22, and C22 must have
`requires_grad=False`.

The optimizer parameter list must be constructed explicitly from the correction
parameters. Blanket optimization over `model.parameters()` is forbidden.

## 10. Frozen arms

Exactly three arms are authorized:

- `G5-C0`
  - unrestricted correction
  - `DeltaW_eff = DeltaW_theta`
- `G5-C1`
  - matched control projection
  - `DeltaW_eff = (I - P_C) DeltaW_theta`
- `G5-M1`
  - causal-role projection
  - `DeltaW_eff = (I - P_R) DeltaW_theta`

Primary later comparison is M1 versus C1.

C0 remains descriptive.

No additional arm is authorized.

## 11. Frozen training seeds and initialization

Exactly these training seeds are authorized:

- `5201`
- `5202`
- `5203`

For each seed:

- A initialization uses the frozen dedicated CPU-generator Kaiming-uniform rule;
- B is exactly zero initialized;
- A/B initialization must be byte-identical across C0/C1/M1 for the same seed;
- global CPU/CUDA RNGs must be reset deterministically for each arm/seed cell so
  stochastic parent-head behavior uses the same seed contract across matched arms.

No seed expansion or replacement is authorized.

## 12. Frozen objective

Only:

`FINAL_3WAY_CROSS_ENTROPY_ONLY`

External class order:

1. `REFUTE`
2. `NOT_ENTITLED`
3. `SUPPORT`

The loss is ordinary unweighted three-way cross entropy on the historical final
classifier logits.

Forbidden training losses include, without limitation:

- Q losses
- D_NEC losses
- D_SUF losses
- R/C coefficient losses
- native state/write reconstruction losses
- PP3/PP5 losses
- frame losses
- predicate losses
- sufficiency losses
- polarity losses
- reason-router auxiliary losses
- ranking losses
- intervention losses
- weighted-label losses
- teacher/distillation losses

No class weighting is authorized.

## 13. Frozen optimizer and schedule

For every arm/seed cell:

- optimizer: `torch.optim.AdamW`
- learning rate: `0.001`
- weight decay: `0.0001`
- scheduler: none
- gradient clipping: global correction-parameter norm `5.0`
- epochs: `20`
- logical optimizer steps: exactly `20`
- checkpoint: fixed state immediately after optimizer step `20`

No early stopping.

No best-dev selection.

No task-metric-based checkpoint selection.

No retry with changed hyperparameters.

## 14. Batch semantics

The scientific training contract is one logical full-train objective and one optimizer
step per epoch over the exact ordered 2880-row training split.

Silent replacement by ordinary minibatch SGD is forbidden.

Gradient accumulation, microbatch partitioning, prefix caching, or any other execution
transformation that changes the realized stochastic/training semantics is not
authorized by this document.

If the exact full-train execution cannot fit the qualified runtime, the run must stop
as:

`GEN5_PHASE2_FULL_BATCH_EXECUTION_FEASIBILITY_BLOCKED`

and return for an explicit design/authority amendment.

No automatic batch-size reduction is allowed.

## 15. Runtime and CUDA backend

Scientific training is authorized only on the qualified Kaggle CUDA runtime:

- Python: `3.12.13`
- PyTorch: `2.10.0+cu128`
- Transformers: `5.0.0`
- CUDA reported by PyTorch: `12.8`
- GPU: Tesla T4 compatible runtime
- default dtype: float32
- autocast: false

Training must use the existing repository's previously qualified exact CUDA kernel
loading/compatibility machinery. No new mutable kernel revision or unverified binary
may be introduced.

The parent native Mamba path must execute on the fast CUDA path.

The Phase 2 correction uses the frozen accelerated streaming/checkpointed backend from
commit `a69a0dd47853e3ec68f7c5aa21bb8d692c89a86a`.

The slow CPU path remains reference-only and is not the scientific training backend.

## 16. Cache semantics

Scientific training is full-sequence/no-cache.

During training:

- `model.train()` must be active;
- Mamba cache use must be false;
- no incremental decoding/cache path may execute.

The representative-parent verification's explicit `use_cache=False` was a bounded
eval-mode adaptation only and does not alter training semantics.

## 17. Preflight required before scientific optimizer step 1

After the runner-opening implementation is committed and pushed, the exact execution
commit must pass all of the following before any scientific optimizer step:

1. repository identity and clean worktree authentication;
2. authority ancestry authentication;
3. frozen source/blob identity checks;
4. runtime identity checks;
5. parent checkpoint/config/basis/data/split authentication;
6. trainable tensor count `2`;
7. trainable numel `50688`;
8. same-seed A/B initialization equality across all three arms;
9. B exact-zero initialization;
10. parent/basis frozen status;
11. exact CUDA fast-path availability;
12. full-batch memory/forward feasibility preflight without optimizer step;
13. no fresh XG1 assay population loaded;
14. no scientific p-value computed.

If any preflight gate fails, scientific training must not start.

## 18. Authorized training execution matrix

Exactly nine cells:

| Seed | C0 | C1 | M1 |
|---:|:---:|:---:|:---:|
| 5201 | train | train | train |
| 5202 | train | train | train |
| 5203 | train | train | train |

Preferred deterministic execution order:

1. seed5201 / G5-C0
2. seed5201 / G5-C1
3. seed5201 / G5-M1
4. seed5202 / G5-C0
5. seed5202 / G5-C1
6. seed5202 / G5-M1
7. seed5203 / G5-C0
8. seed5203 / G5-C1
9. seed5203 / G5-M1

The same parent checkpoint is reloaded/authenticated for every cell or otherwise
reconstructed in a way that proves identical parent bytes at cell start.

State must not leak between cells.

## 19. Authorized outputs

Each cell may write only training/provenance artifacts needed for subsequent frozen
assay reconstruction.

Recommended per-cell directory:

`reports/reason_router_gen5_phase2_training_runs/<seed>/<arm>/`

Required per-cell artifacts:

- `run_provenance.json`
- `training_report.json`
- `final_correction.pt`

`training_report.json` must contain at least:

- execution commit
- authority commit
- parent checkpoint SHA256
- R22/C22 SHA256
- arm
- seed
- runtime identity
- train split identities/counts
- tokenizer/training encoding identity
- exactly 20 training losses
- exactly 20 optimizer steps
- optimizer/lr/weight-decay/clip/scheduler contract
- trainable tensor names/count/numel
- parent signature before and after
- R22/C22 identity before and after
- final correction tensor SHA256 values
- training success/failure
- task evaluation executed: false
- fresh XG1 loaded: false
- scientific p-value count: 0
- scientific conclusion: null

`final_correction.pt` should contain only the correction state required to reconstruct
the trained Phase 2 model plus immutable identity metadata. A duplicate full parent
checkpoint should not be persisted unless technically required.

A top-level matrix manifest/checksum file may be produced.

## 20. Scientific firewall

During this training stage:

- do not load `xg1_fact_8701..xg1_fact_9000`;
- do not compute Q_RR/Q_RC ownership assay endpoints;
- do not compute D_OWN;
- do not compute the confirmatory t-test;
- do not compute any scientific p-value;
- do not interpret arm differences scientifically;
- do not promote any arm;
- do not change the assay cohort.

Training loss traces are execution diagnostics, not scientific conclusions.

Successful training establishes only:

1. code execution success;
2. valid trained correction checkpoints under the frozen contract.

It does not establish the Phase 2 scientific claim.

## 21. Kaggle execution requirements

Scientific training may begin only after:

1. this authority document is committed and pushed;
2. the permitted runner-opening code delta is implemented;
3. dedicated tests/static preflight pass;
4. the runner-opening delta is committed and pushed;
5. `cm ship` is clean/ready before that commit;
6. `cm kaggle` bootstraps the exact runner-opening commit;
7. bootstrap verifies exact full HEAD and clean Kaggle worktree;
8. the exact command is registered with `cm run save`.

The run name must be descriptive and commit-bound.

Recommended run name:

`gen5-phase2-ownership-training-3x3`

The exact shell command will be supplied only after the runner-opening commit is
known and its static/CUDA preflight interface is verified.

## 22. Stop conditions

Stop without scientific training if any of the following occurs:

- repository or authority identity mismatch;
- parent checkpoint/config identity mismatch;
- R22/C22 identity or geometry mismatch;
- dataset/sidecar/split/order mismatch;
- tokenizer/training encoding mismatch;
- trainable tensor count or numel mismatch;
- any parent or basis parameter becomes trainable;
- optimizer contains any non-correction parameter;
- zero-initialization or same-seed cross-arm initialization mismatch;
- final CE is not exactly plain 3-way CE;
- auxiliary loss becomes nonzero or active;
- scheduler becomes active;
- checkpoint selection depends on dev/task metrics;
- cache/incremental decoding path activates;
- exact qualified CUDA fast path is unavailable;
- full logical batch cannot execute within the qualified runtime;
- OOM requires ordinary minibatch SGD or unauthorized accumulation;
- NaN/Inf appears in loss, gradients, parameters, or logits required for training;
- parent/R22/C22 identity changes;
- fresh XG1 assay data is accessed;
- a scientific p-value is computed;
- rescue requires changing rank, layer, arm, seed, projector, loss, data, or optimizer.

## 23. Success condition and next stage

Training execution succeeds only if all nine authorized cells complete exactly
20 optimizer steps and produce valid authenticated final correction artifacts with
the parent and frozen bases unchanged.

After successful collect/import, the next stage is:

`GEN5_PHASE2_FRESH_OWNERSHIP_ASSAY_EXECUTION`

That later stage will use only the already-frozen fresh XG1 cohort
`xg1_fact_8701..xg1_fact_9000`.

No ownership assay or scientific conclusion is authorized by this document.
