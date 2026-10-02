# Gen5 A-Initialization × Training-RNG Causal Intervention Design + Implementation Authority Candidate

## Status

IMPLEMENTATION_AUTHORITY_CANDIDATE

This document authorizes only the bounded implementation and static verification of one Gen5 causal intervention that separates:

- the seed controlling `A_theta` initialization;
- the seed controlling all remaining training RNG.

It does not authorize CUDA execution, model forward execution, backward, optimizer construction, training, task evaluation, Kaggle scientific execution, confirmatory-data access, or scientific interpretation of new runtime results.

## Authority basis

Frozen mechanism-analysis commit:

`155c1898f9f2d6dd0765fc93cc156190d8a08708`

Mechanism report:

`reports/reason_router_gen5_cross_seed_initialization_anchoring_mechanism_report_candidate.md`

Frozen original Phase3A P0 source execution:

`d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`

The static mechanism evidence establishes:

1. final `row(A)` is strongly anchored to its same-seed initialization;
2. the dominant right/input functional direction retains strong same-seed initialization memory;
3. the dominant output/write direction is substantially more shared across seeds;
4. SCALEMATCH near-full recovery tracks recovery of the same-seed dominant rank-1 functional axis.

The remaining causal question is whether final read-side geometry follows the `A` initialization seed when the remaining training RNG seed is changed independently.

## Scientific question

Under the exact frozen Phase3A P0 contract:

> Does final `row(A)` and the dominant right/input functional direction follow the `A_theta` initialization seed more strongly than the remaining training RNG seed?

This is an initialization-path causal intervention.

It is not:

- a new architecture;
- a rank sweep;
- a learning-rate sweep;
- a horizon sweep;
- a pressure sweep;
- an optimizer sweep;
- a data/split intervention;
- a new objective.

## Factorization

Two seed factors are separated.

### Factor 1 — A initialization seed

Exact allowed values:

`6201, 6202, 6203`

This seed controls only deterministic construction of:

`A_theta.weight`

using the existing frozen initialization rule:

`CPU float32 kaiming_uniform_(shape=(2,768), a=sqrt(5), generator=manual_seed(A_INIT_SEED))`

### Factor 2 — training RNG seed

Exact allowed values:

`6201, 6202, 6203`

This seed controls all non-A stochastic runtime/training RNG that was previously controlled by the single Phase3A seed, including the explicit pre-training:

- `torch.manual_seed(TRAINING_RNG_SEED)`
- `torch.cuda.manual_seed_all(TRAINING_RNG_SEED)`

The training RNG seed must not alter `A_theta` initialization.

`B_theta.weight` remains exact zero initialization and is therefore independent of both seed identities except for metadata.

## 3 × 3 factorial and reuse policy

The scientific factorial is:

| A-init seed | training RNG 6201 | training RNG 6202 | training RNG 6203 |
|---|---|---|---|
| 6201 | existing frozen diagonal | NEW | NEW |
| 6202 | NEW | existing frozen diagonal | NEW |
| 6203 | NEW | NEW | existing frozen diagonal |

The three diagonal cells:

- `(6201,6201)`
- `(6202,6202)`
- `(6203,6203)`

already exist as frozen Phase3A P0 evidence and MUST NOT be rerun.

Only the six off-diagonal cells are new execution targets.

Exact off-diagonal cells:

- `A6201-R6202`
- `A6201-R6203`
- `A6202-R6201`
- `A6202-R6203`
- `A6203-R6201`
- `A6203-R6202`

## GPU topology

Exact execution topology for the future execution stage:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

Exactly two GPUs are required.

No DDP, model parallelism, distributed gradient synchronization, shared optimizer, or cross-worker communication is allowed.

Balanced worker assignment:

### GPU worker 0

- `A6201-R6202`
- `A6202-R6203`
- `A6203-R6201`

### GPU worker 1

- `A6201-R6203`
- `A6202-R6201`
- `A6203-R6202`

Each worker therefore executes:

- exactly three cells;
- every A-init seed exactly once;
- every training RNG seed exactly once.

This worker assignment is frozen prospectively and must not be changed after outcome inspection.

## Frozen Phase3A scientific contract

Every new off-diagonal cell must preserve exactly:

- arm: `G5-C0`;
- pressure: `P0`;
- train rows: `3360`;
- dev rows: `840`;
- split seed: `16384`;
- parent checkpoint identity;
- frozen parent parameters;
- exact dataset and row order;
- exact tokenizer/model/runtime identity requirements;
- rank: `2`;
- `A_theta` shape: `[2,768]`;
- `B_theta` shape: `[24576,2]`;
- `B_theta` exact zero initialization;
- trainable tensors: `A_theta.weight`, `B_theta.weight`;
- trainable numel: `50688`;
- objective: final 3-way cross entropy only;
- AdamW;
- learning rate: `0.001`;
- weight decay: `0.0001`;
- gradient clipping norm: `5.0`;
- scheduler: none;
- optimizer steps: exactly `20`;
- final fixed-step checkpoint;
- no early stopping;
- no checkpoint selection;
- frozen Phase3A dev task evaluation only;
- confirmatory IDs `9601..9900` forbidden;
- scientific p-values forbidden.

No pressure other than P0 is authorized for this intervention.

## Seed-separation semantics

The implementation must make seed ownership explicit.

Required public identities:

- `a_init_seed`
- `training_rng_seed`

A single ambiguous `seed` field must not be used as the scientific identity of a new off-diagonal cell.

### Runtime-model construction

The runtime model may use `training_rng_seed` for any temporary constructor RNG needed by the existing loading path, because the parent checkpoint is frozen and authenticated.

Immediately before constructing the correction:

1. reconstruct `A_theta` using only `a_init_seed`;
2. initialize `B_theta` to exact zero.

Immediately before step-0/training stochastic operations:

- call `torch.manual_seed(training_rng_seed)`;
- call `torch.cuda.manual_seed_all(training_rng_seed)`.

No later reseeding from `a_init_seed` is allowed.

## Mandatory initialization authentication

Before training each cell, the implementation must:

1. reconstruct the expected CPU float32 `A_init` from `a_init_seed`;
2. verify exact byte identity of live `A_theta.weight` against that expected tensor after device/dtype transfer under the existing frozen semantics;
3. record `A_init_sha256`;
4. verify `B_theta.weight` is exact zero;
5. record both seed identities in checkpoint, training report, and provenance.

The implementation must also record the hashes of the three canonical A initializations:

- A-init seed 6201;
- A-init seed 6202;
- A-init seed 6203.

## Existing diagonal evidence identity

The future analysis must use, without rerunning, the frozen P0 Phase3A checkpoints:

- seed6201/P0
- seed6202/P0
- seed6203/P0

from:

`reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/`

Their original single seed is interpreted as:

`a_init_seed = training_rng_seed = seed`

for the factorial diagonal only.

No artifact rewriting is required.

## Primary analysis endpoints

The future analysis must be defined prospectively before execution.

### A. Final read-side inheritance

For each new cell:

`row(A_final)`

Compare against all three canonical initialization planes:

- `row(A_init_6201)`
- `row(A_init_6202)`
- `row(A_init_6203)`

Primary statistic:

`AFFINITY_TO_MATCHED_A_INIT - MEAN_AFFINITY_TO_OTHER_A_INITS`

### B. Dominant right/input functional-axis inheritance

Compute the dominant right singular direction of:

`B_final A_final`

without requiring materialization of the full ambient operator.

Compare its captured energy in each canonical A-init plane.

Primary statistic:

`CAPTURE_IN_MATCHED_A_INIT - MEAN_CAPTURE_IN_OTHER_A_INITS`

### C. Factor attribution

Across the full 3 × 3 factorial assembled from:

- 3 frozen diagonal cells;
- 6 new off-diagonal cells;

compare whether geometry clusters more strongly by:

- `a_init_seed`, or
- `training_rng_seed`.

The analysis must report raw pairwise geometry and grouped descriptive means.

No post-hoc threshold may be introduced to define success.

### D. Task behavior

Frozen Phase3A dev CE/accuracy may be recorded for each new cell under the existing evaluation contract.

Task performance is secondary to the geometry inheritance endpoint.

No new evaluation population is allowed.

## Interpretation cases

The following interpretation is prospective.

### Case 1 — A-init seed dominates

If final read-side geometry and dominant right/input direction group strongly by `a_init_seed` across changed training RNG seeds:

Support:

`GEN5_READ_SIDE_GEOMETRY_CAUSALLY_TRACKS_A_INITIALIZATION_UNDER_FIXED_PHASE3A_P0_TRAINING`

This would strengthen the interpretation that a substantial part of solution non-identifiability is optimization-path memory introduced through factor initialization.

### Case 2 — training RNG dominates

If geometry follows `training_rng_seed` rather than `a_init_seed`:

The static initialization anchoring was correlational and does not survive causal decoupling.

Shift interpretation toward stochastic optimization-path selection rather than initialization anchoring.

### Case 3 — both matter

If both factors produce substantial structured effects:

Interpret seed dependence as jointly determined by initialization geometry and stochastic training trajectory.

Do not force a single-cause claim.

### Case 4 — neither cleanly dominates

If off-diagonal cells reorganize idiosyncratically:

Close the simple seed-factor explanation and treat the solution family as more strongly path-dependent/nonlinear than the current static decomposition can resolve.

## Implementation scope

Create exactly three new files:

1. `reports/reason_router_gen5_ainit_rng_causal_intervention_design_implementation_authority_spec_candidate.md`
2. `scripts/train_reason_router_gen5_ainit_rng_causal_intervention.py`
3. `tests/test_reason_router_gen5_ainit_rng_causal_intervention.py`

Do not modify any existing file.

The new runner should reuse the frozen Phase3A primitives wherever possible.

It must not copy or fork the full model/training implementation unnecessarily.

## Required static-verification invariants

Before any later execution authority, static verification must establish:

1. exact repository branch and authorized HEAD ancestry;
2. exact frozen mechanism-report identity;
3. exact six off-diagonal cells;
4. exact three frozen diagonal evidence sources;
5. exact two-worker assignment;
6. worker 0 and worker 1 each contain exactly three cells;
7. each worker sees every A-init seed exactly once;
8. each worker sees every training RNG seed exactly once;
9. no diagonal cell is scheduled for execution;
10. A initialization depends only on `a_init_seed`;
11. training reseeding depends only on `training_rng_seed`;
12. B initialization is exact zero;
13. all Phase3A optimizer/loss/horizon constants are unchanged;
14. pressure is exactly P0;
15. confirmatory 9601–9900 access is forbidden;
16. static verify loads no parent model;
17. static verify performs no CUDA;
18. static verify performs no forward;
19. static verify performs no backward;
20. static verify constructs no optimizer;
21. static verify performs no training or task evaluation.

Expected static terminal marker:

`GEN5_AINIT_RNG_CAUSAL_INTERVENTION_STATIC_VERIFY_PASS`

## Required tests

The dedicated test file must cover at minimum:

1. exact factor levels;
2. exact six off-diagonal cells;
3. exclusion of all three diagonal cells from new execution;
4. exact two-GPU worker assignment;
5. balance of both factors across both workers;
6. deterministic A-init reconstruction for each seed;
7. distinct A-init hashes across all three seeds;
8. proof that changing `training_rng_seed` does not change reconstructed A-init;
9. proof that changing `a_init_seed` changes A-init while B remains zero;
10. exact inherited Phase3A optimizer/loss/horizon constants;
11. P0-only enforcement;
12. exact frozen diagonal source paths/hashes;
13. runtime modes blocked without later execution authority;
14. static mode performs no model/CUDA/backward/optimizer/training/eval;
15. confirmatory population remains inaccessible.

## Validation command

After creating the three files, run only CPU/static validation:

```powershell
python -m pytest tests/test_reason_router_gen5_ainit_rng_causal_intervention.py -q
python scripts/train_reason_router_gen5_ainit_rng_causal_intervention.py `
  --static-verify-only `
  --expected-head <CURRENT_HEAD> `
  --allow-opening-worktree
```

No CUDA preflight is authorized under this document.

## Training / evaluation authority

`CUDA_ALLOWED=NO`

`GPU_COUNT_FOR_FUTURE_EXECUTION=2`

`GPU_TOPOLOGY_FOR_FUTURE_EXECUTION=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

`PARENT_MODEL_INSTANTIATION_ALLOWED=NO`

`MODEL_FORWARD_ALLOWED=NO`

`BACKWARD_ALLOWED=NO`

`OPTIMIZER_ALLOWED=NO`

`TRAINING_ALLOWED=NO`

`TASK_EVALUATION_ALLOWED=NO`

`KAGGLE_SCIENTIFIC_EXECUTION_ALLOWED=NO`

`CONFIRMATORY_9601_9900_ALLOWED=NO`

A separate implementation freeze and a separate execution authority are required before any CUDA/runtime execution.

## Stop conditions

Stop implementation immediately if:

- an existing file must be modified;
- diagonal cells would need to be rerun;
- more than six new training cells are required;
- any pressure other than P0 is needed;
- rank, optimizer, LR, weight decay, horizon, objective, split, or evaluation domain must change;
- A initialization cannot be isolated from training RNG;
- the existing Phase3A parent/checkpoint/data identities cannot be preserved;
- DDP or cross-GPU gradient synchronization is introduced;
- static verification attempts CUDA/model/forward/backward/optimizer/training/evaluation;
- confirmatory data are accessed.

## Required implementation report

After implementation/static validation, report:

- exact HEAD;
- exact three new file paths;
- SHA256 for all three files;
- pytest result;
- static verifier terminal marker;
- exact six off-diagonal cells;
- exact frozen diagonal source identities;
- exact worker 0 and worker 1 assignments;
- exact A-init hashes for 6201/6202/6203;
- proof of A-init/training-RNG separation;
- inherited Phase3A optimizer/loss/horizon constants;
- confirmation that no CUDA/model/forward/backward/optimizer/training/evaluation or confirmatory access occurred.

No scientific conclusion may be drawn from implementation validation.
