# Gen5 M7 — t1 A0/B1 Finite Factor-Swap Causal Authority

SOURCE_HEAD=4841ed1d106f625e690c9cd1ae021ab6040f7afc
SOURCE_TEMPORAL_SYNTHESIS_SHA256=a9aee61287c09bc4181fc79d6805319996706e52f09cb2cb25d0d5b298e6266b
SOURCE_PHASE_A_TRAJECTORY_SHA256=0f7cd4248faa92223597e0816597b59e426f08829dadd366f9603f56a8de809e
SOURCE_RECOVERY_SUMMARY_SHA256=05cd49181ec3874fdcee16c78ce9b7e1ba5f7d907718bf0e7c56d17f9865bfeb
SOURCE_BEHAVIORAL_COORDINATES_SHA256=41518c0dbac345b972bf61920fe98681541f6393d6770494acb0df1cce149101

STATUS=AUTHORIZED_FOR_IMPLEMENTATION_ONLY
SCIENTIFIC_STAGE=M7_T1_FACTOR_SWAP_CAUSAL_COUNTERFACTUAL

TRAINING_ALLOWED=NO
BACKWARD_ALLOWED=NO
PARAMETER_GRADIENTS_ALLOWED=NO
OPTIMIZER_CONSTRUCTION_ALLOWED=NO
OPTIMIZER_STEP_ALLOWED=NO
PARAMETER_UPDATE_ALLOWED=NO
NEW_SEEDS_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
DATA_SPLIT_LABEL_CHANGE_ALLOWED=NO
MODEL_ARCHITECTURE_CHANGE_ALLOWED=NO
POST_HOC_ROWS_ALLOWED=NO
POST_HOC_TIMES_ALLOWED=NO
POST_HOC_THRESHOLDS_ALLOWED=NO
PROJECTOR_RECOVERY_REQUIRED_BEFORE_M7=NO
SCIENTIFIC_EXECUTION_ALLOWED_BEFORE_IMPLEMENTATION_FREEZE=NO
COMMIT_PUSH=MANUAL_ONLY

TARGET_ROW_BATCH_SIZE=32
MINIMUM_ACCEPTABLE_ROW_BATCH_SIZE=16
SCIENTIFIC_FULL_FORWARD_STATES=18
FROZEN_REFERENCE_STATES=9
WORKER_PARTITION=ROW_SHARDED_420_420
COMMON_CONTEXT_REUSE=MANDATORY
A0_LATENT_CACHE=MANDATORY
DURABLE_BATCH_CHUNKS=MANDATORY
MERGE_ONLY_RECOVERY=MANDATORY

## 1. Scientific question

The temporal mechanism is now localized to the first update:

`A0-dependent step-0 gradient -> B1 write -> t1 behavioral birth`

and `B_UPDATE_ONLY=(A0,B1)` reproduces `FULL_T1` predictions exactly with
approximately `1.43e-6` max absolute logit error.

M7 asks:

> When frozen t1 write factors are crossed, does first-update behavior follow
> the recipient A0 coordinate system, the donor B1 write, or the compatibility
> of the matched A0/B1 pair?

This is a finite parameter-state intervention. It is not training, model
selection, hyperparameter tuning, or a new seed experiment.

## 2. Frozen inputs

Use only repository-frozen inputs at `SOURCE_HEAD`.

### Phase A trajectory

Path:

`reports/reason_router_gen5_ainit_temporal_birth_replay_runs/gen5-ainit-temporal-birth-phase-a-numerical-auth-d940e19-r1/temporal_birth_trajectory.pt`

SHA256:

`0f7cd4248faa92223597e0816597b59e426f08829dadd366f9603f56a8de809e`

Required tensors:

- `A0[a]`: t0 `A_theta.weight`, shape `(2,768)`
- `B1[a,r]`: t1 `B_theta.weight`, shape `(24576,2)`

Factor seeds remain exactly:

`{6201,6202,6203}`

The implementation must authenticate that same-A / different-R `A0` tensors
are exactly equal before collapsing A0 to the three unique A-init seeds.

### Frozen matched-anchor logits

Path:

`reports/reason_router_gen5_ainit_temporal_mechanism_recovery_runs/gen5-ainit-temporal-mechanism-d9b790b-r1-partial-recovery/temporal_behavioral_coordinates.pt`

SHA256:

`41518c0dbac345b972bf61920fe98681541f6393d6770494acb0df1cce149101`

The frozen B_UPDATE_ONLY logits in this artifact are the canonical 840-row
matched references. They are not recomputed over all 840 rows during M7.

### Frozen vulnerable subset

The 120 shared vulnerable rows are fixed by the existing
shared-vulnerability artifact. M7 evaluates every new cross-A intervention on
all 840 dev rows and reports the already-frozen 120-row subset separately.

No new row subset is allowed.

## 3. Exact intervention family and no-duplicate execution rule

Define:

`H(a_rec, a_don, r_don) = B1[a_don,r_don] @ A0[a_rec] @ x`

with each factor in `{6201,6202,6203}`.

The conceptual grid contains 27 states.

### Nine matched references — frozen, not fully recomputed

The 9 states

`H(a,a,r)`

are exactly the already frozen B_UPDATE_ONLY states.

Their complete 840-row logits are loaded from the frozen behavioral-coordinate
artifact and serve as recipient/donor references.

They MUST NOT be rerun over all 840 rows.

Runtime integrity is checked by replaying all 9 matched anchors only on the
fixed authentication row slice:

`AUTH_ROWS = [0,1,...,31]`

This slice is fixed before results and is not a scientific subset.

### Eighteen cross-A scientific interventions — the only full new forwards

Only the 18 states with

`a_rec != a_don`

are newly evaluated over all 840 rows.

### Donor-R negative-control axis

Variation of `r_don` at fixed `(a_rec,a_don)` is the preregistered
training-RNG control. No RNG is sampled during M7.

## 4. Efficient forward schedule

The implementation must avoid the inefficiency of the temporal scan.

### Row-sharded two-GPU topology

Use:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

but partition **rows**, not hybrid states:

- worker 0: rows `[0,420)`
- worker 1: rows `[420,840)`

Each worker evaluates all 18 cross-A hybrids on its own rows.

Rationale: hybrid-sharding would cause both GPUs to recompute the expensive
layers-0-through-21 common context over the same 840 rows. Row sharding makes
each dev row pay that common-context cost on exactly one GPU.

No cross-GPU scientific reduction occurs during worker execution.

### Row batch size

Implementation target:

`ROW_BATCH_SIZE = 32`

CUDA preflight must test a real 32-row batch through the exact M7 forward path.

If batch 32 is infeasible on Tesla T4, the implementation may be revised to
batch 16 **before implementation freeze**. This is a resource correction, not
a scientific choice.

After implementation freeze there is no automatic batch-size fallback.

Do not use batch 4 unless a separately documented pre-freeze blocker proves
both 32 and 16 infeasible.

### Common-context reuse

For each worker row batch:

1. compute `_phase_b_prepare_common_context(...)` exactly once;
2. retain that common context while evaluating all 18 cross-A hybrids for the
   row batch;
3. never recompute layers 0..21 separately for each hybrid.

### A0 latent cache

For each row batch compute exactly three recipient latents:

`Z[a_rec] = A0[a_rec] @ mixer_input`

and reuse each `Z[a_rec]` for all donor B1 states.

Do not recompute the `768 -> 2` A0 projection separately for every hybrid.

All nine B1 tensors must be moved to the worker GPU once during initialization,
not once per batch/hybrid.

### Downstream forward count

Every cross-A hybrid / row combination receives exactly one downstream
scientific forward.

No pair metric, recipient/donor comparison, summary statistic, or merge step
may trigger an additional model forward.

Use `torch.inference_mode()` for scientific forwards. Do not change numerical
precision from the authenticated runtime merely for speed.

## 5. Primary finite causal estimands

For every cross-A hybrid at fixed donor RNG `r`:

`X = H(a_rec,a_don,r)`, where `a_rec != a_don`.

References are loaded frozen anchors:

- recipient: `R = H(a_rec,a_rec,r)`
- donor: `D = H(a_don,a_don,r)`

For both centered-logit and two-margin coordinates report:

- `d_rec = ||X-R||`
- `d_don = ||X-D||`
- `S = (d_don-d_rec)/(d_don+d_rec+eps)`

Interpretation:

- `S > 0`: closer to recipient reference;
- `S < 0`: closer to donor reference;
- near zero: neither dominates.

Report for all 840 rows and the frozen 120 vulnerable rows.

Do not introduce a post-hoc "strong ownership" threshold. Report full
distributions, mean/median, and sign counts.

## 6. Factor-effect decomposition

Using the 18 new cross-A states plus 9 frozen reference states, report finite
effects along:

### Recipient-A effect

Change `a_rec` while holding `(a_don,r_don)` fixed.

### Donor-A/B1-history effect

Change `a_don` while holding `(a_rec,r_don)` fixed.

### Donor-R control effect

Change `r_don` while holding `(a_rec,a_don)` fixed.

Report pairwise means/ranges and aggregate recipient-A / donor-A / donor-R
contrasts in two-margin space and prediction disagreement.

No p-value study is authorized.

## 7. Matched-pair compatibility diagnostics

For each cross-A hybrid report:

- prediction agreement with recipient reference;
- prediction agreement with donor reference;
- rows agreeing with neither;
- frozen-120 versions;
- hybrid margin/norm range checks;
- descriptive recipient-to-donor segment extrapolation.

If mismatched hybrids are systematically far from both references, that is
evidence for matched A0/B1 compatibility rather than simple recipient-only or
donor-only ownership.

Do not call this gauge equivalence. Basis/subspace alignment belongs to M8.

## 8. Mandatory authentication

### Repository/provenance

Require:

- exact execution HEAD = implementation-freeze commit;
- clean worktree;
- exact Phase A trajectory SHA;
- exact behavioral-coordinate SHA;
- exact model/checkpoint/tokenizer identities;
- no confirmatory 9601..9900 source loaded.

### Frozen reference authentication

Before main scientific interpretation:

1. authenticate frozen B_UPDATE_ONLY versus FULL_T1 from the committed
   behavioral-coordinate artifact:
   - prediction mismatch exactly `0`;
   - max-abs within `5e-5`.

2. replay all 9 matched `H(a,a,r)` states only on fixed `AUTH_ROWS=0..31`:
   - prediction mismatch versus corresponding frozen B_UPDATE_ONLY slice
     exactly `0`;
   - logit max-abs at most `5e-5`.

This checks every A/B matched source identity without paying for 9 redundant
840-row forward passes.

Do not widen tolerances after results.

### Execution flags

Must finish with:

- `TRAINING_EXECUTED=False`
- `BACKWARD_EXECUTED=False`
- `OPTIMIZER_CONSTRUCTED=False`
- `OPTIMIZER_STEP_EXECUTED=False`
- `PARAMETER_GRADIENTS_ACCUMULATED=False`
- `CONFIRMATORY_9601_9900_LOADED=False`

## 9. Crash-safe execution and merge-only recovery

The prior temporal run demonstrated that completed GPU work must survive a
parent assertion failure.

M7 therefore requires durable row-batch chunks.

For each worker and each completed row batch:

1. evaluate all 18 cross-A hybrids for that row batch;
2. move logits to CPU;
3. atomically write a persistent chunk into the final run directory, e.g.
   `worker0/chunk_rows_000_032.pt`;
4. write/record the chunk SHA256 and row interval;
5. only then proceed to the next row batch.

On worker restart, already existing chunks may be skipped only after exact
schema, source identity, row interval, hybrid-order, and SHA authentication.

This is execution recovery, not reuse across commits.

### Merge-only mode

`--merge-only` must:

- execute no CUDA work;
- execute no model forward;
- consume authenticated durable chunks or final worker results;
- reconstruct the full 18-cross-hybrid × 840-row output;
- combine it with the 9 frozen matched references;
- run all scientific summaries and authentication gates;
- write final artifacts.

Parent merge failure must never require cross-A model forwards to be rerun.

## 10. Expected computational scale

The implementation must print the planned counts before execution.

With row batch 32:

- common-context batches per worker: `ceil(420/32) = 14`
- full new scientific hybrid states: `18`
- downstream cross-hybrid forward calls per worker: `14 × 18 = 252`
- total cross-hybrid downstream calls across two workers: `504`
- full-row matched-anchor forwards: `0`

The 9 matched-anchor integrity replays are only on fixed 32-row auth data.

This is intentionally far smaller than the earlier temporal scan, which
looped over 21 time points, critical-time internal stages, Jacobian/projector
calculations, and repeated intervention resumes.

A materially larger forward count is an implementation defect unless justified
before implementation freeze.

## 11. Implementation scope

Create exactly:

- `scripts/audit_reason_router_gen5_ainit_t1_factor_swap.py`
- `tests/test_reason_router_gen5_ainit_t1_factor_swap.py`

Do not modify the temporal-mechanism implementation as part of M7.

Required modes:

- `--static-verify-only`
- `--cuda-preflight-only`
- `--run-factor-swap`
- `--factor-swap-worker`
- `--merge-only`

Static verify executes no model forward and no CUDA.

CUDA preflight must verify:

- exact frozen identities;
- exact 32-row target batch path;
- one matched anchor replay;
- at least one cross-A hybrid;
- finiteness;
- no parameter gradients;
- durable chunk schema;
- merge-only schema compatibility.

Preflight is not scientific evidence.

## 12. Required scientific outputs

Run directory must contain at minimum:

- `factor_swap_summary.json`
- `factor_swap_logits.pt`
- `factor_swap_pair_metrics.jsonl`
- `run_provenance.json`
- `artifact_manifest.json`
- `worker0/worker_result.pt`
- `worker1/worker_result.pt`
- `worker0/worker.log`
- `worker1/worker.log`
- durable `worker*/chunk_rows_*.pt` files or a manifest proving their
  authenticated consolidation.

The final logits artifact contains the conceptual 27-state grid over all 840
rows, with per-state provenance distinguishing:

- `FROZEN_MATCHED_REFERENCE`
- `NEW_CROSS_A_INTERVENTION`

## 13. Interpretation outcomes

M7 does not assume which factor wins.

Allowed bounded outcomes include:

- recipient-A coordinate dominance;
- donor-B1/history dominance;
- matched A0/B1 compatibility;
- mixed/nonseparable control;
- negligible donor-R contribution;
- failure of this compact factorization.

Claims follow the finite swap evidence, not a preferred narrative.

## 14. Falsification / stop conditions

Stop interpretation if:

- any frozen source identity fails;
- any of 9 fixed-slice matched-anchor replays fails;
- any new seed/data/split/label/model change occurs;
- parameter gradients accumulate;
- training/optimizer/backward executes;
- worker/chunk provenance is inconsistent;
- any of the 18 cross-A × 840 outputs is missing.

A scientifically negative but valid M7 is acceptable.

Do not redesign the intervention post hoc to rescue a preferred conclusion.

## 15. Claim boundary

M7 supports only the finite intervention:

`replace A0 and/or B1 while holding frozen downstream model and dev inputs fixed`

It does not prove:

- global training-dynamics causality;
- uniqueness of the factorization;
- gauge equivalence;
- unseen-seed generalization;
- a universal Mamba mechanism.

## 16. Completion criterion

M7 is complete when:

1. implementation is statically validated and frozen;
2. target batch 32 passes CUDA preflight, or implementation is revised and
   re-frozen at batch 16 before scientific execution;
3. 18 full cross-A interventions complete over all 840 rows;
4. frozen-reference and provenance authentication pass;
5. merge-only reproducibly reconstructs the final 27-state artifact;
6. bounded factor-swap evidence is frozen.

Only after M7 closure proceed to M8 representational-equivalence / quotient
interpretation.
