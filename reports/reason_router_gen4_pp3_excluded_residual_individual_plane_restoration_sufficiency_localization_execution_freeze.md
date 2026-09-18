# Gen4 PP3-Excluded Residual Individual-Plane Restoration Sufficiency Localization — Execution Freeze

## Status

`SCIENTIFIC_EXECUTION_AUTHORIZED_FOR_EXACT_FROZEN_IMPLEMENTATION`

This document authorizes exactly one raw scientific execution of the already
frozen individual residual-plane restoration-sufficiency localization
experiment.

It does not authorize training, backward, task-head evaluation, statistical
inference on GPU, Holm correction on GPU, rescue analysis, endpoint changes,
plane ranking, or scientific interpretation before raw artifact import and
validation.

## Frozen authority chain

Prospective design:

`8c0ff6dbad77ed876fc1481b3b53c3fd47a27d3b`

Static preparation:

`4a1d5871fad17a34951bc433a283c95e682a081b`

Implementation authority:

`25ff56c80ccd943bb386ebc7ab6612fe6e68d470`

Implementation freeze:

`466741783dd1fc9325e7d87a2d1a44ccb0a09de3`

The execution-freeze commit produced by this document must have parent exactly:

`466741783dd1fc9325e7d87a2d1a44ccb0a09de3`

and must change exactly this execution-freeze document.

The resulting execution-freeze commit SHA is the only authorized scientific
execution HEAD.

No descendant is implicitly authorized.

## Exact implementation identities

Runner:

`scripts/reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_fast_cuda.py`

Git blob:

`c208c01d3cff7fa80b44d94444df42b6cd0227be`

SHA256:

`c2dc56d91e114b4af011f53171156f2563b19fa20e365ae32b1c5da9ceda84fe`

Test:

`tests/test_reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_fast_cuda.py`

Git blob:

`266fe544e11215ba624407c0fab0d9c899fc6060`

SHA256:

`7059cd10ee45c876f153fe56c0942ea9772a26e35457ced4b31e438dc10f9d23`

Implementation validation already completed with:

`16 passed`

Scientific model forwards during implementation validation:

`0`

Checkpoint-backed scientific execution:

`0`

Primary inference:

`False`

## Frozen population

Fresh prospective population:

`xg1_fact_2101..xg1_fact_2400`

Pairs:

`300`

Rows:

`1800`

Static preparation established:

- old XG1 `001..2100` byte-regeneration identity: `True`
- pair-ID overlap: `0`
- claim overlap: `0`
- evidence overlap: `0`
- `(claim,evidence)` overlap: `0`
- tokenizer eligibility: `PASS_300_OF_300`
- frozen endpoint basis identity: `PASS`
- all residual vectors reproduced exactly: `True`
- scientific model forwards: `0`
- scientific outcomes observed: `False`

## Frozen conditions

Exactly:

1. `native`
2. `p1_neutralized`
3. `p1_quarter_turn_replacement`
4. `p2_neutralized`
5. `p2_quarter_turn_replacement`
6. `p4_neutralized`
7. `p4_quarter_turn_replacement`
8. `p5_neutralized`
9. `p5_quarter_turn_replacement`

For each plane:

`c_k = a_k p_k+ + b_k p_k-`

`r_k = -b_k p_k+ + a_k p_k-`

Neutralized state:

`B_k = h - c_k`

Matched replacement:

`C_k = h - c_k + r_k`

The replacement correction is:

`delta_C,k = -c_k + r_k`

It is not the old necessity control `-r_k`.

## Frozen endpoint

Shared exact-restoration endpoint:

`Q0 = Q(native)`

Per plane:

`QB,k = Q(pk_neutralized)`

`QC,k = Q(pk_quarter_turn_replacement)`

Supporting quantities:

`S_k = Q0 - QB,k`

`S_C,k = QC,k - QB,k`

Canonical raw primary contrast:

`D_SUF,k = Q0 - QC,k`

The raw runner must not compute t statistics, p-values, Holm decisions,
supported-plane sets, adjusted p-values, rankings, or a scientific conclusion.

## Frozen forward budget

Per direction:

`4 scientific forwards`

Per condition:

`40 scientific forwards`

Per pair:

`360 scientific forwards`

Total:

`108000 scientific forwards`

Baseline forwards:

`0`

Any forward-count mismatch invalidates the raw run.

## Frozen two-GPU topology

GPU 0:

`xg1_fact_2101..xg1_fact_2250`

Pairs:

`150`

Forwards:

`54000`

GPU 1:

`xg1_fact_2251..xg1_fact_2400`

Pairs:

`150`

Forwards:

`54000`

No DDP.

No NCCL.

Each worker loads one model/checkpoint instance.

## Frozen runtime

Accepted Kaggle runtime:

- Python `3.12.13`
- NumPy `2.0.2`
- Torch `2.10.0+cu128`
- Transformers `5.0.0`
- tokenizers `0.22.2`
- kernels `0.10.2`
- CUDA runtime `12.8`
- two Tesla T4 GPUs

Model:

`state-spaces/mamba-130m-hf`

Revision:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Snapshot:

`/kaggle/working/contramamba_runtime/models--state-spaces--mamba-130m-hf/snapshots/40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Checkpoint:

`/kaggle/input/datasets/terryterry9/checkpoint/selected_checkpoint.pt`

Checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Frozen Mamba binary SHA256:

`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

Frozen causal-conv1d binary SHA256:

`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

Exact binary identity is mandatory.

## Authorized environment repair

The Kaggle environment may install exactly:

`kernels==0.10.2`

using an environment-only operation such as:

`python -m pip install --no-deps kernels==0.10.2`

This does not authorize upgrading or replacing Torch, Transformers, NumPy,
tokenizers, CUDA, Mamba, causal-conv1d, or any other scientific dependency.

The git worktree must remain clean.

Preflight must verify:

`importlib.metadata.version("kernels") == "0.10.2"`

and the exact frozen Mamba / causal-conv1d binary SHA256 values above.

## Fresh-bootstrap requirement

Use a fresh Kaggle bootstrap at the exact execution-freeze commit created from
this document.

Do not reuse a repository checkout or scientific artifact from another commit.

The Kaggle repository worktree must be clean.

## Preflight boundary

Before any scientific model forward, verify:

- exact execution HEAD;
- clean worktree;
- exact runner/test blobs and SHA256;
- exact static-input identities;
- exact tokenizer eligibility;
- exact XG2/XG4 frozen basis-plan identities;
- exact residual vectors;
- exact checkpoint SHA256;
- exact accepted runtime versions;
- `kernels==0.10.2`;
- exact frozen Mamba binary SHA256;
- exact frozen causal-conv1d binary SHA256;
- exactly two accepted T4 GPUs;
- run output directory absent.

Preflight must complete with:

`SCIENTIFIC_MODEL_FORWARD_COUNT=0`

`PRIMARY_INFERENCE_EXECUTED=False`

`MULTIPLICITY_CORRECTION_EXECUTED=False`

## Command-capture boundary

The local `cm run save` operation captures clipboard bytes.

Therefore the clipboard passed to `cm run save` must contain only the intended
Kaggle Bash preflight-and-run payload.

It must not contain PowerShell orchestration commands.

The saved command must be inspected/pinned before `cm run`.

## Authorized runner

Invoke as a Python module:

`python -u -m scripts.reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_fast_cuda`

with:

- `--expected-head <EXACT_EXECUTION_FREEZE_COMMIT_SHA>`
- exact frozen model snapshot
- exact frozen tokenizer snapshot
- exact frozen checkpoint
- a new run-specific output directory

The exact full Bash payload will be generated only after this execution-freeze
commit exists.

## Run identity

Authorized run-name template:

`g4k-residual-individual-plane-restoration-sufficiency-xg1-2101-2400-<execution-short-sha>`

The run name is single-use.

If an infrastructure failure occurs before a valid raw artifact exists, do not
reuse the run name. Diagnose first and separately authorize a retry if needed.

## Expected raw artifacts

Exactly the runner-defined artifact set:

`pp3_excluded_residual_individual_plane_restoration_sufficiency_items.jsonl`

`pp3_excluded_residual_individual_plane_restoration_sufficiency_summary.json`

`artifact_manifest.json`

`SHA256SUMS.txt`

Successful raw execution must report:

- pair count `300`
- pair interval `2101..2400`
- total scientific forwards `108000`
- baseline forwards `0`
- GPU count `2`
- exact two shard ranges and budgets
- planned raw confirmatory p-value count `4`
- planned Holm family across exactly P1/P2/P4/P5
- familywise alpha `0.05`
- `primary_inference_executed = false`
- `multiplicity_correction_executed = false`
- `scientific_conclusion = null`

## Later inference boundary

Only after successful collection/import and raw provenance validation may a
separate CPU-only confirmatory inference compute exactly four one-sided
Student t-tests:

`H0,k: mean(D_SUF,k) <= 0`

`H1,k: mean(D_SUF,k) > 0`

for:

`k in {P1,P2,P4,P5}`

Those four p-values form one Holm step-down family at FWER `0.05`.

No fifth p-value is authorized.

## Current authorization

Design freeze: `YES`

Static preparation: `YES`

Implementation freeze: `YES`

Raw scientific execution at the exact execution-freeze commit produced by this
document: `YES`

Training: `NO`

Backward: `NO`

Task-head evaluation: `NO`

GPU statistical inference: `NO`

Outcome-guided plane selection: `NO`

Commit/push of this execution-freeze document: manual only.