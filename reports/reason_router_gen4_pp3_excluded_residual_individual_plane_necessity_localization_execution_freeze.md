# Gen4 PP3-Excluded Residual Individual-Plane Necessity Localization — Execution Freeze

## Status

`SCIENTIFIC_EXECUTION_AUTHORIZED_FOR_EXACT_FROZEN_IMPLEMENTATION`

This document authorizes exactly one raw scientific execution of the already
frozen individual-plane necessity-localization experiment.

It does not authorize training, backward passes, task-head evaluation,
statistical inference on GPU, Holm correction on GPU, subgroup analysis,
rescue analysis, endpoint changes, or scientific interpretation before raw
artifact import and provenance validation.

## Frozen authority chain

Prospective design:

`d046a8e03e7522a72dfbd08cc9129b769cd5686a`

Static preparation:

`ab9a6ebbc95bec20e8682b9538365f53167f108e`

Implementation authority:

`ac13d93785719345fc0362298c9988bf040e694d`

Implementation freeze:

`87acea31459d5b08e2af02e840c098c3a3e188b4`

The execution-freeze commit must have parent exactly:

`87acea31459d5b08e2af02e840c098c3a3e188b4`

and must change exactly this execution-freeze document.

The scientific execution HEAD is the resulting execution-freeze commit SHA,
which is fixed only after this document is manually committed and pushed.

No descendant of that execution-freeze commit is implicitly authorized.

## Exact implementation identities

Runner:

`scripts/reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_fast_cuda.py`

Git blob:

`9cc0f41b24953d94f870aa3e07b993fafd3d83ea`

Frozen external SHA256:

`56b919e54c8cef1d2220abe9652ffa48a77b42d00d28a64890f0ee60807a176d`

Test:

`tests/test_reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_fast_cuda.py`

Git blob:

`f81c9a97fc4579773e521261708042722e0b1cd6`

Frozen external SHA256:

`cc935c0366c8b192510159ae65a464ae3eda703581e00659e42178a63fdef410`

The implementation commit changes exactly these two files.

## Frozen population

Fresh XG1 population:

`xg1_fact_1801..xg1_fact_2100`

Pair count:

`300`

Six-cell row count:

`1800`

Static preparation has already established:

- byte-regeneration identity for frozen XG1 `001..1800`;
- pair-ID overlap with `001..1800`: `0`;
- claim overlap with `001..1800`: `0`;
- evidence overlap with `001..1800`: `0`;
- exact `(claim,evidence)` overlap with `001..1800`: `0`;
- tokenizer eligibility: `PASS_300_OF_300`;
- all frozen residual vectors reproduced exactly;
- no scientific outcomes observed.

## Frozen plane family

Exactly:

`[P1, P2, P4, P5]`

No plane selection, ranking, omission, or weighting is allowed.

## Frozen conditions

Exactly nine:

1. `native`
2. `p1_neutralized`
3. `p1_quarter_turn_control`
4. `p2_neutralized`
5. `p2_quarter_turn_control`
6. `p4_neutralized`
7. `p4_quarter_turn_control`
8. `p5_neutralized`
9. `p5_quarter_turn_control`

## Frozen endpoint

For each plane `Pk`:

`D_k = QC,k - QN,k`

with stored supporting quantities:

`Q0`

`QN,k`

`QC,k`

`A_N,k = Q0 - QN,k`

`A_C,k = Q0 - QC,k`

The raw runner must not calculate any p-value, Holm decision, supported-plane
set, or family-level label.

## Frozen forward budget

Per direction:

`4 scientific model forwards`

Per condition:

`40 scientific model forwards`

Per pair:

`360 scientific model forwards`

Total:

`108000 scientific model forwards`

Baseline model forwards:

`0`

Any count mismatch invalidates the raw run.

## Frozen two-GPU topology

Exactly two independent shards.

GPU 0:

`xg1_fact_1801..xg1_fact_1950`

Pairs:

`150`

Scientific forwards:

`54000`

GPU 1:

`xg1_fact_1951..xg1_fact_2100`

Pairs:

`150`

Scientific forwards:

`54000`

No DDP.

No NCCL.

Each worker loads one model/checkpoint instance.

## Runtime

Use the already validated ContraMamba Kaggle runtime:

- Python `3.12.13`
- NumPy `2.0.2`
- Torch `2.10.0+cu128`
- Transformers `5.0.0`
- tokenizers `0.22.2`
- kernels `0.10.2`
- CUDA runtime `12.8`
- two Tesla T4 GPUs, compute capability `7.5`

Frozen model:

`state-spaces/mamba-130m-hf`

Frozen snapshot revision:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Model/tokenizer snapshot:

`/kaggle/working/contramamba_runtime/models--state-spaces--mamba-130m-hf/snapshots/40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Representative checkpoint:

`/kaggle/input/datasets/terryterry9/checkpoint/selected_checkpoint.pt`

Checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Checkpoint bytes:

`518270455`

Frozen Mamba scientific binary SHA256:

`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

Frozen causal-conv1d scientific binary SHA256:

`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

Exact frozen binary identity is required. Source-revision variation is
acceptable only through the already validated transport identity path that
proves exact frozen binary SHA256 equality.

## Fresh bootstrap requirement

Use a fresh Kaggle bootstrap at the exact execution-freeze commit SHA created
from this document.

Its parent must be:

`87acea31459d5b08e2af02e840c098c3a3e188b4`

The repository worktree must be clean before the scientific command.

Do not reuse a repository checkout or raw artifact from another commit.

## Preflight boundary

Before scientific execution, verify without scientific model forwards:

- execution HEAD exact;
- worktree clean;
- static artifact identities exact;
- runner/test Git blobs exact;
- two T4 GPUs visible;
- accepted runtime versions exact;
- frozen kernel binary identities exact;
- tokenizer eligibility remains `PASS_300_OF_300`;
- checkpoint bytes authenticate exactly;
- output directory does not already exist.

Preflight must report:

`SCIENTIFIC_MODEL_FORWARD_COUNT=0`

`PRIMARY_INFERENCE_EXECUTED=False`

No scientific model forward may be spent merely to preflight the run.

## Authorized raw runner invocation

Invoke as a Python module, not a direct script path:

`python -u -m scripts.reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_fast_cuda`

Required arguments:

- `--expected-head <EXACT_EXECUTION_FREEZE_COMMIT_SHA>`
- exact frozen model snapshot;
- exact frozen tokenizer snapshot;
- exact frozen checkpoint;
- a new run-specific output directory.

## Run naming

Authorized run name:

`g4k-residual-individual-plane-necessity-xg1-1801-2100-<execution-short-sha>`

If and only if an infrastructure/runtime failure occurs before a valid raw
artifact is completed, a later separately frozen retry name may append
`-retryN`.

Do not silently reuse the same run name after a failed scientific attempt.

## Expected raw output

The raw run must produce exactly the runner-defined scientific artifact set:

- `pp3_excluded_residual_individual_plane_necessity_items.jsonl`
- `pp3_excluded_residual_individual_plane_necessity_summary.json`
- `artifact_manifest.json`
- `SHA256SUMS.txt`

The summary must state:

- source pair count `300`;
- pair interval `1801..2100`;
- scientific model forwards `108000`;
- baseline forwards `0`;
- GPU count `2`;
- both exact shard intervals and budgets;
- planned raw confirmatory p-value count `4`;
- planned multiplicity method
  `Holm step-down across exactly P1,P2,P4,P5`;
- planned familywise alpha `0.05`;
- `primary_inference_executed = false`;
- `multiplicity_correction_executed = false`;
- `scientific_conclusion = null`.

## Statistical boundary

No statistical inference is authorized during the Kaggle raw execution.

After successful collect/import and raw artifact validation, a separate
CPU-only confirmatory inference step may compute exactly four one-sided
Student t-test p-values:

`H0,k: mean(D_k) <= 0`

`H1,k: mean(D_k) > 0`

for `k in {P1,P2,P4,P5}`.

Those four p-values form one family and must use the frozen Holm step-down
procedure at FWER `0.05`.

No fifth p-value is allowed.

No subgroup, tail, rescue, alternate endpoint, alternate epsilon, or
outcome-guided plane selection is allowed.

## Failure handling

If the raw run fails:

- preserve the run log and metadata;
- collect/import failure provenance when possible;
- do not reinterpret partial outputs as confirmatory evidence;
- do not alter the frozen endpoint or statistical family;
- do not resume from a partial pair unless a separately reviewed resume
  implementation is explicitly frozen;
- diagnose the failure before authorizing any retry.

A runtime failure does not count as a scientific negative result.

## Scientific interpretation boundary

A successful raw run alone establishes only execution success.

A valid imported artifact establishes artifact/provenance validity.

Only after the later frozen confirmatory inference may scientific claims be
made.

Even a positive final result may establish only individual local necessity for
planes that satisfy all frozen gates after Holm correction.

It may not establish:

- plane sufficiency;
- plane dominance or ranking;
- additive decomposition;
- absence of cross-plane interactions;
- causal necessity of the XG2-like residual-template orientation;
- behavioral or downstream-task necessity;
- global model or architecture universality.

## Current authorization

Design freeze: `YES`

Static preparation: `YES`

Implementation freeze: `YES`

Raw scientific execution at the exact execution-freeze commit produced by
this document: `YES`

Training: `NO`

Backward: `NO`

Task-head evaluation: `NO`

GPU statistical inference: `NO`

Additional p-values beyond the later frozen four-test family: `NO`

Commit / push of this execution-freeze document: manual only.
