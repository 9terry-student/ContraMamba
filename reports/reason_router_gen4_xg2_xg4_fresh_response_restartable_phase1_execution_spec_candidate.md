# Gen4-K XG2/XG4 Fresh Response Restartable Phase-1 Execution Freeze

## Status

This file authorizes only the corrected restartable Phase-1 scientific
baseline execution derived from implementation commit:

`d237953cae76a801c99ff2be8adf7bc1d9300d6d`

Runner:

`scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py`

The execution permits XG2 and XG4 only.

It does not authorize:

- Phase-2 alignment-response execution;
- alignment intervention;
- `R_ALIGN`;
- H1/H2 inference;
- threshold re-estimation;
- training;
- backward;
- task heads;
- logits.

## Scientific workload

For each family:

- 300 fixed fresh source pairs;
- pair IDs `301..600`;
- four baseline branches per pair;
- exactly 1200 scientific model forwards;
- zero alignment model forwards.

Across XG2 and XG4:

- exactly 2400 Phase-1 baseline forwards total;
- zero Phase-2 forwards.

Expected frozen regime counts:

- XG2: LARGE = 92, SMALL = 208
- XG4: LARGE = 57, SMALL = 243

Any mismatch blocks the execution result.

Frozen threshold:

`0.11228626366380845`

Minimum group size:

`30`

## Runtime snapshot

Frozen model/tokenizer revision:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Observed known-good runtime snapshot path:

`/kaggle/working/contramamba_runtime_assets/gen4-k-xg1-full-e551484-r6/40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

The same directory is supplied as both:

- `--model-snapshot`
- `--tokenizer-snapshot`

Required files and SHA256:

`config.json`
`784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`

`tokenizer.json`
`b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`

`tokenizer_config.json`
`9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`

`special_tokens_map.json`
`57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

The runtime snapshot working path is not assumed to survive a Kaggle session
reset. If the directory or any required hash is absent or mismatched, execution
must stop before any scientific model forward.

No substitute revision, mutable latest snapshot, or regenerated tokenizer is
allowed.

## Representative checkpoint

Observed Kaggle Input path:

`/kaggle/input/datasets/terryterry9/contramamba-seed180-g3-group-d-half-checkpoint/selected_checkpoint.pt`

Required SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Representative identity:

- seed: 180
- arm: `G3-GROUP-D-HALF`

A missing or mismatched checkpoint blocks before model construction.

## Backend identity

Known-good scientific runtime:

- Python 3.12.13
- NumPy 2.0.2
- PyTorch 2.10.0+cu128
- Transformers 5.0.0
- CUDA runtime 12.8
- GPU Tesla T4
- capability 7.5
- kernels 0.10.2

The existing exact-kernel compatibility layer remains authoritative.

Frozen Mamba binary SHA256:

`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

Frozen causal-conv1d binary SHA256:

`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

No mutable-main or latest-kernel substitution is allowed.

## Output isolation

The execution writes outside the Git repository so that XG2 completion does
not dirty the repository before XG4 authentication.

Run output root:

`/kaggle/working/g4k_xg2_xg4_restartable_phase1_d237953_r1`

Family outputs:

`/kaggle/working/g4k_xg2_xg4_restartable_phase1_d237953_r1/xg2`

`/kaggle/working/g4k_xg2_xg4_restartable_phase1_d237953_r1/xg4`

The output root must not exist before execution.

Expected files per family:

- `baseline_items.jsonl`
- `alignment_delta_h.pt`
- `regime_summary.json`
- `artifact_manifest.json`
- `SHA256SUMS.txt`

## Run identity

Run name:

`g4k-xg2-xg4-restartable-phase1-d237953-r1`

The formal Kaggle execution must bootstrap the exact execution-freeze commit
created from this document.

The runner argument `--expected-head` must equal that exact bootstrapped full
HEAD.

The final command is instantiated only after this execution-freeze document is
committed and pushed, so that the exact execution HEAD can be supplied.

## Required command structure

The formal command must:

1. fail on any shell error;
2. verify clean repository state;
3. verify exact current HEAD;
4. verify all four runtime snapshot file SHA256 values;
5. verify representative checkpoint SHA256;
6. verify output-root nonexistence;
7. run XG2 restartable Phase-1;
8. confirm repository remains clean;
9. run XG4 restartable Phase-1;
10. confirm repository remains clean;
11. leave Phase-2 unexecuted.

No scientific forward may occur before identity checks pass.

## Post-run acceptance

Each family must independently report:

`PASS_XG2_XG4_FRESH_RESPONSE_RESTARTABLE_PHASE1_FREEZE`

and:

- `baseline_model_forward_count = 1200`
- `alignment_model_forward_count = 0`
- `total_model_forward_count = 1200`
- `phase2_restartable = true`
- support gate PASS
- exact frozen regime counts
- 300 persisted item rows
- 300 alignment-plan entries
- manifest/hash validation PASS.

Successful execution is not yet a Phase-2 scientific conclusion.

Artifacts must be collected/imported and provenance-validated before Phase-2
is authorized.

## Stop conditions

Stop before or during execution on any:

- HEAD mismatch;
- dirty repository;
- runtime snapshot absence;
- tokenizer/config hash mismatch;
- checkpoint absence or hash mismatch;
- backend/runtime gate failure;
- frozen kernel identity failure;
- pair-order mismatch;
- threshold mismatch;
- frozen regime-count mismatch;
- budget mismatch;
- output collision;
- manifest/hash validation failure.

No automatic fallback, asset substitution, threshold change, pair replacement,
or baseline re-execution inside Phase-2 is allowed.
