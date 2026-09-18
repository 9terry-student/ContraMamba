# Gen4 PP3-Excluded Residual Individual-Plane Restoration Sufficiency Localization — Execution Freeze Correction

## Status

`SCIENTIFIC_EXECUTION_AUTHORIZED_AFTER_ZERO_FORWARD_EXTERNAL_SHA_PROVENANCE_CORRECTION`

This document corrects only the cross-platform external SHA256 definition in
the prior execution freeze.

No scientific design, implementation, test, population, intervention,
endpoint, forward budget, statistical family, runtime semantics, or
interpretation boundary is changed.

## Parent execution freeze

Prior execution freeze:

`9467db0d55670f0f6145d28885ccba60177b481b`

This correction-freeze commit must have parent exactly:

`9467db0d55670f0f6145d28885ccba60177b481b`

and must change exactly this correction document.

The resulting correction-freeze commit SHA is the only authorized scientific
execution HEAD.

The prior execution HEAD `9467db0...` must not be used for scientific
execution.

## Zero-forward bootstrap blocker

A fresh dedicated Kaggle checkout successfully reached exactly:

`9467db0d55670f0f6145d28885ccba60177b481b`

and successfully authenticated:

- execution-freeze commit scope;
- runner Git blob;
- test Git blob;
- runner SHA256.

Bootstrap then blocked on the test external SHA256 comparison before:

- runtime scientific preflight;
- checkpoint-backed model loading;
- scientific model forwards;
- raw artifact creation;
- primary inference;
- multiplicity correction.

Scientific model forwards:

`0`

Primary inference:

`False`

Scientific conclusion:

`None`

This blocker is infrastructure/provenance only and is not scientific evidence.

## Provenance defect

The prior freeze recorded the SHA256 of Windows working-tree test bytes after
line-ending conversion.

That value was:

`7059cd10ee45c876f153fe56c0942ea9772a26e35457ced4b31e438dc10f9d23`

It is not the canonical repository-file byte identity.

Git blob identity itself was never mismatched.

## Canonical byte-identity rule

For repository text files, the authoritative cross-platform external SHA256 is
defined as:

`SHA256(git cat-file blob <exact-git-blob-sha>)`

This hashes the exact bytes stored by Git and is independent of checkout
line-ending conversion.

An operating-system working-tree SHA256 is diagnostic only and must not replace
the canonical Git-blob-byte identity.

## Exact implementation identities

Runner:

`scripts/reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_fast_cuda.py`

Git blob:

`c208c01d3cff7fa80b44d94444df42b6cd0227be`

Canonical Git-blob-byte SHA256:

`c2dc56d91e114b4af011f53171156f2563b19fa20e365ae32b1c5da9ceda84fe`

Test:

`tests/test_reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_fast_cuda.py`

Git blob:

`266fe544e11215ba624407c0fab0d9c899fc6060`

Canonical Git-blob-byte SHA256:

`d48a005918ce0b910b88f7814df261a9d5c0c7bd036d4df19fc4bee53d86667f`

Both Git blob SHAs are unchanged from the implementation freeze.

No implementation file is modified by this correction.

## Scientific contract unchanged

Prospective design:

`8c0ff6dbad77ed876fc1481b3b53c3fd47a27d3b`

Static preparation:

`4a1d5871fad17a34951bc433a283c95e682a081b`

Implementation authority:

`25ff56c80ccd943bb386ebc7ab6612fe6e68d470`

Implementation freeze:

`466741783dd1fc9325e7d87a2d1a44ccb0a09de3`

Population:

`xg1_fact_2101..xg1_fact_2400`

Residual planes:

`[P1, P2, P4, P5]`

Conditions:

1. `native`
2. `p1_neutralized`
3. `p1_quarter_turn_replacement`
4. `p2_neutralized`
5. `p2_quarter_turn_replacement`
6. `p4_neutralized`
7. `p4_quarter_turn_replacement`
8. `p5_neutralized`
9. `p5_quarter_turn_replacement`

Canonical per-plane raw endpoint:

`D_SUF,k = Q0 - QC,k`

Scientific model forward budget:

`108000`

Baseline forwards:

`0`

Two independent GPU shards:

- GPU 0: `xg1_fact_2101..xg1_fact_2250`, `54000`
- GPU 1: `xg1_fact_2251..xg1_fact_2400`, `54000`

No DDP.

No NCCL.

## Runtime contract unchanged

The accepted Kaggle runtime remains:

- Python `3.12.13`
- NumPy `2.0.2`
- Torch `2.10.0+cu128`
- Transformers `5.0.0`
- tokenizers `0.22.2`
- kernels `0.10.2`
- CUDA runtime `12.8`
- two Tesla T4 GPUs

The previously authorized environment-only repair:

`python -m pip install --no-deps kernels==0.10.2`

remains authorized.

Exact frozen Mamba and causal-conv1d binary identities remain mandatory.

## Corrected preflight rule

Future Kaggle preflight must authenticate both:

1. exact Git blob SHA; and
2. SHA256 of the corresponding exact Git blob bytes.

It must not compare against a Windows CRLF working-tree SHA.

For the clean Linux checkout, ordinary file SHA256 is expected to equal the
canonical Git-blob-byte SHA256 above.

All other prior preflight gates remain unchanged.

## Command-capture boundary unchanged

`cm run save` captures clipboard bytes.

The clipboard must contain only the intended Kaggle Bash payload.

PowerShell orchestration must never be captured as the Kaggle command.

## Run identity

Authorized run-name template:

`g4k-residual-individual-plane-restoration-sufficiency-xg1-2101-2400-<correction-execution-short-sha>`

The run name is single-use.

## Statistical boundary unchanged

Raw execution computes no t statistics, p-values, Holm decisions, supported
plane sets, rankings, or scientific labels.

Only after successful collect/import and raw artifact validation may CPU-only
confirmatory inference compute exactly four one-sided Student t-tests on
`D_SUF,k`, one each for P1/P2/P4/P5, followed by Holm step-down at FWER `0.05`.

No fifth p-value is authorized.

## Current authorization

Design freeze: `YES`

Static preparation: `YES`

Implementation freeze: `YES`

Prior execution freeze: superseded for execution by this correction

Scientific forwards before this correction: `0`

Raw scientific execution at the exact correction-freeze commit produced by
this document: `YES`

Training: `NO`

Backward: `NO`

Task-head evaluation: `NO`

GPU statistical inference: `NO`

Scientific interpretation before validated import and later inference: `NO`

Commit/push of this correction document: manual only.