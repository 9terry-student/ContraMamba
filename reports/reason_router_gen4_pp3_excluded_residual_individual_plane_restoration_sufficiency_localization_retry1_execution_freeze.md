# Gen4 PP3-Excluded Residual Individual-Plane Restoration Sufficiency Localization — Retry1 Execution Freeze

## Status

`RETRY1_SCIENTIFIC_EXECUTION_AUTHORIZED_AFTER_ZERO_FORWARD_PREFLIGHT_ORCHESTRATION_FAILURE`

This document authorizes one retry of the already frozen raw
restoration-sufficiency execution.

No scientific design, implementation, population, intervention, endpoint,
forward budget, statistical family, runtime identity, or interpretation
boundary is changed.

## Parent authority

Parent correction execution freeze:

`90ea0f3d12a520ea6a28ebd9cae787d19152f77b`

This retry1-freeze commit must have that commit as its exact parent and must
change exactly this retry1 document.

The resulting retry1-freeze commit SHA is the only authorized retry1
scientific execution HEAD.

## Failed attempt provenance

Failed run:

`g4k-residual-individual-plane-restoration-sufficiency-xg1-2101-2400-90ea0f3`

Expected and actual execution commit:

`90ea0f3d12a520ea6a28ebd9cae787d19152f77b`

Pinned command SHA256:

`7833edfc21eef0ecf8203e8e44e801c2d630f6f631f82eee7542b48440df6e38`

Started UTC:

`2026-09-18T10:23:43Z`

Finished UTC:

`2026-09-18T10:24:48Z`

Exit code:

`1`

Failure class:

`ZERO_FORWARD_PREFLIGHT_TRANSFORMERS_BINDING_VALIDATION_ORDER_ERROR`

Observed exception:

`KernelCompatibilityError: TRANSFORMERS_KERNEL_BINDING:mamba_ssm,causal_conv1d`

The failure occurred in the external orchestration preflight before invocation
of the frozen raw runner.

Scientific model forwards:

`0`

Raw scientific artifact created:

`NO`

Primary inference executed:

`False`

Multiplicity correction executed:

`False`

Scientific conclusion:

`None`

The failed run name is single-use and must not be reused.

## Diagnosed orchestration defect

`load_exact_fast_kernels()` authenticates and loads the exact frozen Mamba and
causal-conv modules and patches the direct Transformers kernel functions.

At that point the Transformers module attributes:

`mamba_ssm`

and

`causal_conv1d`

are not yet required to be bound to those exact module objects.

The frozen runner establishes those module bindings through the exact
Transformers lazy-loader path during model construction:

1. `load_exact_fast_kernels()`
2. `exact_transformers_kernel_loader(kernels)`
3. `load_representative_model_external(...)`
4. `validate_runtime_components(model)`
5. verify exact lazy-loader calls
6. `validate_transformers_kernel_bindings(kernels)`
7. move model to device
8. begin scientific forwards

The failed external preflight incorrectly executed
`validate_transformers_kernel_bindings(kernels)` immediately after step 1.

That standalone check therefore tested a state that the frozen runner does not
claim should already exist.

## Authorized retry1 orchestration correction

Retry1 may remove only that premature standalone call to:

`validate_transformers_kernel_bindings(kernels)`

from the external zero-forward preflight.

The external preflight must still verify:

- exact retry1 HEAD;
- clean repository;
- exact runner/test Git blobs;
- canonical Git-blob-byte SHA256 values;
- no output collision;
- `kernels==0.10.2`;
- accepted Python/NumPy/Torch/Transformers/tokenizers/CUDA runtime;
- two Tesla T4 GPUs;
- exact static inputs;
- tokenizer eligibility `PASS_300_OF_300`;
- exact residual vectors and XG2/XG4 bases;
- exact checkpoint bytes and SHA256;
- exact frozen Mamba binary SHA256;
- exact frozen causal-conv1d binary SHA256;
- `load_exact_fast_kernels()` succeeds;
- kernel transport status is
  `EXACT_FROZEN_BINARY_SHA256_MATCH`.

The frozen raw runner itself remains unchanged and must perform its existing
model-construction-time exact lazy-loader and Transformers binding validation
before any scientific model forward.

Any failure in that frozen runner validation blocks execution before valid
scientific evidence exists.

## Scientific contract unchanged

Population:

`xg1_fact_2101..xg1_fact_2400`

Residual planes:

`[P1, P2, P4, P5]`

Conditions:

`native`

`p1_neutralized`
`p1_quarter_turn_replacement`

`p2_neutralized`
`p2_quarter_turn_replacement`

`p4_neutralized`
`p4_quarter_turn_replacement`

`p5_neutralized`
`p5_quarter_turn_replacement`

Canonical per-plane endpoint:

`D_SUF,k = Q0 - QC,k`

Total scientific model forwards:

`108000`

Baseline forwards:

`0`

GPU 0:

`xg1_fact_2101..xg1_fact_2250`, `54000`

GPU 1:

`xg1_fact_2251..xg1_fact_2400`, `54000`

No DDP.

No NCCL.

## Implementation identity unchanged

Runner Git blob:

`c208c01d3cff7fa80b44d94444df42b6cd0227be`

Runner canonical SHA256:

`c2dc56d91e114b4af011f53171156f2563b19fa20e365ae32b1c5da9ceda84fe`

Test Git blob:

`266fe544e11215ba624407c0fab0d9c899fc6060`

Test canonical SHA256:

`d48a005918ce0b910b88f7814df261a9d5c0c7bd036d4df19fc4bee53d86667f`

No implementation-file modification is authorized.

## Retry1 identity

Authorized run-name template:

`g4k-residual-individual-plane-restoration-sufficiency-xg1-2101-2400-<retry1-short-sha>-retry1`

The failed original run name must not be reused.

Retry1 is single-use.

## Raw statistical boundary

The raw runner remains forbidden from calculating:

- t statistics;
- p-values;
- Holm decisions;
- adjusted p-values;
- supported-plane sets;
- plane rankings;
- scientific labels.

Only after successful collect/import and provenance validation may the frozen
CPU-only four-test Holm confirmatory inference be performed.

## Current authorization

Design freeze: `YES`

Static preparation: `YES`

Implementation freeze: `YES`

Correction execution freeze: `YES`

Failed original execution scientific forwards: `0`

Retry1 raw execution at exact retry1-freeze commit: `YES`

Training: `NO`

Backward: `NO`

Task-head evaluation: `NO`

GPU statistical inference: `NO`

Implementation modification: `NO`

Commit/push of this retry1 document: manual only.