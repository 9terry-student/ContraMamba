# Gen4 PP3-Excluded Residual Aggregate Restoration Sufficiency — Execution Freeze

## Status

`SCIENTIFIC_RAW_EXECUTION_AUTHORIZED_AT_EXACT_EXECUTION_FREEZE_HEAD`

This document authorizes exactly one raw scientific execution of the frozen
aggregate residual restoration-sufficiency experiment after implementation
validation.

It does not authorize confirmatory inference, statistical conclusion,
training, backward, task-head evaluation, logits analysis, implementation
modification, endpoint modification, or exploratory analysis.

## Exact parent requirement

This execution-freeze commit must have exact parent:

`ae0bb32ed0afcfb6d3f5cc5d83f079ee71375b68`

and must change exactly this execution-freeze document.

The resulting execution-freeze commit SHA is the only authorized scientific
execution HEAD for this run.

Scientific execution must not occur at the implementation commit itself.

## Frozen chain

Prospective design:

`14d71b742488dd088ca22c416962522235d8d67b`

Static preparation:

`7744ddabe05e0180428d3e753a4453dea531ef6a`

Implementation authority:

`ca4a0cd25e6fe21b79ac8c545d68ec60e26fefcd`

Implementation:

`ae0bb32ed0afcfb6d3f5cc5d83f079ee71375b68`

## Exact implementation identity

Runner:

`scripts/reason_router_gen4_pp3_excluded_residual_aggregate_restoration_sufficiency_fast_cuda.py`

Runner Git blob:

`47afa603cc940a363a5f5d4f1d3ca52be283893a`

Runner canonical Git-blob-byte SHA256:

`d17d6c8f0622f965bf050dc7f62ce6f95245a3429ed8adaa09ce9c38e40cb757`

Test:

`tests/test_reason_router_gen4_pp3_excluded_residual_aggregate_restoration_sufficiency_fast_cuda.py`

Test Git blob:

`eb83d068001b272241746ae5b5931d8c4bd67fd6`

Test canonical Git-blob-byte SHA256:

`d3a4a1dc3d59d3dd4dff5fc375a12f807b88e92d442de88c95b22098bde0c2`

Implementation authority blob:

`aacf3d68804d85640372f1522138ca4f84104c6b`

No implementation-file modification is authorized after this freeze.

## Implementation validation

`python -m py_compile`:

`PASS`

Targeted pytest:

`11 passed`

Scientific model forwards during implementation validation:

`0`

Checkpoint-backed scientific execution:

`0`

GPU scientific execution:

`0`

Primary inference executed:

`False`

Restoration semantics source gate:

`PASS`

The implementation validation initially produced one synthetic-test failure.

That failure was not a scientific model execution and did not modify the
runner.

The failing test intentionally used an extreme floating-point scale to
demonstrate that:

`Q0 - Q_C`

and:

`(Q0 - Q_B) - (Q_C - Q_B)`

need not be bitwise identical.

The test incorrectly invoked the normal-scale endpoint validator on that
deliberately pathological synthetic example.

The repair removed only that final validator invocation from the pathological
test case.

The runner remained unchanged.

The normal-scale endpoint test continues to exercise the full validator.

Final targeted suite:

`11 passed`

Scientific model forwards across both validation attempts:

`0`

## Frozen population

Fresh prospective XG1:

`xg1_fact_2401..xg1_fact_2700`

Pair count:

`300`

Rows:

`1800`

GPU 0 shard:

`xg1_fact_2401..xg1_fact_2550`

Pairs:

`150`

Scientific forwards:

`18000`

GPU 1 shard:

`xg1_fact_2551..xg1_fact_2700`

Pairs:

`150`

Scientific forwards:

`18000`

No DDP.

No NCCL.

## Frozen residual geometry

Residual subspace:

`R = P1 ⊕ P2 ⊕ P4 ⊕ P5`

For native branch-local state `h`:

`c_R = sum_k (a_k p_k+ + b_k p_k-)`

`r_R = sum_k (-b_k p_k+ + a_k p_k-)`

Conditions are exactly:

1. `native`
2. `residual_neutralized`
3. `residual_quarter_turn_replacement`

Neutralized background:

`B = h - c_R`

Exact native restoration:

`B + c_R = h`

Matched restoration replacement:

`C = B + r_R`

Therefore:

`C = h - c_R + r_R`

and the direct correction relative to native is exactly:

`delta_C = -c_R + r_R`

The old aggregate-necessity correction:

`-r_R`

is forbidden for this experiment.

## Frozen endpoint

`Q = E_XG2 - E_XG4`

with exactly five frozen XG2 directions and five frozen XG4 directions.

Finite-difference epsilon:

`0.025`

Per pair:

`Q0 = Q(native)`

`Q_B = Q(residual_neutralized)`

`Q_C = Q(residual_quarter_turn_replacement)`

Aggregate native-restoration gain:

`S_R = Q0 - Q_B`

Matched replacement gain:

`S_C = Q_C - Q_B`

Canonical primary endpoint:

`D_RES_SUF = Q0 - Q_C`

The runner stores the canonical endpoint directly using `Q0-Q_C`.

The algebraically equivalent expanded form may only be checked within the
frozen numerical tolerance.

## Exact scientific budget

Directions per condition:

`10`

Scientific model forwards per direction:

`4`

Scientific model forwards per condition:

`40`

Conditions:

`3`

Scientific forwards per pair:

`120`

Pairs:

`300`

Total scientific model forwards:

`36000`

Baseline model forwards:

`0`

No extra scientific forward is authorized.

## Frozen static provenance

Source SHA256:

`eb77056732740f501066026a3b65ea3522cd203916828f9d8d56a6e746c79a87`

Rows SHA256:

`de175b9817e6b589f4580247775adf760ece7a5b929c59dc28bdad6a7fe763e7`

Structural manifest SHA256:

`85ff2f3f47c9b104f7e20529e520f5d5b294a14398764dba2b0bc7c3e7a1a0b8`

Geometry manifest SHA256:

`d0d43655e3fe5f432a7878ed46a526c6bde4ba220b78d75aedafad4a256bc2e6`

Tokenizer anchor manifest SHA256:

`d57f55af9ebe5ccb83982c58fb803952cd37ec4d9f40ad1cf2ffc8da2c47fc0f`

Tokenizer eligibility summary SHA256:

`7052785940b10699eeffad4dea5bc4b0877d856ae3adbdd5844ec77e956d0527`

Preparation manifest SHA256:

`b58ca182fe66b91017ce008e56ea8bd68d4d595b8a265ac4a6958afe481ced02`

Static preparation script canonical SHA256:

`449180e53d61c35766b7b121ce2adf79474ffe3878c3ea255973c21319012606`

Tokenizer eligibility:

`PASS_300_OF_300`

## Frozen external model/checkpoint identity

Model:

`state-spaces/mamba-130m-hf`

Revision / snapshot:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Representative checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Representative checkpoint bytes:

`518270455`

## Accepted runtime

Execution runtime is frozen to the already validated native-Mamba CUDA stack:

- Python: `3.12.13`
- NumPy: `2.0.2`
- Torch: `2.10.0+cu128`
- Transformers: `5.0.0`
- tokenizers: `0.22.2`
- kernels: `0.10.2`
- CUDA runtime: `12.8`
- GPU topology: exactly `2 x Tesla T4`

Frozen Mamba binary SHA256:

`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

Frozen causal-conv1d binary SHA256:

`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

The only authorized environment repair is:

`python -m pip install --no-deps kernels==0.10.2`

No dependency upgrade, environment-wide reinstall, or alternative kernel build
is authorized.

## Kernel-binding orchestration requirement

External zero-forward preflight may verify:

- exact Mamba binary identity;
- exact causal-conv1d binary identity;
- exact `kernels==0.10.2`;
- successful `load_exact_fast_kernels()`;
- exact kernel transport status.

External zero-forward preflight must not invoke:

`validate_transformers_kernel_bindings(kernels)`

before model construction.

The frozen runner owns the valid ordering:

1. exact kernel load;
2. exact Transformers lazy-loader context;
3. model construction;
4. checkpoint validation;
5. runtime-component validation;
6. lazy-loader call verification;
7. exact Transformers kernel-binding validation;
8. move model to device;
9. scientific forwards.

Any failure before step 9 implies zero valid scientific forwards.

## Required pre-execution gates

Before the raw runner begins:

- exact execution-freeze HEAD;
- clean repository;
- exact runner Git blob;
- exact runner canonical SHA256;
- exact test Git blob;
- exact test canonical SHA256;
- output directory does not already exist;
- accepted runtime versions;
- exactly two Tesla T4 GPUs;
- exact model snapshot;
- exact tokenizer snapshot;
- exact checkpoint bytes and SHA256;
- exact static artifact hashes;
- tokenizer eligibility `PASS_300_OF_300`;
- exact residual-vector identities;
- exact frozen XG2/XG4 bases;
- exact Mamba binary SHA256;
- exact causal-conv1d binary SHA256;
- exact kernel transport identity.

Any mismatch fails closed.

## Raw artifact boundary

The raw runner must produce exactly:

- `300` canonical item rows;
- exact pair order `2401..2700`;
- exact condition order;
- exact ten-direction order;
- exactly `36000` scientific forwards;
- exactly `0` baseline forwards;
- `Q0`;
- `Q_B`;
- `Q_C`;
- `S_R`;
- `S_C`;
- `D_RES_SUF`;
- aggregate intervention audits;
- two-shard provenance;
- manifest;
- checksums.

Raw runner must record:

`primary_inference_executed = false`

`multiplicity_correction_executed = false`

`training_executed = false`

`backward_executed = false`

`task_heads_executed = false`

`logits_read = false`

`scientific_conclusion = null`

## Statistical boundary

Raw execution must not compute:

- t statistics;
- p-values;
- scientific support labels;
- individual-plane tests;
- interaction tests;
- additive-decomposition tests;
- exploratory subgroup statistics.

After successful collect/import and provenance validation only, CPU-only
confirmatory inference may execute exactly one test:

`H0: mean(D_RES_SUF) <= 0`

against:

`H1: mean(D_RES_SUF) > 0`

using a one-sample one-sided Student t-test with:

- `N = 300`;
- `df = 299`;
- alpha `0.05`;
- confirmatory p-value count exactly `1`;
- multiplicity correction: none.

Positive support additionally requires:

- `mean(Q0) > 0`;
- `mean(S_R) > 0`;
- `mean(D_RES_SUF) > 0`;
- one-sided `p < 0.05`.

## Authorized run identity

Run-name template:

`g4k-residual-aggregate-restoration-sufficiency-xg1-2401-2700-<execution-freeze-short-sha>`

Run name is single-use.

A failed run name must never be reused.

## Current authorization

Design freeze: `YES`

Static preparation: `YES`

Implementation authority: `YES`

Implementation validation: `YES`

Raw scientific execution at the exact resulting execution-freeze commit:
`YES`

Confirmatory inference during raw run: `NO`

Training: `NO`

Backward: `NO`

Task-head evaluation: `NO`

Implementation modification: `NO`

Commit/push of this execution-freeze document: manual only.