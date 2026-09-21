# ContraMamba Gen4 — Mamba-1.4B P5 Cross-Block Reference-FD One-Row Equivalence Execution Spec

## Status

`TECHNICAL_EXECUTION_AUTHORITY`

Parent implementation HEAD:

`f03cc01176674871b6bb9c2ef91baec7caec1e4d`

Authorized implementation:

`scripts/reason_router_gen4_mamba14b_p5_cross_block_reference_fd_one_row_equivalence.py`

Authorized implementation git blob:

`3dc94d833f81d829e81c85c0c3641fce861e8eb7`

Validated test:

`tests/test_reason_router_gen4_mamba14b_p5_cross_block_reference_fd_one_row_equivalence.py`

Validated test git blob:

`65f4f482b302264844ab054ee684ebe7dbaa6009`

Prospective program:

`reports/reason_router_gen4_next_mechanistic_program_prospective_freeze.md`

Program git blob:

`487bf844e27279392a4203c1dc70e5702e3ad325`

Prior reverse-over-reverse execution authority:

`reports/reason_router_gen4_mamba14b_p5_cross_block_reverse_over_reverse_one_row_execution_spec_candidate.md`

Prior authority git blob:

`ae42198e7c55a560aec0aa72bf770a02cdb225a8`

## Prior technical gate disposition

The authorized reverse-over-reverse run:

`g4k-mamba14b-p5-crossblock-ror-gate-560837b-2t4`

reached the intended double-backward estimator boundary and ended with:

`DOUBLE_BACKWARD_UNSUPPORTED`

The exact fast-CUDA selective-scan backward path did not support the required second reverse.

That failed run is not collected/imported and its identity is retired.

This is a technical backend/autodiff result only.

It is not a scientific result for cross-layer causal transport.

## Authorized question

Exactly one technical question is authorized:

> On the same frozen one-row block35-to-block36 map and canonical P5 plus/minus directions, does the fixed `epsilon=0.025` authenticated fast-CUDA symmetric finite-difference estimator agree with an exact differentiable reference implementation closely enough to authorize the estimator for later population transport?

This is an estimator-equivalence gate only.

## Frozen row and geometry

Use exactly:

- model: Mamba-1.4B;
- family: `xg2`;
- source pair: `xg2_fact_301`;
- cell: `C2_NAME`;
- anchor: `A_IDENTITY`;
- target offset: `+2`;
- source block: `35`;
- target block: `36`;
- source tensor: block35 `mixer.in_proj` content half;
- target tensor: block36 `mixer.in_proj` content half;
- ambient dimension: `4096`;
- source plane: canonical `P5`;
- directions: `plus`, `minus`.

No row, layer, token, plane, or direction search is allowed.

## Frozen model identity

Model repository:

`state-spaces/mamba-1.4b-hf`

Revision:

`6e46eae61c27280517feef46f536d16b91076f08`

Compact checkpoint:

`reports/reason_router_gen4_mamba14b_training_runs/g4k-mamba14b-train-g3d-seed181-dualt4-cache-ae6ab9a/selected_downstream_checkpoint.pt`

Checkpoint SHA256:

`915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a`

## Reference implementation

The reference local map uses the unfused differentiable semantics of:

- repository: `huggingface/transformers`;
- tag: `v5.0.0`;
- commit: `08810b1e278938278c50153ee1edfd7a20a759da`;
- source: `src/transformers/models/mamba/modeling_mamba.py`;
- source git blob: `ae80aa74f651f15a82bfc41ece60ba51ed5bb206`.

The implementation begins immediately after block35 `in_proj` and reproduces the v5.0.0 slow-path semantics through:

1. depthwise convolution;
2. activation;
3. x projection;
4. dt projection and softplus;
5. sequential selective-state recurrence;
6. D skip;
7. gate activation;
8. out projection;
9. block35 residual addition;
10. block36 RMSNorm;
11. block36 `in_proj`;
12. block36 target-token content-half readout.

The exact reference directional derivative is computed with:

`torch.func.jvp`

only on this unfused pure-PyTorch reference path.

## Fast estimator

Use the exact authenticated fast-CUDA local map already frozen by the prior gate.

For each P5 direction `v`:

`Jv_hat = [Phi_fast(x + epsilon v) - Phi_fast(x - epsilon v)] / (2 epsilon)`

with exactly:

`epsilon = 0.025`

No epsilon sweep is allowed.

No alternative epsilon is allowed.

No result-dependent epsilon choice is allowed.

## Frozen PASS thresholds

The reference and fast unperturbed local-map outputs must satisfy:

- `atol = 1e-4`;
- `rtol = 1e-4`.

For each of canonical P5 `plus` and `minus`, the fast finite-difference derivative must satisfy all of:

- cosine similarity `>= 0.9998`;
- norm ratio `fast/reference` in `[0.98, 1.02]`;
- relative L2 error `<= 0.02`.

These thresholds are frozen before execution.

No threshold relaxation after observing output is allowed.

A failure of any threshold closes this fixed finite-difference estimator route under the current program.

## Runtime

Use the frozen Gen4 fast-CUDA runtime:

- Python `3.12.13`;
- NumPy `2.0.2`;
- Torch `2.10.0+cu128`;
- Transformers `5.0.0`;
- CUDA runtime `12.8`;
- `kernels==0.10.2`;
- exactly `2 x Tesla T4`.

The fast side must use the exact authenticated causal-conv and selective-scan kernel identities already frozen by the Gen4 fast-CUDA program.

## Allowed execution

This gate may perform only:

- one baseline full-model boundary capture;
- two reference exact JVPs;
- one unperturbed fast local-map evaluation;
- four perturbed fast local-map evaluations;
- two fixed-epsilon central-difference vectors;
- technical equivalence calculations;
- rank diagnostics for the two reference vectors and two FD vectors;
- provenance and accounting.

## Allowed persisted outputs

The gate may persist:

- `reference_exact_p5_plus.f64le`;
- `reference_exact_p5_minus.f64le`;
- `fast_fd_p5_plus.f64le`;
- `fast_fd_p5_minus.f64le`;
- `reference_fd_equivalence_report.json`;
- `SHA256SUMS.txt`.

These are technical estimator-validation artifacts.

They are not population scientific evidence.

## Prohibited

This authority does not permit:

- XG1 Experiment-5 response access;
- population transport;
- XG2/XG4 population execution;
- adjacent P5 geometry access;
- principal angles;
- projector overlap;
- Procrustes alignment;
- transported-basis response;
- statistical testing;
- p-values;
- epsilon sweep;
- tolerance search;
- row/layer/token/plane search;
- training;
- parameter updates;
- scientific conclusion.

## PASS

Expected PASS label:

`PASS_MAMBA14B_P5_CROSS_BLOCK_ONE_ROW_REFERENCE_FD_EQUIVALENCE`

PASS requires:

1. exact frozen identities;
2. reference local-map JVP available for both P5 directions;
3. unperturbed reference/fast local-map equivalence within frozen primal tolerance;
4. both directional comparisons satisfy every frozen directional threshold;
5. no prohibited access or scientific metric;
6. no model parameter gradients or updates.

A PASS establishes only:

> the fixed `epsilon=0.025` fast-CUDA symmetric finite-difference estimator is technically validated against the frozen differentiable reference on the prospectively frozen one-row gate.

It does not establish cross-layer transport geometry.

## FAIL

If the reference exact JVP is unavailable, the gate fails technically.

If reference/fast primal equivalence fails, the gate fails.

If either P5 direction fails any frozen directional equivalence threshold, the gate fails.

No rescue epsilon, tolerance relaxation, alternate row, or alternate backend is authorized by this document.

A failed run is not collected/imported and its run identity is not reused.

## Authority boundary

`ONE_ROW_REFERENCE_FD_EQUIVALENCE_EXECUTION_AUTHORIZED = TRUE`

`POPULATION_TRANSPORT_EXECUTION_AUTHORIZED = FALSE`

`STATISTICAL_TESTING_AUTHORIZED = FALSE`

`TRAINING_AUTHORIZED = FALSE`

`SCIENTIFIC_CONCLUSION_AUTHORIZED = FALSE`
