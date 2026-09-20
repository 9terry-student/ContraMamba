# ContraMamba Gen4 — Mamba-1.4B P5 Cross-Block One-Row JVP Technical Execution Spec

## Status

`TECHNICAL_EXECUTION_AUTHORITY_CANDIDATE`

This specification authorizes exactly one bounded technical feasibility execution
of the already implemented one-row forward-JVP gate.

It does **not** authorize a population transport study, principal-angle analysis,
projector-overlap analysis, Procrustes analysis, statistical inference, model
training, or any scientific conclusion.

Implementation parent:

`bf901817abbbf67e179080f99b699bf89861fe6c`

Frozen implementation blob:

`scripts/reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_feasibility.py`

`db523f66d9a05297077af18b547b4f2b2c9379ea`

Frozen test blob:

`tests/test_reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_feasibility.py`

`b58cd546f6cdcf0fa7e3e8a9c953048b19468199`

Required feasibility audit blob:

`reports/reason_router_gen4_mamba14b_p5_cross_block_jvp_transport_feasibility_audit_candidate.md`

`f461b14d34ca58f8cce3711795dc7bd0cab3cd1a`

## Authorized gate identity

Exactly one response-free row:

- family: `XG2`
- source pair: `xg2_fact_301`
- contrast cell: `C2_NAME`
- anchor: `A_IDENTITY`
- target offset: `+2`
- source site: block-35 `mixer.in_proj` content half
- target site: block-36 `mixer.in_proj` content half
- source directions: frozen canonical `P5_plus` and `P5_minus`

No other row, plane, token, block, offset, or direction may be substituted or
searched.

## Authorized estimator

The only authorized estimator is the implemented exact analytic forward-mode:

`torch.func.jvp`

applied to the block-35-content to block-36-content local map defined by the
frozen gate implementation.

The implementation must use the already frozen exact project CUDA kernel
functions for:

- causal convolution;
- selective scan.

No finite difference, reverse-over-forward reconstruction, slow-backend fallback,
CPU fallback, alternate kernel bundle, or estimator substitution is authorized
by this specification.

If forward-mode AD is unsupported by the exact frozen fast-CUDA path, the gate
must fail technically and stop.

Such a failure is **not** a scientific failure of the cross-layer transport
hypothesis.

## Frozen model/runtime identity

Model:

`state-spaces/mamba-1.4b-hf`

revision:

`6e46eae61c27280517feef46f536d16b91076f08`

Compact downstream checkpoint:

`reports/reason_router_gen4_mamba14b_training_runs/g4k-mamba14b-train-g3d-seed181-dualt4-cache-ae6ab9a/selected_downstream_checkpoint.pt`

checkpoint SHA256:

`915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a`

The runtime gate inherited from the frozen Mamba-1.4B geometry implementation
must pass exactly.

Because that inherited runtime gate requires the original two-GPU topology,
the technical execution must use Kaggle with exactly the already validated
`2 x Tesla T4` environment, even though the gate itself executes on logical GPU 0.

No GPU is needed for bootstrap or collection.

## Authorized execution accounting

The gate may perform:

- one baseline full-model forward for boundary capture;
- two local forward-JVP evaluations, one for canonical `P5_plus` and one for
  canonical `P5_minus`.

The gate must report:

- `baseline_full_model_forward_count = 1`;
- `local_jvp_count = 2`;
- `scientific_model_forward_count = 0`;
- `scientific_inference_count = 0`;
- `p_value_count_added = 0`.

No training, optimizer step, parameter gradient, or parameter update is authorized.

## Authorized outputs

The gate may write only its bounded technical output directory containing:

- `jvp_feasibility_report.json`;
- `transported_p5_plus.f64le`;
- `transported_p5_minus.f64le`;
- `SHA256SUMS.txt`.

The two transported vectors are technical feasibility artifacts only.

This execution must not compute or write:

- adjacent P5 vectors;
- principal angles;
- projector overlap;
- Procrustes alignment;
- Experiment-5 `D_CAN`;
- Experiment-5 `D_ADJ`;
- paired specificity `S`;
- any p-value;
- any population aggregate;
- any scientific conclusion.

## PASS condition

Technical PASS requires all of the following:

1. repository identity and frozen blobs authenticate;
2. exact frozen runtime/model/checkpoint authenticate;
3. the fixed response-free row is loaded exactly;
4. both canonical P5 ambient directions are valid unit directions;
5. `torch.func.jvp` succeeds for both directions;
6. both transported vectors are finite and nonzero;
7. the transported two-vector span has numerical rank `2`;
8. no parameter gradient is created;
9. no scientific transport comparison is performed.

Expected result label:

`PASS_MAMBA14B_P5_CROSS_BLOCK_ONE_ROW_JVP_FEASIBILITY`

## FAIL handling

Any runtime, provenance, exact-kernel, forward-AD, nonfinite, zero-vector, or
rank-deficiency failure stops this gate.

A failed run must not be collected/imported as successful evidence and its run
identity must not be reused.

No fallback estimator or backend may be attempted under this specification.

## Post-PASS boundary

A technical PASS establishes only that the intended local cross-block JVP is
computable on the one frozen response-free row.

It does not establish:

- cross-layer alignment;
- cross-layer rotation;
- preservation of the canonical causal plane;
- explanation of the Experiment-5 sign reversal.

A separate prospective scientific design is required before any population
transport measurement or adjacent-P5 comparison.

`ONE_ROW_TECHNICAL_EXECUTION_AUTHORIZED = TRUE`

`POPULATION_TRANSPORT_EXECUTION_AUTHORIZED = FALSE`

`STATISTICAL_TESTING_AUTHORIZED = FALSE`

`TRAINING_AUTHORIZED = FALSE`

`SCIENTIFIC_CONCLUSION_AUTHORIZED = FALSE`
