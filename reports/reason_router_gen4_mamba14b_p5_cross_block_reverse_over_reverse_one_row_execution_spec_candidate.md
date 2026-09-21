# ContraMamba Gen4 — Mamba-1.4B P5 Cross-Block Reverse-over-Reverse One-Row Execution Spec

## Status

`TECHNICAL_EXECUTION_AUTHORITY`

Parent implementation HEAD:

`4e40337cc8c4a1803aaf752e3c9173705215db44`

Authorized implementation:

`scripts/reason_router_gen4_mamba14b_p5_cross_block_reverse_over_reverse_one_row_feasibility.py`

Authorized implementation git blob:

`47525ce1d6bf7706ba0ea3ecd0d00ba3cb89e634`

Validated test:

`tests/test_reason_router_gen4_mamba14b_p5_cross_block_reverse_over_reverse_one_row_feasibility.py`

Validated test git blob:

`13705ca9d9b7ecab3a2ff6ca7f5f7e515bb134a5`

Prospective program:

`reports/reason_router_gen4_next_mechanistic_program_prospective_freeze.md`

Program git blob:

`487bf844e27279392a4203c1dc70e5702e3ad325`

Direct forward-JVP technical closure:

`reports/reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_technical_closure_report_candidate.md`

Closure git blob:

`eba094c517b38d8730eb9b8362f9fa6f69b295b5`

## Authorized question

Exactly one technical question is authorized:

> Can the already frozen row-conditioned block35-to-block36 local map produce the same mathematical `Jv` quantity for canonical P5 plus/minus using reverse-over-reverse exact autodiff on the authenticated fast-CUDA path?

This is a technical feasibility gate only.

## Frozen execution identity

Model:

`state-spaces/mamba-1.4b-hf`

Revision:

`6e46eae61c27280517feef46f536d16b91076f08`

Compact checkpoint:

`reports/reason_router_gen4_mamba14b_training_runs/g4k-mamba14b-train-g3d-seed181-dualt4-cache-ae6ab9a/selected_downstream_checkpoint.pt`

Checkpoint SHA256:

`915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a`

Frozen row:

- family: `xg2`
- source pair: `xg2_fact_301`
- cell: `C2_NAME`
- anchor: `A_IDENTITY`
- target offset: `+2`

Frozen transport:

- source block: `35`
- target block: `36`
- source tensor: block35 `mixer.in_proj` content half
- target tensor: block36 `mixer.in_proj` content half
- ambient dimension: `4096`
- source plane: canonical `P5`
- directions: `plus`, `minus`

Estimator:

`grad_w(v^T grad_x(w^T Phi(x))) = Jv`

## Runtime

Use the already frozen Gen4 fast-CUDA runtime:

- Python `3.12.13`
- NumPy `2.0.2`
- Torch `2.10.0+cu128`
- Transformers `5.0.0`
- CUDA `12.8`
- `kernels==0.10.2`
- exactly `2 x Tesla T4`

The exact authenticated causal-conv and selective-scan kernel path must be used.

## Allowed outputs

The gate may report only technical feasibility quantities:

- whether both exact `Jv` vectors are obtainable;
- finite/nonfinite status;
- transported vector norms;
- numerical rank and its rank-only singular values;
- exact runtime/model/kernel provenance;
- forward/reverse accounting.

## Prohibited

This authority does not permit:

- population transport execution;
- XG1 Experiment-5 response access;
- principal angles;
- projector overlap;
- Procrustes analysis;
- adjacent P5 comparison;
- transported-basis response measurement;
- finite difference;
- direct `torch.func.jvp`;
- slow/CPU fallback;
- alternate kernel bundle;
- epsilon search;
- layer/token/plane/row search;
- statistical testing;
- p-values;
- training;
- parameter updates;
- scientific conclusion.

## PASS

Technical PASS requires:

1. exact frozen identities;
2. both canonical P5 directions produce finite exact transported vectors;
3. transported span numerical rank is `2`;
4. no parameter gradient remains on model parameters;
5. no unauthorized fallback or scientific metric is executed.

Expected PASS label:

`PASS_MAMBA14B_P5_CROSS_BLOCK_ONE_ROW_REVERSE_OVER_REVERSE_FEASIBILITY`

## FAIL

If the exact fast-CUDA operator stack does not support the required double backward, the gate fails technically and stops.

That failure is not a scientific negative result for cross-layer causal transport.

No fallback estimator is authorized by this document.

A failed run is not collected/imported and its run identity is not reused.

## Authority boundary

`ONE_ROW_REVERSE_OVER_REVERSE_TECHNICAL_EXECUTION_AUTHORIZED = TRUE`

`POPULATION_TRANSPORT_EXECUTION_AUTHORIZED = FALSE`

`STATISTICAL_TESTING_AUTHORIZED = FALSE`

`TRAINING_AUTHORIZED = FALSE`

`SCIENTIFIC_CONCLUSION_AUTHORIZED = FALSE`
