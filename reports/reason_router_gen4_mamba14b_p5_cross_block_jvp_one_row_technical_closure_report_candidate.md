# ContraMamba Gen4 — Mamba-1.4B P5 Cross-Block One-Row JVP Technical Gate Closure

## Status

`TECHNICAL_GATE_CLOSED_FAIL_FORWARD_JVP_UNSUPPORTED`

Execution authority HEAD:

`62daec8b43ea68c1722ea19b48b5d1cf44e0fcbd`

Gate implementation:

`scripts/reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_feasibility.py`

Failed technical run:

`g4k-mamba14b-p5-crossblock-jvp-gate-62daec8-2t4-retry2`

Pinned run command SHA256:

`1a06621dfb481934c866272d3d056e80d5e83a36bb207ee2936177d384b3257d`

Exit code:

`1`

## What executed successfully

Before the failure:

- exact execution HEAD matched `62daec8b43ea68c1722ea19b48b5d1cf44e0fcbd`;
- the authenticated Mamba-1.4B snapshot path existed;
- the frozen one-row gate script started;
- the Mamba-1.4B backbone checkpoint weights were materialized;
- execution reached the first `torch.func.jvp` call for the frozen block35-to-block36 local map.

The failure therefore occurred after environment/snapshot/model provisioning and at the intended forward-JVP feasibility boundary.

## Exact blocker

The first local JVP entered the exact frozen fast-CUDA causal-convolution wrapper and failed in the custom `torch.autograd.Function` before a transported vector was produced.

Observed runtime error:

`RuntimeError: In order to use an autograd.Function with functorch transforms (vmap, grad, jvp, jacrev, ...), it must override the setup_context staticmethod.`

The gate reclassified that exception as:

`FORWARD_JVP_UNSUPPORTED`

The failing operator was the authenticated causal-conv1d fast-CUDA path used by:

`kernels["causal_conv1d_fn"]`

inside the frozen local block35-to-block36 map.

## Technical conclusion

The currently frozen exact fast-CUDA operator stack does **not** support the authorized direct `torch.func.jvp` transport path.

Result:

`DIRECT_FAST_CUDA_TORCH_FUNC_JVP_FEASIBILITY = FAIL`

This is a technical backend/autodiff compatibility result.

It is not evidence that the block35 canonical P5 plane fails to transport to block36.

## Scientific boundary

No transported P5 vector was produced.

Therefore this run establishes none of the following:

- principal angles;
- projector overlap;
- Procrustes alignment;
- transported-plane rotation;
- transported-plane preservation;
- explanation of the adjacent-site causal sign reversal;
- any population effect;
- any p-value;
- any scientific conclusion.

`SCIENTIFIC_CONCLUSION = NONE`

## Execution/accounting boundary

The failed run is not collected or imported.

The failed run identity is single-use and must not be reused.

No fallback estimator was executed under the failed authority.

No finite difference, slow-backend fallback, CPU fallback, alternate kernel bundle, or estimator substitution was performed.

## Next technical question

A new bounded technical gate is required before any scientific transport study.

The preferred next candidate is an **exact autodiff fallback feasibility check**, distinct from `torch.func.jvp`, that asks whether the same local Jacobian-vector product can be obtained without changing:

- model/checkpoint;
- source/target sites;
- row/token/directions;
- exact frozen kernel bytes;
- scientific quantity.

Any fallback must be prospectively frozen and independently validated before population transport execution.

No scientific population execution is authorized by this closure.

`POPULATION_TRANSPORT_EXECUTION_AUTHORIZED = FALSE`

`STATISTICAL_TESTING_AUTHORIZED = FALSE`

`TRAINING_AUTHORIZED = FALSE`
