# Gen4 PP3-Excluded Residual Individual-Plane Necessity Localization — Retry2 Execution Freeze

## Status

`RETRY2_SCIENTIFIC_EXECUTION_AUTHORIZED_AFTER_ZERO_FORWARD_RUNTIME_PACKAGE_FAILURE`

This document authorizes one retry of the already frozen raw scientific
execution after retry1 failed during zero-forward runtime preflight because the
required `kernels` Python package metadata was absent.

No scientific design, static input, implementation, endpoint, population,
condition, multiplicity, or interpretation change is authorized.

## Parent retry1 freeze

Parent retry1 execution-freeze commit:

`fceffc87571116f0007dd6a7ef0346fd6f4d6883`

The retry2-freeze commit must have this commit as its parent and must change
exactly this retry2-freeze document.

The scientific retry2 execution HEAD is the resulting retry2-freeze commit SHA.

## Retry1 failure provenance

Failed run name:

`g4k-residual-individual-plane-necessity-xg1-1801-2100-fceffc8-retry1`

Expected and actual repository commit:

`fceffc87571116f0007dd6a7ef0346fd6f4d6883`

Pinned command SHA256:

`c38c8736c5a11ee1577e6d884748f4b3a34756b658ee5676544a93ca5977fa2b`

Started UTC:

`2026-09-18T07:41:45Z`

Finished UTC:

`2026-09-18T07:42:07Z`

Exit code:

`1`

Failure class:

`KERNELS_PACKAGE_MISSING_DURING_ZERO_FORWARD_RUNTIME_GATE`

Observed failure:

`importlib.metadata.PackageNotFoundError: No package metadata was found for kernels`

Frozen runtime gate then raised:

`KernelCompatibilityError: KERNELS_PACKAGE_MISSING`

The failure occurred inside the preflight call to
`runtime_gate_for_device(runtime, gpu_id)` before:

- `PREFLIGHT_RESULT=PASS`;
- checkpoint-backed model load;
- scientific model forwards;
- raw artifact creation;
- primary inference;
- multiplicity correction.

Scientific model forward count for retry1:

`0`

Primary inference executed:

`False`

Multiplicity correction executed:

`False`

Scientific conclusion:

`None`

Retry1 remains execution/provenance failure only, not scientific evidence.

## Authorized runtime repair

Before retry2 raw execution, the Kaggle Python environment may install exactly:

`kernels==0.10.2`

This is an environment repair only.

It does not authorize installing or upgrading Torch, Transformers, NumPy, CUDA,
Mamba, causal-conv1d, or any other scientific dependency.

The repair must not modify the git worktree.

After installation, preflight must verify:

`importlib.metadata.version("kernels") == "0.10.2"`

The package version alone is not sufficient scientific identity.

The already frozen kernel compatibility layer must additionally load and
authenticate the exact frozen kernel binary identities:

Mamba scientific binary SHA256:

`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

causal-conv1d scientific binary SHA256:

`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

If exact binary identity cannot be established, retry2 is blocked.

## Retry2 identity

Authorized run name:

`g4k-residual-individual-plane-necessity-xg1-1801-2100-<retry2-execution-short-sha>-retry2`

The exact short SHA is derived from the retry2-freeze commit.

The original run and retry1 run names are single-use and must not be reused.

## Scientific contract unchanged

Fresh population:

`xg1_fact_1801..xg1_fact_2100`

Residual planes:

`[P1, P2, P4, P5]`

Conditions:

1. `native`
2. `p1_neutralized`
3. `p1_quarter_turn_control`
4. `p2_neutralized`
5. `p2_quarter_turn_control`
6. `p4_neutralized`
7. `p4_quarter_turn_control`
8. `p5_neutralized`
9. `p5_quarter_turn_control`

Primary raw endpoint for each plane:

`D_k = QC,k - QN,k`

Total scientific model forward budget:

`108000`

Baseline model forwards:

`0`

Two-GPU topology:

- GPU 0: `xg1_fact_1801..xg1_fact_1950`, `54000` forwards
- GPU 1: `xg1_fact_1951..xg1_fact_2100`, `54000` forwards

No DDP.

No NCCL.

## Retry2 preflight

Retry2 must fail closed before any scientific forward unless all of the
following pass:

- exact retry2 execution HEAD;
- clean repository worktree;
- `kernels==0.10.2`;
- exact accepted Python/NumPy/Torch/Transformers/CUDA runtime;
- two Tesla T4 GPUs;
- exact frozen static-input identities;
- runner Git blob `9cc0f41b24953d94f870aa3e07b993fafd3d83ea`;
- test Git blob `f81c9a97fc4579773e521261708042722e0b1cd6`;
- runner SHA256
  `56b919e54c8cef1d2220abe9652ffa48a77b42d00d28a64890f0ee60807a176d`;
- test SHA256
  `cc935c0366c8b192510159ae65a464ae3eda703581e00659e42178a63fdef410`;
- frozen residual vectors;
- frozen XG2/XG4 bases;
- exact frozen Mamba binary SHA256;
- exact frozen causal-conv1d binary SHA256;
- exact tokenizer eligibility;
- exact representative checkpoint SHA;
- new retry2 output directory absent.

Preflight must report:

`SCIENTIFIC_MODEL_FORWARD_COUNT=0`

`PRIMARY_INFERENCE_EXECUTED=False`

`MULTIPLICITY_CORRECTION_EXECUTED=False`

## Raw execution boundary

The retry2 raw runner remains forbidden from computing:

- t statistics;
- p-values;
- Holm decisions;
- adjusted p-values;
- supported-plane sets;
- family-level scientific labels;
- plane rankings.

A successful retry2 must end with:

`PRIMARY_INFERENCE_EXECUTED=False`

`MULTIPLICITY_CORRECTION_EXECUTED=False`

`SCIENTIFIC_CONCLUSION=None`

## Later inference boundary

Only after successful collect/import and raw artifact validation may a separate
CPU-only confirmatory inference compute exactly four one-sided Student t-test
p-values, one for P1/P2/P4/P5, followed by the frozen Holm step-down procedure
at familywise alpha `0.05`.

No fifth p-value is authorized.

## Failure handling

If retry2 fails before a valid raw artifact is completed:

- preserve retry2 run provenance;
- do not overwrite or reuse retry2;
- do not interpret partial outputs scientifically;
- diagnose the failure before further retry authorization.

## Current authorization

Design freeze: `YES`

Static preparation: `YES`

Implementation freeze: `YES`

Original execution freeze: `YES`

Original run scientific forwards: `0`

Retry1 scientific forwards: `0`

Exact `kernels==0.10.2` environment repair: `YES`

Retry2 raw scientific execution at exact retry2-freeze commit: `YES`

Training: `NO`

Backward: `NO`

Task-head evaluation: `NO`

GPU statistical inference: `NO`

Commit / push of this retry2-freeze document: manual only.
