# ContraMamba K0-RVG Layer-22 Four-Tap Convolution Runtime-Preflight Authority Candidate

## 1. Phase

This document authorizes only a bounded runtime instrumentation preflight for the frozen layer-22 four-tap convolution decomposition implementation.

It does **not** authorize the full scientific execution.

## 2. Frozen authorities

Static-design authority commit:

`7e9e4f5b4ebf0676568822d3aa855278c8c1f192`

Static-design SHA256:

`781b08bc5f2e2d812f10d6bf8f464fc38bbc8a13f738ad791c6f9c7edb870d6d`

Implementation commit:

`3dd3791cab718421fd30ef119d44d3d2a4defde9`

Runner path:

`scripts/longterm_k0_rvg_layer22_four_tap_convolution_decomposition_audit.py`

Runner SHA256:

`bc48b1bcb222dbf75828ad61fb61c0d766a5024e2456d19c8ccee6ccdfcbe088`

Parent evidence freeze commit:

`dabb9422dbf0e111319828cf027e8dc5d82fe326`

Branch:

`longterm-k-series-native-state-kinematics`

## 3. Authorized runtime action

Exactly one invocation of the runner in `--runtime-preflight` mode is authorized.

The preflight may:

- load the frozen model/checkpoint/runtime lineage;
- access exactly one fixed pair-role row selected by the runner's preflight path;
- execute exactly two model forwards, one matched and one swapped;
- authenticate the actual layer-22 four-tap depthwise-convolution kernel;
- verify all four causal lag mappings;
- reconstruct the direct `delta_C` boundary from the four tap terms;
- check squared-norm, interaction-normalization, and addition-factor identities;
- check the frozen parent `H_RF` / `C` boundary bridge for the exercised row;
- report scalar residuals, kernel RMS values, and exact-zero tap flags.

## 4. Explicit prohibitions

The runtime preflight must not:

- run the full 336-item / 672-pair-role scientific execution;
- emit scientific population metrics or scientific conclusions;
- create a scientific run directory or evidence artifacts;
- persist raw vectors;
- invoke a tokenizer;
- read logits or task heads;
- train or evaluate a model;
- perform a causal intervention;
- perform PCA, SVD, whitening, learned probes, or learned geometry;
- perform post-hoc layer, lag, channel, item, or window search;
- touch the unrelated K1 files;
- use Kaggle or GPU.

## 5. Required pass conditions

The runtime preflight is valid only if all of the following hold:

- `model_forward_count = 2`;
- `scientific_population_accessed = True`;
- `scientific_evidence_emitted = False`;
- `raw_vectors_persisted = False`;
- source layer is exactly `22`;
- convolution is authenticated as depthwise width `1536`, kernel size `4`;
- all four lag-to-kernel-index mappings are reported;
- no layer-23 lag-3 zero-tap assumption is imported;
- actual layer-22 exact-zero tap status is reported from the authenticated kernel;
- four-tap reconstruction relative residual is `<= 2e-5`;
- squared-norm closure satisfies the frozen implementation tolerance;
- interaction-normalization closure satisfies the frozen implementation tolerance;
- addition-factor identity satisfies the frozen implementation tolerance;
- parent boundary reconstruction/bridge checks pass;
- no unexpected worktree drift exists beyond the two pre-existing unrelated K1 untracked files.

## 6. Stop conditions

Stop immediately and do not proceed to full execution if:

- the implementation commit or runner SHA256 mismatches;
- the static-design authority identity mismatches;
- parent evidence identities mismatch;
- the runtime preflight performs other than exactly two forwards;
- scientific evidence is emitted;
- raw vectors are persisted;
- any algebraic/reconstruction gate fails;
- any unrelated tracked worktree change exists;
- either K1 untracked file changes state.

## 7. Boundary after PASS

A runtime-preflight PASS establishes only that the frozen implementation is instrumented consistently enough for a later execution decision.

It does **not** itself establish the scientific claim and does **not** authorize the full 1344-forward observational execution.

A separate execution authorization decision is required after reviewing the preflight output.
