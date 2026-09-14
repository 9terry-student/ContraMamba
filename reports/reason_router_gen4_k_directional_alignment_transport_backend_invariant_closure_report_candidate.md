# Gen4 × K Directional-Alignment Transport — Backend-Invariant Closure Report Candidate

## Status

**CLOSED — DIRECTIONAL ALIGNMENT CAUSAL TRANSPORT NOT ESTABLISHED**

This report closes the bounded Gen4 × K directional-alignment transport question tested on the frozen Gen4 layer-17 `POST4_PATH_EFFICIENCY` phenotype.

The result is negative under both the frozen CPU slow/sequential backend and the separately validated fast-CUDA backend. The CUDA execution is itemwise equivalent to the CPU reference within the pre-frozen backend-equivalence tolerances, and the inferential decisions and final outcome agree exactly.

## Scientific question

Test whether K's independently causal directional-alignment → current-write/post-state mechanism class transports to the frozen Gen4 layer-17 `DELTA_NAME / POST4_PATH_EFFICIENCY` phenotype under the frozen four-branch factorial bridge.

This report does **not** address or establish:

- behavioral mediation;
- output-failure or hallucination causation;
- universality across tasks, layers, anchors, endpoints, or checkpoints;
- absence of directional-alignment effects outside this frozen bridge;
- superiority or inferiority of CPU vs CUDA as a general execution backend.

## Frozen bridge design

- Gen4 scientific parent: `a738b4c169d30b0a7e21563c3aba2c29c481c456`
- K causal parent: `5f5f4d6a80085ad8baf43445475d1c3049535c22`
- target pair: `C2_NAME - C0_SHAM`
- reference pair: `C5_TITLE_NAME - C1_TITLE`
- source / target / intervention roles: layer `15 → 16 → 17`
- relative intervention coordinate: `A_IDENTITY + 2`
- target branches satisfy frozen `A_IDENTITY == A_NAME`
- intervention site: layer-17 x branch, post-`in_proj`, pre-depthwise-conv
- fixed strong set: 395 channels
- downstream endpoint: frozen layer-17 `POST4_PATH_EFFICIENCY`
- family:
  1. `mean(R_ALIGN) > 0`
  2. `mean(R_ALIGN - R_MAG) > 0`
- correction: Holm-2, alpha `0.05`
- support required both one-sided hypotheses to reject in the positive direction with provenance, baseline reproduction, and manipulation checks passing.

No endpoint, layer, offset, or channel scan was performed.

## CPU reference execution

Execution identity:

- HEAD: `8496ece911e0d461f0abbdf1a0fa619f8a2f22ab`
- source pairs: 300
- model forwards: 2400
- backend: CPU sequential/slow Mamba
- CPU reference ZIP SHA256:
  `25a9e6a862f7c1ad7c272d85cb5000ba8542cca2c8976b785021e5e4159ebcaf`

Validation:

- full 2400-forward execution completed;
- independent full-bundle validator: PASS;
- exact 300-item artifact surface recovered and verified;
- frozen baseline mean reproduced exactly;
- all mandatory manipulation checks passed.

Frozen baseline:

- observed mean: `-0.013998957453394283`
- frozen mean: `-0.013998957453394283`
- maximum baseline reproduction residual: `0.0`

H1 — `R_ALIGN > 0`:

- n: 300
- mean: `-7.34395589648736e-06`
- sample SD: `9.20270596965561e-05`
- t(299): `-1.3822135340630874`
- one-sided raw p: `0.9160311013709048`
- Holm-adjusted p: `1.0`
- dz: `-0.07980213559688674`
- reject: **false**

H2 — `ALIGNMENT_SPECIFICITY > 0`:

- n: 300
- mean: `-2.888124477688825e-05`
- sample SD: `0.00016675783638366254`
- t(299): `-2.9997860624860326`
- one-sided raw p: `0.9985355055456305`
- Holm-adjusted p: `1.0`
- dz: `-0.17319272906875985`
- reject: **false**

CPU outcome:

`DIRECTIONAL_ALIGNMENT_CAUSAL_TRANSPORT_NOT_ESTABLISHED`

Both estimated means are opposite in sign to the pre-specified positive transport hypotheses.

## Fast-CUDA backend validation

Exact fast backend was established separately before the full run.

Runtime/backend identity:

- Python `3.12.13`
- NumPy `2.0.2`
- PyTorch `2.10.0+cu128`
- Transformers `5.0.0`
- GPU: Tesla T4, capability `(7, 5)`
- `kernels==0.10.2`
- Mamba kernel revision:
  `c8ffc584c147878a6eb978ae0e8db4d116c93a8c`
- Mamba CUDA binary SHA256:
  `dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`
- causal-conv revision:
  `f2651e776f66069cdcf842840db637583def1223`
- causal-conv CUDA binary SHA256:
  `6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

Kernel-level synthetic validation:

- selective-scan final-state max abs diff: `2.384185791015625e-07`
- POST4 state reconstruction max abs diff: `4.76837158203125e-07`
- causal-conv max abs diff: `0.0`

One-pair CPU-slow ↔ CUDA-fast gate:

- gate commit: `d6e1521c3f2c92f16aa88bbaf3f6b6c332a85ae6`
- canonical pair: `orion_approval`
- CPU forwards: 8
- CUDA forwards: 8
- max state abs diff: `1.710955984890461e-06`
- max geometry abs diff: `4.800659422730946e-07`
- max PE abs diff: `2.871362522194332e-07`
- result: `PASS_FAST_CUDA_ONE_PAIR_EQUIVALENCE`

## CUDA full execution

Execution identity:

- HEAD: `60f7485d37a7afb848b11b31a30242f5b89db534`
- source pairs: 300
- GPU model forwards: 2400
- CPU model forwards in full CUDA run: 0
- CUDA bundle SHA256:
  `816cc586f8c6a1eed8e730937f30032befbcb6b40b59bd56b4d763a0cc2177c5`

Execution result:

- `PASS_FAST_CUDA_FULL_EXECUTION`
- exact artifact surface: PASS
- item count: 300
- backend outcome:
  `DIRECTIONAL_ALIGNMENT_CAUSAL_TRANSPORT_NOT_ESTABLISHED`

The full CUDA run was explicitly marked backend-equivalence-only and not as an independent scientific authority.

## Full CPU ↔ CUDA itemwise equivalence

Comparator report:

- file: `reports/reason_router_gen4_k_directional_alignment_transport_backend_equivalence_report.json`
- SHA256: `ca7f787b1ce3ec6450bc71e0aa159c08e11677f36de3c9cac0ba329756f7f893`
- schema: `gen4-k-fast-cuda-full-equivalence-report-v1`
- result: `PASS_FAST_CUDA_FULL_BACKEND_EQUIVALENCE`

Observed full-population differences:

- max item geometry abs diff:
  `3.166055559500336e-06`
- max item PE abs diff:
  `1.6212262369252883e-06`
- baseline mean abs diff:
  `9.497736354335817e-09`
- max hypothesis-mean abs diff:
  `1.4363876632191572e-08`
- Holm decisions match: `true`
- CPU outcome equals CUDA outcome: `true`
- backend-invariant negative result: `true`

Pre-frozen comparator tolerances:

- geometry atol: `1e-4`
- geometry rtol: `1e-4`
- PE atol: `1e-4`

The observed discrepancies are comfortably below the frozen backend-equivalence limits.

## Scientific conclusion

Under the exact frozen Gen4 × K bridge tested here, the K directional-alignment causal mechanism **does not establish transport** to the Gen4 layer-17 `DELTA_NAME / POST4_PATH_EFFICIENCY` phenotype.

The negative conclusion is not attributable to the CPU slow backend: an exact fast-CUDA implementation reproduced the 300-item geometry/endpoints within frozen numerical tolerances and produced the same Holm decisions and final outcome.

The most defensible closure statement is therefore:

> K's frozen strong-side directional-alignment causal effect does not transport, under this pre-specified four-branch mapping and endpoint, to the frozen Gen4 layer-17 NAME `POST4_PATH_EFFICIENCY` phenotype.

This is a falsification of the tested transport hypothesis, not a universal claim that directional alignment is irrelevant to Gen4 or to other Mamba mechanisms.

## Provenance classification

These full CPU and CUDA executions were run from exact committed code and produced internally checksummed artifacts whose outer ZIP SHA256 identities are recorded above. The CPU bundle also passed the independent full validator, and the two bundles passed a 300-item independent backend comparator.

However, the executions were performed manually in Kaggle rather than through the registered:

`cm run → cm collect → cm import`

handoff chain.

Therefore:

- classify these as **validated manual-execution evidence**;
- do not claim a canonical `cm import` handoff that did not occur;
- do not retroactively fabricate or bypass run-registry provenance;
- retain the two raw ZIP files externally under the recorded SHA256 identities;
- this closure report and comparator report are the repository-visible evidence anchors.

No rerun is required solely to manufacture missing workflow metadata for this already falsified bridge claim.

## Backend consequence

For this exact Gen4 state-observation/intervention workload, fast CUDA has now been demonstrated to preserve the scientific quantities used by the bridge within the frozen equivalence contract.

This supports using the exact validated fast-CUDA backend for future comparable experiments **only when**:

1. the same semantic recurrent-state observation can be reconstructed;
2. the exact kernel/runtime identity is bound;
3. a task-appropriate equivalence gate is passed before scientific promotion.

It does not grant unrestricted replacement of frozen CPU backends for unrelated experiments.

## Closure

Bridge status: **CLOSED / NEGATIVE**

No additional tuning, endpoint scanning, layer scanning, or post-hoc reformulation is authorized under this bridge question.

Any next scientific study should begin as a new question rather than modifying this failed transport hypothesis after observing its outcome.
