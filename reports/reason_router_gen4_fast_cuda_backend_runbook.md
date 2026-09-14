# Gen4 Native-State Fast CUDA Runbook

## Purpose

This document records the exact fast-CUDA backend that was validated against the frozen Gen4 CPU slow/sequential native-state implementation.

Use this runbook for future **comparable Gen4 native-Mamba state observation/intervention workloads** so that the already-solved CUDA loader, kernel identity, recurrent-state reconstruction, and provenance problems are not rediscovered through long experiments.

This runbook does not authorize arbitrary replacement of a frozen CPU backend in unrelated studies. A new workload must preserve the same semantic state location or pass a task-appropriate equivalence gate before scientific promotion.

---

## Validated scientific equivalence

The validated bridge used:

- CPU reference HEAD:
  `8496ece911e0d461f0abbdf1a0fa619f8a2f22ab`
- fast-CUDA one-pair gate HEAD:
  `d6e1521c3f2c92f16aa88bbaf3f6b6c332a85ae6`
- fast-CUDA full backend-equivalence HEAD:
  `60f7485d37a7afb848b11b31a30242f5b89db534`
- bridge closure HEAD:
  `ad375c88385eef507e151f95717e951919a6a3fe`

The 300-pair CPU ↔ CUDA comparator passed with:

- max item geometry abs diff:
  `3.166055559500336e-06`
- max item PE abs diff:
  `1.6212262369252883e-06`
- baseline mean abs diff:
  `9.497736354335817e-09`
- max hypothesis-mean abs diff:
  `1.4363876632191572e-08`
- Holm decisions: identical
- final scientific outcome: identical

Frozen comparator tolerances were:

- geometry atol `1e-4`
- geometry rtol `1e-4`
- PE atol `1e-4`

This established backend equivalence for the exact observed/intervened quantities used by the Gen4 × K bridge.

---

## Known-good runtime

The validated GPU runtime was:

```text
Python       3.12.13
NumPy        2.0.2
PyTorch      2.10.0+cu128
Transformers 5.0.0
CUDA runtime 12.8
GPU          Tesla T4
Capability   7.5
kernels      0.10.2
```

Do not silently substitute package versions or a different CUDA build when reproducing this backend.

The frozen CPU state-observation runtime used `torch 2.10.0+cpu`; do not mutate that historical CPU environment merely to obtain CUDA.

---

## Exact CUDA kernel identity

### Mamba

Repository:

```text
kernels-community/mamba-ssm
```

Immutable revision:

```text
c8ffc584c147878a6eb978ae0e8db4d116c93a8c
```

Build variant:

```text
torch210-cxx11-cu128-x86_64-linux
```

Scientific CUDA `.so` SHA256:

```text
dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587
```

Required functions:

```text
selective_scan_fn
selective_state_update
mamba_inner_fn
```

### causal-conv1d

Repository:

```text
kernels-community/causal-conv1d
```

Immutable revision:

```text
f2651e776f66069cdcf842840db637583def1223
```

Build variant:

```text
torch210-cxx11-cu128-x86_64-linux
```

Scientific CUDA `.so` SHA256:

```text
6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6
```

Required functions:

```text
causal_conv1d_fn
causal_conv1d_update
```

Do not record an entire Hugging Face cache `blobs/` tree as the scientific kernel identity. Cache trees can contain unrelated blobs and generated `__pycache__` files. Bind the immutable Hub revision, build variant, and loaded scientific `.so` SHA256.

---

## Why the Transformers 5.0.0 default loader failed

Transformers 5.0.0 requests the Mamba Hub kernel at the historical revision:

```text
v0.0.4
```

For the validated Kaggle combination:

```text
torch210-cxx11-cu128-x86_64-linux
```

that pinned revision does **not** contain the required build.

The observed failure was:

```text
FileNotFoundError:
Kernel `kernels-community/mamba-ssm` at revision v0.0.4
does not have build: torch210-cxx11-cu128-x86_64-linux
```

This is not evidence that:

- Tesla T4 is unsupported;
- CUDA 12.8 is unsupported;
- PyTorch 2.10 CUDA is scientifically invalid.

The exact current immutable Mamba revision listed above contains the required build and was validated against CPU slow execution.

Do **not** solve this by repeatedly guessing package versions.

---

## Exact kernel load procedure

In an isolated Kaggle GPU session, if the `kernels` package is absent or wrong:

```bash
python -m pip install --no-deps "kernels==0.10.2"
```

Do this only in the GPU execution environment. Do not mutate the historical frozen CPU environment.

Load exact revisions:

```python
from kernels import get_kernel

mamba = get_kernel(
    "kernels-community/mamba-ssm",
    revision="c8ffc584c147878a6eb978ae0e8db4d116c93a8c",
)

conv = get_kernel(
    "kernels-community/causal-conv1d",
    revision="f2651e776f66069cdcf842840db637583def1223",
)
```

Then require:

```python
assert mamba.selective_scan_fn is not None
assert mamba.selective_state_update is not None
assert mamba.mamba_inner_fn is not None
assert conv.causal_conv1d_fn is not None
assert conv.causal_conv1d_update is not None
```

For Transformers 5.0.0 fast inference, the validated harness injects these exact functions into:

```python
transformers.models.mamba.modeling_mamba
```

before the CUDA scientific forwards.

Never use mutable Hub `main` as the recorded scientific backend identity.

---

## Interpreting the fast-path warning

During model construction, Transformers may print:

```text
The fast path is not available because one of
(selective_state_update, selective_scan_fn, causal_conv1d_fn,
causal_conv1d_update, mamba_inner_fn) is None.
Falling back to the sequential implementation...
```

This warning by itself does not establish which backend the later scientific forward used.

In the validated harness:

1. the model can be constructed while Transformers globals still contain the default unavailable kernel values;
2. the exact immutable kernels are then loaded and injected;
3. the model is moved to `cuda:0`;
4. the layer-17 fast scan is explicitly intercepted;
5. the fast-state reconstruction must be observed;
6. the one-pair equivalence gate must pass.

Therefore, judge backend validity from the explicit runtime/kernel/dispatch gates, not from the presence or absence of this construction-time warning.

If the scientific CUDA forward cannot prove the fast scan was intercepted, fail closed.

---

## Reconstructing the frozen recurrent state on CUDA

Do not use the internal chunk-level scan buffer as though it were the full tokenwise recurrent-state sequence.

The validated semantic reconstruction for frozen state `s_t` is:

1. capture layer-17 fast scan inputs;
2. run CUDA `selective_scan_fn(..., return_last_state=True)` through token `a`;
3. treat the returned last state as `s[a]`;
4. apply CUDA `selective_state_update` sequentially for:
   - `a+1`
   - `a+2`
   - `a+3`
   - `a+4`
5. use the reconstructed:
   - `s[a]`
   - `s[a+1]`
   - `s[a+2]`
   - `s[a+3]`
   - `s[a+4]`
   to calculate frozen `POST4_PATH_EFFICIENCY`.

This matches the frozen semantic state timing:

```text
post recurrent update
before C readout
```

The exact intervention remains the x branch:

```text
post in_proj
pre depthwise convolution
```

with the gate branch untouched.

---

## Mandatory equivalence gates

For a new comparable workload, do not begin with a long full run.

### Gate 1 — kernel arithmetic

Synthetic CUDA recurrence / convolution smoke:

- `selective_scan_fn` last state vs explicit recurrence;
- prefix scan state;
- four `selective_state_update` steps;
- causal-conv vs reference grouped convolution.

The validated bridge observed:

```text
full scan last state max abs diff   2.384185791015625e-07
POST4 reconstruction max abs diff  4.76837158203125e-07
causal conv max abs diff            0.0
```

### Gate 2 — bounded model equivalence

Run a canonical item first.

The validated bridge used:

```text
pair: orion_approval
CPU slow forwards: 8
CUDA fast forwards: 8
total: 16
```

Observed:

```text
max state abs diff     1.710955984890461e-06
max geometry abs diff  4.800659422730946e-07
max PE abs diff        2.871362522194332e-07
```

Only after the bounded gate passes may the full GPU run be authorized.

### Gate 3 — full backend comparison

After the full CUDA bundle is produced, compare the complete item population against the frozen CPU reference when a CPU reference exists.

Do not accept conclusion-only agreement.

Compare at minimum:

- provenance identity;
- item order/count;
- geometry;
- endpoints;
- causal reductions;
- manipulation residuals;
- summary means;
- Holm decisions;
- final outcome.

---

## Normal future Kaggle workflow

The bridge validation runs were manually executed and are retained as validated manual-execution evidence. **Do not copy that provenance shortcut into future work.**

Future authorized GPU runs must use the normal project chain:

```text
commit/push exact code
→ cm kaggle
→ run safe bootstrap
→ confirm exact full commit SHA
→ GPU ON in Kaggle web only when needed
→ copy the exact authorized command
→ cm run save <descriptive-run-name>
→ cm run <descriptive-run-name>
→ execute the generated pinned cell
→ cm collect <descriptive-run-name>
→ run the collector cell
→ download the handoff ZIP
→ cm import <handoff.zip>
→ verify IMPORT PASS
→ interpret
→ cm ship
→ explicitly stage only intended result/report files
```

Run names should identify stage and purpose, for example:

```text
gen4-native-fastcuda-<purpose>-<shortsha>
```

Do not use vague names such as `test`, `new`, `run1`, or `final`.

GPU control remains manual:

- GPU OFF for local/static/preflight work that does not need CUDA;
- GPU ON immediately before authorized GPU execution;
- GPU OFF / terminate the Kaggle session when GPU-dependent work ends.

---

## Repository bootstrap rules

Do not assume a historical Kaggle repo path still exists.

If the intended repo directory already exists:

1. verify it is a Git repo;
2. verify `origin`;
3. require a clean worktree;
4. fetch the exact branch;
5. fast-forward only;
6. verify the full 40-character HEAD.

Do not delete or overwrite an existing Kaggle repo merely because a bootstrap script expected a fresh path.

Use `cm kaggle fresh` only when destructive fresh bootstrap is actually authorized and prior outputs have already been recovered.

---

## Artifact validation rules

### File sets

Do not compare newline strings whose order depends on locale/sort behavior.

Use set equality for exact artifact surfaces.

Bad pattern:

```bash
[ "$OBSERVED_FILES" = "$EXPECTED_FILES" ]
```

when each side depends on independently sorted text.

Preferred pattern:

- enumerate file names;
- compare exact sets in Python;
- separately validate expected row counts.

### ZIP paths

ZIP member names always use POSIX `/` semantics.

When Python analyzes ZIP contents on Windows, use:

```python
PurePosixPath
```

not host `Path`, for archive member names.

This exact portability issue previously caused a Windows-only comparator failure.

### Checksums

Validate both:

1. outer handoff/bundle SHA256;
2. internal artifact checksums.

Do not weaken or bypass checksum failures.

---

## Failure handling

### `v0.0.4` missing build

Meaning:

- Transformers 5.0.0 historical default kernel revision lacks the exact torch/CUDA build.

Action:

- do not package-roulette;
- load the exact validated immutable revisions above;
- verify `.so` identities;
- rerun only the bounded equivalence gate before any full run.

### Hub current revision loads but historical pin fails

Meaning:

- infrastructure generation mismatch, not scientific equivalence.

Action:

- freeze an immutable Hub commit;
- freeze the actual loaded `.so`;
- synthetic smoke;
- bounded model equivalence;
- only then scientific execution.

### Exact kernels load but layer-17 fast scan is not intercepted

Meaning:

- fast dispatch is not proven.

Action:

- BLOCK;
- do not run full.

### Backend difference exceeds frozen tolerance

Meaning:

- backend sensitivity is present.

Action:

- do not force equivalence;
- keep CPU/CUDA evidence separate;
- report the discrepancy.

### Existing output path

Meaning:

- output collision.

Action:

- do not overwrite;
- use the authorized run identity and collection workflow to determine whether the existing result must be recovered.

### Dirty Kaggle repo

Action:

- stop;
- inspect/recover outputs if necessary;
- do not silently reset or clean.

---

## Validated reusable code

Reference implementation:

```text
scripts/reason_router_gen4_k_fast_cuda_one_pair_equivalence.py
scripts/reason_router_gen4_k_fast_cuda_full.py
scripts/reason_router_gen4_k_fast_cuda_full_compare.py
```

The exact bridge is closed and should not be reopened by tuning.

Reuse the backend implementation pattern, not the closed scientific hypothesis.

---

## Practical default

For a future comparable Gen4 native-state causal experiment:

1. design/freeze the scientific question without regard to backend outcome;
2. use fast CUDA as the preferred execution backend;
3. bind the exact runtime/kernel identities;
4. run the smallest sufficient equivalence gate;
5. if PASS, run the authorized GPU experiment through the normal `cm` provenance chain;
6. if FAIL, treat it as backend sensitivity and stop promotion.

This is the default path unless a new authority explicitly requires CPU slow execution.
