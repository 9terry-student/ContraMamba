# Gen4 PP3 Necessity — Implementation Authority

## Status

`IMPLEMENTATION_AUTHORIZED_NO_SCIENTIFIC_EXECUTION`

This authority permits only the bounded implementation and CPU/static validation
needed to realize the frozen PP3 necessity design.

It does not authorize scientific model execution, checkpoint loading, CUDA
execution, Kaggle execution, primary inference, or scientific interpretation.

## Frozen authority chain

Necessity design:

`f4419f1e7efbeeb9e50e84b67accc56422580cf2`

Design document blob:

`0dacef464eb5802d121de2db0ab2bfcfd5281c98`

Static preparation freeze:

`e470f132a37c731754feb4333dcb2e48e29af53b`

The implementation must treat the above design and static preparation as fixed.

## Frozen population

Data root:

`data/reason_router_gen4_xg1_necessity_v1/`

Population:

- generator family: XG1
- source pairs: `xg1_fact_601..xg1_fact_900`
- N = 300
- six-cell rows = 1800

Frozen identities:

- source SHA256:
  `49bec37150630d31bb5f502f49ef23ffc9a75bb93e079127c8c430aae3da6abd`
- rows SHA256:
  `e03534599c07201e371eb07938d8492d22a30de39dcbbb3c0c22300a4ff94224`
- structural manifest SHA256:
  `219994e7ec757148ffcae30a7d50357e347530e06cecbb618db619b73bbe8b76`

Tokenizer/anchor preparation:

- result: `PASS_300_OF_300`
- anchor manifest SHA256:
  `67d74cd2b3ab8daa8065227b1e85ea125ca88af4de64c872e5d47f60d9c40ec2`
- tokenizer revision:
  `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

No pair substitution or post-response exclusion is allowed.

## Frozen geometry

PP3+ SHA256:

`66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`

PP3- SHA256:

`ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

PP5+ SHA256:

`7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2`

PP5- SHA256:

`311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855`

Frozen static audits:

- PP3 Gram max residual:
  `1.3322676295501878e-15`
- PP5 Gram max residual:
  `1.1102230246251565e-15`
- PP3-vs-PP5 cross-plane max absolute dot:
  `2.723515857283587e-16`
- coefficient-transfer max L2 mismatch:
  `1.7763568394002505e-15`

Implementation must not recompute, rotate, select, or replace the scientific
planes from response data.

## Frozen broad endpoint

Reuse the already validated XG2/XG4 top-5 basis reconstruction.

Frozen Phase-1 plan SHA256 values:

- XG2:
  `b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c`
- XG4:
  `792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f`

For condition `c`:

`E_XG2(c) = (1/5) * sum_j J(v_XG2,j;c)^2`

`E_XG4(c) = (1/5) * sum_j J(v_XG4,j;c)^2`

`Q(c) = E_XG2(c) - E_XG4(c)`

Finite-difference epsilon remains exactly:

`0.025`

No endpoint redesign is allowed.

## Authorized implementation files

Exactly two new implementation files are authorized:

1. `scripts/reason_router_gen4_pp3_necessity_fast_cuda.py`
2. `tests/test_reason_router_gen4_pp3_necessity_fast_cuda.py`

No existing tracked file may be modified.

In particular, do not modify:

- `scripts/reason_router_gen4_k_directional_alignment_transport_runtime.py`
- `scripts/reason_router_gen4_k_directional_alignment_transport_runner.py`
- existing family-subspace or holdout runners
- frozen data
- frozen static preparation artifacts
- frozen PP3/PP5 vector artifacts

If implementation cannot be completed without changing an existing tracked
runtime file, stop and return to authority review.

## Required runner dependency reuse

The runner must reuse, without modifying:

- frozen XG2/XG4 basis reconstruction from
  `reason_router_gen4_family_subspace_sensitivity_fast_cuda.py`;
- validated model/runtime setup from the existing fast-CUDA runners;
- existing path-efficiency definition;
- existing target token and layer-17 intervention coordinate;
- existing `parent.capture_branch`;
- existing exact kernel/runtime gates.

The new runner may define its own temporary layer-17 `in_proj` hook.

It must not duplicate or replace the full model-forward stack.

## State-conditioned hook semantics

For each target branch forward, let `h` be the unmodified native strong-channel
vector read from the layer-17 `in_proj` output at the frozen target token before
any intervention.

`h` has dimension 395.

For every forward compute:

`a = <h, pp3_plus>`

`b = <h, pp3_minus>`

### Condition 0

No condition correction:

`delta_condition = 0`

### Condition 3

PP3 neutralization:

`delta_condition = -a * pp3_plus - b * pp3_minus`

Before adding the finite-difference probe, the implementation must audit:

`<h + delta_condition, pp3_plus> ~= 0`

and:

`<h + delta_condition, pp3_minus> ~= 0`

within the frozen numerical tolerance.

### Condition 5

PP5 coefficient-transfer control:

`delta_condition = -a * pp5_plus - b * pp5_minus`

The coefficients must be the PP3 coefficients from the same native `h`.

The implementation must never derive coefficients from native PP5 projections.

## Signed finite-difference probe

For orientation `o in {-1,+1}`, frozen unit direction `v`, and branch sign:

- target-plus branch sign = `+1`
- target-minus branch sign = `-1`

the direct correction applied to the strong channels must be:

`delta_total = delta_condition + branch_sign * o * epsilon * v`

The implementation must apply `delta_total` directly.

It must not pass `delta_total` through the existing `0.5/-0.5` intervention
scaling again.

Exactly one model forward is allowed per branch for this operation.

## Hook isolation requirements

The custom hook must verify:

- exact target token only;
- strong x channels only;
- gate half unchanged exactly;
- non-strong x channels unchanged exactly;
- all other tokens unchanged exactly;
- correction finite;
- applied-correction residual within existing runtime cast tolerance.

The hook must be installed only around the single existing
`parent.capture_branch(..., delta_h=None)` invocation and removed in `finally`.

The shared runtime's `install_inproj_hook` must not be monkey-patched.

## Condition-matching audits

For the same native branch state, PP3 treatment and PP5 control use the same
`(a,b)` coefficients.

The runner must expose sufficient raw audit fields to validate:

- PP3 condition-correction L2;
- PP5-control condition-correction L2;
- absolute L2 mismatch;
- PP3 post-neutralization residual projections;
- native PP3 coefficients `a,b`.

The item validator must require the treatment/control condition-correction L2
mismatch to remain within the frozen execution tolerance.

No magnitude rescaling is allowed.

## Measurement order

For every pair use fixed condition order:

1. `native`
2. `pp3_neutralized`
3. `pp5_coefficient_control`

Within every condition use fixed direction order:

1. XG2 basis 0
2. XG2 basis 1
3. XG2 basis 2
4. XG2 basis 3
5. XG2 basis 4
6. XG4 basis 0
7. XG4 basis 1
8. XG4 basis 2
9. XG4 basis 3
10. XG4 basis 4

Within each direction evaluate orientation `+1` before `-1`.

No response-guided ordering is allowed.

## Per-pair raw quantities

The runner must record:

- `Q0`
- `Q3`
- `Q5`
- `A3 = Q0 - Q3`
- `A5 = Q0 - Q5`
- `D_NEC = A3 - A5`
- algebraic identity `D_NEC = Q5 - Q3`

For each condition it must retain enough raw direction-probe data to reconstruct
`E_XG2`, `E_XG4`, and `Q`.

All values must be finite.

## Exact forward budget

Per direction:

- positive orientation: 2 branch forwards
- negative orientation: 2 branch forwards
- total: 4

Per condition:

- 10 basis directions
- 40 forwards

Per pair:

- 3 conditions
- 120 forwards

Full population:

- 300 pairs
- exact scientific forward budget: `36000`

Baseline model forward count this run:

`0`

No extra forward is authorized for native-state capture.

## Raw observation boundary

GPU execution, when separately authorized later, is observation-only.

The runner must report:

- `scientific_model_forward_count_this_run = 36000`
- `baseline_model_forward_count_this_run = 0`
- `primary_inference_executed = false`
- `multiplicity_correction_executed = false`
- `training_executed = false`
- `backward_executed = false`
- `task_heads_executed = false`
- `logits_read = false`
- `scientific_conclusion = null`

The runner must not import or call scipy statistical tests.

## Expected output artifacts

The runner must produce exactly:

1. `pp3_necessity_items.jsonl`
2. `pp3_necessity_summary.json`
3. `artifact_manifest.json`
4. `SHA256SUMS.txt`

The validator must reconstruct all endpoint identities from the raw probe data
and fail closed on any mismatch.

## Required CPU/static tests

The authorized test file must cover at least:

1. exact frozen constants and artifact identities;
2. exact population `601..900`;
3. exact forward-budget algebra `120/pair`, `36000/total`;
4. condition order and direction order;
5. synthetic orthonormal PP3/PP5 coefficient-transfer L2 equality;
6. synthetic PP3 neutralization residual;
7. condition 5 uses PP3-derived coefficients, not PP5 projections;
8. direct correction formula including branch sign/orientation/epsilon;
9. hook changes only target-token strong x channels;
10. gate channels remain exact;
11. non-strong channels and other tokens remain exact;
12. `Q0/Q3/Q5/A3/A5/D_NEC` algebra;
13. artifact validator rejects altered endpoint algebra;
14. raw observation boundary fields remain false/null;
15. no scipy/t-test/inference implementation in the GPU runner.

Tests must use synthetic tensors/mocks only.

They must not:

- load the representative checkpoint;
- instantiate the scientific model;
- run CUDA;
- execute model forwards;
- observe scientific endpoint values.

## Validation before implementation freeze

Required:

`python -m py_compile scripts/reason_router_gen4_pp3_necessity_fast_cuda.py`

and:

`python -m pytest -q tests/test_reason_router_gen4_pp3_necessity_fast_cuda.py`

plus a static boundary audit confirming:

- checkpoint load count = 0
- model forward count = 0
- CUDA scientific execution = false
- no output scientific artifact directory created

## Prohibited implementation changes

Do not introduce:

- training;
- backward passes;
- optimizers;
- task-head evaluation;
- logits reads;
- alternative planes;
- alternative controls;
- epsilon sweep;
- layer/token/checkpoint sweep;
- response-guided selection;
- rescue analyses;
- subgroup analyses;
- statistical inference.

## Execution authority

A passing implementation and test suite does not authorize scientific
execution.

After implementation is separately frozen and pushed, an execution freeze must
bind the exact runner/test identities and exact execution command before Kaggle
GPU use.
