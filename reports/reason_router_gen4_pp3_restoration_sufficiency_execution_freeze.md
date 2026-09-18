# Gen4 PP3 Restoration Sufficiency — Execution Freeze

## Status

`EXECUTION_READY`

This is the sole scientific execution freeze for the prospectively frozen
PP3 restoration-sufficiency experiment on fresh XG1 `901..1200`.

It does not alter the scientific question, population, frozen geometry,
intervention, matched PP5 replacement, broad endpoint, finite-difference
epsilon, primary hypothesis, decision rule, no-rescue policy, or
interpretation boundary.

Scientific execution authorized by this document is raw observation only.
Statistical inference and scientific interpretation remain prohibited until
successful collection/import and provenance/artifact validation.

## Frozen authority chain

Restoration-sufficiency design:

`854bcd5585512155776c34e46846817c54b75cf6`

Design Git blob:

`f58931df229a09705edb94346c23c0dd9e67e799`

Static preparation:

`0f907574f1ca25ec12e35573a83e1b499ed1b53b`

Implementation authority:

`cae566ed458e5c6f93c86ce03b029950035652dc`

Implementation-authority Git blob:

`090b68503c3f10327065bc275f7b6daffdccfc59`

Implementation freeze:

`3b0154439aac51e1199cfa18842818430b8406c7`

Branch:

`gen4-k-xg2-basis-holdout`

Scientific execution is authorized only from the clean pushed descendant HEAD
that contains this execution-freeze document and preserves every frozen
identity below.

That post-freeze full 40-character commit SHA is the sole valid
`--expected-head` for the concrete execution command.

## Exact implementation identity

Scientific runner:

`scripts/reason_router_gen4_pp3_restoration_sufficiency_fast_cuda.py`

Git blob at implementation freeze:

`527235cb9ae1ad0b5417f3dcc2a72de259b09a09`

File SHA256:

`887482f1415acd211fe689550380f8389ccd4b48a6483f6db4506c6372617656`

CPU/static test:

`tests/test_reason_router_gen4_pp3_restoration_sufficiency_fast_cuda.py`

Git blob at implementation freeze:

`01a72c424bcf21eba020b497ae54252d7e16af95`

File SHA256:

`ab652a253ead3d57fa7741e014075d03201f47e2d6ad89bd77422ff64b8fa99f`

Pre-freeze validation:

- `9 passed in 6.70s`
- `py_compile` PASS
- checkpoint loads: `0`
- scientific model forwards: `0`
- CUDA scientific execution: `false`
- training: `false`
- backward: `false`
- primary inference: `false`

No repeat of these validations is required merely to create this freeze.

## Frozen scientific population

Exact population:

`xg1_fact_901..xg1_fact_1200`

Cardinality:

- source pairs: `300`
- six-cell rows: `1800`

Static preparation result:

`PASS_PP3_RESTORATION_SUFFICIENCY_STATIC_PREPARATION`

Frozen structural identities:

- source facts SHA256:
  `2c700452d818531c46a8ffd473eb6d64d8af29f3a9284da8371f5c9eb2610c21`
- six-cell rows SHA256:
  `7ec2ea86f35562394244f6df6e8b098ea3ba8a9bf358868fc614f5029746241c`
- structural manifest SHA256:
  `a1c6957d7d48fb93b53f7b93bb5378caf0bbdc0adb96c438f6aa56a779ae9922`
- tokenizer anchor manifest SHA256:
  `dc4f2cd4ca2806249467407c7c980411d8fa02051418f9a8b625c1b1c4756253`
- tokenizer eligibility summary SHA256:
  `0fb8da67687f223b6c72ea7bc946542e6543bb85c050163056696a8e945e5a89`
- geometry manifest SHA256:
  `4fdb7778738db109138199ab2e9725cfb7db3cb2f173a88b96a045c595c17da7`
- static preparation manifest SHA256:
  `19fce1d109b8ab2d8e7faddd9ea5b2f5571e0ea5f1c2c30927a4d7a10e0e061c`

Prospective overlap against all previously used XG1 `001..900`:

- pair-ID overlap: `0`
- claim overlap: `0`
- evidence overlap: `0`
- exact claim/evidence-row overlap: `0`

No outcome-dependent exclusions, substitutions, or pair replacement are allowed.

## Frozen tokenizer / anchor identity

Eligibility:

`PASS_300_OF_300`

Anchor rows:

`1800`

Tokenizer revision:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Tokenizer files:

- `tokenizer.json` SHA256:
  `b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`
- `tokenizer_config.json` SHA256:
  `9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`
- `special_tokens_map.json` SHA256:
  `57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

Active serialization:

`claim[:63]+EOS(0)+evidence[:64]`

Post4 eligibility rule:

`a+4 <= terminal_index-1`

Frozen `tokenizers` package version:

`0.22.2`

Any tokenizer revision, file hash, serialization, package-contract, or
eligibility mismatch blocks execution.

## Frozen PP3 / PP5 geometry

Strong-state dimension:

`395`

PP3 plus SHA256:

`66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`

PP3 minus SHA256:

`ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

PP5 plus SHA256:

`7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2`

PP5 minus SHA256:

`311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855`

Frozen static geometry gates include:

- PP3 Gram max residual:
  `1.3322676295501878e-15`
- PP5 Gram max residual:
  `1.1102230246251565e-15`
- PP3/PP5 cross-plane max absolute dot:
  `2.7235158572835871e-16`
- coefficient-transfer max L2 mismatch:
  `1.7763568394002505e-15`
- R3-equals-native-h max absolute residual:
  `2.7755575615628914e-17`
- R5-minus-background-equals-c5 max absolute residual:
  `1.1102230246251565e-16`
- restoration-addition-norm max absolute mismatch:
  `1.7763568394002505e-15`

PP3 and PP5 remain fixed. No response-guided rotation, reselection, or
replacement is authorized.

## Frozen broad XG2/XG4 endpoint

Frozen XG2 top-5 basis-plan SHA256:

`b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c`

Frozen XG4 top-5 basis-plan SHA256:

`792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f`

Directions, in exact order:

1. `xg2_0`
2. `xg2_1`
3. `xg2_2`
4. `xg2_3`
5. `xg2_4`
6. `xg4_0`
7. `xg4_1`
8. `xg4_2`
9. `xg4_3`
10. `xg4_4`

Finite-difference epsilon:

`0.025`

For condition `c`:

`E_XG2(c) = (1/5) * sum_j J(v_XG2,j;c)^2`

`E_XG4(c) = (1/5) * sum_j J(v_XG4,j;c)^2`

`Q_c = E_XG2(c) - E_XG4(c)`

No epsilon, basis, direction-order, layer, token, or endpoint change is
authorized.

## Frozen restoration conditions

Exact condition order:

1. `pp3_neutralized`
2. `pp3_restored`
3. `pp5_replacement`

Let the unmodified native layer-17 target-token frozen strong-channel state be:

`h`

Native PP3 coefficients:

`a = <h, pp3_plus>`

`b = <h, pp3_minus>`

Native PP3 component:

`c3 = a*pp3_plus + b*pp3_minus`

Coefficient-transferred PP5 component:

`c5 = a*pp5_plus + b*pp5_minus`

The exact same native PP3-derived `(a,b)` must be used to construct `c3`
and `c5`.

Native PP5 coordinates must not be substituted.

Because both bases are orthonormal and the same `(a,b)` are transferred:

`||c3||_2 = ||c5||_2`

### PP3 neutralized

Background:

`B = h - c3`

Direct final-state correction:

`delta_B = -c3`

### PP3 restored

Restoration:

`R3 = B + c3 = h`

The runner directly realizes this algebraically exact final state as:

`delta_R3 = 0`

### Matched PP5 replacement

Replacement:

`R5 = B + c5`

Equivalently:

`R5 = h - c3 + c5`

Direct final-state correction:

`delta_R5 = -c3 + c5`

No extra scientific forward may be added to obtain `h`, `a`, or `b`.

## Intervention boundary

The intervention is restricted to the frozen layer-17 `mixer17.in_proj`
hook and modifies only:

- the frozen target token;
- the x-half;
- the frozen strong 395 channels.

Execution must verify:

- gate half unchanged exactly;
- non-strong x channels unchanged exactly;
- all other token positions unchanged exactly;
- applied-cast residual within the frozen runtime tolerance.

Per-branch raw audits must retain sufficient information to verify at least:

- native PP3 coefficient `a`;
- native PP3 coefficient `b`;
- PP3 component L2;
- PP5 component L2;
- restoration-addition norm mismatch;
- direct final-state correction L2;
- PP3 post-condition residual coordinates;
- neutralized construction residual;
- R3 native identity residual;
- R5 construction residual;
- probe correction L2;
- applied correction residual.

## Exact forward budget

Per signed orientation:

- target-plus forward: `1`
- target-minus forward: `1`
- total: `2`

Per direction:

- two orientations;
- forwards: `4`

Per condition:

- ten directions;
- forwards: `40`

Per pair:

- three conditions;
- forwards: `120`

Population:

- pairs: `300`

Exact scientific model forwards:

`36000`

Baseline model forwards:

`0`

No additional native-state or coefficient-capture forward is authorized.

## Frozen raw endpoint algebra

Per pair:

`Q_B = Q(pp3_neutralized)`

`Q_R3 = Q(pp3_restored)`

`Q_R5 = Q(pp5_replacement)`

Restoration by native PP3:

`S3 = Q_R3 - Q_B`

Restoration by matched PP5 replacement:

`S5 = Q_R5 - Q_B`

Primary restoration-sufficiency contrast:

`D_SUF = S3 - S5`

Equivalent frozen identity:

`D_SUF = Q_R3 - Q_R5`

The raw artifact validator must preserve and validate these identities.

## Frozen model and checkpoint

Model:

`state-spaces/mamba-130m-hf`

Representative checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Expected checkpoint size:

`518270455` bytes

Canonical Kaggle checkpoint source:

`/kaggle/input/datasets/terryterry9/contramamba-seed180-g3-group-d-half-checkpoint/selected_checkpoint.pt`

The repository historical checkpoint placeholder must not be used as the
scientific checkpoint.

Before scientific execution the external checkpoint must be verified for both
exact byte size and SHA256.

Any mismatch blocks execution.

## Frozen execution runtime contract

Established Gen4 fast-CUDA runtime identity:

- Python: `3.12.13`
- NumPy: `2.0.2`
- Torch: `2.10.0+cu128`
- Transformers: `5.0.0`
- CUDA runtime: `12.8`
- GPU: `Tesla T4`
- capability: `7.5`
- kernels: `0.10.2`

Exact Mamba kernel revision:

`c8ffc584c147878a6eb978ae0e8db4d116c93a8c`

Exact Mamba binary SHA256:

`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

Exact causal-conv revision:

`f2651e776f66069cdcf842840db637583def1223`

Exact causal-conv binary SHA256:

`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

The prior successful Gen4 execution required `kernels==0.10.2`; absence or
metadata mismatch must be corrected during CPU/preflight before the scientific
run rather than bypassed.

Package versions must not be opportunistically upgraded.

Any runtime, kernel, model, tokenizer, checkpoint, frozen vector, basis-plan,
or input identity mismatch blocks scientific execution.

## Raw observation artifact contract

Successful execution result label:

`PASS_PP3_RESTORATION_SUFFICIENCY_RAW_OBSERVATION`

Exactly four raw output files are permitted:

1. `pp3_restoration_sufficiency_items.jsonl`
2. `pp3_restoration_sufficiency_summary.json`
3. `artifact_manifest.json`
4. `SHA256SUMS.txt`

The raw summary must retain at least:

- exact execution HEAD;
- implementation-authority commit;
- static-preparation commit;
- source-pair count `300`;
- population bounds `901..1200`;
- exact condition order;
- exact direction order;
- epsilon `0.025`;
- scientific forward count `36000`;
- baseline forward count `0`;
- representative checkpoint SHA256;
- `primary_inference_executed = false`;
- `multiplicity_correction_executed = false`;
- `training_executed = false`;
- `backward_executed = false`;
- `task_heads_executed = false`;
- `logits_read = false`;
- `scientific_conclusion = null`.

The Kaggle execution is therefore RAW OBSERVATION ONLY.

No p-value, inferential result, supported/not-established scientific label,
or other scientific interpretation may be produced during execution.

## Post-import confirmatory inference

Only after:

1. the run completes;
2. collection succeeds;
3. local import reports PASS;
4. commit/run/command/log/meta/artifact provenance is validated;
5. all 300 pair rows and exact ordering are validated;
6. finite-value validation passes;
7. intervention-audit validation passes;
8. endpoint algebra passes;
9. exact forward-budget validation passes;

may the single frozen confirmatory test be executed.

Primary values:

the `300` imported `D_SUF_i` values.

Hypotheses:

`H0: mean(D_SUF) <= 0`

`H1: mean(D_SUF) > 0`

Test:

- one-sample Student t-test;
- one-sided greater;
- N = `300`;
- df = `299`;
- alpha = `0.05`;
- confirmatory hypotheses = `1`;
- multiplicity correction = none.

Exactly one confirmatory p-value is authorized.

No inferential p-value is authorized for `Q_B`, `Q_R3`, `Q_R5`, `S3`, or
`S5`.

Descriptive reporting may include:

- mean `Q_B`;
- mean `Q_R3`;
- mean `Q_R5`;
- mean `S3`;
- mean `S5`;
- mean `D_SUF`;
- SD `D_SUF`.

## Positive-label gates

A positive label requires all provenance/runtime/completeness gates and all:

1. `mean(Q_R3) > 0`
2. `mean(S3) > 0`
3. `mean(D_SUF) > 0`
4. one-sided primary `p < 0.05`

If all pass, exact label:

`PP3_RESTORATION_SUFFICIENCY_OVER_MATCHED_PP5_REPLACEMENT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Otherwise exact label:

`PP3_RESTORATION_SUFFICIENCY_OVER_MATCHED_PP5_REPLACEMENT_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

## No-rescue policy

Do not perform:

- subgroup analysis;
- pair dropping or replacement;
- alternative tail;
- additional p-values;
- alternate endpoint;
- alternate control after outcomes;
- PP1/PP2/PP4 rescue;
- response-guided plane selection;
- response-guided rotation;
- epsilon sweep;
- layer sweep;
- token sweep;
- checkpoint sweep;
- hyperparameter tuning;
- task-head/logit analysis.

A failed positive gate remains a valid negative/non-established result and is
not authorization to modify the frozen question.

## Interpretation boundary

A positive result would support only the following bounded statement:

On the PP3-neutralized frozen layer-17 target-token background, restoring the
native PP3 component restores the broad XG2-vs-XG4 local susceptibility
contrast more strongly than an equal-coefficient, matched-addition-norm PP5
replacement.

This supports PP3 as a local restoration-sufficient contributor relative to
the frozen matched PP5 replacement.

It does not establish:

- PP3 as the sole mechanism;
- PP3 alone as sufficient in an empty state;
- global behavioral sufficiency;
- task-head sufficiency;
- arbitrary-generator sufficiency;
- sufficiency across arbitrary checkpoints;
- sufficiency across arbitrary layers or token positions;
- architecture-wide or all-Mamba universality.

The distributed-mechanism interpretation remains mandatory.

## Failure policy

Any mismatch in:

- execution HEAD;
- branch/clean-worktree requirement;
- runner or test identity;
- authority ancestry;
- static-preparation identity;
- population or ordering;
- input hashes;
- tokenizer identity or anchor eligibility;
- PP3/PP5 vector identities;
- XG2/XG4 basis identities;
- model/checkpoint identity;
- runtime/kernel identity;
- intervention formulas;
- intervention isolation audits;
- restoration norm match;
- endpoint algebra;
- exact forward budget;
- finite-value validation;
- raw artifact schema or checksums;

blocks promotion of the execution to scientific evidence.

Do not bypass provenance/hash/commit/dirty-worktree blockers.

A failed or partial run may be collected for provenance when possible, but it
must not be promoted into scientific evidence.

## Execution authorization

Scientific GPU raw observation is authorized only after this document itself
is committed and pushed on `gen4-k-xg2-basis-holdout`.

The resulting clean pushed full HEAD is the exact execution identity and must
be passed to the runner as `--expected-head`.

The concrete Kaggle shell command and command SHA256 are deliberately not
frozen here before that commit exists.

After this execution freeze is committed and pushed, the concrete command
must be constructed from the actual new full HEAD and the runner's required
CLI:

- `--expected-head`
- `--model-snapshot`
- `--tokenizer-snapshot`
- `--checkpoint`
- `--output-dir`

The approved command must then be registered with a new descriptive run name
using the normal `cm run save` / `cm run` provenance chain.

A run from another commit, a dirty repository, a reused command identity, or
a materially edited pinned Kaggle command is invalid.
