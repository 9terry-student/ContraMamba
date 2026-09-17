# Gen4 PP3 Necessity — Execution Freeze

## Status

`EXECUTION_READY`

This is the sole scientific execution freeze for the prospectively frozen
PP3 necessity experiment on fresh XG1 `601..900`.

It does not alter the scientific question, population, intervention,
matched control, broad endpoint, epsilon, primary hypothesis, decision rule,
failure policy, or interpretation boundary frozen previously.

## Frozen authority chain

Necessity design:

`f4419f1e7efbeeb9e50e84b67accc56422580cf2`

Static preparation:

`e470f132a37c731754feb4333dcb2e48e29af53b`

Implementation authority:

`0f428d84a5065268c9a4dfaf80855483c4e99c42`

Implementation freeze:

`f1896e0d3669bac832038e1486b2b04d322af0aa`

## Exact implementation identity

Scientific runner:

`scripts/reason_router_gen4_pp3_necessity_fast_cuda.py`

Git blob:

`26ca67ad8603799a849c151a39728368227326df`

File SHA256:

`62fe61af7fe4933c68bc36813e6ff96fda69dbe23a2cb063574eeb0565f2b6d3`

Static/mock test:

`tests/test_reason_router_gen4_pp3_necessity_fast_cuda.py`

Git blob:

`f9d0142460d232a4b7f471f2499c2877d8a8883b`

File SHA256:

`2e8fa6899c8278bacdecf2792b62b8923f753ff22a49389c15c5acce40be2d05`

Pre-freeze validation:

`9 passed`

Implementation validation executed:

- checkpoint loads: `0`
- scientific model forwards: `0`
- CUDA scientific execution: `false`
- primary inference: `false`

## Frozen scientific population

Exact population:

`xg1_fact_601..xg1_fact_900`

Cardinality:

- source pairs: `300`
- six-cell rows: `1800`

Frozen structural identities:

- source facts SHA256:
  `49bec37150630d31bb5f502f49ef23ffc9a75bb93e079127c8c430aae3da6abd`
- six-cell rows SHA256:
  `e03534599c07201e371eb07938d8492d22a30de39dcbbb3c0c22300a4ff94224`
- structural manifest SHA256:
  `219994e7ec757148ffcae30a7d50357e347530e06cecbb618db619b73bbe8b76`

Prospective non-overlap gates frozen before model execution:

- pair-ID overlap with XG1 `001..600`: `0`
- claim overlap with XG1 `001..600`: `0`
- evidence overlap with XG1 `001..600`: `0`
- exact claim/evidence-row overlap with XG1 `001..600`: `0`

## Frozen tokenizer / anchor eligibility

Eligibility:

`PASS_300_OF_300`

Anchor rows:

`1800`

Anchor manifest SHA256:

`67d74cd2b3ab8daa8065227b1e85ea125ca88af4de64c872e5d47f60d9c40ec2`

Tokenizer revision:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Tokenizer files:

- tokenizer.json:
  `b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`
- tokenizer_config.json:
  `9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`
- special_tokens_map.json:
  `57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

Active serialization:

`claim[:63]+EOS(0)+evidence[:64]`

Post4 eligibility rule:

`a+4 <= terminal_index-1`

## Frozen PP3 / PP5 geometry

PP3+ SHA256:

`66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`

PP3- SHA256:

`ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

PP5+ SHA256:

`7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2`

PP5- SHA256:

`311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855`

Static geometry gates:

- PP3 Gram max residual:
  `1.3322676295501878e-15`
- PP5 Gram max residual:
  `1.1102230246251565e-15`
- PP3-vs-PP5 cross-plane max absolute dot:
  `2.7235158572835871e-16`
- coefficient-transfer max L2 mismatch:
  `1.7763568394002505e-15`

## Frozen broad basis endpoint

Reuse exactly the previously frozen top-5 XG2/XG4 bases.

Phase-1 plan SHA256:

XG2:

`b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c`

XG4:

`792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f`

For each condition `c`:

`E_XG2(c) = (1/5) * sum_j J(v_XG2,j;c)^2`

`E_XG4(c) = (1/5) * sum_j J(v_XG4,j;c)^2`

`Q(c) = E_XG2(c) - E_XG4(c)`

Finite-difference epsilon:

`0.025`

## Frozen intervention conditions

Exact condition order:

1. `native`
2. `pp3_neutralized`
3. `pp5_coefficient_control`

For each unmodified native strong-channel state `h`:

`a = <h, pp3_plus>`

`b = <h, pp3_minus>`

### Native

`delta_condition = 0`

### PP3 neutralized

`delta_condition = -a*pp3_plus - b*pp3_minus`

The post-condition PP3 residual projections must remain within frozen tolerance.

### PP5 coefficient-transfer control

`delta_condition = -a*pp5_plus - b*pp5_minus`

The same PP3-derived `(a,b)` from that native branch state must be used.

Native PP5 projection coefficients must not be used.

Treatment/control condition-correction L2 values must match within the frozen
runtime tolerance.

## Direct signed-probe intervention

For branch sign `s`:

- target-plus: `s = +1`
- target-minus: `s = -1`

For orientation `o in {+1,-1}` and frozen unit direction `v`:

`delta_total = delta_condition + s * o * 0.025 * v`

The correction is applied directly to the target-token strong x channels.

It must not pass through an additional `0.5/-0.5` scaling.

The execution must preserve:

- gate half exactly;
- non-strong x channels exactly;
- all other tokens exactly.

## Frozen direction order

Within every condition:

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

Within each direction:

1. orientation `+1`
2. orientation `-1`

No response-guided reordering is allowed.

## Exact forward budget

Per orientation:

- target-plus: `1`
- target-minus: `1`
- forwards per signed orientation: `2`

Per direction:

- two orientations
- forwards: `4`

Per condition:

- ten directions
- forwards: `40`

Per pair:

- three conditions
- forwards: `120`

Full population:

- pairs: `300`
- exact scientific forwards: `36000`

Baseline model forwards this run:

`0`

No additional native-state capture forward is authorized.

## Frozen per-pair endpoint

For pair `i`:

`Q0_i = Q_i(native)`

`Q3_i = Q_i(pp3_neutralized)`

`Q5_i = Q_i(pp5_coefficient_control)`

PP3 attenuation:

`A3_i = Q0_i - Q3_i`

Matched-control attenuation:

`A5_i = Q0_i - Q5_i`

Primary necessity contrast:

`D_NEC_i = A3_i - A5_i`

Equivalent identity:

`D_NEC_i = Q5_i - Q3_i`

The artifact validator must reconstruct these identities from raw probe data.

## Frozen model / checkpoint

Model:

`state-spaces/mamba-130m-hf`

Representative checkpoint repository path:

`reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt`

Checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

## Frozen runtime

Expected runtime:

- Python: `3.12.13`
- NumPy: `2.0.2`
- Torch: `2.10.0+cu128`
- Transformers: `5.0.0`
- CUDA runtime: `12.8`
- device: `Tesla T4`
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

Any runtime, kernel, model, tokenizer, checkpoint, input, or frozen vector
identity mismatch blocks execution.

## Raw observation boundary

The GPU execution is observation-only.

Required raw summary fields:

- `scientific_model_forward_count_this_run = 36000`
- `baseline_model_forward_count_this_run = 0`
- `primary_inference_executed = false`
- `multiplicity_correction_executed = false`
- `training_executed = false`
- `backward_executed = false`
- `task_heads_executed = false`
- `logits_read = false`
- `scientific_conclusion = null`

Expected output files exactly:

1. `pp3_necessity_items.jsonl`
2. `pp3_necessity_summary.json`
3. `artifact_manifest.json`
4. `SHA256SUMS.txt`

No statistical test or scientific result label may be produced during the GPU
run.

## Post-import confirmatory inference

Only after successful collection/import plus provenance, completeness,
finite-value, intervention-audit, and exact-forward-budget validation may the
single frozen confirmatory test be executed.

Endpoint:

the 300 `D_NEC_i` values

Hypothesis:

`H1: mean(D_NEC) > 0`

Test:

- one-sample Student t-test
- one-sided
- N = `300`
- df = `299`
- alpha = `0.05`
- confirmatory hypotheses = `1`
- multiplicity correction = none

No inferential test is authorized for `Q0`, `Q3`, `Q5`, `A3`, or `A5`.

## Positive-label gates

A positive necessity label requires all provenance/runtime/completeness gates
and all of:

1. `mean(Q0) > 0`
2. `mean(A3) > 0`
3. `mean(D_NEC) > 0`
4. one-sided primary `p < 0.05`

Positive label:

`PP3_NECESSITY_OVER_MATCHED_PP5_CONTROL_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Otherwise:

`PP3_NECESSITY_OVER_MATCHED_PP5_CONTROL_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

## No-rescue policy

Do not perform:

- PP1/PP2/PP4 rescue;
- alternative-control rescue;
- response-guided plane selection;
- response-guided rotation;
- epsilon sweep;
- layer sweep;
- token sweep;
- checkpoint sweep;
- subgroup testing;
- tail testing;
- pair dropping or replacement;
- additional p-values.

## Interpretation boundary

A positive result would support a local causal necessity contribution of PP3 to
the frozen broad cross-family susceptibility contrast at the frozen
layer-17/target-token intervention.

It would not establish:

- PP3 as the sole mechanism;
- global behavioral necessity;
- downstream-task necessity;
- PP3 sufficiency;
- universal necessity across arbitrary generators, layers, tokens,
  checkpoints, or architectures.

The distributed-mechanism interpretation remains in force.

## Failure policy

Any mismatch in:

- execution HEAD;
- runner/test identity;
- authority/static-preparation ancestry;
- data hash;
- tokenizer identity;
- anchor eligibility;
- PP3/PP5 vectors;
- XG2/XG4 basis identity;
- runtime/kernel identity;
- checkpoint identity;
- population/order;
- intervention audit;
- correction matching;
- exact forward budget;
- finite-value validation;
- artifact validation

blocks the run.

No failed or partial run may be promoted into scientific evidence.

## Execution authorization

Scientific GPU observation is authorized only from the clean pushed repository
HEAD containing this execution freeze while preserving all frozen identities
above.

That exact full execution HEAD must be passed as `--expected-head`.

The concrete Kaggle command and its command SHA256 must be generated only after
this execution freeze is committed and pushed, so the final execution HEAD can
be bound exactly.

A run from another commit, a dirty worktree, or a changed command is invalid.
