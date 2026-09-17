# Gen4 PP3 vs PP5 Fresh-XG1 Specificity — Execution Freeze

## Status

`EXECUTION_READY`

This is the sole execution freeze for the prospectively frozen PP3-vs-PP5
specificity experiment on fresh XG1 `301..600`.

It does not alter the scientific question, population, control selection,
directions, epsilon, endpoint, primary hypothesis, decision rule, failure
policy, or interpretation boundary frozen previously.

## Frozen authority chain

- specificity design:
  `0bc49ab95cbb2c8735b4bc79422d660fa64e3e01`
- static preparation:
  `ddb1404800af6dbd89982bbbcdd8262d203577f6`
- fresh tokenizer eligibility:
  `3613d3a6a21d6fb2bef3692e3c8203b5ce0f37ec`
- implementation authority:
  `1db39ffbbcc0c0558edc859d3260e50132d8c858`
- runner implementation:
  `4e7c3a17c20d954c1fb899e0b5258c41034ec0cc`

## Exact implementation identity

Scientific runner:

`scripts/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_fast_cuda.py`

Git blob:

`a72bcb516e6cac8ff5eeec345a88f5bddbfb0064`

Unit/static/mock test:

`tests/test_reason_router_gen4_pp3_pp5_fresh_xg1_specificity_fast_cuda.py`

Git blob:

`6074529f4dc4d340bf793ed7185454af21609129`

Pre-freeze validation:

`31 passed`

No scientific model execution occurred during implementation validation.

## Frozen scientific population

Exact population:

- `xg1_fact_301..xg1_fact_600`
- 300 source pairs
- 1800 six-cell rows

Frozen structural identities:

- source facts SHA256:
  `aa5b8e3cfcbf19e71335ecdbea659326925f8bea33312c2354de670fa7a15cf7`
- six-cell rows SHA256:
  `3f28d8a75008d383855313a08168fef1a2b9b37257103636a7f2edb65ce76ad6`
- structural manifest SHA256:
  `92ae0137641691c2a9d0254e87739e3e89d90556dc0579a2877433b86ff97dfb`

Fresh tokenizer/anchor eligibility:

- verdict: `PASS_300_OF_300`
- anchor manifest SHA256:
  `4f7eb6b8660212a9db64b679370262fe4ec9b57cff0ab99121b039c0db141e04`
- eligibility summary SHA256:
  `d546db6cce692e7000d382570ef15802770f5e6267d266e66f08b5efe4fbe5d2`
- tokenizer revision:
  `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`
- active serialization:
  `claim[:63]+EOS(0)+evidence[:64]`

## Frozen geometry

PP3:

- `s3 = 0.98692852916688512`
- PP3+ SHA256:
  `66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`
- PP3- SHA256:
  `ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

PP5 max-separation control:

- `s5 = 0.99986792842854511`
- PP5+ SHA256:
  `7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2`
- PP5- SHA256:
  `311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855`
- preparation manifest SHA256:
  `f403969f2c099d5227c28edae70696347b57bb7d96e1ab18fd5dcf24299365be`

## Frozen model/runtime identity

Model family:

`state-spaces/mamba-130m-hf`

Representative scientific checkpoint:

`reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt`

Checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

The exact model construction, native-Mamba runtime binding, CUDA kernel
compatibility, checkpoint identity validation, paired-intervention audit, and
fast-CUDA capture semantics are inherited unchanged from the already validated
PP3 runtime dependency chain and are enforced by the runner.

Any runtime/kernel/model/checkpoint identity mismatch blocks execution.

## Authorized observation

Exact epsilon:

`0.025`

Exact direction order:

1. `PP3+`
2. `PP3-`
3. `PP5+`
4. `PP5-`

For every direction:

- 2 model forwards for `F(+epsilon)`
- 2 model forwards for `F(-epsilon)`
- 4 scientific forwards per direction

Therefore:

- 16 scientific forwards per pair
- 300 pairs
- exactly `4800` scientific model forwards
- exactly `0` baseline forwards

Per direction:

`J_i(w) = [F_i(+epsilon;w) - F_i(-epsilon;w)] / (2 epsilon)`

Per pair:

`C_PP3_i = (s3/5) * (J_PP3_PLUS_i^2 - J_PP3_MINUS_i^2)`

`C_PP5_i = (s5/5) * (J_PP5_PLUS_i^2 - J_PP5_MINUS_i^2)`

`D_SPEC_i = C_PP3_i - C_PP5_i`

All finite observations must be retained regardless of sign.

## Raw observation boundary

The GPU execution is observation-only.

The runner must preserve:

- `scientific_model_forward_count_this_run = 4800`
- `baseline_model_forward_count_this_run = 0`
- `primary_inference_executed = false`
- `multiplicity_correction_executed = false`
- `training_executed = false`
- `backward_executed = false`
- `task_heads_executed = false`
- `logits_read = false`
- `scientific_conclusion = null`

Expected artifact files:

- `pp3_pp5_fresh_xg1_specificity_items.jsonl`
- `pp3_pp5_fresh_xg1_specificity_summary.json`
- `artifact_manifest.json`
- `SHA256SUMS.txt`

No t-test or scientific result label may be produced during the GPU run.

## Post-import primary inference

Only after collection/import and provenance, completeness, finite-value, and
forward-budget validation may the prospectively frozen single primary test be
executed:

- endpoint: the 300 `D_SPEC_i` values
- one-sample Student t-test
- one-sided alternative: `mean(D_SPEC) > 0`
- alpha: `0.05`
- exactly one confirmatory hypothesis
- no multiplicity correction

The positive label requires all gates plus:

1. `mean(C_PP3) > 0`
2. `mean(D_SPEC) > 0`
3. one-sided `p < 0.05`

Positive label:

`PP3_SPECIFICITY_OVER_MAX_SEPARATION_PP5_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Otherwise:

`PP3_SPECIFICITY_OVER_MAX_SEPARATION_PP5_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

No rescue test is authorized.

## Failure policy

Any mismatch in commit, runner blob, input hash, tokenizer identity, anchor
eligibility, PP3/PP5 vector identity, runtime/kernel identity, checkpoint
identity, population, forward budget, finite-value validation, or artifact
validation blocks the run.

Do not:

- change epsilon;
- replace/drop pairs;
- rotate or sign-change directions after observing responses;
- substitute PP1/PP2/PP4;
- select another control;
- sweep checkpoint/layer/token/epsilon;
- perform subgroup or tail mining;
- perform additional p-values.

## Execution authorization

Scientific GPU observation is authorized only from the clean pushed repository
HEAD that contains this execution freeze and preserves all frozen identities
above.

That exact full execution HEAD must be passed as `--expected-head`.

A run produced from any other commit or dirty worktree is invalid for this
experiment.
