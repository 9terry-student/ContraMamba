# Gen4 PP3 vs PP5 Fresh-XG1 Specificity — Implementation Authority

Status: `IMPLEMENTATION_AUTHORITY_ONLY_NO_SCIENTIFIC_EXECUTION`

## Authority chain

This authority is subordinate to the prospectively frozen specificity design:

- design freeze:
  `0bc49ab95cbb2c8735b4bc79422d660fa64e3e01`
- static preparation freeze:
  `ddb1404800af6dbd89982bbbcdd8262d203577f6`
- fresh tokenizer/anchor eligibility freeze:
  `3613d3a6a21d6fb2bef3692e3c8203b5ce0f37ec`

The scientific question, control selection, endpoint, population, direction
order, forward budget, primary hypothesis, decision rule, forbidden
adaptations, and interpretation boundary remain exactly those frozen in the
design. This document does not revise them.

## Frozen prerequisites

Fresh XG1 population:

- exact pair IDs: `xg1_fact_301..xg1_fact_600`
- pair count: 300
- row count: 1800
- source SHA256:
  `aa5b8e3cfcbf19e71335ecdbea659326925f8bea33312c2354de670fa7a15cf7`
- six-cell rows SHA256:
  `3f28d8a75008d383855313a08168fef1a2b9b37257103636a7f2edb65ce76ad6`
- structural manifest SHA256:
  `92ae0137641691c2a9d0254e87739e3e89d90556dc0579a2877433b86ff97dfb`

Fresh tokenizer/anchor gate:

- result: `PASS_300_OF_300`
- required anchor rows: 1800
- exclusions: 0
- anchor manifest SHA256:
  `4f7eb6b8660212a9db64b679370262fe4ec9b57cff0ab99121b039c0db141e04`
- eligibility summary SHA256:
  `d546db6cce692e7000d382570ef15802770f5e6267d266e66f08b5efe4fbe5d2`
- active serialization:
  `claim[:63]+EOS(0)+evidence[:64]`
- post4 rule:
  `a+4 <= terminal_index-1`

Frozen PP3:

- `s3 = 0.98692852916688512`
- PP3+ SHA256:
  `66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`
- PP3- SHA256:
  `ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

Frozen PP5 specificity control:

- geometry-only selection rule:
  PP5 is the principal plane with maximal `sin(theta)` among PP1..PP5
- `s5 = 0.99986792842854511`
- PP5+ SHA256:
  `7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2`
- PP5- SHA256:
  `311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855`
- preparation manifest SHA256:
  `f403969f2c099d5227c28edae70696347b57bb7d96e1ab18fd5dcf24299365be`

Reference implementation:

- existing PP3 XG1 observation runner:
  `scripts/reason_router_gen4_pp3_xg1_external_transport_fast_cuda.py`
- frozen Git blob:
  `1b81deacc330beb9a7cf1d09b520ec55aaf2cc0d`

The existing PP3 runtime semantics, paired intervention audit, finite-difference
construction, forward-budget accounting, exact runtime/kernel binding, and
artifact validation are to be reused rather than redesigned.

## Authorized implementation delta

Exactly one new scientific runner and its tests may be implemented:

- `scripts/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_fast_cuda.py`
- `tests/test_reason_router_gen4_pp3_pp5_fresh_xg1_specificity_fast_cuda.py`

Supporting code may import already frozen helpers, but existing frozen
scientific/runtime files must not be modified.

The runner must enforce:

1. exact population `xg1_fact_301..xg1_fact_600`;
2. exact epsilon `0.025`;
3. exact direction order:
   - `pp3_plus`
   - `pp3_minus`
   - `pp5_plus`
   - `pp5_minus`
4. four scientific forwards per direction;
5. sixteen scientific forwards per pair;
6. exactly `4800` scientific model forwards total;
7. exactly zero baseline forwards in this run;
8. no training, backward pass, task-head evaluation, or logits read.

For each direction `w`:

`J_i(w) = [F_i(+epsilon;w) - F_i(-epsilon;w)] / (2 epsilon)`

with the already frozen paired-intervention half-delta semantics.

Per pair:

`C_PP3_i = (s3/5) * (J_PP3_PLUS_i^2 - J_PP3_MINUS_i^2)`

`C_PP5_i = (s5/5) * (J_PP5_PLUS_i^2 - J_PP5_MINUS_i^2)`

`D_SPEC_i = C_PP3_i - C_PP5_i`

The observation runner may record these item-level quantities and descriptive
fields required by the frozen design.

It must not perform the confirmatory t-test or emit a scientific conclusion.
Primary inference remains a separate post-import step.

## Required implementation validation

Before implementation freeze, CPU/static/mock tests must establish at least:

- exact frozen file/blob identities;
- exact fresh pair order and count;
- exact PP3/PP5 vector hashes and unit/orthogonality checks;
- exact direction order;
- exact `4 / 16 / 4800` forward accounting;
- exact finite-difference `J` identity;
- exact `C_PP3`, `C_PP5`, and `D_SPEC` algebra;
- paired-intervention half-delta semantics;
- output item/summary/manifest validation;
- inference boundary:
  `primary_inference_executed == False`;
- conclusion boundary:
  `scientific_conclusion == None`;
- no adaptive control/direction/epsilon/population logic.

Mocked capture calls and synthetic numerical values are allowed for tests.

## Explicitly not authorized

This authority does not authorize:

- tokenizer execution;
- checkpoint loading;
- real model construction;
- model forward execution;
- CUDA scientific execution;
- Kaggle execution;
- primary or secondary hypothesis testing;
- scientific result interpretation;
- PP1/PP2/PP4 rescue;
- alternative control selection;
- response-guided direction or sign changes;
- epsilon/checkpoint/layer/token sweeps;
- subgroup/tail analysis;
- pair dropping or replacement.

A subsequent execution freeze is required before any real scientific model
execution.
