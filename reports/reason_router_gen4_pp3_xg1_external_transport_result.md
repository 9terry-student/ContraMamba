# Gen4 PP3 XG1 External Transport Result

## Status

`PP3_PROJECTOR_CONTRAST_TRANSPORT_SUPPORTED_ON_XG1_EXTERNAL_GENERATOR`

## Execution provenance

- Execution commit: `8e33d96e32a86fbf94d2b0222b48a48146a4b4a1`
- Run: `pp3-xg1-external-transport-8e33d96-retry2`
- Command SHA256: `f4f8caeb5da9f22003c7a6dfff3fa7e0acacaacd3638a5022cca48a895ccc77b`
- Imported handoff ZIP SHA256: `9d741aad939ff55d32f6260340954113d38d7b3869a0ab1ebe1f88566c17fb70`
- Representative checkpoint SHA256: `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`
- Population: `xg1_fact_001` through `xg1_fact_300`
- Pair count: 300
- Scientific model forwards: 2400
- Baseline model forwards: 0
- Training/backward/task-head/logit execution: none

## Artifact validation

Post-import validation passed:

- item count: 300
- pair order: exact
- manifest/SHA256: exact
- endpoint identity:
  `C_PP3 = (s3 / 5) * (J_PP3_PLUS^2 - J_PP3_MINUS^2)`
- row scientific-forward sum: 2400
- row baseline-forward sum: 0
- all endpoint values finite

Imported artifact SHA256:

- `pp3_xg1_external_transport_items.jsonl`:
  `6a5242508034ba53ae37552039cca6cd8ffcf216052c69b16fbb17d6a5498053`
- `pp3_xg1_external_transport_summary.json`:
  `7e52293020a02b143dbb2a66e3c8691d7826339eb7f1536da871b0b40c8d2448`
- `artifact_manifest.json`:
  `42ee6707ac215de840aa3906db3893833fbad73613c2c6dd4d15a651c1165405`

## Frozen primary inference

Prespecified hypothesis:

- `H1: mean(C_PP3) > 0`
- one-sample Student t-test
- one-sided
- `alpha = 0.05`
- exactly one primary hypothesis
- no multiplicity correction
- `N = 300`

Observed:

- mean `C_PP3`: `7.1202761928785655e-08`
- SD: `6.4998373256093212e-08`
- SE: `3.7526828296293188e-09`
- t statistic: `18.97382890091431`
- df: `299`
- one-sided p-value: `1.5194095747784581e-53`
- mean positive: `true`
- reject H0: `true`

Therefore the frozen positive-label criterion is satisfied:

`PP3_PROJECTOR_CONTRAST_TRANSPORT_SUPPORTED_ON_XG1_EXTERNAL_GENERATOR`

## Descriptive PP3+ diagnostic

- mean `J_PP3_PLUS`: `0.00069627895420726338`
- mean absolute `J_PP3_PLUS`: `0.00069627895420726338`
- fraction `J_PP3_PLUS > 0`: `1.0`
- SD `J_PP3_PLUS`: `0.00027851811616604337`

These PP3+ quantities are descriptive only and do not define an additional inferential hypothesis.

## Interpretation boundary

The result supports prospective transport of the frozen PP3 projector-contrast susceptibility contrast to the structurally independent XG1 generator.

It does not establish that PP3 is the sole explanation of the broader family-specific geometry, and it does not by itself establish transport to generators other than XG1. No rescue analysis, subgroup/tail mining, epsilon sweep, checkpoint sweep, direction rotation, or additional hypothesis test was performed.
