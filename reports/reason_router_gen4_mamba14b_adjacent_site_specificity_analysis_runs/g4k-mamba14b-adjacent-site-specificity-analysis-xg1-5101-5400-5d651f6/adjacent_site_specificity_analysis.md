# ContraMamba Gen4 Experiment 5 — One-Shot Adjacent-Site Specificity

Result: `PASS_MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_ANALYSIS`

Scientific conclusion: `MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_SUPPORTED`

## Frozen test

- Cohort: `xg1_fact_5101..xg1_fact_5400`, N=300.
- Canonical site: `(33,34,35)`.
- Adjacent site: `(34,35,36)`.
- Fixed rank-aligned causal candidate: `P5`.
- Canonical control: `P4`.
- Adjacent geometry-only control: `P4`.
- epsilon: `0.025`.
- Primary endpoint: `S = D_CAN - D_ADJ`.
- Exactly one inferential p-value was computed.

## Descriptive results

| Endpoint | Mean | SD | 95% t-CI | Cohen dz | Positive fraction |
| --- | ---: | ---: | --- | ---: | ---: |
| D_CAN | 9.1027831979425761e-09 | 9.9310258755280052e-09 | [7.9744352084201466e-09, 1.0231131187465005e-08] | 0.91660049143297662 | 0.77666666666666662 |
| D_ADJ | -1.5510695754416869e-09 | 9.2954059832763453e-10 | [-1.6566825582743089e-09, -1.445456592609065e-09] | -1.6686410235682707 | 0.0066666666666666671 |
| S | 1.0653852773384264e-08 | 9.9984407144957309e-09 | [9.5178452128400069e-09, 1.1789860333928522e-08] | 1.0655514272278794 | 0.81000000000000005 |

## Primary inference

- Test: one-sample Student t-test on paired `S`.
- Null: `E[S] <= 0`.
- Alternative: `E[S] > 0`.
- Tail: one-sided greater.
- t(299) = 18.455892100362185.
- p = 1.3359277323396141e-51.
- Multiplicity correction: none; the inferential family contains one test.

## Decision gates

- Fresh canonical sign gate `mean(D_CAN)>0`: PASS.
- Paired specificity gate `mean(S)>0` and `p<0.05`: PASS.

## Adjacent geometry

- Strong dimension: `1205`.
- Strong-index SHA256: `3c1d39df9b9b7a9acd3200b9b8cb31576bd6cc81fcd4518781f2cbbee3533141`.
- Geometry-only response-blind control: `P4`.
- `P1` lambda_plus: `0.96226310619693978`.
- `P2` lambda_plus: `0.97332929387559231`.
- `P3` lambda_plus: `0.98819701331746213`.
- `P4` lambda_plus: `0.99470306568545441`.
- `P5` lambda_plus: `0.99770810884887617`.

## Forward accounting

- Adjacent geometry: `2400` new scientific model forwards.
- Canonical geometry reruns: `0`.
- Canonical fresh response: `24000` forwards.
- Adjacent fresh response: `24000` forwards.
- Paired response total: `48000` forwards.
- Experiment 5 total new scientific model forwards: `50400`.
- Static analysis model forwards: `0`.

## Claim boundary

On the prospectively frozen Mamba-1.4B XG1 5101..5400 cohort, the canonical (33,34,35) homologous site carried a stronger positive rank-aligned P5 core signal than the single architecture-predeclared adjacent +1 site (34,35,36) under the matched frozen measurement procedure.

This experiment does not rescue Experiments 1–3 and does not establish a global layer optimum or uniqueness across all layers.
