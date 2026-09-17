# Gen4 PP3 Necessity — Validated Result

## Status

`VALIDATED_CONFIRMATORY_RESULT`

Scientific conclusion:

`PP3_NECESSITY_OVER_MATCHED_PP5_CONTROL_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

## Frozen chain

- corrected design: `f4419f1e7efbeeb9e50e84b67accc56422580cf2`
- static preparation: `e470f132a37c731754feb4333dcb2e48e29af53b`
- implementation authority: `0f428d84a5065268c9a4dfaf80855483c4e99c42`
- implementation freeze: `f1896e0d3669bac832038e1486b2b04d322af0aa`
- execution freeze / execution HEAD:
  `dc85079cb21b78610b4de6a12179ea5e7d72f1b4`

## Accepted execution

Run:

`gen4-pp3-necessity-xg1-601-900-dc85079-retry2`

Command SHA256:

`f0e0f9a2926fac614e1e89e2ec93e1ecff469a1bc7b62baa5b9e06bc2b58aab8`

Imported handoff ZIP SHA256:

`3c56ccd649380ce0fc2c78704d731ea0a2906eaec507cb047460bfc34b9bc9a6`

Run log SHA256:

`72e3eb1e865fa08a056c3abeb85dd5f7904d1e0be61ce75005ee42f55aedf9c4`

Run metadata SHA256:

`7cc9e654139fb8f39692f7d5c5055d992a9651beb8cba611847b0d9ee72d6664`

Execution:

- exit code: `0`
- scientific model forwards: `36000`
- baseline forwards: `0`
- primary inference during GPU run: `false`
- scientific conclusion during GPU run: `null`

The earlier failed attempts are not scientific evidence. The accepted execution
is retry2 only.

## Representative checkpoint

Checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Checkpoint bytes:

`518270455`

The imported checkpoint is provenance input only and is not part of this result
commit.

## Raw artifact identities

`pp3_necessity_items.jsonl`

SHA256:

`67d1d550c45e73a346a0a3b4b3fe4fea39c8d7ac89104ba74b1528b265f8cce0`

`pp3_necessity_summary.json`

SHA256:

`a6fc7e10cb8856c2951a90e57ea2daa06744049be20c2964c561187a0e28476d`

`artifact_manifest.json`

SHA256:

`96db0221d6c697d8dcec10a1feb7b4049fb6214937250559fcf3a4f97f627ab5`

`SHA256SUMS.txt`

SHA256:

`d8381cbecb6454d6428d8fa1a446998496b1845d29d419343f9a6ed22d0f945c`

Post-import artifact validation:

`PASS`

Validated item count:

`300`

## Prospectively frozen primary inference

Exactly one confirmatory hypothesis was tested.

Endpoint:

`D_NEC_i = (Q0_i - Q3_i) - (Q0_i - Q5_i) = Q5_i - Q3_i`

Test:

- one-sample Student t-test
- one-sided alternative: `mean(D_NEC) > 0`
- N: `300`
- df: `299`
- alpha: `0.05`
- multiplicity correction: none

Observed descriptive quantities:

- mean Q0:
  `1.8678970016861961e-07`
- mean A3:
  `8.4608268084375035e-08`
- mean D_NEC:
  `4.7414371121837106e-08`
- SD D_NEC:
  `3.4241842639833269e-08`

Primary statistic:

- t:
  `23.983551544160797`
- one-sided p:
  `6.2242759364566024e-72`

Frozen positive gates:

- `mean(Q0) > 0`: PASS
- `mean(A3) > 0`: PASS
- `mean(D_NEC) > 0`: PASS
- one-sided `p < 0.05`: PASS

Therefore the prospectively frozen positive label applies:

`PP3_NECESSITY_OVER_MATCHED_PP5_CONTROL_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

No rescue test, subgroup test, tail test, alternative-control test, epsilon
sweep, layer sweep, token sweep, checkpoint sweep, or additional p-value was
performed.

## Scientific interpretation

On fresh XG1 `601..900`, selectively removing the frozen PP3 component
attenuated the established broad XG2-vs-XG4 susceptibility contrast more than
the intervention-magnitude-matched PP5 coefficient-transfer control.

This supports PP3 as a local causal necessity contributor to the frozen
layer-17 / target-token susceptibility mechanism.

It does not establish PP3 as:

- the sole mechanism;
- globally behaviorally necessary;
- sufficient by itself;
- necessary across arbitrary generators, layers, tokens, checkpoints, or
  architectures.

The distributed-mechanism interpretation remains appropriate.
