# Gen4 PP3 Restoration Sufficiency — Validated Result

## Status

`VALIDATED_CONFIRMATORY_RESULT`

Scientific conclusion:

`PP3_RESTORATION_SUFFICIENCY_OVER_MATCHED_PP5_REPLACEMENT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

## Frozen chain

- restoration-sufficiency design:
  `854bcd5585512155776c34e46846817c54b75cf6`
- static preparation:
  `0f907574f1ca25ec12e35573a83e1b499ed1b53b`
- implementation authority:
  `cae566ed458e5c6f93c86ce03b029950035652dc`
- implementation freeze:
  `3b0154439aac51e1199cfa18842818430b8406c7`
- execution freeze / execution HEAD:
  `53b6ccea8a06fbf9be699bffa034034807970878`

## Accepted execution

Run:

`gen4-pp3-restoration-sufficiency-xg1-901-1200-53b6cce-retry2`

Command SHA256:

`2a310ee599eb02088d8ed99719b70ee70d95f2091d80c72d7f9593802185d805`

Imported handoff ZIP SHA256:

`a3e7d1e93d674eca3c9dbf5d8298ff7b81bf62d5a2bc995f9bf43a96d5522acc`

Run log SHA256:

`3608ee518531a5e27d573ae16c3c8a8c5d9faff3cfeb30179202a3110b940849`

Run metadata SHA256:

`c8e9dfcdaa427860ac1e9047939f568cd05de01c2592286dded40bcdecbdf68a`

Execution:

- exit code: `0`
- scientific model forwards: `36000`
- baseline forwards: `0`
- primary inference during GPU run: `false`
- scientific conclusion during GPU run: `null`
- local import: `PASS`
- imported files validated: `4`

The earlier failed attempts are not scientific evidence.
The accepted execution is `retry2` only.

## Representative checkpoint

Checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Checkpoint bytes:

`518270455`

## Raw artifact identities

`pp3_restoration_sufficiency_items.jsonl`

SHA256:

`8c4fade5f0e2c02ed183352a573fe9dc70af210610ec0ab5d00cb2f27371444d`

`pp3_restoration_sufficiency_summary.json`

SHA256:

`1f78cae47d27730c0c988df877afc861461930dd9f978c26d194172908c0597d`

`artifact_manifest.json`

SHA256:

`562c51ea985721cd89e8f4bc0c7ccf26e20cb5ccd3bb6b159c8d042095ed2857`

`SHA256SUMS.txt`

SHA256:

`f88efc1b0d3b8ebe3e8ecbc4c5820b3233ebf3f654d601eac588dae65c4d6e88`

Post-import frozen artifact validation:

`PASS`

Validated item count:

`300`

Execution HEAD:

`53b6ccea8a06fbf9be699bffa034034807970878`

Raw result:

`PASS_PP3_RESTORATION_SUFFICIENCY_RAW_OBSERVATION`

## Prospectively frozen primary inference

Exactly one confirmatory hypothesis was tested.

Endpoint:

`D_SUF_i = (Q_R3_i - Q_B_i) - (Q_R5_i - Q_B_i) = Q_R3_i - Q_R5_i`

Hypotheses:

- `H0: mean(D_SUF) <= 0`
- `H1: mean(D_SUF) > 0`

Test:

- one-sample Student t-test
- one-sided alternative: greater
- N: `300`
- df: `299`
- alpha: `0.05`
- multiplicity correction: none
- confirmatory p-value count: `1`

Observed descriptive quantities:

- mean Q_B:
  `1.0507711352505444e-07`
- mean Q_R3:
  `1.8577153849600195e-07`
- mean Q_R5:
  `1.4498281493001444e-07`
- mean S3:
  `8.069442497094754e-08`
- mean S5:
  `3.9905701404959994e-08`
- mean D_SUF:
  `4.078872356598753e-08`
- SD D_SUF:
  `3.5139279965303874e-08`

Primary statistic:

- t:
  `20.105176219299192`
- one-sided p:
  `9.010901911776928e-58`

Frozen positive gates:

- `mean(Q_R3) > 0`: PASS
- `mean(S3) > 0`: PASS
- `mean(D_SUF) > 0`: PASS
- one-sided `p < 0.05`: PASS

Therefore the prospectively frozen positive label applies:

`PP3_RESTORATION_SUFFICIENCY_OVER_MATCHED_PP5_REPLACEMENT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

No rescue test, subgroup test, alternative tail, additional p-value,
alternative control, epsilon sweep, layer sweep, token sweep, checkpoint
sweep, response-guided plane modification, or task-head/logit analysis was
performed.

## Scientific interpretation

On fresh XG1 `901..1200`, after removal of the native PP3 component from the
frozen layer-17 target-token strong-channel state, restoring that exact native
PP3 component recovered the broad XG2-vs-XG4 local susceptibility contrast
more strongly than inserting the frozen equal-coefficient,
equal-addition-norm PP5 replacement.

This supports PP3 as a local restoration-sufficient contributor relative to
the prospectively frozen matched PP5 replacement.

Together with the previously validated fresh-XG1 matched-control necessity
result, the accumulated evidence supports PP3 as both a locally necessary
contributor and a locally restoration-sufficient contributor within the
frozen layer-17 / target-token susceptibility mechanism.

This does not establish PP3 as:

- the sole mechanism;
- sufficient in an otherwise empty state;
- globally behaviorally sufficient;
- sufficient for downstream task behavior;
- universally sufficient across arbitrary generators;
- sufficient across arbitrary checkpoints, architectures, layers, or token
  positions.

The distributed-mechanism interpretation remains mandatory.
