# Gen4 PP3-Excluded Residual Individual-Plane Restoration Sufficiency Localization — Confirmatory Result

## Status

`PP3_EXCLUDED_INDIVIDUAL_RESIDUAL_PLANE_RESTORATION_SUFFICIENCY_LOCALIZATION_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

The completed prospective fresh-XG1 experiment supports individual local
restoration sufficiency for the unordered pre-registered set:

`{P1, P2, P4, P5}`

No plane ranking is inferred.

Exactly four one-sided Student t-tests were executed across the frozen family
`{P1, P2, P4, P5}`, followed by Holm step-down control of familywise error
at `alpha = 0.05`.

No fifth p-value or additional confirmatory analysis was executed.

## Frozen chain

Prospective design:

`8c0ff6dbad77ed876fc1481b3b53c3fd47a27d3b`

Static preparation:

`4a1d5871fad17a34951bc433a283c95e682a081b`

Implementation authority:

`25ff56c80ccd943bb386ebc7ab6612fe6e68d470`

Implementation:

`466741783dd1fc9325e7d87a2d1a44ccb0a09de3`

Original execution freeze:

`9467db0d55670f0f6145d28885ccba60177b481b`

Byte-provenance correction freeze:

`90ea0f3d12a520ea6a28ebd9cae787d19152f77b`

Retry1 execution freeze / accepted execution HEAD:

`09be091bce8a4ba10ef5010c16840826ab4d8aa2`

## Accepted execution

Run:

`g4k-residual-individual-plane-restoration-sufficiency-xg1-2101-2400-09be091-retry1`

Pinned command SHA256:

`27224ec2b13e5ea30ea7e21c2b046a75ff583ab5178a7b84379557543e8a66e4`

Imported handoff ZIP SHA256:

`fdf247431318ab5a9dec2aa80d413a1c1c2721fa58a390da49770b3027d4b9a4`

Run log SHA256:

`ad781d6a5353709a7d2ffb674e93c085a7f520a3e5674fe8f86b527f903b9d01`

Run metadata SHA256:

`352c078fc4d9cae74307cef2c72bae3d6fee15edeaede0428f071f2699e800fc`

Started UTC:

`2026-09-18T10:28:00Z`

Finished UTC:

`2026-09-18T11:10:49Z`

Exit code:

`0`

Scientific model forwards:

`108000`

Baseline model forwards:

`0`

Raw primary inference:

`False`

Raw multiplicity correction:

`False`

Raw scientific conclusion:

`None`

## Population

Prospective fresh XG1 holdout:

`xg1_fact_2101..xg1_fact_2400`

Pair count:

`N = 300`

Degrees of freedom:

`df = 299`

## Frozen endpoint

For each plane `Pk`:

`B_k = h - c_k`

`R_k = B_k + c_k = h`

`C_k = B_k + r_k`

with the frozen equal-norm orthogonal within-plane quarter-turn replacement
`r_k`.

Endpoints:

`Q0 = Q(native)`

`Q_B,k = Q(B_k)`

`Q_C,k = Q(C_k)`

Exact native restoration gain:

`S_k = Q0 - Q_B,k`

Matched replacement gain:

`S_C,k = Q_C,k - Q_B,k`

Primary contrast:

`D_SUF,k = S_k - S_C,k = Q0 - Q_C,k`

The stored canonical endpoint and expanded restoration form were revalidated
item-by-item before confirmatory inference.

## Raw artifact validation

Import validation:

`PASS`

Frozen runner artifact validation:

`PASS`

Validated item count:

`300`

Endpoint revalidation:

`PASS`

Raw scientific model forwards:

`108000`

Raw baseline forwards:

`0`

No raw artifact contained primary inference, multiplicity correction, or a
scientific conclusion.

## Confirmatory family

Frozen hypotheses for each `k in {P1,P2,P4,P5}`:

`H0,k: mean(D_SUF,k) <= 0`

`H1,k: mean(D_SUF,k) > 0`

Test:

one-sample Student t-test, one-sided greater

Raw confirmatory p-value count:

`4`

Additional p-value count:

`0`

Multiplicity:

`Holm step-down across exactly P1,P2,P4,P5`

Familywise alpha:

`0.05`

SciPy version used for confirmatory computation and cross-check:

`1.18.1`

## Shared positive gate

Mean native endpoint:

`mean(Q0) = 1.864160988613634e-07`

Shared `mean(Q0) > 0` gate:

`PASS`

## P1 result

Mean neutralized-background endpoint:

`mean(Q_B,P1) = 1.5865887026271439e-07`

Mean quarter-turn replacement endpoint:

`mean(Q_C,P1) = 1.4691196899759734e-07`

Mean exact native restoration gain:

`mean(S_P1) = 2.7757228598649021e-08`

Mean matched replacement gain:

`mean(S_C,P1) = -1.1746901265117048e-08`

Mean restoration-specific contrast:

`mean(D_SUF,P1) = 3.9504129863766069e-08`

Sample standard deviation:

`sd(D_SUF,P1) = 3.4811629781005159e-08`

Student t statistic:

`t(299) = 19.655259022137678`

One-sided raw p-value:

`p = 4.2852749015640899e-56`

Holm step-down threshold at its ordered step:

`0.012500000000000001`

Holm rejection:

`PASS`

Frozen gates:

- `mean(Q0) > 0`: `PASS`
- `mean(S_P1) > 0`: `PASS`
- `mean(D_SUF,P1) > 0`: `PASS`
- Holm rejection: `PASS`

P1 individual local restoration sufficiency:

`SUPPORTED`

## P2 result

Mean neutralized-background endpoint:

`mean(Q_B,P2) = 1.5921102209838953e-07`

Mean quarter-turn replacement endpoint:

`mean(Q_C,P2) = 1.6842987499465523e-07`

Mean exact native restoration gain:

`mean(S_P2) = 2.7205076762973886e-08`

Mean matched replacement gain:

`mean(S_C,P2) = 9.2188528962656873e-09`

Mean restoration-specific contrast:

`mean(D_SUF,P2) = 1.79862238667082e-08`

Sample standard deviation:

`sd(D_SUF,P2) = 1.9816848472378611e-08`

Student t statistic:

`t(299) = 15.720488359624252`

One-sided raw p-value:

`p = 2.6290903850154124e-41`

Holm step-down threshold at its ordered step:

`0.016666666666666666`

Holm rejection:

`PASS`

Frozen gates:

- `mean(Q0) > 0`: `PASS`
- `mean(S_P2) > 0`: `PASS`
- `mean(D_SUF,P2) > 0`: `PASS`
- Holm rejection: `PASS`

P2 individual local restoration sufficiency:

`SUPPORTED`

## P4 result

Mean neutralized-background endpoint:

`mean(Q_B,P4) = 1.8458051998315244e-07`

Mean quarter-turn replacement endpoint:

`mean(Q_C,P4) = 1.8360230203055157e-07`

Mean exact native restoration gain:

`mean(S_P4) = 1.8355788782109717e-09`

Mean matched replacement gain:

`mean(S_C,P4) = -9.7821795260087353e-10`

Mean restoration-specific contrast:

`mean(D_SUF,P4) = 2.8137968308118452e-09`

Sample standard deviation:

`sd(D_SUF,P4) = 1.2751456938511514e-08`

Student t statistic:

`t(299) = 3.8220252765181728`

One-sided raw p-value:

`p = 8.0520282867827463e-05`

Holm step-down threshold at its ordered step:

`0.050000000000000003`

Holm rejection:

`PASS`

Frozen gates:

- `mean(Q0) > 0`: `PASS`
- `mean(S_P4) > 0`: `PASS`
- `mean(D_SUF,P4) > 0`: `PASS`
- Holm rejection: `PASS`

P4 individual local restoration sufficiency:

`SUPPORTED`

## P5 result

Mean neutralized-background endpoint:

`mean(Q_B,P5) = 1.7260996504110435e-07`

Mean quarter-turn replacement endpoint:

`mean(Q_C,P5) = 1.709353343703456e-07`

Mean exact native restoration gain:

`mean(S_P5) = 1.3806133820259049e-08`

Mean matched replacement gain:

`mean(S_C,P5) = -1.6746306707587528e-09`

Mean restoration-specific contrast:

`mean(D_SUF,P5) = 1.5480764491017802e-08`

Sample standard deviation:

`sd(D_SUF,P5) = 2.0648495920540842e-08`

Student t statistic:

`t(299) = 12.985677378940377`

One-sided raw p-value:

`p = 3.5028143739779681e-31`

Holm step-down threshold at its ordered step:

`0.025000000000000001`

Holm rejection:

`PASS`

Frozen gates:

- `mean(Q0) > 0`: `PASS`
- `mean(S_P5) > 0`: `PASS`
- `mean(D_SUF,P5) > 0`: `PASS`
- Holm rejection: `PASS`

P5 individual local restoration sufficiency:

`SUPPORTED`


## Holm family conclusion

Holm-rejected planes:

`{P1, P2, P4, P5}`

Frozen supported set, unordered:

`{P1, P2, P4, P5}`

All four prospectively registered PP3-excluded residual planes satisfy their
individual restoration-sufficiency gates under familywise error control.

## Scientific conclusion

On fresh XG1 `2101..2400`, P1, P2, P4, and P5 each show an individually
detectable local restoration-sufficiency contribution to the frozen
layer-17 / target-token native-Mamba susceptibility endpoint.

For each supported plane, starting from that plane's exact native-component-
neutralized state, restoring the exact removed native component recovers the
endpoint more strongly than adding its pre-specified equal-norm orthogonal
within-plane quarter-turn replacement.

Together with the previously validated individual-plane necessity result,
the accumulated evidence now supports both individual local necessity and
individual local restoration sufficiency for each of P1, P2, P4, and P5
within the frozen PP3-excluded residual mechanism.

This remains a distributed local mechanism statement.

## Interpretation boundary

This result does **not** establish:

- a ranking or dominance ordering among P1, P2, P4, and P5;
- that smaller p-value implies greater scientific importance;
- equality of plane effects;
- additive decomposition of the aggregate residual mechanism;
- independence of plane contributions;
- absence of interactions among residual planes;
- aggregate residual restoration sufficiency;
- causal status of the XG2-like residual-template orientation itself;
- behavioral or downstream-task sufficiency;
- benchmark improvement;
- checkpoint-, layer-, token-, generator-, dataset-, architecture-, or
  model-wide universality.

The supported set is intentionally unordered.

## Failed-attempt provenance

The original `90ea0f3...` execution attempt failed during zero-forward
external preflight because `validate_transformers_kernel_bindings()` was
invoked before model-construction lazy-loader binding of `mamba_ssm` and
`causal_conv1d`.

Scientific model forwards:

`0`

Scientific conclusion:

`None`

Retry1 corrected only that external preflight ordering error. The frozen
scientific runner and scientific design were unchanged.

The first local read-only confirmatory-analysis invocation after import also
stopped before any p-value computation because its orchestration compared the
frozen multiplicity string against an abbreviated string. The corrected
read-only invocation used the frozen runner constant verbatim and completed
the exact four pre-registered tests.

No scientific artifact was modified by either failed diagnostic attempt.

## Raw artifact identities

`pp3_excluded_residual_individual_plane_restoration_sufficiency_items.jsonl`

Original bytes:

`164201415`

Original SHA256:

`f81a90f2f33a7b6f13c09a28a21c0372ae61293155d6c08612daf3f5e33b0adc`

`pp3_excluded_residual_individual_plane_restoration_sufficiency_summary.json`

SHA256:

`e2667887301dad0871e6235908152022678aae0a93b593520d490fd2f9fec14c`

`artifact_manifest.json`

SHA256:

`cb1d3936b5c763a1901269eb928714beaf879489c6000f3bbbd055851f001ce5`

`SHA256SUMS.txt`

SHA256:

`c0e58631be068fa2a4b1cdf84b1a485c7ba8196926e695db171b259636a03927`

## Repository archival form for oversized raw item artifact

The original raw JSONL exceeds GitHub's 100 MiB object limit.

The repository therefore stores the deterministic gzip archival copy:

`pp3_excluded_residual_individual_plane_restoration_sufficiency_items.jsonl.gz`

Deterministic gzip settings:

- compression level: `9`
- `mtime = 0`
- embedded filename: empty

Archived gzip bytes:

`5659579`

Archived gzip SHA256:

`9d7127f26eced14471a7fdcd1a6b70027b0e44c1c1a81cfd2eaffc8cbc3d8ee7`

Decompressing the archive reproduces exactly:

- bytes: `164201415`
- SHA256: `f81a90f2f33a7b6f13c09a28a21c0372ae61293155d6c08612daf3f5e33b0adc`

The original uncompressed JSONL remains preserved locally and is excluded from
Git history solely because of the remote object-size limit.

The original scientific `artifact_manifest.json` and `SHA256SUMS.txt` remain
unchanged and continue to describe the original uncompressed raw artifact set.

## Final status

Raw execution:

`PASS`

Artifact/provenance validation:

`PASS`

Endpoint revalidation:

`PASS`

Confirmatory inference:

`PASS`

Raw confirmatory p-values:

`4`

Additional p-values:

`0`

Holm familywise correction:

`PASS`

Supported planes, unordered:

`{P1, P2, P4, P5}`

Scientific conclusion:

`PP3_EXCLUDED_INDIVIDUAL_RESIDUAL_PLANE_RESTORATION_SUFFICIENCY_LOCALIZATION_SUPPORTED_ON_FRESH_XG1_HOLDOUT`
