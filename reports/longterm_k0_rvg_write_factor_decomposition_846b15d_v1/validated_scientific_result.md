# K0-RVG Write-Factor Symmetric Decomposition

## Status

VALIDATED SCIENTIFIC EVIDENCE

Execution commit:

`846b15dabead9c207169535c5b76a9a2f5a3f067`

Parent carry/write freeze:

`aecd6d93cbb4edaa6cae6f68418a30bfecda1ceb`

The experiment reused the frozen-token, divergence-aligned,
equal-length-prefix CPU sequential Mamba protocol.

No tokenizer, logits, task heads, training, causal intervention, PCA,
probe, or learned geometry was executed.

## Factorization

The validated layer-23 write term was:

`W = discrete_B * hidden_states`

Define:

`D = discrete_B`

`U = conv-activated hidden_states`

The matched-versus-swapped write difference was decomposed by the exact
symmetric bilinear identity:

`delta_W = Q_U + Q_D`

where:

`Q_U = mean(D_m,D_s) * (U_m-U_s)`

`Q_D = mean(U_m,U_s) * (D_m-D_s)`

The interaction term was:

`2 * dot(Q_U, Q_D)`

## Artifact authentication

execution_manifest.json SHA256:

`87db94441117bf533571559725327f0016ee85774c2f84f723c2529c75a955b4`

summary.json SHA256:

`583fb225b62bab5c3e19010bd4c9869061c6d30edb5a71218d697bbb60c1c2c0`

Full write_factor_metrics.jsonl remains in the validated local run artifact.

SHA256:

`da85f69c173790d8a37693974e68ad0f7d70e2c3219cdfdd0181637a46060a7e`

## Validation

Rows:

`5376`

Common paired DDSSSSS cohort:

`330`

At k=-1:

- discrete_B exact identity: 672/672
- hidden_states exact identity: 672/672
- write exact identity: 672/672

Maximum symmetric vector composition relative residual:

`2.5611153157125365e-06`

Maximum energy-identity relative error:

`1.3419456910389242e-06`

The symmetric factor decomposition therefore closes numerically within the
validated tolerance.

## Common-330 trajectory

Median magnitudes normalized by pair-specific k=0 write divergence:

corr:

- k=1: Q_U 0.3630, Q_D 0.3790, W 0.4659
- k=2: Q_U 0.9681, Q_D 0.6579, W 0.8677
- k=3: Q_U 0.1286, Q_D 0.0629, W 0.1286

ctrl:

- k=1: Q_U 0.3016, Q_D 0.3339, W 0.4639
- k=2: Q_U 0.2708, Q_D 0.1822, W 0.2296
- k=3: Q_U 0.1034, Q_D 0.0554, W 0.1072

## Corr k=1 -> k=2 rebound

Within the common 330-item cohort:

- write increased: 313/330
- Q_U increased: 302/330
- Q_D increased: 302/330
- write and Q_U increased: 302/330
- write and Q_D increased: 297/330
- Q_U and Q_D both increased during write rebound: 286/330

Thus the corr write rebound is generally accompanied by amplification of both
factor-side contributions.

## Corr k=2 factor dominance

At corr k=2:

- Q_U > Q_D: 323/330
- Q_D > Q_U: 7/330
- median Q_U / Q_D magnitude ratio: 1.4085723613

Thus the U-side contribution is usually larger, but the D-side contribution is
not negligible.

## Corr versus ctrl at k=2

For the common paired cohort:

- corr write > ctrl write: 330/330
- corr Q_U > ctrl Q_U: 328/330
- corr Q_D > ctrl Q_D: 330/330

Median corr/ctrl ratios:

- write: 3.8762485511
- Q_U: 3.4743458589
- Q_D: 3.6038825577

The role-associated k=2 write separation is therefore present in both the
conv-activated input-stream contribution and the selective discrete_B
contribution.

## Interaction and cancellation

At corr k=2:

- positive Q_U/Q_D interaction: 0/330
- negative interaction: 330/330
- median cosine between Q_U and Q_D: -0.4768861565

Median energy fractions at corr k=2:

- Q_U squared contribution / W squared: 1.192508
- Q_D squared contribution / W squared: 0.544567
- interaction / W squared: -0.746427

Thus Q_U and Q_D are both large, but their vector contributions systematically
oppose one another. The final write divergence is substantially determined by
destructive composition between the two factors.

This prevents interpreting the write rebound as a simple single-factor
amplification.

## Corr k=2 -> k=3 collapse

Within the common cohort:

- write decreased: 330/330
- Q_U decreased: 330/330
- Q_D decreased: 330/330
- both Q_U and Q_D decreased: 330/330

Median ratios:

- W(k=3) / W(k=2): 0.1350069799
- Q_U(k=3) / Q_U(k=2): 0.1402929502
- Q_D(k=3) / Q_D(k=2): 0.0871640961

Thus the immediate collapse reflects collapse of both factor-side
contributions.

## Scientific interpretation

The validated evidence supports the following narrow conclusion:

The layer-23 corr-specific k=2 write rebound is a two-factor transient.
Both the conv-activated hidden-state contribution and the selective
discrete_B contribution increase strongly. The hidden-state contribution is
usually larger, but discrete_B also shows a strong and cohort-wide
corr-versus-ctrl separation.

The two factor contributions exhibit systematic destructive alignment,
especially at corr k=2, so the observed write magnitude is not the sum of two
independent positive magnitude effects.

The k=3 collapse is likewise shared by both factors.

These results do not establish causal downstream relevance.

## Next scientific boundary

Because discrete_B makes a substantial and systematic contribution, the next
minimal native-Mamba question is to decompose:

`discrete_B = discrete_time_step * B`

using the same exact symmetric bilinear decomposition.

Only after that decomposition should any deeper interpretation of selective
time-step versus B modulation be made.