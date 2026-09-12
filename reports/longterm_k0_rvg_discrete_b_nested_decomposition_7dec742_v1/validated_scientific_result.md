# K0-RVG Nested Discrete-B Decomposition

## Status

VALIDATED SCIENTIFIC EVIDENCE

Execution commit:

`7dec7420b2a925ded93ede241eee680e7c608f8c`

Parent write-factor freeze:

`076ffca0e2305442327f698cf012e9070e3bf69c`

The experiment reused the frozen-token, divergence-aligned,
equal-length-prefix CPU sequential Mamba protocol.

No tokenizer, logits, task heads, training, causal intervention, PCA,
probe, or learned geometry was executed.

## Parent term

The previous validated decomposition defined:

`Q_D = mean(U_m,U_s) * (D_m-D_s)`

where:

`D = discrete_B`

and established that Q_D is a substantial component of the layer-23
corr-specific k=2 write transient.

This experiment preserved that exact parent Q_D term.

## Nested factorization

The frozen Mamba slow path computes:

`discrete_B = discrete_time_step * B`

Define:

`T = discrete_time_step`

The exact nested symmetric decomposition was:

`Q_D = Q_T + Q_B`

with:

`Q_T = mean(U_m,U_s) * mean(B_m,B_s) * (T_m-T_s)`

`Q_B = mean(U_m,U_s) * mean(T_m,T_s) * (B_m-B_s)`

Interaction:

`2 * dot(Q_T, Q_B)`

## Artifact authentication

execution_manifest.json SHA256:

`f577aad10c81f4b7837ad8b5004c66f07eff618cdbf26d023f2ce2c1629dcd7f`

summary.json SHA256:

`b537e9b92aeb3070d0b9dadbd3835d2f070a60c73704a8e88fb0e4fb51056715`

The full discrete_b_nested_metrics.jsonl remains in the validated local run artifact.

SHA256:

`9896f352faa9467c63fa6cdeb5f6265d566327fb461201d63d3856b837de8b08`

## Validation

Rows:

`5376`

Common paired DDSSSSS cohort:

`330`

At k=-1 all 672 pair-role cases showed exact identity for:

- hidden_states
- discrete_time_step
- B
- discrete_B
- write

Maximum nested vector composition relative residual:

`4.519292652146182e-06`

Maximum energy-identity relative error:

`1.8329013840647644e-06`

All common-cohort summary medians were independently recomputed.

All parent Q_D role/k medians exactly reproduced the frozen parent
write-factor summary.

## Common-330 decomposition

Median quantities normalized by pair-specific k=0 write divergence.

corr:

- k=1: Q_T 0.180223, Q_B 0.329327, Q_D 0.379018
- k=2: Q_T 0.539309, Q_B 0.278598, Q_D 0.657867
- k=3: Q_T 0.040379, Q_B 0.038158, Q_D 0.062947

ctrl:

- k=1: Q_T 0.155310, Q_B 0.276403, Q_D 0.333874
- k=2: Q_T 0.149992, Q_B 0.108668, Q_D 0.182233
- k=3: Q_T 0.037067, Q_B 0.036366, Q_D 0.055388

## Corr k=1 -> k=2 rebound

Within the common 330-item cohort:

- Q_D increased: 302/330
- Q_T increased: 326/330
- Q_B increased: 112/330
- Q_D rebound and Q_T increase: 302/330
- Q_D rebound and Q_B increase: 112/330
- Q_D rebound with both Q_T and Q_B increasing: 112/330
- Q_D rebound with Q_T increasing but Q_B not increasing: 190/330
- Q_D rebound with Q_B increasing but Q_T not increasing: 0/330

Thus every observed Q_D rebound case was accompanied by an increase in
the time-step contribution, whereas most did not require an increase in
the B contribution.

## Corr k=2 time-step versus B dominance

At corr k=2:

- Q_T > Q_B: 330/330
- Q_B > Q_T: 0/330
- median Q_T / Q_B magnitude ratio: 1.8844492717

This establishes cohort-wide time-step-side dominance within the
discrete-B contribution.

## Corr versus ctrl at k=2

Within the common paired cohort:

- corr Q_D > ctrl Q_D: 330/330
- corr Q_T > ctrl Q_T: 329/330
- corr Q_B > ctrl Q_B: 328/330

Median corr/ctrl ratios:

- Q_D: 3.6038825577
- Q_T: 3.5609017237
- Q_B: 2.6383791660

Both nested factors participate in role separation, but the time-step
contribution is larger and more tightly associated with the corr k=2
transient.

## Interaction

At corr k=2:

- positive Q_T/Q_B interaction: 330/330
- negative interaction: 0/330
- median cosine: 0.1500312741

Median energy fractions relative to Q_D squared:

- Q_T squared contribution: 0.694838
- Q_B squared contribution: 0.194889
- interaction: 0.108899

The interaction is systematic and constructive, but unlike the parent
Q_U/Q_D decomposition it is not the dominant structure.

## Corr k=2 -> k=3 collapse

Within the common cohort:

- Q_D decreased: 330/330
- Q_T decreased: 330/330
- Q_B decreased: 330/330
- both nested contributions decreased: 330/330

Median k3/k2 ratios:

- Q_D: 0.0871640961
- Q_T: 0.0691592417
- Q_B: 0.1333086398

The time-step contribution collapses even more sharply than the B
contribution.

## Scientific interpretation

The validated evidence supports the following narrow conclusion:

The layer-23 corr-specific k=2 discrete-B transient is primarily associated
with selective time-step modulation rather than selective B modulation.

The evidence is cohort-wide at k=2: Q_T exceeds Q_B in all 330 common
items, every observed Q_D rebound is accompanied by Q_T growth, and most
Q_D rebound cases occur without Q_B growth.

B remains nonzero and role-associated, so the result does not imply that B
is irrelevant. The conclusion is specifically that the time-step pathway is
the dominant component of the validated discrete-B-side transient.

This is an observational factor decomposition and does not establish causal
downstream relevance.

## Next scientific boundary

The frozen slow path defines discrete_time_step as the softplus-transformed
output of dt_proj(time_step).

The next minimal question is therefore whether the corr k=2 time-step
transient is already present in the pre-softplus dt projection difference,
or whether changing local softplus sensitivity materially shapes the
observed selective time-step transient.

No deeper claim should be made until that distinction is measured.