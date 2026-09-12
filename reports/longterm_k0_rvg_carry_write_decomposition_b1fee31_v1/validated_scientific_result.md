# K0-RVG Carry / Write Recurrence Decomposition

## Status

VALIDATED SCIENTIFIC EVIDENCE

Execution commit:

`b1fee3118375174d26000c86f99553292e95ea4a`

Parent magnitude-trajectory freeze:

`c01cdf8bd1c61f244bb761b2de624b794578a6ba`

The experiment reused the frozen-token, equal-length-prefix, CPU sequential
Mamba recurrence protocol.

No tokenizer, logits, task heads, training, causal intervention, PCA, probe,
or learned geometry was executed.

## Recurrence decomposition

The layer-23 recurrence was decomposed as:

`S_post = carry + write`

with:

`carry = G * S_prev`

`write = W`

Matched-versus-swapped divergence was measured for:

- delta carry
- delta write
- delta S_post
- `2 * dot(delta_carry, delta_write)`

The common primary cohort contains 330 items with token-equality signature:

`DDSSSSS`

Thus token IDs differ at k=0 and k=1 and are identical from k=2 through k=6.

## Artifact authentication

execution_manifest.json SHA256:

`593fd2a4d4225398c529f837b669c220f378c12a2fdec3815c776219b5552643`

summary.json SHA256:

`437c615b7fd63f6f49ca6f531202371fa084f528ec6f4f52aafe133b0236cc91`

Full carry_write_metrics.jsonl remains in the validated local run artifact.

SHA256:

`57ad5df7a464686c750cd2cffc6e5fae24663b8ca641f0465d783cac8f58d803`

## Validation

Artifact hashes and manifest contract passed.

Common corr/ctrl DDSSSSS cohort identity passed at 330/330 items.

At k=-1, carry, write, and S_post were exact-identical for all 672
role-level comparisons.

Maximum vector composition relative residual:

`1.2166710768220438e-06`

Maximum energy-identity relative error:

`6.093907972638283e-07`

The decomposition therefore closes numerically within the validated tolerance.

## Common-330 result

Median normalized magnitudes relative to pair-specific k=0 S_post divergence:

corr:

- k=1: carry 0.0362, write 0.4666, state 0.4684
- k=2: carry 0.0273, write 0.8692, state 0.8688
- k=3: carry 0.0278, write 0.1289, state 0.1337

ctrl:

- k=1: carry 0.0376, write 0.4641, state 0.4639
- k=2: carry 0.0282, write 0.2303, state 0.2314
- k=3: carry 0.0206, write 0.1074, state 0.1097

Write energy accounts for nearly all state-divergence energy at the key
coordinates. Median carry/write interaction energy is small and does not
explain the rebound or collapse.

## Corr k=1 -> k=2 rebound

Within the common 330-item cohort:

- S_post divergence increased: 313/330
- write divergence increased: 313/330
- carry divergence increased: 7/330
- S_post and write both increased: 313/330
- S_post and write increased while carry decreased: 306/330

Thus the corr rebound is not a carry-amplification phenomenon.

## Corr versus ctrl at k=2

For every common-cohort item:

- corr S_post divergence > ctrl S_post divergence: 330/330
- corr write divergence > ctrl write divergence: 330/330

Carry showed no comparable universal role separation:

- corr carry divergence > ctrl carry divergence: 173/330

Median corr/ctrl ratios:

- S_post divergence: 3.8488838104
- write divergence: 3.8779645883
- carry divergence: 1.0183866692

The corr-versus-ctrl state transient therefore tracks write divergence rather
than carry divergence.

## Corr k=2 -> k=3 collapse

Within the common 330-item cohort:

- S_post divergence decreased: 330/330
- write divergence decreased: 330/330
- carry divergence decreased: 162/330

Median ratios:

- write(k=3) / write(k=2): 0.1350069799
- carry(k=3) / carry(k=2): 1.0054028513

Thus the cohort-wide state collapse is dominated by collapse of write
divergence while carry divergence is approximately preserved at the median.

## Interaction

At corr k=2, interaction was negative for 323/330 items, but the median
interaction-energy contribution was approximately -0.19 percent of S_post
divergence energy.

The interaction term is therefore real but too small to explain the main
magnitude transient.

## Scientific interpretation

The validated evidence supports a narrow mechanism-level conclusion:

The layer-23 corr-specific k=2 rebound and immediate k=3 collapse are dominated
by transient modulation of matched-versus-swapped write-term divergence, not
by recurrent carry amplification.

Because current token IDs are identical from k=2 onward in the common cohort,
this write divergence cannot be attributed simply to renewed current-token
value difference.

This does not establish that the write modulation is autonomous or causal for
the downstream decision. Upstream/context representations can already differ
despite identical current token IDs.

## Next scientific boundary

The next minimal question is what produces the write-term transient.

Since the observed write term is W = deltaB_u, the next decomposition should
separate the factors entering W rather than moving to PCA, probes, logits, or
causal intervention.