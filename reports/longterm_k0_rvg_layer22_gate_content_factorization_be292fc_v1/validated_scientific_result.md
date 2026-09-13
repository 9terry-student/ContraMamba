# K0-RVG Layer-22 Gate / Content Factorization

## Status

VALIDATED SCIENTIFIC EVIDENCE

Runner commit:

`be292fc12b745dc8d7f2ac04a7b4e61822b5ee15`

Runner SHA256:

`91505b337c0f208ee2606aace7caa4551d0e6d8666c208207e35f420cb089ec6`

Parent evidence freeze:

`f456e31f2730e0bee1633b30f7d7776bae7de05d`

No tokenizer execution, logits, task heads, training, causal intervention,
PCA, or learned probe was executed. Raw vectors were not persisted.

## Authenticated boundary

For layer 22 on the authenticated CPU slow path:

`C22 = recurrent scan readout + D * hidden_states`

`A22 = SiLU(gate)`

`V22 = C22 * A22`

with elementwise multiplication.

The exact symmetric matched/swapped difference decomposition is:

`delta_V22 = Q_content + Q_gate`

where:

`Q_content = Abar22 * delta_C22`

`Q_gate = Cbar22 * delta_A22`

and arithmetic bars denote matched/swapped means.

The activation object was authenticated as SiLU and exactly matched
`torch.nn.functional.silu` on the synthetic runtime probe.

## Artifact authentication

Full local metrics SHA256:

`1c9e344eacd547f0ab9d1463a5f7c98dfac5868c6519d55d9cb123a37f6c5b2a`

summary.json SHA256:

`cf1137b9b0fa92b497696ac81537ca883805218a38d17766544ff5badbb6e594`

execution_manifest.json SHA256:

`4326b36a4ce0e4aeea42f7b7252ad49dcd2672e20395a8482b684ed52e44f0f0`

Trajectory rows:

`5376`

Common DDSSSSS cohort:

`330`

Scientific forwards:

`1344`

Parent delta-V22 reproduction:

`PASS`

At k=-1 all delta-C22, delta-A22, delta-V22, Q-content, Q-gate,
Q-sum, and product-delta quantities are zero in 672/672 pair-role rows.

Maximum symmetric product closure relative residual:

`1.9960937224129005e-16`

Maximum float64 product-delta residual relative to observed delta-V22:

`1.1852412322894514e-06`

Maximum float64 product-delta residual relative to branch scale:

`3.882675549364276e-08`

Maximum float32 branch product reconstruction residual:

`0.0`

## Common-cohort k2 structure

corr k2 medians:

- delta-C22: 49.529481
- delta-A22: 2.850880
- delta-V22: 45.973612
- Q-content: 41.356796
- Q-gate: 19.301164
- content energy fraction: 0.831546
- normalized cross: +0.006312
- addition factor: 1.003151

ctrl k2 medians:

- delta-C22: 23.195208
- delta-A22: 1.923192
- delta-V22: 10.261545
- Q-content: 8.298828
- Q-gate: 4.840796
- content energy fraction: 0.753180
- normalized cross: +0.123281
- addition factor: 1.059850

Thus both content and gate terms carry substantial k2 difference energy,
with the content term larger in the median for both roles.

## Corr versus ctrl at k2

Counts:

- Q-content corr > ctrl: 323/330
- Q-gate corr > ctrl: 330/330
- Q-sum corr > ctrl: 330/330
- corr Q-content > Q-gate: 286/330
- ctrl Q-content > Q-gate: 330/330
- corr content-energy fraction > 0.5: 286/330
- ctrl content-energy fraction > 0.5: 330/330

Role-effect absolute-log comparison:

- content term larger: 215/330
- gate term larger: 115/330

Median paired log corr/ctrl ratios:

- Q-content: +1.464697
- Q-gate: +1.302865
- RSS term envelope: +1.468501
- addition factor: -0.050781
- Q-sum / delta-V22: +1.420642

RSS role effect is larger than interaction/addition-factor role effect
in 330/330 common-cohort items.

Therefore the k2 V22 role separation is jointly represented in both the
pre-gate content and activated-gate terms, but the content term is the
larger term in most items and has the stronger role effect more often.

The content/gate interaction is not the source of the large k2 role
separation. It slightly attenuates the median corr-versus-ctrl difference.

## Corr k1 -> k2

Counts:

- Q-content up: 219/330
- Q-content down: 111/330
- Q-gate up: 185/330
- Q-gate down: 145/330
- Q-sum up: 222/330
- Q-sum down: 108/330

Q-content has the larger absolute temporal log change in 215/330 items.

RSS term-envelope change dominates addition-factor change in 284/330 items.

Median paired log changes:

- Q-content: +0.200453
- Q-gate: +0.040391
- RSS: +0.153442
- addition factor: +0.025423
- Q-sum: +0.207979

Thus the corr k1-to-k2 V22 increase is primarily associated with changes in
the content/gate term envelope rather than interaction geometry, with the
content term providing the larger temporal effect more often.

## Corr k2 -> k3

Counts:

- Q-content down: 328/330
- Q-gate down: 330/330
- Q-sum down: 330/330

Q-content has the larger absolute temporal log change in 234/330 items;
Q-gate has the larger one in 96/330.

RSS term-envelope change dominates interaction/addition-factor change in
330/330 items.

Median paired log changes:

- Q-content: -2.363330
- Q-gate: -2.470202
- RSS: -2.500466
- addition factor: -0.082248
- Q-sum: -2.524986

Therefore the strong corr k2-to-k3 V22 collapse is shared by both content
and gate terms and is overwhelmingly a collapse of the term envelope,
not a change in their interaction geometry.

## Interaction structure

At k2, corr median normalized cross is approximately +0.0063 and the
addition factor is approximately 1.0032.

At k2, ctrl median normalized cross is approximately +0.1233 and the
addition factor is approximately 1.0598.

The interaction is therefore modest and more constructive in ctrl than
corr. This interaction geometry slightly reduces, rather than creates,
the corr-versus-ctrl role separation.

## Validated scientific conclusion

The strong layer-22 pre-output-projection V22 role separation is not a
pure gate-only or interaction-generated effect.

Both exact symmetric product terms are role selective:

- Q-content is larger for corr in 323/330 items.
- Q-gate is larger for corr in 330/330 items.

However, the content term is larger than the gate term in most corr items,
in all ctrl items, carries the majority of term energy in most items, and
has the larger paired role-effect magnitude in 215/330 items.

Interaction/addition geometry is secondary: the RSS term envelope dominates
the interaction role effect in 330/330 items.

Accordingly, the next primary localization boundary is inside C22, while
the activated-gate path remains a validated secondary carrier rather than
being discarded.

All statements are observational/algebraic and do not establish causal
importance for downstream task behavior.

## Updated local evidence chain

The currently localized chain is:

`layer22 pre-gate content C22 + activated gate A22`
`-> exact multiplicative V22 representation`
`-> output projection partially attenuates k2 role separation`
`-> layer22 mixer update Y22`
`-> constructive residual addition produces R23`
`-> layer23 RMSNorm further amplifies role separation`
`-> layer23 hidden in-projection partially attenuates it`

Within V22, both C22 and A22 carry role-selective structure, but C22 is the
primary next boundary because its exact term is larger and more often
dominant, while interaction geometry is already ruled out as the primary
source.

## Next scientific boundary

The next source audit should authenticate the internal C22 identity:

`C22 = recurrent SSM readout + D * hidden_states`

before any new scientific execution.

The next exact decomposition should separate the recurrent-readout term from
the direct D-skip term and quantify their vector interaction.

The A22 gate branch remains open as a secondary localization branch and
should not be interpreted as null.