# K0-RVG Layer-22 Recurrent Readout State / Readout-C Factorization

## Status

VALIDATED SCIENTIFIC EVIDENCE

Runner commit:

`d5243a907517fe4911b8df026647a7b15880626b`

Runner SHA256:

`b9a307b031f085f454203e7cd3c3bbe06b26552f6389d072d68f9996f6f182b0`

Parent recurrent-skip evidence freeze:

`f672fd6d9b15fa24465d39b07ca7c8c21178b99d`

The execution used 1344 CPU sequential slow-path scientific forwards.

No tokenizer execution, logits, task heads, training, causal intervention,
PCA, or learned probe was used. Raw state/readout vectors were not persisted.

## Authenticated boundary

The active layer-22 sequential recurrent readout is:

`R_t[d] = sum_n S_t[d,n] * C_t[n]`

where:

- `S_t` is the direct post-update recurrent state from the existing
  `RawRecurrenceCollector.s_post`;
- `C_t` is directly observed from the same authenticated slow_forward
  readout-line frame local `C[:, i, :]`.

For matched/swapped branches the exact symmetric bilinear identity is:

`delta_R = Q_state + Q_readout_C`

with:

`Q_state = <C_bar, delta_S>`

and:

`Q_readout_C = <S_bar, delta_C>`.

The previously frozen recurrent-readout complement is used only as a
numerical bridge and parent-reproduction check. It is not used to define
the new direct readout operands.

## Artifact authentication

Full local metrics SHA256:

`20d686c88a195c855bac32f1d7b251ee9d2b34953bd3c84171593f0d96aaf1d7`

summary.json SHA256:

`1654370d647e3a729fb9ec0a8c69254774176d24c4351cbbcc269535fd80d652`

execution_manifest.json SHA256:

`2fb569b73b8fa3b95b9e612e168e5dd4dfa367cc8b2d68249693348404cf3912`

Trajectory rows:

`5376`

Common DDSSSSS cohort:

`330`

Parent recurrent-readout delta reproduction:

`PASS`

At k=-1 all 672 pair-role rows have exactly zero:

- delta-S;
- delta-C;
- delta-R;
- Q_state;
- Q_readout_C;
- RSS;
- parent recurrent-readout-complement delta.

Maximum bilinear closure relative residual:

`4.197963885301822e-15`

Maximum branch direct-readout versus parent-complement relative residual:

`4.7516117920474735e-08`

Maximum direct-operand delta versus parent delta relative residual:

`2.735175776845672e-06`

## Common-330 term dominance

### k1

corr:

- Q_state > Q_readout_C: 2/330
- median state energy fraction: 0.262879
- median readout-C energy fraction: 0.737121
- median normalized cross: +0.413281
- median addition factor: 1.188815
- median state additive share: 0.333782
- median readout-C additive share: 0.666218

ctrl:

- Q_state > Q_readout_C: 0/330
- median state energy fraction: 0.203201
- median readout-C energy fraction: 0.796799
- median normalized cross: +0.344648
- median addition factor: 1.159590
- median state additive share: 0.281071
- median readout-C additive share: 0.718929

Thus the recurrent readout at k1 is readout-C-term dominated.

### k2

corr:

- Q_state > Q_readout_C: 330/330
- median state energy fraction: 0.949710
- median readout-C energy fraction: 0.050290
- median state additive share: 0.909692
- median readout-C additive share: 0.090308

ctrl:

- Q_state > Q_readout_C: 311/330
- median state energy fraction: 0.874806
- median readout-C energy fraction: 0.125194
- median state additive share: 0.852755
- median readout-C additive share: 0.147245

Thus by k2 the recurrent-readout difference has switched to strong
post-update-state-term dominance.

### k3

corr:

- Q_state > Q_readout_C: 330/330
- median state energy fraction: 0.950175

ctrl:

- Q_state > Q_readout_C: 312/330
- median state energy fraction: 0.909470

State-term dominance persists at k3.

## corr k1 -> k2: factor handoff

Counts:

- Q_state up: 251/330
- Q_state down: 79/330
- Q_readout_C down: 330/330
- delta-R down: 305/330
- Q_readout_C abs-log change exceeds Q_state: 312/330
- RSS change exceeds addition-factor change: 266/330

Median paired log changes:

- Q_state: +0.311027
- Q_readout_C: -1.779785
- RSS: -0.324792
- addition factor: -0.131263
- delta-R: -0.453867
- delta-S: +0.309404
- delta-C: -1.689549

This transition is not a uniform attenuation of both factors.

The state-associated term generally strengthens while the readout-C term
collapses universally. The overall recurrent-readout difference decreases
because the much larger readout-C contraction overwhelms the state increase.

Accordingly, k1 -> k2 is a factor-dominance handoff from readout-C-term
dominance to post-update-state-term dominance.

## corr k2 -> k3

Counts:

- Q_state down: 330/330
- Q_readout_C down: 316/330
- delta-R down: 330/330
- Q_state abs-log change exceeds Q_readout_C: 171/330
- Q_readout_C abs-log change exceeds Q_state: 159/330
- RSS change exceeds addition-factor change: 330/330

Median paired log changes:

- Q_state: -1.983509
- Q_readout_C: -1.867887
- RSS: -1.977368
- addition factor: -0.024057
- delta-R: -1.950367
- delta-S: -1.424113
- delta-C: -1.043324

Both terms contract strongly. There is no evidence here that only one
factor explains the k2 -> k3 contraction.

The contraction is overwhelmingly in the term-magnitude envelope rather
than vector-addition geometry.

## corr versus ctrl at k2

Counts:

- Q_state corr > ctrl: 313/330
- Q_readout_C corr > ctrl: 210/330
- delta-R corr > ctrl: 307/330
- corr Q_state > Q_readout_C: 330/330
- ctrl Q_state > Q_readout_C: 311/330
- state role-effect abs-log > readout-C role-effect: 210/330
- readout-C role-effect abs-log > state role-effect: 120/330
- RSS role effect > addition-factor role effect: 322/330

Median paired log corr/ctrl ratios:

- Q_state: +0.743438
- Q_readout_C: +0.162296
- RSS: +0.659226
- addition factor: +0.013713
- delta-R: +0.663181
- delta-S: +0.839760
- delta-C: +0.460218

The k2 corr-versus-ctrl recurrent-readout separation is therefore primarily
associated with the state term.

The readout-C term remains nonzero and role associated, but its median
corr/ctrl role contrast is materially smaller.

Vector interaction is secondary to term magnitude.

## Validated scientific conclusion

Layer-22 recurrent-readout role separation exhibits a temporal factor
handoff.

At k1, the recurrent-readout difference is dominated by the readout-C term.

Between k1 and k2, the state-associated term generally increases while the
readout-C term collapses in all 330 common-cohort items. This causes the
readout to transition from readout-C-term dominance to post-update-state-term
dominance.

At k2, the post-update state term is the primary factor associated with the
corr-versus-ctrl recurrent-readout role separation:

- Q_state exceeds Q_readout_C in 330/330 corr items;
- median corr state energy fraction is 0.949710;
- median log corr/ctrl Q_state ratio is +0.743438 versus +0.162296 for
  Q_readout_C;
- state role-effect magnitude exceeds readout-C role-effect magnitude in
  210/330 items.

At k2 -> k3, both state and readout-C terms contract strongly, with neither
term universally dominating the rate of contraction.

Interaction geometry is not the primary explanation for the k2 role
separation or the k2 -> k3 decline.

These conclusions are observational/algebraic. They localize factor
association within the authenticated recurrent-readout computation and do
not establish causal importance for downstream task behavior.

## Updated localized chain

The current primary path is:

`layer22 post-update recurrent state S_t`
`+ secondary readout projection C_t`
`-> recurrent readout R_t`
`+ small D-skip`
`-> pre-gate content C22`
`+ activated gate`
`-> V22`
`-> output projection`
`-> Y22`
`-> residual stream`
`-> layer23 RMSNorm`
`-> layer23 mixer input`

At k1, the readout-C term temporarily dominates the recurrent-readout
difference. By k2, ownership has shifted decisively to the post-update state
term.

## Next scientific boundary

The next primary boundary is inside the layer-22 post-update recurrent state:

`S_t = G_t * S_(t-1) + W_t`

The existing authenticated `RawRecurrenceCollector` already observes:

- `S_(t-1)`;
- `G_t`;
- `W_t`;
- `S_t`.

Therefore no new state recurrence implementation is needed.

Before full execution, the next audit should statically define and validate
the exact layer-22 carry/write decomposition of the post-update state and
preserve the distinction between:

- carried prior-state contribution;
- current write contribution;
- vector interaction.

No claim from the earlier layer-23 carry/write audit should be transferred
to layer 22 without new layer-22 evidence.