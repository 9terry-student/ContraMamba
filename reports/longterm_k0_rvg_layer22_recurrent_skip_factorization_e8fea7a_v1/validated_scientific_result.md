# K0-RVG Layer-22 Recurrent-Readout / D-Skip Factorization

## Status

VALIDATED SCIENTIFIC EVIDENCE

Runner commit:

`e8fea7ae927f294a20adf0e1c19d49370adb30bb`

Runner SHA256:

`fd7f02fa966af3359c479754f9d90e49df2e160aeca79ed43131b52f42c8d2fb`

Parent evidence freeze:

`bf15fc1b7fa48081930df3f3c0e070241413dabd`

No tokenizer execution, logits, task heads, training, causal intervention,
PCA, or learned probe was executed. Raw vectors were not persisted.

## Epistemic boundary

The authenticated layer-22 source identity is:

`C22 = recurrent SSM readout + D * hidden_states`

This audit uses:

`K22 := D * hidden_states`

where K22 is reconstructed from the authenticated slow_forward frame operands.

C22 is inherited as a direct runtime observation from the frozen parent audit.

The recurrent-readout quantity in this audit is:

`R22_complement := C22 - K22`

R22_complement is NOT claimed to be a directly captured recurrent-readout
runtime tensor.

It is a source-defined algebraic complement.

Therefore:

`delta_C22 = delta_R22_complement + delta_K22`

and:

`||delta_C22||^2 =
 ||delta_R22_complement||^2 +
 ||delta_K22||^2 +
 2 <delta_R22_complement, delta_K22>`

All conclusions below preserve this distinction.

## Artifact authentication

Full local metrics SHA256:

`d462f53c8cda35b11b971fcf7e3744b789b9e41c804c67dc6ddffb368b106859`

summary.json SHA256:

`220e01efe6358621f8bc3c33bdbad9b12a449c2fe0a52b2e75b9deab8e2414ba`

execution_manifest.json SHA256:

`a6e8d2e79389212754559365a9efd7993627655e22cb60e977a6f485c2e98914`

Trajectory rows:

`5376`

Scientific forwards:

`1344`

Common DDSSSSS cohort:

`330`

Parent delta-C22 reproduction:

`PASS`

At k=-1, delta-R22-complement, delta-K22, delta-C22, and RSS are zero
in 672/672 pair-role rows.

Maximum additive closure relative residual:

`0.0`

## Common-cohort k2 structure

corr k2 medians:

- delta-R22-complement: 49.275561
- delta-K22: 0.571691
- delta-C22: 49.529481
- recurrent energy fraction: 0.999860
- skip energy fraction: 0.000140
- normalized cross: +0.008581
- addition factor: 1.004282
- R22-complement / K22 norm ratio: 84.412

ctrl k2 medians:

- delta-R22-complement: 23.169319
- delta-K22: 0.195587
- delta-C22: 23.195208
- recurrent energy fraction: 0.999924
- skip energy fraction: 0.0000755
- normalized cross: +0.002535
- addition factor: 1.001267
- R22-complement / K22 norm ratio: 115.053

Thus the C22 matched/swapped difference energy is overwhelmingly carried
by the source-defined recurrent-readout complement.

## Corr versus ctrl at k2

Counts:

- R22-complement corr > ctrl: 307/330
- K22 corr > ctrl: 329/330
- C22 corr > ctrl: 307/330
- corr R22-complement > K22: 330/330
- ctrl R22-complement > K22: 330/330
- corr recurrent energy fraction > 0.99: 330/330
- ctrl recurrent energy fraction > 0.99: 330/330

Median paired log corr/ctrl ratios:

- R22-complement: +0.663181
- K22: +0.959805
- RSS term envelope: +0.663186
- addition factor: +0.002933
- C22: +0.666326

RSS role-effect magnitude is larger than addition-factor role-effect
magnitude in 330/330 items.

The D-skip term is therefore role selective in a proportional sense:
its corr/ctrl ratio is substantial and is larger than the recurrent-complement
log ratio in the median.

However, its absolute norm and energy contribution are tiny relative to the
recurrent-readout complement. It does not explain the large magnitude of
the C22 role separation.

## Corr k1 -> k2

Counts:

- R22-complement down: 305/330
- K22 down: 231/330
- C22 down: 305/330
- RSS dominates addition-factor change: 330/330

Median paired log changes:

- R22-complement: -0.453867
- K22: -0.221262
- RSS: -0.453841
- addition factor: +0.002425
- C22: -0.452279

Thus the C22 decline from k1 to k2 is almost entirely associated with the
term-magnitude envelope and closely tracks the recurrent-readout complement.

## Corr k2 -> k3

Counts:

- R22-complement down: 330/330
- K22 down: 330/330
- C22 down: 330/330
- RSS dominates addition-factor change: 330/330

Median paired log changes:

- R22-complement: -1.950367
- K22: -1.198150
- RSS: -1.950000
- addition factor: +0.003956
- C22: -1.947611

Thus the strong k2-to-k3 C22 contraction is overwhelmingly associated with
collapse of the recurrent-readout-complement-dominated term envelope, not
with residual-addition geometry.

## Interaction structure

At k2, the normalized cross term is small and positive:

- corr median: +0.008581
- ctrl median: +0.002535

The addition factors are correspondingly close to one:

- corr median: 1.004282
- ctrl median: 1.001267

Cross is positive in 330/330 items for both roles.

This is weak constructive addition. It is not the source of the large C22
role separation.

## Validated scientific conclusion

Within layer-22 pre-gate content C22, the matched/swapped role separation is
overwhelmingly localized in magnitude to the source-defined recurrent-readout
complement rather than the direct D-skip term or their vector interaction.

The D-skip term remains a validated secondary carrier because it is itself
role selective:

- K22 corr > ctrl in 329/330 k2 items;
- median log corr/ctrl ratio is +0.959805.

But its energy is negligible relative to the recurrent complement:

- corr median recurrent energy fraction: 0.999860;
- ctrl median recurrent energy fraction: 0.999924;
- recurrent-complement norm exceeds K22 by approximately 84x and 115x in
  the corr and ctrl k2 medians respectively.

The interaction geometry is also secondary: RSS role effects dominate
addition-factor role effects in 330/330 items.

Accordingly, the next primary localization boundary is inside the recurrent
SSM readout path.

This conclusion is observational/algebraic. R22 in this audit is a
source-defined complement, not a directly observed recurrent-readout tensor,
and no causal importance for downstream task behavior is established.

## Updated evidence chain

The current localized chain is:

`layer22 recurrent-readout-dominated C22`
`+ smaller role-selective D-skip K22`
`-> pre-gate content C22`
`+ activated gate A22`
`-> V22`
`-> output projection`
`-> Y22`
`-> R23 residual addition`
`-> layer23 RMSNorm`
`-> layer23 hidden in-projection`

Within C22, direct D-skip and recurrent/skip interaction have now been
excluded as primary magnitude explanations.

## Next scientific boundary

Before any new scientific execution, statically authenticate the actual
layer-22 recurrent-readout producer on the active slow path.

The source audit must distinguish the active runtime branch and identify
the exact operands and tensor shapes for the recurrent readout, including
the state/readout tensor and C projection used immediately before the
D-skip addition.

Only after that source identity is authenticated should a new runner
directly observe or reconstruct the recurrent-readout operands.

No direct-runtime-read claim should be retroactively assigned to
R22_complement from this audit.