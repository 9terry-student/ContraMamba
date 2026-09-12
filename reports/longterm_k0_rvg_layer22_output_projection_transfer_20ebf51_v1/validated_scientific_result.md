# K0-RVG Layer-22 Output Projection Transfer Audit

## Status

VALIDATED SCIENTIFIC EVIDENCE

Runner commit:

`20ebf5185bb324e457b2d4c0b293f08e1a017ccf`

Runner SHA256:

`2c2f2a228ae49a77b4ddaafd162a6b211de664796afe7b27f007dc32fc8dc2a1`

Parent evidence freeze:

`e27e27fb5b4d6b3a5ac64973d8b9a566679f5775`

No tokenizer execution, logits, task heads, training, causal intervention,
PCA, or learned probe was executed. Raw vectors were not persisted.

## Authenticated boundary

The authenticated CPU slow path ends with:

`V22 := scan_output.transpose(1, 2)`

followed by the bias-free projection:

`Y22 = W_O V22`

where:

- `W_O` shape: `(768, 1536)`
- output-projection bias: absent
- mixer forward SHA256:
  `63450e056b1ebbaef584f8025b9c26766f5b97a69cd0c579213f095970dab826`
- slow-forward SHA256:
  `283d2f21854a8f0804e01fe03c8d0ece29c86dc3128bdf6527a1d6b62d8cd7d9`
- Mamba source SHA256:
  `23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

The source-level difference identity is:

`delta_Y22 = W_O delta_V22`

The primary scalar transfer metric is defined from observed runtime
differences as:

`output_projection_transfer =
 ||delta_Y22|| / ||delta_V22||`

Float64 `W_O delta_V22` reconstruction is retained as a numerical diagnostic,
not as a cancellation-sensitive blocking criterion.

## Artifact authentication

Full local metrics:

`layer22_output_projection_transfer_metrics.jsonl`

SHA256:

`7dd57a37109a10fc2e3b2d71a212647875b2583fa3aad27c4fe9ce480218b34a`

summary.json SHA256:

`7abeb6eec6d5060be57d53689a23f8e5d3b9bd35b40e234cab87a91b92efb536`

execution_manifest.json SHA256:

`d53efe388c93066d7f04a9b2bf792e8d048aaecf0e43ef6ebffca6907545c8c6`

Trajectory rows:

`5376`

Common DDSSSSS cohort:

`330`

Scientific forwards:

`1344`

Parent delta-Y22 reproduction:

`PASS`

At k=-1:

- delta-V22 zero: 672/672
- delta-Y22 zero: 672/672
- projected delta-V22 zero: 672/672

Maximum float32 branch reconstruction relative residual:

`8.739592775484828e-07`

Maximum float64 difference-projection residual relative to observed delta-Y22:

`1.1347976775530973e-05`

Maximum float64 difference-projection residual relative to branch scale:

`3.266769983338318e-07`

The latter two are diagnostic-only.

## Common-cohort medians

corr k1:

- delta-V22: 36.580004
- delta-Y22: 115.364063
- output-projection transfer: 3.178520

corr k2:

- delta-V22: 45.973612
- delta-Y22: 90.540620
- output-projection transfer: 1.980063

corr k3:

- delta-V22: 3.729455
- delta-Y22: 10.058254
- output-projection transfer: 2.475611

ctrl k1:

- delta-V22: 26.904172
- delta-Y22: 90.390588
- output-projection transfer: 3.390650

ctrl k2:

- delta-V22: 10.261545
- delta-Y22: 27.960542
- output-projection transfer: 2.641872

ctrl k3:

- delta-V22: 2.645883
- delta-Y22: 7.396159
- output-projection transfer: 2.949565

## Corr versus ctrl at k2

Counts:

- corr delta-V22 > ctrl: 330/330
- corr projection transfer > ctrl: 0/330
- corr projection transfer < ctrl: 330/330
- corr delta-Y22 > ctrl: 327/330

Absolute-log dominance:

- delta-V22: 327/330
- output-projection transfer: 3/330

Direction:

- V22 favors corr while projection transfer favors ctrl: 330/330
- both favor corr: 0/330
- V22 favors ctrl while transfer favors corr: 0/330

Median paired log corr/ctrl ratios:

- V22: +1.420642
- output-projection transfer: -0.305135
- Y22: +1.090169

Thus the strong corr-vs-ctrl k2 mixer-update separation is already present
before the output projection.

The fixed output projection does not generate this role separation. Its
direction-conditioned transfer is lower for corr than ctrl in every
common-cohort item and therefore partially attenuates the upstream V22 role
separation.

## Corr k1 -> k2

Counts:

- V22 up: 222/330
- V22 down: 108/330
- projection transfer up: 0/330
- projection transfer down: 330/330
- Y22 up: 80/330
- Y22 down: 250/330

Absolute-log dominance:

- V22: 103/330
- projection transfer: 227/330

Median paired log ratios:

- V22: +0.207979
- output-projection transfer: -0.472396
- Y22: -0.255227

Therefore k1-to-k2 differs from the k2 role-separation result.

Although the median V22 difference rises, output-projection transfer contracts
universally and is the larger absolute-log factor in 227/330 items. The
resulting Y22 difference decreases in 250/330 items.

This is an algebraic direction-conditioned projection effect, not a causal
downstream claim.

## Corr k2 -> k3

Counts:

- V22 down: 330/330
- projection transfer up: 317/330
- projection transfer down: 13/330
- Y22 down: 330/330

Absolute-log dominance:

- V22: 330/330
- projection transfer: 0/330

Median paired log ratios:

- V22: -2.524986
- output-projection transfer: +0.266424
- Y22: -2.099191

Thus the strong k2-to-k3 mixer-update collapse is already present before the
output projection.

The projection transfer usually increases and therefore partially compensates
rather than causes this collapse.

## Numerical diagnostic agreement

Observed and algebraic transfer estimates agree closely.

Median relative transfer gaps:

- corr k1: 4.48e-08
- corr k2: 6.76e-08
- corr k3: 1.86e-07
- ctrl k1: 1.08e-07
- ctrl k2: 2.69e-07
- ctrl k3: 2.28e-07

Maximum relative gaps remain small and are retained only as numerical
diagnostics.

## Validated scientific conclusion

The layer-22 output projection has different effects across the temporal
coordinates, but it is not the source of the primary k2 role-selective mixer
update.

For corr versus ctrl at k2, delta-V22 is larger for corr in 330/330
common-cohort items. Output-projection transfer is simultaneously lower for
corr in 330/330 items. The pre-projection V22 role difference is the larger
absolute-log factor in 327/330 items.

Therefore the large layer-22 mixer-update role separation is already formed
upstream of out_proj, while out_proj partially attenuates it.

For corr k2 -> k3, V22 collapses in 330/330 items and is the dominant
absolute-log factor in 330/330. Output-projection transfer usually rises,
partially compensating the collapse.

For corr k1 -> k2, projection transfer instead contracts universally and often
dominates the local factorization, contributing to the observed Y22 decline.

All conclusions are observational/algebraic and do not establish causal
importance for downstream task behavior.

## Updated local evidence chain

The validated local chain is now:

`layer22 pre-output representation V22`
`-> output projection partially attenuates k2 corr-vs-ctrl separation`
`-> layer22 mixer update Y22`
`-> constructive residual addition produces R23`
`-> layer23 RMSNorm further amplifies role separation`
`-> layer23 hidden in-projection partially attenuates it`

For corr k2 role separation, the current earliest localized strong factor is
therefore V22, not Y22 or the output projection.

## Next scientific boundary

The next primary boundary is immediately inside V22.

Authenticated slow-path source defines V22 after:

1. the recurrent state readout;
2. the direct `D * hidden_states` skip contribution;
3. their addition;
4. multiplication by `SiLU(gate)`.

The next step should first authenticate this exact source-level structure and
identify the narrowest exact decomposition of V22 into pre-gate SSM/skip
content and gate modulation.

No new scientific execution should precede freezing the present evidence.