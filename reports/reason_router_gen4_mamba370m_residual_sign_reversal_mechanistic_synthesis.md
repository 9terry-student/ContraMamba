# Mamba-370M Residual Sign Reversal — Mechanistic Synthesis

## Status

This report records a **descriptive mechanistic synthesis** of the frozen Mamba-370M cross-backbone residual-sign result.

It is not a new confirmatory test, does not add p-values, does not reopen the failed joint cross-backbone recurrence criterion, and does not promote a new plane by outcome-guided selection.

Frozen synthesis inputs:

- Mamba-370M cross-backbone confirmation evidence:
  `bbff1a0f348ef8e2c00a845cde2d99678ad6f92f`
- fresh residual-decomposition holdout:
  `1429c61e3ac5f4bdd8f18cd789a385bad8832acd`
- residual-decomposition implementation:
  `f42c94fe310c4bb739366eb0994d8da9597720af`
- residual-decomposition evidence:
  `b92ed1aded48de898cfac9cc85da373cc675b484`
- native coefficient census implementation:
  `02a25e4a279962e033fbd10ae526e50346064d31`
- native coefficient census evidence:
  `2a9dd5bcb4c10f86d88fa5289fe074ab4fec49d9`

No training, backward pass, model selection, rescue analysis, layer sweep, token sweep, epsilon sweep, or additional inferential test is introduced by this report.

## Question

Why did the Mamba-370M preregistered aggregate residual endpoint become negative even though the dominant P3 component recurred positively?

The decomposition asks whether the negative aggregate residual is primarily:

1. a nonlinear interaction created only when multiple residual planes are neutralized together;
2. an approximately additive signed mixture of individual residual-plane effects;
3. a consequence of native state coefficient mass concentrating in a different residual rank;
4. a change in the functional coupling between residual-plane content and the XG2-vs-XG4 susceptibility endpoint;
5. or a combination of the above.

The residual planes are the independently reconstructed 370M principal planes:

`R = P1 ⊕ P2 ⊕ P4 ⊕ P5`

with P3 held as the frozen dominant candidate.

## 1. Fresh same-pair decomposition

Fresh XG1 decomposition cohort:

`xg1_fact_3601..xg1_fact_3900`

Pair count:

`N = 300`

For each pair:

`S_Pk = Q_native - Q_Pk-neutralized`

`S_RES = Q_native - Q_all-residual-neutralized`

`S_INDIVIDUAL_SUM = S_P1 + S_P2 + S_P4 + S_P5`

`I_RES = S_RES - S_INDIVIDUAL_SUM`

Observed means:

| quantity | mean |
|---|---:|
| `S_P1` | `+5.7574578555749163e-09` |
| `S_P2` | `-1.4613450843693005e-08` |
| `S_P4` | `+2.8695081507774797e-08` |
| `S_P5` | `-6.2875830220286845e-08` |
| `S_RES` | `-3.9308513194261373e-08` |
| `S_INDIVIDUAL_SUM` | `-4.3036741700630129e-08` |
| `I_RES` | `+3.7282285063687634e-09` |

Pair-level descriptive structure:

- `corr(S_RES, S_INDIVIDUAL_SUM) = 0.96307146815124978`
- `abs(mean(I_RES)) / abs(mean(S_RES)) = 0.094845319840615214`
- `P4-P5 effect correlation = -0.783186340562306`
- canonical sign pattern `P1>0, P2<0, P4>0, P5<0` occurs on `167/300 = 0.5566666666666666`
- P5 is the largest absolute individual contributor on `224/300 = 0.7466666666666667` pairs
- omitting P5 from mean individual bookkeeping changes the residual-plane sum to `+1.983908851965671e-08`

Interpretation:

The aggregate negative residual is not primarily created by a large nonlinear multi-plane interaction. The same-pair individual-plane sum tracks the aggregate residual closely, with a small positive mean interaction residual relative to the aggregate mean and a high pair-level correlation.

The observed sign reversal is therefore **predominantly an approximately additive signed residual composition** under this frozen local intervention and endpoint.

This is a descriptive mechanistic localization, not a formal equivalence or non-inferiority claim about exact additivity.

## 2. Which residual planes drive the negative sum?

The fresh 370M decomposition gives a stable opponent pattern:

- P1: small positive contribution
- P2: negative contribution
- P4: substantial positive contribution
- P5: dominant negative contribution

P5 contributes the largest absolute individual effect on approximately three quarters of pairs.

P4 and P5 exhibit a strong negative pairwise effect correlation (`-0.7832`), while P2 is also predominantly negative.

Thus the most compact 370M residual organization is:

`small P1 positive + P4 positive` opposed by `P2 negative + dominant P5 negative`.

This opponent organization explains the negative aggregate residual without requiring a large aggregate-only interaction.

## 3. Native coefficient mass is strongly redistributed in 370M

A separate frozen native coefficient census reused the same `3601..3900` pairs and read only the native target-plus and target-minus branch-local coefficients.

Scientific model forwards:

`600`

No p-values or selection were performed.

Mean residual-plane coefficient energies and shares:

| plane | mean coefficient energy | residual energy share | largest-energy fraction |
|---|---:|---:|---:|
| P1 | `4.7547475914015163` | `0.063984156771484474` | `0.0` |
| P2 | `10.961272063076528` | `0.1475047279832685` | `0.0` |
| P4 | `6.9660841763872607` | `0.093741889229064013` | `0.0` |
| P5 | `51.629223086871654` | `0.69476922601618296` | `1.0` |

P5 is the largest-energy residual plane on all 300 pairs.

The plane with the largest coefficient energy is also the plane with the largest absolute individual effect on `224/300 = 0.7466666666666667` pairs.

Therefore native state-mass redistribution is a substantial part of the 370M mechanism: the P5-indexed residual rank contains the majority of the native residual coefficient energy.

However, coefficient mass alone does not determine causal effect, because P5 is not the largest absolute effect on the remaining approximately one quarter of pairs despite being the largest-energy plane on every pair.

## 4. Coupling is also reorganized

Within the 370M cohort, the descriptive mean-effect / mean-coefficient-energy ratios are:

| plane | mean effect / mean energy |
|---|---:|
| P1 | `+1.2108861185369127e-09` |
| P2 | `-1.3331893195972194e-09` |
| P4 | `+4.1192556364796319e-09` |
| P5 | `-1.2178341346429246e-09` |

The energy-effect correlations are:

- P1: `+0.27416831929823787`
- P2: `-0.18938345987173158`
- P4: `+0.69791332171866305`
- P5: `-0.26502607532575317`

These quantities show that native coefficient magnitude is not sufficient to explain the signed causal response. In particular, P2 and P5 carry negative effect coupling while P4 carries strongly positive coupling.

The ratio `mean(effect) / mean(energy)` is only a descriptive coupling proxy. It is not a normalized causal coefficient with backbone-independent units.

## 5. Relation to the earlier 130M mechanism

The earlier 130M residual individual-plane necessity experiment used a different backbone, state dimension, layer, and independently reconstructed principal planes. Therefore `P1`, `P2`, `P4`, and `P5` are **rank labels within each independently reconstructed backbone geometry**, not established one-to-one homologous semantic planes across the two models.

With that boundary, the earlier 130M raw native-neutralization attenuations were all positive:

| rank label | 130M raw native-neutralization attenuation |
|---|---:|
| P1 | `+2.8320710131173835e-08` |
| P2 | `+2.8646069304788239e-08` |
| P4 | `+1.8540349335461023e-09` |
| P5 | `+1.2380632555142974e-08` |

The 130M native residual coefficient-energy shares from the archived raw artifact were:

| rank label | 130M residual energy share |
|---|---:|
| P1 | `0.77755641823901611` |
| P2 | `0.10838051634505845` |
| P4 | `0.017786867395789541` |
| P5 | `0.096276198020135997` |

Thus, at the rank-aligned descriptive level:

- 130M residual coefficient mass was P1-dominant;
- 370M residual coefficient mass is P5-dominant;
- 130M raw native-neutralization effects were positive for all four residual ranks;
- 370M P2 and P5 effects are negative, while P4 is strongly positive.

Because the absolute coefficient scales are backbone-specific, raw energy magnitudes and effect-per-energy magnitudes should not be interpreted as directly calibrated cross-backbone physical quantities.

The robust cross-backbone statement is instead that **both the rank-aligned residual coefficient distribution and the signed residual causal profile are reorganized at 370M**.

## 6. XG2/XG4 family localization

Read-only decomposition of the already frozen 130M and 370M raw condition artifacts separated each individual effect as:

`S_k = ΔE_XG2,k - ΔE_XG4,k`

For the two negative 370M residual ranks:

P5:

- 130M: `ΔE_XG2 = +1.2621534355182687e-08`
- 130M: `ΔE_XG4 = +2.4090180003970645e-10`
- 370M: `ΔE_XG2 = -8.1747622756932468e-08`
- 370M: `ΔE_XG4 = -1.887179253664559e-08`

P2:

- 130M: `ΔE_XG2 = +2.8763642132923868e-08`
- 130M: `ΔE_XG4 = +1.1757282813564761e-10`
- 370M: `ΔE_XG2 = -4.1400781718507524e-08`
- 370M: `ΔE_XG4 = -2.6787330874814526e-08`

The sign reversal is therefore driven primarily by the XG2-side response moving negative; the XG4-side response also changes but partially offsets the negative XG2 shift in the final `ΔE_XG2 - ΔE_XG4` endpoint.

This supports a functional interpretation in which the 370M residual becomes an opponent organization around XG2 susceptibility rather than a simple uniformly positive residual mechanism.

## 7. Mechanistic synthesis

The evidence supports the following descriptive mechanism for the Mamba-370M residual sign reversal:

1. The dominant P3 component remains positively recurrent relative to the frozen response-blind control.
2. The P3-excluded residual does not preserve the earlier uniformly positive rank-aligned causal profile.
3. In 370M, the residual is organized as opposing signed components: P4 positive versus P5 strongly negative, with P2 also negative and P1 small positive.
4. These individual effects combine approximately additively; the aggregate-only interaction residual is small relative to the aggregate mean.
5. Native residual coefficient mass is heavily concentrated in the P5-indexed rank in 370M.
6. P5 mass concentration alone is insufficient: the signed effect coupling is also reorganized, with P5 and P2 negative and P4 strongly positive.
7. The negative P5/P2 effect is driven primarily by reversal of the XG2-side susceptibility response.
8. Therefore the aggregate residual sign reversal is best described as a **joint state-distribution and functional-coupling reorganization**, not as a large nonlinear interaction artifact and not as simple disappearance of residual structure.

A concise claim is:

> Across the two tested Mamba scales, the dominant component recurs, while the residual mechanism reorganizes. In the 370M model, residual state mass becomes concentrated in a P5-indexed mode whose XG2-side causal coupling is negative, opposed by a positive P4 mode; the resulting signed components combine mostly additively to produce the preregistered negative aggregate residual.

## 8. Interpretation boundaries

This synthesis does not establish:

- semantic one-to-one identity of same-numbered principal planes across 130M and 370M;
- that the 370M P5 plane is mathematically the same object as the 130M P5 plane;
- a backbone-independent calibrated causal coefficient from effect/energy ratios;
- exact additivity or absence of interaction;
- statistical significance of the descriptive decomposition, coefficient census, or cross-backbone comparison;
- a significant opposite-tail reversal for the failed preregistered residual recurrence test;
- universality across other Mamba sizes, checkpoints, layers, token positions, generators, datasets, or architectures;
- behavioral or task-level consequences beyond the separately frozen behavioral evidence;
- rescue of the failed preregistered full cross-backbone qualitative recurrence criterion.

The preregistered joint cross-backbone criterion remains not established because the positive residual recurrence endpoint failed.

## Final status

Code correctness for the decomposition and coefficient census:

`PASS`

Scientific executions:

`PASS`

Imported artifact/provenance validation:

`PASS`

Residual decomposition:

`DESCRIPTIVE — NO P-VALUES`

Native coefficient census:

`DESCRIPTIVE — NO P-VALUES`

Cross-backbone mechanistic synthesis:

`STATIC DESCRIPTIVE INTERPRETATION`

Failed preregistered joint recurrence claim rescued:

`NO`

Mechanistic localization:

`RESIDUAL SIGN REVERSAL IS CONSISTENT WITH MOSTLY ADDITIVE SIGNED COMPONENTS PLUS P5-DOMINANT STATE-MASS REDISTRIBUTION AND SIGNED COUPLING REORGANIZATION`
