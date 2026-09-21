# ContraMamba Gen4 — Three-Scale AVeriTeC External-Transfer Descriptive Synthesis

Status: STATIC_DESCRIPTIVE_SYNTHESIS_ONLY

Frozen synthesis base commit:

`b2733b4acbb1c8a483046aeec81c8b93a1ec953c`

This report adds:

- no model execution;
- no training;
- no forward or backward pass;
- no new p-value;
- no pooled three-scale hypothesis test;
- no trend or monotonicity test;
- no zero-crossing or parameter-threshold estimate;
- no subgroup inferential test;
- no cohort, tokenizer, anchor, plane, or layer reselection.

It only places already-frozen AVeriTeC external-transfer results side by side.

## 1. Frozen primary sources

### Mamba-130M / Mamba-370M

Primary external-transfer analysis:

`reports/reason_router_gen4_averitec_external_transfer_runs/g4k-averitec-external-transfer-130m370m-dev462-c5b470f-postcheck-recovery2/external_transfer_analysis.json`

Frozen primary scientific conclusion:

`CROSS_SCALE_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_THROUGH_370M_SUPPORTED`

Frozen post-result descriptive analysis:

`reports/reason_router_gen4_averitec_external_transfer_static_post_result_analysis.json`

### Mamba-1.4B

Primary external-transfer analysis:

`reports/reason_router_gen4_averitec_mamba14b_negative_sign_analysis_v1/external_transfer_analysis.json`

Frozen analysis SHA256:

`e9847177d78858ef400bc196f4b811e3e9ad401ae777a4315aea1bf9dd215d1e`

Frozen analysis commit:

`b2733b4acbb1c8a483046aeec81c8b93a1ec953c`

Frozen primary scientific conclusion:

`MAMBA14B_AVERITEC_NEGATIVE_SIGN_TRANSFER_NOT_ESTABLISHED`

## 2. Shared external domain

All three scale results use the official AVeriTeC development source restricted to
the same compatible three-label population:

- `Refuted`: `305`;
- `Supported`: `122`;
- `Not Enough Evidence`: `35`;
- total `N = 462`.

All use gold-evidence serialization with a response-blind claim/evidence boundary
anchor and target offset `+2`.

The 1.4B tokenizer is byte-distinct from the tokenizer used by the completed
130M/370M family and was independently gated `462/462` before execution.

## 3. Three-scale descriptive table

| scale | selected/control | mean D_EXT | sign fraction | standardized effect | frozen inferential status |
|---|---|---:|---:|---:|---|
| Mamba-130M | `P3 / P5` | `+0.00023104868881158465` | positive `0.658008658008658` | `dz = +0.11964886938686847` | positive external transfer supported in the frozen 130M/370M Holm family |
| Mamba-370M | `P3 / P5` | `+0.0000455500221672976` | positive `0.6060606060606061` | `dz = +0.10631034549402116` | positive external transfer supported in the frozen 130M/370M Holm family |
| Mamba-1.4B | `P5 / P4` | `-0.00007850951577129465` | negative `0.538961038961039` | `dz = -0.06686547612930485` | negative-sign external transfer not established; one-sided `p = 0.0756670802614973` |

The 1.4B primary test was the prospectively frozen one-sided `less` test with exactly
one new p-value and no multiplicity correction. Its mean-sign gate passed, but its
alpha gate did not.

## 4. Scientific synthesis

The AVeriTeC external-transfer evidence is asymmetric across the three frozen scales.

At Mamba-130M and Mamba-370M, the frozen selected-versus-control displacement has a
small positive mean effect on correct-class margin and the prospectively defined
positive external-transfer tests were supported.

At Mamba-1.4B, the frozen point estimate changes sign:

`mean D_EXT_14B = -7.850951577129465e-05`.

However, the predeclared negative-sign test did not pass its alpha gate:

`t(461) = -1.4372189314350718`,
one-sided `p = 0.0756670802614973`.

Therefore the correct three-scale statement is:

> Positive AVeriTeC external transfer is supported at the frozen 130M and 370M
> checkpoints. The frozen 1.4B point estimate is negative, but negative-sign
> external transfer is not established under the prospectively specified test.

The descriptive sign change at 1.4B must not be promoted into a statistically
established three-scale external sign reversal.

## 5. Margin-level rather than prediction-level behavior at 1.4B

At Mamba-1.4B:

- native accuracy: `0.12987012987012986`;
- dominant-neutralized accuracy: `0.12987012987012986`;
- dominant-control accuracy: `0.12987012987012986`;
- native-versus-neutralized prediction flip rate: `0.0`;
- native-versus-control prediction flip rate: `0.0`.

Thus the observed 1.4B intervention effect is confined to margin perturbation on this
cohort and does not alter the argmax prediction for any of the `462` examples.

This does not invalidate a margin-level causal effect, but it sharply limits any claim
about benchmark decision changes or accuracy improvement.

## 6. Source-label heterogeneity is descriptive only

Frozen source-label means are:

| source label | 130M mean D_EXT | 370M mean D_EXT | 1.4B mean D_EXT |
|---|---:|---:|---:|
| Refuted | `+0.00048267237377375914` | `+0.00005504660555624334` | `+0.00004571420127443863` |
| Supported | `-0.00032780590463815196` | `-0.0000022896524122367483` | `-0.00042802124940848085` |
| Not Enough Evidence | `-0.000013664554405425277` | `+0.00012954951831229016` | `+0.000057267564365507234` |

A notable descriptive pattern is that the `Supported` subgroup has a negative mean at
all three frozen scales, while the overall 130M/370M results remain positive.

No subgroup hypothesis was prospectively declared for the 1.4B extension. These
label-conditioned means therefore remain descriptive and must not be assigned subgroup
p-values or promoted into a supported subgroup mechanism.

## 7. Relation to frozen synthetic/readout evidence

The previously frozen Mamba-1.4B synthetic behavioral bridge and local readout
alignment were both negative:

- synthetic behavioral `mean D_BEH = -0.002012885312239329`;
- local readout `mean Delta_L = -0.0011954475058862238`.

The AVeriTeC external point estimate is directionally consistent with those two
negative 1.4B quantities, but its magnitude is substantially smaller and the
prospective external negative-sign test was not supported.

Accordingly, the natural-language external domain provides evidence of attenuation,
not confirmation, of the stronger negative synthetic/readout pattern.

This is a descriptive interpretation of the frozen point estimates and not a new
inferential test of cross-domain attenuation.

## 8. Interpretation boundary

Supported statements:

- positive AVeriTeC external transfer is supported at 130M and 370M under their
  completed prospective family;
- the 1.4B AVeriTeC point estimate is negative;
- the 1.4B negative mean sign gate passed;
- the 1.4B alpha gate failed;
- the 1.4B negative-sign external-transfer claim is therefore not established;
- all 1.4B intervention conditions preserve the same argmax prediction on all 462
  examples;
- source-label effects are heterogeneous descriptively.

Not established:

- a three-scale AVeriTeC sign reversal;
- a monotonic scaling law;
- a parameter-count threshold;
- a zero crossing between 370M and 1.4B;
- a universal negative 1.4B external effect;
- subgroup-specific causal transfer;
- benchmark accuracy improvement;
- complete causal mediation;
- semantic identity of the scale-local principal planes.

## 9. Stop rule

Do not rescue the 1.4B result by:

- adding a second p-value;
- changing the test direction;
- filtering the 462-item cohort;
- testing source-label subgroups post hoc;
- changing tokenizer, anchor, target offset, plane, layer, or checkpoint;
- pooling 130M/370M/1.4B into a new inferential family.

Any future study of source-label heterogeneity or cross-domain attenuation must be a
new prospectively defined scientific question, not a continuation of this primary
test.
