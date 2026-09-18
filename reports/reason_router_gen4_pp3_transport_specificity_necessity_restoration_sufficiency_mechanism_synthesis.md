# Gen4 PP3 Local Causal Mechanism + PP3-Excluded Residual Transport Synthesis

## Status

`PP3_TRANSPORT_SPECIFICITY_LOCAL_NECESSITY_AND_RESTORATION_SUFFICIENCY_SUPPORTED`

Residual extension status:

`PP3_EXCLUDED_XG2_LIKE_RESIDUAL_TEMPLATE_TRANSPORT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

This document synthesizes already validated evidence only.

It introduces no new model execution, statistical test, subgroup analysis,
rescue analysis, hyperparameter sweep, response-guided selection, scientific
endpoint, or execution authorization.

## Evidence sequence

The completed evidence chain is:

`cross-family susceptibility`
→ `PP3 localization`
→ `external-generator transport`
→ `geometry-control specificity`
→ `matched-control local necessity`
→ `matched-replacement local restoration sufficiency`
→ `PP3-excluded residual-template transport`

### 1. Cross-family susceptibility

Fresh-index XG2-vs-XG4 experiments established a broad asymmetry in the frozen
native-Mamba local susceptibility geometry.

Validated result commit:

`6252ffca422b15227863b069b3efc4ad5580dfba`

This established the phenomenon but did not identify the responsible internal
geometric components.

### 2. Static PP3 localization

Static decomposition localized the dominant shared positive contrast component
to principal pair 3.

Localization commit:

`83b8cf7e73eb68343cc4c97e3f0698df4b587408`

Frozen localization included:

- dominant positive contrast mode: `PP3+`
- dominant positive-net principal plane: `PP3`
- PP3 positive-net share on XG2: `0.527178977905`
- PP3 positive-net share on XG4: `0.555490065950`

Substantial response remained outside PP3, so the mechanism remained
distributed rather than one-dimensional.

### 3. Prospective external-generator transport

PP3 was frozen before inspection of XG1 outcomes and prospectively tested on
independent XG1 examples.

Population:

`xg1_fact_001..xg1_fact_300`

Validated result commit:

`7ae74c656922663569d84a7e060308a77c450eab`

Primary endpoint:

`C_PP3 = (s3 / 5) * (J_PP3_PLUS^2 - J_PP3_MINUS^2)`

Observed:

- N: `300`
- mean C_PP3: `7.1202761928785655e-08`
- SD: `6.4998373256093212e-08`
- t(299): `18.97382890091431`
- one-sided p: `1.5194095747784581e-53`

Frozen conclusion:

`PP3_PROJECTOR_CONTRAST_TRANSPORT_SUPPORTED_ON_XG1_EXTERNAL_GENERATOR`

Thus the localized PP3 susceptibility geometry transported prospectively to an
independent generator population.

### 4. Prospective geometry-specificity test

A second non-overlapping XG1 population tested whether PP3 transport could be
explained by generic principal-plane geometric separation.

Population:

`xg1_fact_301..xg1_fact_600`

PP5 was frozen prospectively as the response-independent max-separation control.

Validated result commit:

`858d61fe719d9cffba193053b11f102f6a8f7e92`

Specificity synthesis commit:

`cdbffa37f6ac0cf78a7cd76cc5ad96cb3e971068`

Primary endpoint:

`D_SPEC = C_PP3 - C_PP5`

Observed:

- N: `300`
- mean C_PP3: `7.123485238267578e-08`
- mean C_PP5: `4.8194178690694314e-08`
- mean D_SPEC: `2.3040673691981465e-08`
- SD D_SPEC: `5.212684036610186e-08`
- t(299): `7.655867341055923`
- one-sided p: `1.3366883647357252e-13`

Frozen conclusion:

`PP3_SPECIFICITY_OVER_MAX_SEPARATION_PP5_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Therefore PP3 transport is not adequately explained by the simple hypothesis
that a more geometrically separated principal plane necessarily carries the
stronger susceptibility contrast.

### 5. Prospective matched-control local necessity

A third non-overlapping XG1 population tested whether selectively removing the
native PP3 component attenuated the established broad XG2-vs-XG4 susceptibility
contrast more than a magnitude-matched orthogonal PP5 coefficient-transfer
control.

Population:

`xg1_fact_601..xg1_fact_900`

Validated result commit:

`fc4bf45ed7f2a0e635a7e7503cf3ca6280774024`

Primary contrast:

`D_NEC = (Q0 - Q3) - (Q0 - Q5) = Q5 - Q3`

Observed:

- N: `300`
- mean Q0: `1.8678970016861961e-07`
- mean A3: `8.4608268084375035e-08`
- mean D_NEC: `4.7414371121837106e-08`
- SD D_NEC: `3.4241842639833269e-08`
- t(299): `23.983551544160797`
- one-sided p: `6.2242759364566024e-72`

Frozen conclusion:

`PP3_NECESSITY_OVER_MATCHED_PP5_CONTROL_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Thus selective PP3 removal caused a larger attenuation of the broad
susceptibility contrast than the prospectively matched PP5 control.

Necessity synthesis commit:

`543bd8590cc0e62d4fac75c9c9ab13f37fb2df76`

### 6. Prospective matched-replacement local restoration sufficiency

A fourth non-overlapping XG1 population tested the complementary restoration
question.

Population:

`xg1_fact_901..xg1_fact_1200`

Starting from the same PP3-neutralized background:

`B = h - c3`

the experiment compared:

`R3 = B + c3 = h`

against the prospectively frozen matched PP5 replacement:

`R5 = B + c5 = h - c3 + c5`

where `c3` and `c5` used exactly the same native PP3-derived coefficients
`(a,b)` and therefore had equal addition norm.

Execution HEAD:

`53b6ccea8a06fbf9be699bffa034034807970878`

Accepted run:

`gen4-pp3-restoration-sufficiency-xg1-901-1200-53b6cce-retry2`

Validated result commit:

`9561a23ccf3cd8430542dd7d688112fee7a01611`

Primary contrast:

`D_SUF = (Q_R3 - Q_B) - (Q_R5 - Q_B) = Q_R3 - Q_R5`

Observed:

- N: `300`
- mean Q_B: `1.0507711352505444e-07`
- mean Q_R3: `1.8577153849600195e-07`
- mean Q_R5: `1.4498281493001444e-07`
- mean S3: `8.069442497094754e-08`
- mean S5: `3.9905701404959994e-08`
- mean D_SUF: `4.078872356598753e-08`
- SD D_SUF: `3.5139279965303874e-08`
- t(299): `20.105176219299192`
- one-sided p: `9.010901911776928e-58`

All prospectively frozen positive gates passed:

1. `mean(Q_R3) > 0`
2. `mean(S3) > 0`
3. `mean(D_SUF) > 0`
4. one-sided `p < 0.05`

Frozen conclusion:

`PP3_RESTORATION_SUFFICIENCY_OVER_MATCHED_PP5_REPLACEMENT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Thus, on a PP3-neutralized local state background, restoring the exact native
PP3 component recovered the broad susceptibility contrast more strongly than an
equal-coefficient, equal-addition-norm PP5 replacement.


### 7. Prospective PP3-excluded residual-template transport

A fifth non-overlapping XG1 population tested a genuinely new question about
the distributed response remaining after PP3 was excluded.

Population:

`xg1_fact_1201..xg1_fact_1500`

Prospective design commit:

`d3cc008fad221862e6fe9718b67b6ba0c87d0368`

Validated result commit:

`a32ae557b7b0e1797cb632e23ce023c78938b018`

The residual coordinate order was frozen as:

`[P1, P2, P4, P5]`

For each item, the PP3-excluded residual vector was compared against unit
templates constructed only from the already frozen XG2 and XG4 `601..900`
residual mean-net vectors.

Primary endpoint:

`D_TEMPLATE = C_XG2 - C_XG4`

where `C_XG2` and `C_XG4` are cosine alignments of the item-level residual
vector with the frozen XG2 and XG4 residual templates.

Observed:

- N: `300`
- mean C_XG2: `0.86608386493388712`
- SD C_XG2: `0.11622169218083186`
- mean C_XG4: `0.28077128559825382`
- SD C_XG4: `0.2604699022409554`
- mean D_TEMPLATE: `0.5853125793356333`
- SD D_TEMPLATE: `0.3453234709049785`
- fraction D_TEMPLATE > 0: `0.81333333333333335`
- t(299): `29.357724311692319`
- one-sided p: `2.2607093023100906e-90`
- confirmatory p-value count: `1`
- additional p-values: `0`

All frozen positive-label gates passed:

1. `mean(C_XG2) > 0`
2. `mean(D_TEMPLATE) > 0`
3. one-sided `p < 0.05`

Frozen conclusion:

`PP3_EXCLUDED_XG2_LIKE_RESIDUAL_TEMPLATE_TRANSPORT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Thus the PP3-excluded residual response is not merely an unstable leftover on
fresh XG1 examples. Its orientation prospectively transported in an XG2-like
direction relative to the frozen XG4 residual template.

This is a geometric transport result, not a causal result for P1/P2/P4/P5.
It does not establish necessity, sufficiency, or causal ownership of any
secondary residual plane, and no individual residual plane is promoted.


## Combined mechanism conclusion

Across the first four prospectively separated XG1 populations, PP3 satisfies
four distinct causal-mechanism evidential roles:

1. its localized contrast transports outside the discovery generator families;
2. its effect is stronger than the prospectively frozen max-separation PP5
   geometric control;
3. selectively removing its native component attenuates the broad
   susceptibility contrast more than the matched PP5 control;
4. restoring that native component from a PP3-neutralized background recovers
   the contrast more strongly than the matched PP5 replacement.

The fifth prospective XG1 population addresses the residual mechanism rather
than re-testing PP3. It shows that after PP3 exclusion, the remaining
four-plane residual orientation transports as XG2-like relative to the frozen
XG4 residual template.

Together these results support PP3 as a **transportable, geometry-specific,
locally necessary, and locally restoration-sufficient contributor** to the
frozen layer-17 / target-token native-Mamba susceptibility mechanism.

Necessity and restoration sufficiency are complementary but not equivalent.

The necessity result shows that PP3 removal causes a specific loss relative to
the matched PP5 control.

The restoration result shows that adding the exact removed PP3 component back
to the neutralized background causes a specific recovery relative to the
matched PP5 replacement.

The combination therefore strengthens the local causal mechanism
interpretation without converting it into a global or one-dimensional claim.

## Distributed-mechanism interpretation

The mechanism remains distributed.

The original localization assigned only about half of the positive-net
principal-plane response to PP3. Secondary principal planes retained substantial
response.

Neither the necessity experiment nor the restoration experiment establishes
that PP3 alone accounts for the entire susceptibility phenomenon.

The evidence is therefore most consistent with a distributed local mechanism in
which PP3 is a reproducible and causally important component, not the complete
mechanism.

The fresh `1201..1500` confirmatory result further resolves the previously
unspecified remainder: once PP3 is excluded, the aggregate residual orientation
is prospectively XG2-like rather than XG4-like under the frozen template test.

The combined picture is therefore:

`shared PP3 causal core + structured XG2-like secondary residual geometry on XG1`

The second term is geometric and transportable under the tested protocol, but
it is not yet causally localized. No individual secondary plane should be
interpreted as necessary, sufficient, or promoted.

## What is now established

Within the frozen model, checkpoint, layer, target token, intervention
semantics, generator families, and evaluated populations, the evidence supports:

- a broad XG2-vs-XG4 local susceptibility asymmetry;
- localization of a dominant shared component to PP3;
- prospective PP3 transport to an independent generator population;
- PP3 specificity beyond a response-independent max-separation PP5 control;
- stronger attenuation after PP3 removal than after the matched PP5 control;
- stronger recovery after native PP3 restoration than after the matched PP5
  replacement;
- prospective XG2-like transport of the PP3-excluded aggregate residual
  orientation on a fifth fresh XG1 population.

The PP3 necessity/restoration claims are causal local-intervention results.
The PP3-excluded residual-template result is a distinct prospective geometric
transport result and does not promote the secondary residual planes to causal
status.

## What is not established

The completed evidence does not establish:

- PP3 as the sole internal mechanism;
- causal necessity or sufficiency of P1, P2, P4, or P5 from the residual
  template result;
- identification of a single secondary residual plane as the XG2-like
  mechanism;
- PP3 alone as sufficient in an otherwise empty state;
- complete elimination of susceptibility after PP3 removal;
- global behavioral necessity or sufficiency;
- downstream-task necessity or sufficiency;
- benchmark improvement;
- universal transport across arbitrary generators or datasets;
- invariance across arbitrary checkpoints, architectures, layers, or token
  positions;
- architecture-wide or all-Mamba universality;
- a general causal law for arbitrary hidden-state geometry.

## Branch conclusion

The PP3 transport/specificity/necessity/restoration-sufficiency sequence is
complete for its bounded local-mechanism question.

The first prospective PP3-excluded residual-template transport question is also
complete and supports an XG2-like aggregate secondary residual orientation on
fresh XG1 examples.

No additional replication, rescue control, subgroup mining, alternative tail,
or p-value is required for either completed claim.

A subsequent scientific experiment should therefore address a genuinely new
question: causal localization of the structured secondary residual mechanism,
without outcome-guided selection of P1/P2/P4/P5 and without re-testing PP3 under
minor variants.
