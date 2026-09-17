# Gen4 PP3 Transport + Specificity + Necessity Mechanism Synthesis

## Status

`PP3_TRANSPORT_SPECIFICITY_AND_LOCAL_NECESSITY_SUPPORTED`

This document synthesizes already validated evidence only.

No new model execution, statistical test, subgroup analysis, rescue analysis,
hyperparameter sweep, response-guided selection, scientific endpoint, or
execution authorization is introduced here.

## Evidence sequence

### 1. Cross-family susceptibility

Fresh-index XG2-vs-XG4 experiments established an asymmetry in the frozen
native-Mamba local susceptibility geometry.

Validated result commit:

`6252ffca422b15227863b069b3efc4ad5580dfba`

This established the broad susceptibility phenomenon but did not identify which
internal geometric component contributed to it.

### 2. Static PP3 localization

Static decomposition of already observed XG2/XG4 responses localized the
dominant shared positive contrast component to principal pair 3.

Localization commit:

`83b8cf7e73eb68343cc4c97e3f0698df4b587408`

Frozen localization included:

- dominant positive contrast mode: `PP3+`
- dominant positive-net principal plane: `PP3`
- PP3 positive-net share on XG2: `0.527178977905`
- PP3 positive-net share on XG4: `0.555490065950`

The broader response remained distributed across secondary principal planes.

Therefore localization did not establish PP3 as the sole mechanism.

### 3. Prospective external-generator transport

PP3 was frozen before inspecting XG1 responses and prospectively tested on an
independent XG1 generator population.

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

Thus the localized PP3 susceptibility geometry transported beyond the generator
families used for its discovery.

### 4. Prospective geometry-specificity test

A second non-overlapping fresh XG1 population tested whether PP3 transport could
be explained merely by principal-plane geometric separation.

Population:

`xg1_fact_301..xg1_fact_600`

PP5 was prospectively frozen as the response-independent max-separation
principal-plane control.

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
- fraction D_SPEC > 0: `0.6166666666666667`
- t(299): `7.655867341055923`
- one-sided p: `1.3366883647357252e-13`

Frozen conclusion:

`PP3_SPECIFICITY_OVER_MAX_SEPARATION_PP5_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Therefore PP3 transport is not adequately explained by the simple hypothesis
that the principal plane with greater geometric separation necessarily carries
the stronger susceptibility contrast.

At this point PP3 was supported as a specific distributed component, but its
necessity contribution remained unresolved.

### 5. Prospective matched-control necessity test

The necessity experiment addressed the distinct causal question:

Does selectively removing the frozen PP3 component attenuate the already
established broad susceptibility contrast more than an intervention-magnitude
matched PP5 coefficient-transfer control?

Fresh population:

`xg1_fact_601..xg1_fact_900`

Corrected design commit:

`f4419f1e7efbeeb9e50e84b67accc56422580cf2`

Static preparation commit:

`e470f132a37c731754feb4333dcb2e48e29af53b`

Implementation freeze:

`f1896e0d3669bac832038e1486b2b04d322af0aa`

Execution freeze:

`dc85079cb21b78610b4de6a12179ea5e7d72f1b4`

Validated result commit:

`fc4bf45ed7f2a0e635a7e7503cf3ca6280774024`

#### Frozen intervention

For each native strong-channel state `h`:

`a = <h, pp3_plus>`

`b = <h, pp3_minus>`

PP3 treatment:

`delta_PP3 = -a*pp3_plus - b*pp3_minus`

Matched PP5 coefficient-transfer control:

`delta_PP5CTRL = -a*pp5_plus - b*pp5_minus`

The PP5 control used the same PP3-derived coefficients `(a,b)`, so the
condition-correction magnitude was matched without neutralizing PP5 using its
own native projection coefficients.

#### Broad endpoint

For each condition:

`Q = E_XG2 - E_XG4`

with frozen five-dimensional XG2 and XG4 bases.

Per pair:

`A3 = Q0 - Q3`

`A5 = Q0 - Q5`

Primary necessity contrast:

`D_NEC = A3 - A5 = Q5 - Q3`

#### Execution boundary

Accepted execution:

`gen4-pp3-necessity-xg1-601-900-dc85079-retry2`

Execution HEAD:

`dc85079cb21b78610b4de6a12179ea5e7d72f1b4`

Scientific model forwards:

`36000`

Baseline forwards:

`0`

Training:

`none`

Backward:

`none`

GPU-run primary inference:

`none`

Raw scientific conclusion during execution:

`null`

#### Prospectively frozen single primary inference

- N: `300`
- df: `299`
- alternative: `mean(D_NEC) > 0`
- alpha: `0.05`
- confirmatory hypothesis count: `1`
- multiplicity correction: none

Observed:

- mean Q0:
  `1.8678970016861961e-07`
- mean A3:
  `8.4608268084375035e-08`
- mean D_NEC:
  `4.7414371121837106e-08`
- SD D_NEC:
  `3.4241842639833269e-08`
- t(299):
  `23.983551544160797`
- one-sided p:
  `6.2242759364566024e-72`

All prospectively frozen positive-label gates passed:

1. `mean(Q0) > 0`
2. `mean(A3) > 0`
3. `mean(D_NEC) > 0`
4. one-sided `p < 0.05`

Frozen conclusion:

`PP3_NECESSITY_OVER_MATCHED_PP5_CONTROL_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

No rescue, subgroup, tail, alternative-control, epsilon, layer, token,
checkpoint, or additional inferential test was performed.

## Combined mechanism conclusion

The completed evidence chain is:

`cross-family susceptibility`
→ `PP3 localization`
→ `external-generator transport`
→ `geometry-control specificity`
→ `matched-control local necessity`

Together, these results support the following bounded mechanism claim.

PP3 is a reproducibly identifiable component of the frozen layer-17,
target-token native-Mamba susceptibility geometry.

Its contrast transports prospectively to an independent generator population.

Its effect is more specific than a response-independent principal plane with
greater geometric separation.

Most importantly, selectively removing its native component attenuates the broad
XG2-vs-XG4 susceptibility contrast more than a coefficient- and
intervention-magnitude-matched orthogonal PP5 control.

Therefore PP3 is supported as a **transportable, geometry-specific, locally
necessary contributor** to the observed cross-generator susceptibility
mechanism.

## Distributed-mechanism interpretation

The correct interpretation remains distributed rather than one-dimensional.

The original localization showed that PP3 accounted for only approximately half
of the positive-net principal-plane response, with substantial contribution
remaining outside PP3.

The necessity experiment demonstrates a causal contribution of PP3 relative to
the matched PP5 control. It does not imply that eliminating PP3 eliminates the
entire susceptibility phenomenon.

Accordingly, the evidence is consistent with a distributed local mechanism in
which PP3 is an important causal component rather than the complete mechanism.

## What is now established

Within the frozen model, checkpoint, layer, token location, intervention
semantics, generator families, and evaluated populations, the evidence supports:

- a broad XG2-vs-XG4 local susceptibility asymmetry;
- localization of a dominant shared component to PP3;
- prospective transport of PP3 contrast to independent XG1 examples;
- PP3 specificity beyond the max-separation PP5 geometric control;
- stronger attenuation after PP3 removal than after the matched PP5
  coefficient-transfer control.

These statements are supported by separate non-overlapping XG1 populations for
transport, specificity, and necessity.

## What is not established

The completed evidence does not establish:

- PP3 as the sole internal mechanism;
- PP3 sufficiency for the broad susceptibility effect;
- complete elimination of susceptibility under PP3 removal;
- global behavioral necessity;
- downstream-task necessity or sufficiency;
- improved benchmark accuracy;
- universal transport across arbitrary generators or datasets;
- invariance across arbitrary checkpoints, architectures, layers, or token
  locations;
- a general causal law for all Mamba hidden-state geometry.

No claim beyond the frozen local susceptibility mechanism should be inferred.

## Branch conclusion

The PP3 mechanism sequence is complete for its current bounded scientific
question.

The previously unresolved necessity question has now been answered positively
under a prospectively frozen, non-overlapping fresh-XG1, matched-control
protocol.

No additional replication, rescue control, subgroup mining, or p-value is
required to establish this exact bounded claim.

Any future experiment should address a genuinely new scientific question rather
than re-test transport, specificity, or local necessity under minor variants.
