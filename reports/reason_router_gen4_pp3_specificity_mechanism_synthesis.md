# Gen4 PP3 Specificity Mechanism Synthesis

## Status

`PP3_CROSS_GENERATOR_LOCAL_SUSCEPTIBILITY_SPECIFICITY_SUPPORTED`

This document synthesizes already validated evidence only.

No new model execution, statistical test, subgroup analysis, rescue analysis,
hyperparameter sweep, response-guided selection, or scientific endpoint is
introduced here.

## Evidence chain

### 1. Fresh-index cross-family susceptibility

Frozen XG2-vs-XG4 basis evidence established positive fresh-index susceptibility
advantage across the frozen XG2 and XG4 source families.

Result commit:

`6252ffca422b15227863b069b3efc4ad5580dfba`

This established fresh-index cross-family sensitivity but did not localize the
responsible internal geometric component.

### 2. Static projector-contrast localization

Static decomposition of already observed responses localized the shared
dominant positive contrast component to principal pair 3.

Localization commit:

`83b8cf7e73eb68343cc4c97e3f0698df4b587408`

Frozen observations included:

- dominant positive contrast mode: `PP3+`
- dominant positive-net principal plane: `PP3`
- PP3 positive-net share on XG2: `0.527178977905`
- PP3 positive-net share on XG4: `0.555490065950`

The broader response remained distributed across secondary planes.

Therefore this evidence did not establish PP3 as the sole mechanism.

### 3. Prospective external-generator transport

PP3 was frozen before inspecting XG1 responses and prospectively tested on the
structurally independent XG1 generator.

Result commit:

`7ae74c656922663569d84a7e060308a77c450eab`

Population:

`xg1_fact_001..xg1_fact_300`

Primary endpoint:

`C_PP3 = (s3 / 5) * (J_PP3_PLUS^2 - J_PP3_MINUS^2)`

Observed:

- N = `300`
- mean C_PP3 = `7.1202761928785655e-08`
- SD = `6.4998373256093212e-08`
- t(299) = `18.97382890091431`
- one-sided p = `1.5194095747784581e-53`

Frozen conclusion:

`PP3_PROJECTOR_CONTRAST_TRANSPORT_SUPPORTED_ON_XG1_EXTERNAL_GENERATOR`

This established transport of the localized PP3 susceptibility geometry beyond
the generator families used to discover it.

### 4. Prospective specificity against max-separation PP5 control

The next frozen question was whether PP3 transport represented specificity
beyond a response-blind geometry control rather than merely larger principal
separation.

PP5 was prospectively frozen as the geometry-only max-separation control.

Fresh specificity population:

`xg1_fact_301..xg1_fact_600`

Execution commit:

`95d2561aa09871279f80e3ffac24c61c0eceb3c9`

Validated result commit:

`858d61fe719d9cffba193053b11f102f6a8f7e92`

Exact observation budget:

- N = `300`
- directions = `PP3+`, `PP3-`, `PP5+`, `PP5-`
- scientific model forwards = `4800`
- baseline model forwards = `0`
- training = none
- backward = none
- GPU-run primary inference = none

Specificity endpoint:

`D_SPEC = C_PP3 - C_PP5`

where:

`C_PPk = (s_k / 5) * (J_PLUS^2 - J_MINUS^2)`

Observed descriptive values:

- mean C_PP3 = `7.123485238267578e-08`
- SD C_PP3 = `6.59802732329506e-08`
- mean C_PP5 = `4.8194178690694314e-08`
- SD C_PP5 = `3.375197271128536e-08`
- mean D_SPEC = `2.3040673691981465e-08`
- SD D_SPEC = `5.212684036610186e-08`
- fraction D_SPEC > 0 = `0.6166666666666667`

Prospectively frozen single confirmatory inference:

- H1: `mean(D_SPEC) > 0`
- one-sample Student t-test
- one-sided
- N = `300`
- df = `299`
- alpha = `0.05`
- multiplicity correction = none
- confirmatory hypothesis count = `1`

Observed:

- t(299) = `7.655867341055923`
- one-sided p = `1.3366883647357252e-13`

All frozen positive-label requirements passed:

1. mean C_PP3 > 0
2. mean D_SPEC > 0
3. one-sided p < 0.05

Frozen conclusion:

`PP3_SPECIFICITY_OVER_MAX_SEPARATION_PP5_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

No rescue, subgroup, tail, alternative-control, epsilon, checkpoint, layer, or
token test was executed.

## Combined scientific claim

The validated evidence supports the following bounded mechanism claim.

A local hidden-state susceptibility contrast plane, PP3, was discovered from
frozen XG2/XG4 native-Mamba response geometry, localized without new
response-guided model execution, prospectively frozen, and shown to transport
to an independent XG1 generator.

On a second non-overlapping fresh XG1 holdout, PP3 also produced greater
susceptibility contrast than PP5, even though PP5 was the frozen principal
plane with greater projector separation.

Therefore the observed PP3 transport is not adequately explained by the simple
hypothesis that larger principal-plane separation alone produces the effect.

The evidence supports PP3 as a specific distributed component of the observed
cross-generator local susceptibility geometry.

## What this does not establish

The evidence does not establish:

- that PP3 is the sole mechanism;
- that PP3 is necessary for the full susceptibility effect;
- that PP3 is sufficient for the full susceptibility effect;
- that the mechanism universally transports to arbitrary generators or data;
- universal signed PP3+ causality;
- improved downstream task accuracy;
- benchmark state of the art;
- downstream behavioral necessity or sufficiency.

The response remains distributed across secondary principal planes, so the
current result should not be reframed as a one-dimensional mechanism.

## Current branch conclusion

The original sequence

`cross-family sensitivity`
→ `PP3 localization`
→ `external-generator transport`
→ `max-separation-control specificity`

is now complete.

The specificity question posed by the prior synthesis is answered positively
under the prospectively frozen fresh-XG1 protocol.

No further replication or control rescue is required to establish this exact
claim.

## Next unresolved causal question

The next distinct scientific question is necessity rather than further
specificity:

Does selectively removing or neutralizing the frozen PP3 component materially
attenuate the already established local susceptibility contrast, relative to a
prospectively frozen matched control intervention?

That question is not answered by the current evidence and is not authorized by
this synthesis.
