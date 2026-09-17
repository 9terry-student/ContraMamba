# Gen4 PP3 Cross-Generator Mechanism Synthesis

## Status

`PP3_CROSS_GENERATOR_LOCAL_SUSCEPTIBILITY_MECHANISM_SUPPORTED`

This document synthesizes already validated evidence only.
No new model execution, statistical test, subgroup analysis, rescue analysis,
hyperparameter sweep, or response-guided selection is performed here.

## Evidence chain

### 1. Fresh-index cross-family holdout

The frozen XG2 five-dimensional basis showed positive susceptibility advantage
over the frozen XG4 basis on fresh indices in both XG2 and XG4 source families.

Result commit:

`6252ffca422b15227863b069b3efc4ad5580dfba`

The confirmatory endpoint was:

`Q = E_XG2 - E_XG4`

with positive mean Q and confirmatory rejection in both source families.

This established fresh-index cross-family sensitivity, but did not identify the
responsible internal geometric component.

### 2. Static projector-contrast localization

Static decomposition of the already observed holdout responses localized the
shared dominant component to principal pair 3 of the frozen XG2/XG4 projector
contrast.

Localization commit:

`83b8cf7e73eb68343cc4c97e3f0698df4b587408`

In both XG2 and XG4 source families:

- dominant positive contrast mode: `PP3+`
- dominant positive-net principal plane: `PP3`

PP3 positive-net share:

- XG2: `0.527178977905`
- XG4: `0.555490065950`

The broader response remained distributed across secondary planes, so the
evidence does not support a pure single-mode explanation.

PP3 was also not the maximally separated principal plane, and XG2 did not have
greater total top-5 Phase-1 mass than XG4. Therefore the observed PP3 dominance
was not reduced to either of those geometric explanations.

### 3. Prospective XG1 external-generator transport

The PP3 projector-contrast plane was frozen before inspecting XG1 model
responses and then directly probed on the structurally independent XG1
generator.

Result commit:

`7ae74c656922663569d84a7e060308a77c450eab`

Population:

`xg1_fact_001..xg1_fact_300`

Primary endpoint:

`C_PP3 = (s3 / 5) * (J_PP3_PLUS^2 - J_PP3_MINUS^2)`

Prespecified primary inference:

- N = 300
- one-sample Student t-test
- one-sided alternative `mean(C_PP3) > 0`
- alpha = 0.05
- exactly one primary hypothesis

Observed:

- mean C_PP3 = `7.1202761928785655e-08`
- SD = `6.4998373256093212e-08`
- t(299) = `18.97382890091431`
- one-sided p = `1.5194095747784581e-53`

Frozen conclusion:

`PP3_PROJECTOR_CONTRAST_TRANSPORT_SUPPORTED_ON_XG1_EXTERNAL_GENERATOR`

The signed PP3+ diagnostic was positive for all 300 XG1 pairs, but that
observation remains descriptive and is not promoted to an additional
inferential hypothesis.

## Combined scientific claim

Taken together, the validated evidence supports a cross-generator local
susceptibility mechanism:

A principal contrast plane discovered from frozen XG2/XG4 native-Mamba
response geometry, localized without new model responses, and frozen
prospectively before XG1 execution preserves positive susceptibility contrast
on a structurally independent XG1 generator.

This supports transport of the PP3 local hidden-state susceptibility geometry
beyond the source-generator families used to discover it.

## Boundaries

The evidence does not establish:

- benchmark state of the art;
- improved task accuracy or downstream utility;
- that PP3 is the sole mechanism;
- universal transport to arbitrary generators or datasets;
- universal signed PP3+ causality;
- necessity or sufficiency of PP3 for downstream model behavior.

## Next scientific question

The next prospective question is specificity rather than another unconstrained
replication:

Does frozen PP3 transport more local susceptibility contrast than a
response-blind, deterministically frozen matched control geometry on external
generator data?

Any control must be selected without XG1 response information and frozen before
new scientific execution. No PP1/PP2/PP4/PP5 rescue selection, response-guided
rotation, epsilon sweep, checkpoint sweep, or subgroup selection is allowed.
