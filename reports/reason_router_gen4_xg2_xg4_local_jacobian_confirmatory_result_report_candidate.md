# Gen4-K XG2/XG4 Unit-Direction Local Jacobian — Confirmatory Result Report

## Status

Result class: prespecified read-only confirmatory analysis over frozen local-Jacobian artifacts.

Artifact freeze commit:

`62f4b2b800ff4ca369847f97b6e755b666522b03`

No new model forwards, no baseline forwards, no training, and no additional scientific execution were performed for this analysis.

Primary endpoint and decision rule were frozen before inference:
- endpoint: `J_0.025`
- test: one-sample Student t-test, one-sided `mean(J_0.025) < 0`
- multiplicity family: exactly XG2 and XG4
- correction: Holm, alpha = 0.05
- `epsilon=0.05`: descriptive scale-consistency only, with no additional p-value
- replication label requires both families to have negative `J_0.025` means with Holm rejection, and both families to have negative `J_0.05` means.

This study is prospective only for the previously unobserved local-Jacobian outcomes. It is not an independent replication study.

## Frozen population

- XG2: N = 300
- XG4: N = 300
- exact frozen pair population: 301..600 per family
- no subgroup, tail, PCA, direction, epsilon, layer, offset, channel, or checkpoint selection was performed after observing the outcomes.

## Primary confirmatory result

### XG2

For `J_0.025`:

- N = 300
- mean = `9.63921110782214e-08`
- median = `-7.103457568269356e-06`
- sample SD = `0.0003495134594824075`
- standard error = `2.017916899175653e-05`
- t(299) = `0.004776812717986499`
- one-sided raw p = `0.5019040725905776`
- Holm-adjusted p = `1.0`
- Holm reject at alpha=0.05: `FALSE`

The prespecified negative-mean hypothesis is not supported for XG2.

### XG4

For `J_0.025`:

- N = 300
- mean = `2.3520075836721185e-05`
- median = `-8.959011941644945e-05`
- sample SD = `0.0005146835797878925`
- standard error = `2.9715270333801998e-05`
- t(299) = `0.7915147859168693`
- one-sided raw p = `0.7853644519057472`
- Holm-adjusted p = `1.0`
- Holm reject at alpha=0.05: `FALSE`

The prespecified negative-mean hypothesis is not supported for XG4.

## Prespecified replication decision

The frozen replication criterion is not met.

Final label:

`LOCAL_DIRECTIONAL_SENSITIVITY_REPLICATION_NOT_ESTABLISHED`

Reasons:
1. XG2 does not reject the prespecified `mean(J_0.025) < 0` hypothesis after Holm correction.
2. XG4 does not reject the prespecified `mean(J_0.025) < 0` hypothesis after Holm correction.
3. XG4 has a positive mean at both tested radii.
4. Therefore the conjunctive cross-generator replication rule fails.

This result does not revise the already frozen full-intervention confirmatory result.

## Descriptive epsilon=0.05 scale consistency

No p-value was computed for `epsilon=0.05`.

### XG2

For `J_0.05`:
- mean = `-1.4531306523994185e-07`
- median = `-3.4885500854731077e-06`
- sample SD = `0.00034945713075913286`

Cross-radius consistency:
- Pearson correlation between `J_0.025` and `J_0.05` = `0.9998938049573476`
- Spearman correlation = `0.9981457571750796`
- sign agreement fraction = `1.0`
- zero counts: 0 at both radii

### XG4

For `J_0.05`:
- mean = `2.436257019904353e-05`
- median = `-8.16850430240823e-05`
- sample SD = `0.0005138636266114009`

Cross-radius consistency:
- Pearson correlation between `J_0.025` and `J_0.05` = `0.9997485946538284`
- Spearman correlation = `0.997388415426838`
- sign agreement fraction = `1.0`
- zero counts: 0 at both radii

The item-level local-Jacobian measurements are therefore extremely stable between the two prespecified radii. The failure of the negative-mean confirmatory hypothesis is not explained by a sign-unstable or radius-sensitive local derivative estimate over `epsilon=0.025` versus `0.05`.

This scale-consistency observation does not establish a generator-invariant causal direction. It only shows that the measured local directional responses are internally stable across the two prespecified probe radii.

## Descriptive curvature summaries

No inferential test was prespecified for curvature.

### XG2

`K_0.025`:
- mean = `-6.0904240761180014e-05`
- median = `-1.0974221709147965e-05`
- sample SD = `0.0006820070862633753`

`K_0.05`:
- mean = `-1.4738301016355612e-05`
- median = `-9.430093217410727e-06`
- sample SD = `0.00019315888447141833`

### XG4

`K_0.025`:
- mean = `-6.578642400908071e-06`
- median = `3.2082168033298326e-06`
- sample SD = `0.0014076118447011628`

`K_0.05`:
- mean = `4.438394034966817e-06`
- median = `6.269341046305497e-07`
- sample SD = `0.00036275071901108505`

These curvature quantities are descriptive only. No claim about curvature significance is authorized by this analysis.

## Scientific interpretation

The local-Jacobian experiment does not support the specific prospective hypothesis that the frozen intervention direction has a generator-invariant negative first-order effect on the path-efficiency contrast.

For XG2, the population mean local derivative is essentially centered at zero at both radii. For XG4, the population mean is positive at both radii. Neither family supports the prespecified negative-mean endpoint, and both Holm-adjusted p-values are 1.0.

At the same time, the near-perfect cross-radius correlations and exact item-level sign agreement show that the observed first-order responses are highly reproducible with respect to the two prespecified local radii. Thus the null confirmatory outcome should not be attributed to an unstable finite-difference radius choice within the tested range.

Together with the previously frozen orientation diagnostics, the current result is compatible with generator-dependent directional geometry and does not support a single frozen intervention direction that transports as a common adverse local mechanism across XG2 and XG4.

This conclusion is limited to the tested frozen direction, checkpoint, layer, intervention point, population, and two prespecified local radii. It does not establish absence of all causal state directions, and it does not justify posthoc direction search, subgroup rescue, epsilon search, or architecture modification.

## Boundary with prior evidence

This report must remain separate from:
- the frozen full-intervention confirmatory result;
- the posthoc orientation diagnostic;
- any future generator-specific geometry study.

The local-Jacobian result provides a direct first-order test of the frozen direction. It does not overwrite earlier results; it narrows the surviving mechanism hypothesis.

## Final conclusion

`LOCAL_DIRECTIONAL_SENSITIVITY_REPLICATION_NOT_ESTABLISHED`

The prespecified cross-generator negative local directional sensitivity claim is not established.

The stable cross-radius response indicates that the failed replication criterion is not attributable to obvious sign instability across the two prespecified finite-difference radii. The next scientific stage should therefore change the hypothesis class—from a single transported frozen direction to generator-specific directional geometry—rather than continue rescue analysis on the same direction.
