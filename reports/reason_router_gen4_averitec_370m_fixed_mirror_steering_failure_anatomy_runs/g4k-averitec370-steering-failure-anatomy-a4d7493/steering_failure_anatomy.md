# Experiment 3 — Static Steering Failure Anatomy

Result: `PASS_AVERITEC_370M_FIXED_MIRROR_STEERING_FAILURE_ANATOMY`

This is descriptive anatomy of the frozen failed steering experiment. It adds no p-values and performs no model execution.

## Native-error decomposition

- Native errors: `2468`.
- Helpful margin direction (`M_steer > M_native`): `1257` / `2468` = `0.5093192868719612`.
- Non-helpful or zero direction: `1211` / `2468` = `0.4906807131280389`.
- Actual prediction flips among native errors: `0`.

## Continuous mirror displacement

- Mean `M_steer-M_native` over all examples: `3.804646130231427e-05`.
- Mean `M_native-M_control` over all examples: `4.031383967802844e-05`.
- Mean mirror asymmetry: `-2.2673783757141742e-06`.
- `M_steer > M_native > M_control`: `0.49803501250446586` of all examples.

## Local-linear leverage diagnostic

- Helpful-direction median `lambda*`: `56955.90752118644`.
- Q25/Q75: `[8283.927008412535, 325718.9648022432]`.
- Q90/Q95: `[2120684.0448883325, 6027351.460143295]`.
- Range: `[18.773561468671986, 120965025217.79333]`.

`lambda*` is not an executed coefficient. It is only the local-linear amount of the observed unit displacement that would be required to close the native correct-class margin deficit if the measured displacement scaled linearly.

## Interpretation boundary

This anatomy separates two descriptive failure modes: itemwise direction can fail to improve the correct-class margin, and even when the direction is favorable the observed unit displacement can be too small to reach the decision boundary. It does not rescue Experiment 3 and does not authorize a magnitude sweep.
