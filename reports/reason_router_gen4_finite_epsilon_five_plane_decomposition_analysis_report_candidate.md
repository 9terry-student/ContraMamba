# Gen4 finite-epsilon five-plane decomposition analysis

## Provenance

- run: `g4k-finite-epsilon-five-plane-decomposition-xg1-2401-2700-e27dbbd-retry1`
- execution HEAD: `e27dbbd45635da4beb03691f5a266a7d72824424`
- N: 300
- epsilon: 0.025
- new scientific forwards: 12000
- prior native Q0 reused: `True`
- prior items SHA256: `9d5dabbef82a8fcfaccc4e610bbb4d91f1e7ee9ea2f2627a92f999c88034ac49`

## Reconstruction

- mean Q0: `1.8756264438819782e-07`
- mean Q_principal: `1.8775139776958283e-07`
- mean residual: `-1.8875338138496055e-10`
- mean absolute residual: `2.8531396523418933e-09`
- median absolute residual: `1.3244640761284971e-09`
- residual RMSE: `6.8769472374305173e-09`
- Q0 RMS: `2.2111576489139539e-07`
- normalized RMSE / RMS(Q0): `0.031101116832661126`
- normalized MAE / mean(|Q0|): `0.015211662544257794`
- mean residual / mean(Q0): `-0.0010063484762685381`
- Pearson(Q0, Q_principal): `0.99827755149854303`
- sign agreement: `1`

## Per-item absolute relative residual quantiles

- q50: `0.010794005732471159`
- q90: `0.091562475677804636`
- q95: `0.14795219050155792`
- q99: `0.47852926994961981`

## Mean finite-epsilon plane contributions

- P1: `2.8557904254582005e-08`
- P2: `3.4693191890669348e-08`
- P3: `7.2858080938013261e-08`
- P4: `2.8744010983894693e-09`
- P5: `4.8767819587928708e-08`

## Analysis boundary

- descriptive finite-epsilon reconstruction analysis only
- inferential test count: 0
- p-value count: 0
- no post-hoc support threshold was applied
- causal additivity/interaction is not evaluated here
- plane contribution magnitudes are not causal rankings
## Scientific interpretation

The direct principal-direction measurement at epsilon = 0.025 yields a
high-fidelity finite-epsilon reconstruction of the frozen native Q endpoint,
but not an exact finite-epsilon identity.

Observed reconstruction diagnostics over N=300:

- normalized RMSE / RMS(Q0): `0.031101116832661126`
- normalized MAE / mean(|Q0|): `0.015211662544257794`
- mean residual / mean(Q0): `-0.0010063484762685381`
- Pearson(Q0, Q_principal): `0.99827755149854303`
- sign agreement: `1.0`
- median absolute relative residual: `0.010794005732471159`
- q90 absolute relative residual: `0.091562475677804636`
- q95 absolute relative residual: `0.14795219050155792`
- q99 absolute relative residual: `0.47852926994961981`

Therefore the five-plane spectral coordinates capture the overwhelming
majority of finite-epsilon Q geometry at the tested intervention scale,
while a nonzero and heavy-tailed item-level reconstruction defect remains.

This does not establish an exact finite-epsilon decomposition, causal
additivity, plane independence, or absence of interactions. Plane
contribution magnitudes are descriptive spectral contributions and are not
a causal ranking.

Result label:

`FINITE_EPSILON_FIVE_PLANE_SPECTRAL_RECONSTRUCTION_HIGH_FIDELITY_BUT_NOT_EXACT_ON_XG1_2401_2700`
