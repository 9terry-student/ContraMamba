# Gen5 Temporal Mechanism Partial Recovery Validated Evidence Report

## Identity

- execution HEAD: `d9b790b62f8db9f875cfa13d7ba155ac75137a2b`
- source temporal authority: `adad44c6fb304500c700b9ea59d727cefd71c2d9`
- recovery archive SHA256: `3af11ca1f0fe97182199fc80cb8549031d365f5ddd2828b15d1bc344bf42e784`
- worker0 SHA256: `3f49147ff5d8a1bff7b6e5d3d607e9342ea1f328f00abba41d3d300f92eff3e9`
- worker1 SHA256: `f8ac8f0dcb20e1da1b1e93696137f9a329ff469fd3d32c2f77e8f61a12910e91`
- t20 endpoint diagnostic SHA256: `e64ca120d7ae1ea55ee3145b600d0cad4c48566fb36ec80c76f5396cce300afc`
- recovery mode: no GPU reexecution; no model-forward reexecution

Recovered repository artifact root:

`reports/reason_router_gen5_ainit_temporal_mechanism_recovery_runs/gen5-ainit-temporal-mechanism-d9b790b-r1-partial-recovery/`

## Execution status separation

The two independent Tesla T4 workers completed and wrote their worker results.
The parent process then failed in a post-worker runtime assertion.

The recovered workers authenticate:

- `FROZEN_PREDICTION_MISMATCH_COUNT=0`
- `A_DECAY_ONLY_T0_LOGIT_MAX_ABS=0`
- `FULL_T1_INTERNAL_LOGIT_MAX_ABS=0`

No GPU calculation or model forward was rerun during recovery.

## t=1 micro-origin result

Recovered and authenticated:

- B_UPDATE_ONLY prediction disagreement versus FULL_T1: `0`
- B_UPDATE_ONLY logit max abs versus FULL_T1:
  `1.430511474609375e-06`
- t1 same-training-RNG / different-A disagreement aggregate: `144`
- t1 same-A / different-training-RNG disagreement aggregate: `0`

Bounded conclusion:

> Under the frozen four-state t=1 counterfactual, holding A at A0 while using
> B1 reproduces FULL_T1 predictions exactly and reproduces the logits to
> approximately 1.43e-6 max absolute error. The first behavioral birth is
> already reproduced by the B1-driven write. The A1 decay-only state produces
> no standalone t1 output change because B0 is exactly zero.

## Reconvergence

Recovered behavioral evidence authenticates permanent all-factor prediction
reconvergence from t=17 through t=20.

## t=20 endpoint authentication

Recovered valid-token geometry authentication: `PASS`.

Recovered finite-effect/replay authentication: `PASS`.

The strict task-visible endpoint bundle did not fully authenticate because
exactly three fields failed:

1. `raw_write:task_row_energy`
2. `raw_write:control_task_row_energy`
3. `recurrent_state:normalized_residual`

At the same time:

- c_readout_pre_gate: strict stage authentication PASS
- gated_scan: strict stage authentication PASS
- layer22_out_proj: strict stage authentication PASS
- all five executed finite-intervention gates: PASS
- full replay errors remained approximately 1e-7

Code inspection localized a semantic inconsistency: the temporal stage-local
task-visible accumulator did not apply `attention_mask`, whereas the geometry
path did.

The exact contribution of padded coordinates to each failed scalar is not
retroactively estimated. No post-hoc threshold relaxation is used.

## Executed task-visible values: descriptive only where local-projector
authentication is affected

### t=1

| stage | energy | control | enrichment | margin R_visible | R_complement | R_interaction |
|---|---:|---:|---:|---:|---:|---:|
| raw_write | 0.00111279572736 | 4.77083272673e-06 | 233.249789105 | 0.993588805940 | 0.000449942833 | 0.001196001842 |
| recurrent_state | 0.000864736816627 | 4.35442821130e-06 | 198.587914341 | 0.996222189797 | 0.000171017131 | 0.000505325784 |
| c_readout_pre_gate | 0.00702545847021 | 8.69843955331e-05 | 80.766882694 | 0.995282310541 | 0.000180937620 | 0.000508236494 |
| gated_scan | 0.00223556723628 | 5.43183277952e-05 | 41.156775752 | 0.995426360993 | 0.000172822452 | 0.000471342923 |
| layer22_out_proj | 0.00291391385238 | 0.000107745355621 | 27.044449717 | 0.995447374499 | 0.000202170956 | 0.000549015937 |

### t=17

| stage | energy | control | enrichment | margin R_visible | R_complement | R_interaction |
|---|---:|---:|---:|---:|---:|---:|
| raw_write | 0.000727610869360 | 3.10008314261e-05 | 23.470688878 | 0.882148148519 | 0.010172564385 | 0.020435032036 |
| recurrent_state | 0.000949320178870 | 6.62003554243e-06 | 143.401069796 | 0.899460570957 | 0.003527621500 | 0.004559948389 |
| c_readout_pre_gate | 0.00677175269108 | 8.40278339259e-05 | 80.589399663 | 0.844075239415 | 0.009135800332 | 0.010520726656 |
| gated_scan | 0.00434965422789 | 5.10905830291e-05 | 85.136124311 | 0.869727391361 | 0.005553495364 | 0.007969372770 |
| layer22_out_proj | 0.00528123807343 | 0.000104429436696 | 50.572312181 | 0.865376796384 | 0.006639201973 | 0.013452757140 |

These rows are preserved as executed measurements but do not authorize the
fully authenticated `earliest task-visible stage` claim.

## Scientific boundary

The strongest recovered new result is the t=1 micro-origin result:
B1 with A held at A0 already reproduces FULL_T1 behavior essentially exactly.

The run also validly supplies behavioral trajectories, continuous task
coordinates, and critical-time valid-token geometry.

Do not use this recovered run alone to claim that the fully authenticated
task-visible precursor gate first passes at raw_write at t=1 or t=17.

The original frozen endpoint raw-write precursor result remains a separate
previously validated endpoint result; this recovery does not rewrite it.
