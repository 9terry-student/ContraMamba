# ContraMamba Gen4 Native Mamba State Bridge
# NAME Direct Cross-Layer Paired-Difference Statistical Analysis Report - Candidate

## Provenance

- Scientific specification: `b3e0ade126622f244b557e1db07296c622bd7202`
- Implementation authority: `724c28b528b0f182bc0c79cf5ee0b3adfca76ec2`
- Statistical execution authority: `b5da30c051c268696fb65da0edfdbc6e2bef3a83`
- Implementation commit: `cdf9f2117bc0d5fa119fbf24b85266c1f9ced448`
- Script SHA256: `bccf4b91f8b566808315ce15fafa5c948a8b20bde751c75c8af0da9671c83fff`
- Phase F input: `reports/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis_e917e4c_v1/phase_f_pair_level_contrasts.csv`
- Phase F input SHA256: `abb7e837395138d095e285e40ec8782ea863061087b58844bec6fad5d4ef5e73`
- Q1/Q3 input: `reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_retry1_v1/name_q1_q3_pair_level_contrasts.csv`
- Q1/Q3 input SHA256: `c0c917560b5a37c5df82ad4e203a87441370c129622afc668b77712b98698f82`
- Source pairs: `300`

## Direct cross-layer confirmatory family

| Layer contrast | Endpoint | Estimand | Mean | t | raw p | Holm p | Reject | d_z |
|---|---|---|---:|---:|---:|---:|:---:|---:|
| 5_MINUS_11 | POST4_SPEED | DELTA_NAME | 0.0017060299714406332 | 0.27870200486562835 | 0.7806661771167187 | 1.0 | False | 0.01609086775328589 |
| 5_MINUS_11 | POST4_TURNING | DELTA_NAME | -0.003942878246307373 | -0.8994117830537655 | 0.36915712817547064 | 1.0 | False | -0.05192756350584127 |
| 5_MINUS_11 | POST4_PATH_EFFICIENCY | DELTA_NAME | 0.0006204978883891845 | 0.40786798392388723 | 0.6836624941794077 | 1.0 | False | 0.02354826903122862 |
| 17_MINUS_11 | POST4_SPEED | DELTA_NAME | -0.015996098121007284 | -1.6142604034890364 | 0.10752573207243876 | 0.5376286603621938 | False | -0.0931993678496549 |
| 17_MINUS_11 | POST4_TURNING | DELTA_NAME | 0.0006377017498016358 | 0.15435335460303642 | 0.8774352642521903 | 1.0 | False | 0.008911595083038482 |
| 17_MINUS_11 | POST4_PATH_EFFICIENCY | DELTA_NAME | -0.01586177605040945 | -5.798308334873094 | 1.7017630662606788e-08 | 1.0210578397564072e-07 | True | -0.3347654877983432 |

## Family decision

DIRECT_CROSS_LAYER_NAME_DIFFERENCE_SUPPORTED_FOR_AT_LEAST_ONE_PRESPECIFIED_LAYER_PAIR_ENDPOINT

## Inference boundary

The Holm correction controls only this frozen six-member adaptive
direct cross-layer family. Overall adaptive-program FWER across Phase F,
the Q1/Q3 follow-up, and this phase is not claimed.

A supported member establishes only the exact prespecified between-layer
difference for that layer pair and NAME kinematic endpoint.

The broad unqualified claim that NAME is depth-selective is prohibited.
No causal mediation, necessity, sufficiency, or state-to-output causation
claim follows from this analysis.
