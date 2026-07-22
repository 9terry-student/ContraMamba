# Stage196-B2-B2 Row-Level Paired Treatment-Path Probe

## Executive decision

`STAGE196B2B2_SEED_SPECIFIC_MULTIPATH_EFFECT`

## Authorized interpretation

The observed treatment effect follows seed-specific or multiple row-level paths without a stable cross-seed path.

## Stage196-B2-B1 source closure

Exact eight-file closure, completed decision, empty blockers, 23 passed gates, and the exact 16 rows were required.

## Stage196-B2-A and P0 closure

Exact nine-file B2-A closure and exact six-run P0/20-sidecar/720-row closure were required.

## Provenance normalization

{"margin": "support_logit - not_entitled_logit", "roles": {"framegate_implementation_origin_git_commit": "5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8", "stage196b1_runtime_git_commit": "9835cbbf86d83aca0964821669e63f7f6deb1c59", "stage196b2a_analyzer_git_commit": "833752ec25890e76ee3c2dc3f3bc3c3c1b7428a6", "stage196b2b1_analyzer_git_commit": "85f1de8f9e0393ccdca5da4bc0725d88d8f427c9", "stage196b2p0_runtime_git_commit": "e9aaff24054f1d409119b70df13b94159a34a8e4", "support_vs_not_entitled_margin_source": "support_logit - not_entitled_logit"}, "warnings": []}

## Primary rows and seed roles

{"contrast": [183], "counts": {"184": {"harm": 6, "recovery": 5}, "185": {"harm": 3, "recovery": 2}}, "positive": [184, 185]}

## Paired trajectory construction

Each primary identity is aligned between joint and frame_local_only at every epoch; all deltas are intervention minus joint.

## Fixed thresholds and event definitions

Probability boundaries are 0.5; polarity and composition boundaries are zero. Event times use exact signs and persistent fixed-boundary disagreement.

## Tail-three directional effects

{"184": {"harm": {"COMPOSITION_DIVERGENCE_WITHOUT_LOCAL_BOUNDARY": 1, "FRAME_ENTITLEMENT_LOSS": 2, "POLARITY_OVERRIDE_DESPITE_FRAME_GAIN": 3}, "recovery": {"MULTI_CHANNEL_CONFLICT": 5}}, "185": {"harm": {"MULTI_CHANNEL_CONFLICT": 3}, "recovery": {"FRAME_ENTITLEMENT_GAIN": 2}}}

## First terminal-sign-stable epochs

{"by_seed_and_role": {"184_harm": {"median_first_persistent_boundary_divergence_epoch": {"composition": 12.0, "entitlement": 18.0, "frame": 16.5, "polarity": null, "predicate": 19.0, "sufficiency": null}, "median_first_terminal_sign_stable_epoch": {"entitlement": 14.5, "frame": 15.0, "margin": 16.5, "polarity": 7.0, "predicate": 14.0, "sufficiency": 4.0}}, "184_recovery": {"median_first_persistent_boundary_divergence_epoch": {"composition": 18.0, "entitlement": 17.0, "frame": 17.0, "polarity": null, "predicate": 19.0, "sufficiency": null}, "median_first_terminal_sign_stable_epoch": {"entitlement": 11.0, "frame": 11.0, "margin": 13.0, "polarity": 7.0, "predicate": 13.0, "sufficiency": 4.0}}, "185_harm": {"median_first_persistent_boundary_divergence_epoch": {"composition": null, "entitlement": null, "frame": 20.0, "polarity": null, "predicate": null, "sufficiency": null}, "median_first_terminal_sign_stable_epoch": {"entitlement": 14.0, "frame": 14.0, "margin": 14.0, "polarity": 16.0, "predicate": 20.0, "sufficiency": 13.0}}, "185_recovery": {"median_first_persistent_boundary_divergence_epoch": {"composition": 18.0, "entitlement": null, "frame": 19.5, "polarity": null, "predicate": 18.0, "sufficiency": null}, "median_first_terminal_sign_stable_epoch": {"entitlement": 17.5, "frame": 17.5, "margin": 17.5, "polarity": null, "predicate": 17.0, "sufficiency": 13.5}}}, "row_count": 16, "tail_epochs": [18, 19, 20]}

## Persistent boundary-divergence ordering

Individual local and composition boundary event times remain separate.

## Recovery path analysis

Recovery requires positive tail-three margin movement for margin-direction concordance; discordant rows remain included.

## Preservation-harm path analysis

Harm requires negative tail-three margin movement for margin-direction concordance, while prediction instability is reported separately.

## Frame-entitlement path rule

{"passed": false, "per_seed": {"184": {"harm_loss_rate": 0.3333333333333333, "recovery_gain_rate": 0.0}, "185": {"harm_loss_rate": 0.0, "recovery_gain_rate": 1.0}}, "timing_passed": true}

## Polarity-override harm rule

{"passed": false, "polarity_override_harm_count": 3, "recovery_override_count": 0, "timing_passed": true}

## Composition-without-local-precursor rule

{"passed": false, "per_seed_role_rates": {"184_harm": 0.16666666666666666, "184_recovery": 0.0, "185_harm": 0.0, "185_recovery": 0.0}}

## Intervention-type path audit

{"none_and_paraphrase_recovery_rows_share_a_path": false, "paraphrase_harm_rows_follow_polarity_override": true, "paths_stable_across_both_positive_seeds": ["MULTI_CHANNEL_CONFLICT"], "polarity_flip_remains_harm_only": true}

## Seed183 contrast

Seed183 is contrast-only and is excluded from every primary denominator and decision rule.

## Decision-rule evaluation

{"composition": {"passed": false, "per_seed_role_rates": {"184_harm": 0.16666666666666666, "184_recovery": 0.0, "185_harm": 0.0, "185_recovery": 0.0}}, "frame_entitlement": {"passed": false, "per_seed": {"184": {"harm_loss_rate": 0.3333333333333333, "recovery_gain_rate": 0.0}, "185": {"harm_loss_rate": 0.0, "recovery_gain_rate": 1.0}}, "timing_passed": true}, "polarity_override": {"passed": false, "polarity_override_harm_count": 3, "recovery_override_count": 0, "timing_passed": true}}

## Remaining uncertainty

This artifact-only frozen-Mamba probe is descriptive; event ordering does not establish formal mediation.

## Prohibited claims

- formal causal mediation
- mathematical necessity or sufficiency
- deployable routing
- a safe intervention
- unfrozen behavior
- external/OOD validity
- architectural superiority
- authorization of a new intervention, loss, router, or training regime

## Recommended next stage

`STAGE196B2B3_NO_PROMOTION_INFERENCE_ONLY_COMPONENT_SWAP_PROBE`
