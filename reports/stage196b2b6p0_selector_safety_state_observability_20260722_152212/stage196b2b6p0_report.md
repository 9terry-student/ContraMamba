# Stage196-B2-B6P0: Selector Safety-State Observability

## Executive decision

`STAGE196B2B6P0_CURRENT_SAFETY_OBSERVABILITY_INSUFFICIENT`

## Authorized interpretation

Within the frozen-Mamba, frozen-composer controlled population, the analyzer reports exact categorical separation and cross-seed transfer. Only an epoch-20 recipient-local gate satisfying every ordered rule is potentially eligible for the recommended counterfactual audit; no integration, training, or promotion is authorized.

## B2-B6 source result

The successful B2-B6 unsafe result, exact three nondominated candidates, primary closure, and both-seed nondiscovery failures are required.

## Source closure

{"b2b4_files":["stage196b2b4_analysis.json","stage196b2b4_report.md","stage196b2b4_primitive_coalition_rows.csv","stage196b2b4_primitive_mobius_terms.csv","stage196b2b4_primitive_tail_summary.csv","stage196b2b4_residual_coalition_rows.csv","stage196b2b4_residual_mobius_terms.csv","stage196b2b4_localization_summary.csv","stage196b2b4_contract.csv"],"b2b5_files":["stage196b2b5_analysis.json","stage196b2b5_report.md","stage196b2b5_feature_dictionary.csv","stage196b2b5_row_action_sets.csv","stage196b2b5_recipient_signature_rows.csv","stage196b2b5_recipient_selector_summary.csv","stage196b2b5_paired_delta_signature_rows.csv","stage196b2b5_paired_delta_selector_summary.csv","stage196b2b5_contract.csv"],"b2b6_files":["stage196b2b6_analysis.json","stage196b2b6_report.md","stage196b2b6_candidate_feature_subsets.csv","stage196b2b6_signature_action_map.csv","stage196b2b6_primary_policy_validation.csv","stage196b2b6_clean_dev_signature_audit.csv","stage196b2b6_clean_dev_application_summary.csv","stage196b2b6_policy_dominance.csv","stage196b2b6_contract.csv"],"b2b6_metric_reproduction":{"disagreements":[],"extra_stored_rows":[],"generated_rows":27,"stored_rows":27},"p0_counts":{"composer rows":86400,"composer sidecars":120,"trajectory rows":86400,"trajectory sidecars":120}}

## Population semantics

{"discovery_data_identity_key":["id","source_row_id","dev_position"],"discovery_identities":{"cross_seed_intersection":3,"seed184":11,"seed185":5,"union":13},"per_seed_clean_dev":{"all":720,"discovery_identity_union":13,"nondiscovery":707},"primary_case_key":["seed","stable_row_id"],"seed183":"contrast-only","seed_conditioned_primary_cases":{"seed184":11,"seed185":5,"total":16}}

## Nondominated selector candidates

[{"feature_subset_mask":"00100000000000","feature_subset_members":["HEAD_MARGIN_SIGN_SEQUENCE"]},{"feature_subset_mask":"01000000000000","feature_subset_members":["FINAL_MARGIN_SIGN_SEQUENCE"]},{"feature_subset_mask":"10000000000000","feature_subset_members":["RECIPIENT_PREDICTION_SEQUENCE"]}]

## Row-level selector reconstruction

Every assigned action is applied to the paired epoch-20 frame-local-only primitive state and all entitlement, head, residual, final-logit, and prediction quantities are recomputed.

## Safety-target definition

Outcome-derived labels are evaluation targets only and never enter a safety signature.

## MUST_ALLOW rows

{"00100000000000":7,"01000000000000":7,"10000000000000":7}

## MUST_BLOCK rows

{"00100000000000":57,"01000000000000":57,"10000000000000":57}

## OPTIONAL rows

{"00100000000000":2096,"01000000000000":2096,"10000000000000":2096}

## Single-checkpoint recipient feature dictionary

[{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_RECIPIENT_PREDICTION","formula":"exact epoch-20 recipient prediction","integration_authorized":true,"natural_threshold":"none","outcome_derived":false,"source_fields":["final_native_prediction"],"unavailable_reason":"","value_domain":["REFUTE","NOT_ENTITLED","SUPPORT"]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_FINAL_MARGIN_SIGN","formula":"sign(final_support_logit-final_not_entitled_logit)","integration_authorized":true,"natural_threshold":"zero tolerance 1e-12","outcome_derived":false,"source_fields":["final_support_logit","final_not_entitled_logit"],"unavailable_reason":"","value_domain":["NEGATIVE","ZERO","POSITIVE"]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_HEAD_MARGIN_SIGN","formula":"sign(decision_head_support_logit-decision_head_not_entitled_logit)","integration_authorized":true,"natural_threshold":"zero tolerance 1e-12","outcome_derived":false,"source_fields":["decision_head_support_logit","decision_head_not_entitled_logit"],"unavailable_reason":"","value_domain":["NEGATIVE","ZERO","POSITIVE"]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_HEAD_FINAL_MARGIN_SIGN_CONFLICT","formula":"HEAD_MARGIN_SIGN != FINAL_MARGIN_SIGN","integration_authorized":true,"natural_threshold":"sign zero tolerance 1e-12","outcome_derived":false,"source_fields":["decision_head_support_logit","decision_head_not_entitled_logit","final_support_logit","final_not_entitled_logit"],"unavailable_reason":"","value_domain":[false,true]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_FRAME_HALFSPACE","formula":"halfspace(frame_prob)","integration_authorized":true,"natural_threshold":"0.5; equality tolerance 1e-12","outcome_derived":false,"source_fields":["frame_prob"],"unavailable_reason":"","value_domain":["BELOW_HALF","AT_HALF","ABOVE_HALF"]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_PREDICATE_HALFSPACE","formula":"halfspace(predicate_coverage_prob)","integration_authorized":true,"natural_threshold":"0.5; equality tolerance 1e-12","outcome_derived":false,"source_fields":["predicate_coverage_prob"],"unavailable_reason":"","value_domain":["BELOW_HALF","AT_HALF","ABOVE_HALF"]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_SUFFICIENCY_HALFSPACE","formula":"halfspace(sufficiency_prob)","integration_authorized":true,"natural_threshold":"0.5; equality tolerance 1e-12","outcome_derived":false,"source_fields":["sufficiency_prob"],"unavailable_reason":"","value_domain":["BELOW_HALF","AT_HALF","ABOVE_HALF"]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_ENTITLEMENT_HALFSPACE","formula":"halfspace(entitlement_prob_native)","integration_authorized":true,"natural_threshold":"0.5; equality tolerance 1e-12","outcome_derived":false,"source_fields":["entitlement_prob_native"],"unavailable_reason":"","value_domain":["BELOW_HALF","AT_HALF","ABOVE_HALF"]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_ENTITLEMENT_BOTTLENECK","formula":"deterministic sorted argmin set; ties preserved","integration_authorized":true,"natural_threshold":"tie tolerance 1e-12","outcome_derived":false,"source_fields":["frame_prob","predicate_coverage_prob","sufficiency_prob"],"unavailable_reason":"","value_domain":["sorted nonempty subset of FRAME,PREDICATE,SUFFICIENCY"]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_POLARITY_ENERGY_ORDER","formula":"compare positive_energy and negative_energy","integration_authorized":true,"natural_threshold":"equality tolerance 1e-12","outcome_derived":false,"source_fields":["positive_energy","negative_energy"],"unavailable_reason":"","value_domain":["POSITIVE_DOMINANT","EQUAL","NEGATIVE_DOMINANT"]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_PREDICATE_MISMATCH","formula":"exact exported boolean","integration_authorized":true,"natural_threshold":"none","outcome_derived":false,"source_fields":["predicate_mismatch_active"],"unavailable_reason":"","value_domain":[false,true]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_TEMPORAL_MISMATCH","formula":"exact exported boolean","integration_authorized":true,"natural_threshold":"none","outcome_derived":false,"source_fields":["temporal_mismatch_active"],"unavailable_reason":"","value_domain":[false,true]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_TEMPORAL_ADAPTER_ACTIVITY","formula":"exact exported boolean","integration_authorized":true,"natural_threshold":"none","outcome_derived":false,"source_fields":["temporal_adapter_active"],"unavailable_reason":"","value_domain":[false,true]},{"available":true,"diagnostic_only":false,"epoch_scope":[20],"feature_family":"single_state","feature_name":"E20_TEMPORAL_CHANNEL_ACTIVITY","formula":"exact exported boolean","integration_authorized":true,"natural_threshold":"none","outcome_derived":false,"source_fields":["temporal_channel_active"],"unavailable_reason":"","value_domain":[false,true]}]

## Single-checkpoint exact safety-gate search

[]

## Cross-seed safety-gate transfer

[]

## Tail-trajectory diagnostic result

[]

## Paired-delta diagnostic result

[]

## Conservative gated-policy audit

No inclusion-minimal feasible safety gate was available for conservative application or seed183 contrast auditing.

The zero-row audits are vacuously complete, not missing.

Seed183 was not used to retrofit or discover a gate.

{"auditable_gate_count":0,"completed":true,"duplicate_gate_keys":[],"duplicate_summary_keys":[],"expected_summary_or_audit_rows":0,"extra_gate_keys":[],"missing_gate_keys":[],"observed_summary_or_audit_rows":0,"represented_gate_count":0,"vacuous_completion":true}

## Seed183 contrast

No inclusion-minimal feasible safety gate was available for conservative application or seed183 contrast auditing.

The zero-row audits are vacuously complete, not missing.

Seed183 was not used to retrofit or discover a gate.

{"auditable_gate_count":0,"completed":true,"contrast_only":true,"duplicate_gate_identity_rows":[],"duplicate_gate_keys":[],"expected_audit_rows":0,"expected_rows":0,"extra_gate_keys":[],"gate_summaries":[],"missing_gate_keys":[],"observed_audit_rows":0,"observed_rows":0,"represented_gate_count":0,"vacuous_completion":true,"wrong_row_counts":[]}

## Decision-rule evaluation

{"localized_single_state_gates":[],"ordered_rules":["contract_failure","cross_seed_single_state","single_state_in_sample_only","seed_specific_single_state","tail_trajectory_only","paired_delta_only","current_observability_insufficient"],"paired_delta_bidirectional_gate_exists":false,"pooled_single_state_gate_exists":false,"seed_specific_single_state":{"00100000000000":{"184":false,"185":false},"01000000000000":{"184":false,"185":false},"10000000000000":{"184":false,"185":false}},"shared_auditable_gate_count":0,"tail_trajectory_bidirectional_gate_exists":false}

## Remaining uncertainty

- The evidence is internal to the frozen controlled population.
- Seed183 is contrast-only and does not authorize selection or promotion.
- Set-valued selector actions are evaluated conservatively row by row.

## Prohibited claims

- formal causal mediation
- external or OOD validity
- unfrozen-Mamba validity
- training improvement
- promotion
- deployability from tail-checkpoint trajectories
- deployability from paired-treatment deltas
- safety from pooled in-sample separation alone
- safety from partial cross-seed coverage
- OPTIONAL rows as successful activation evidence
- unseen default blocking as positive selector evidence
- gold correctness as an inference-time feature
- seed or row identity as a safety feature
- an arbitrary threshold as a mechanistic gate

## Recommended next stage

`STAGE196B2B6P1_ADDITIONAL_SAFETY_STATE_OBSERVABILITY_DESIGN`

No training, integration, or promotion is authorized.
