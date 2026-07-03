# Stage91 - SUPPORT-Preserving Entitlement Recovery Design

## Decision

`STAGE91_SUPPORT_PRESERVING_RECOVERY_DESIGN_READY`

## Summary

| stage   | decision                                         | basis_stage89_decision                          | basis_stage90_decision                                        | primary_policy       | stage88c_policy                     | diagnosis                                                                                                   |   support_prediction_mass_stage73 |   support_prediction_mass_stage88c |   support_prediction_mass_delta |   refute_prediction_mass_stage73 |   refute_prediction_mass_stage88c |   refute_prediction_mass_delta |   support_losses_stage73_support_to_stage88c_non_support |   support_stage73_support_to_stage88c_refute |   support_stage73_support_to_stage88c_ne |   refute_gains_stage73_wrong_to_stage88c_refute | recommended_next_stage                                                         |
|:--------|:-------------------------------------------------|:------------------------------------------------|:--------------------------------------------------------------|:---------------------|:------------------------------------|:------------------------------------------------------------------------------------------------------------|----------------------------------:|-----------------------------------:|--------------------------------:|---------------------------------:|----------------------------------:|-------------------------------:|---------------------------------------------------------:|---------------------------------------------:|-----------------------------------------:|------------------------------------------------:|:-------------------------------------------------------------------------------|
| Stage91 | STAGE91_SUPPORT_PRESERVING_RECOVERY_DESIGN_READY | STAGE89_REJECT_STAGE88C_AS_PRIMARY_KEEP_STAGE71 | STAGE90_SUPPORT_SUPPRESSION_REFUTE_OVERCORRECTION_AUDIT_READY | KEEP_STAGE71_PRIMARY | USEFUL_PARTIAL_RECOVERY_NOT_PRIMARY | Stage88C recovered REFUTE and improved macro-F1, but suppressed SUPPORT prediction mass and SUPPORT recall. |                               393 |                                294 |                             -99 |                              219 |                               283 |                             64 |                                                       79 |                                           45 |                                       34 |                                              45 | Stage92A support-preserving counter-bridge generation + encoder preflight only |

## SUPPORT / REFUTE delta

| metric                        |   value |
|:------------------------------|--------:|
| support_prediction_mass_delta |     -99 |
| refute_prediction_mass_delta  |      64 |
| ne_prediction_mass_delta      |      35 |
| support_false_ne_delta        |      19 |
| support_false_refute_delta    |      35 |
| refute_false_support_delta    |     -37 |

## Design constraints

| constraint                             | rule                                                                                                                               | status    |
|:---------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------|:----------|
| keep_stage71_primary                   | Stage71 retry2 remains the primary checkpoint until a candidate beats it on clean-dev preservation and external accuracy/false_NE. | MANDATORY |
| no_external_train_leakage              | Stage90 example files are diagnostic only. Do not use VitaminC external claim/evidence rows as training bridge data.               | MANDATORY |
| no_generic_balanced_entitlement_repeat | Do not repeat Stage88A-style generic SUPPORT/REFUTE-balanced entitlement bridge; it shifted mass away from SUPPORT.                | MANDATORY |
| support_preservation_first             | Next bridge must primarily protect SUPPORT recall against REFUTE/NE overcorrection.                                                | MANDATORY |
| retain_refute_gain_guarded             | REFUTE recovery is useful, but must be retained through narrow contrastive guardrails rather than extra REFUTE-heavy pressure.     | MANDATORY |
| false_entitlement_guardrail            | Track false_SUPPORT_on_NE and false_REFUTE_on_NE; do not recover SUPPORT by simply over-entitling NE rows.                         | MANDATORY |

## Candidate designs

| option                                      | description                                                                                                                                              | proposed_distribution                   | intended_effect                                                                                    | risk                                                                                          |   priority |
|:--------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------|:---------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|-----------:|
| Stage92A_support_preserving_counter_bridge  | Small synthetic/internal bridge emphasizing SUPPORT cases that look superficially refutable but are actually entailed, with paired REFUTE/NE guardrails. | SUPPORT 120, REFUTE 60, NOT_ENTITLED 60 | Restore SUPPORT prediction mass and SUPPORT recall while avoiding a large REFUTE mass increase.    | May increase false_SUPPORT_on_REFUTE or false_SUPPORT_on_NE if SUPPORT pressure is too broad. |          1 |
| Stage92B_support_refute_minimal_pair_bridge | Generate minimal SUPPORT/REFUTE pairs differing only in the decisive predicate/value/date, with SUPPORT count greater than REFUTE count.                 | SUPPORT 100, REFUTE 50, NOT_ENTITLED 50 | Teach polarity distinction without allowing REFUTE to dominate SUPPORT.                            | Could become template-shallow and reduce external generalization.                             |          2 |
| Stage92C_posthoc_threshold_diagnostic       | No retraining. Analyze whether Stage88C can be adjusted by SUPPORT-vs-REFUTE logit margin thresholds on clean-dev only.                                  | No new rows                             | Determine whether SUPPORT suppression is decision-boundary-level rather than representation-level. | External tuning would be leakage; threshold must be learned only from clean-dev.              |          3 |

## Stage92A requirements

| requirement                  | detail                                                                                                                                                                                   |
|:-----------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| synthetic_internal_only      | Rows must be newly generated synthetic/internal examples, not copied or paraphrased from VitaminC external predictions.                                                                  |
| support_preserving_templates | Include SUPPORT rows where evidence explicitly confirms the claim despite lexical distractors, nearby alternative entities, nearby dates, or nearby numeric values.                      |
| paired_refute_guardrails     | For each support-preserving family, include fewer REFUTE rows that make the contradiction explicit, not ambiguous.                                                                       |
| matched_ne_guardrails        | Include NE rows only as matched insufficiency guards, not as additional NE-safety expansion.                                                                                             |
| no_stage83a                  | Stage92 clean-dev candidate should use Stage57 + Stage66 + Stage92A, not Stage83A and not Stage88A unless explicitly testing a combined ablation.                                        |
| preflight_before_training    | Before training, require row counts, label counts, exact duplicate check, exact overlap check against train bridges and external sample, encode_label_tensors, and encode_mamba_records. |

## Promotion criteria

| criterion                   | threshold                                                                                                     |
|:----------------------------|:--------------------------------------------------------------------------------------------------------------|
| clean_dev_preservation      | candidate clean-dev macro_f1 >= Stage71 macro_f1 - 0.003                                                      |
| external_accuracy           | candidate VitaminC external accuracy >= Stage73/Stage71 external accuracy 0.353                               |
| external_macro              | candidate VitaminC external macro_f1_all3 >= Stage73/Stage71 external macro 0.3261787                         |
| false_ne_total              | candidate false_NE_total <= Stage73 false_NE_total 323                                                        |
| support_recall              | candidate SUPPORT recall should not fall below Stage73 SUPPORT recall 0.432 by more than 0.02                 |
| refute_recall               | candidate should retain meaningful REFUTE recall gain over Stage73 if possible, but not at SUPPORT's expense. |
| false_entitlement_guardrail | candidate false_entitlement_total <= Stage73 false_entitlement_total + 10                                     |

## Risk register

| risk                         | symptom                                                                               | mitigation                                                                                         |
|:-----------------------------|:--------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|
| support_overcorrection       | SUPPORT prediction mass exceeds Stage73 sharply and false_SUPPORT_on_REFUTE/NE rises. | Keep REFUTE and NE guardrails paired and explicit.                                                 |
| refute_overcorrection_repeat | REFUTE prediction mass rises again while SUPPORT recall falls.                        | Limit REFUTE count and avoid generic REFUTE-heavy templates.                                       |
| ne_suppression_repeat        | false_NE_on_SUPPORT remains above Stage73 despite new bridge.                         | Use support-preserving evidence patterns with explicit predicate coverage and sufficient evidence. |
| external_leakage             | Bridge rows reuse external VitaminC claim/evidence content.                           | Exact-overlap check and source policy: synthetic_internal_only.                                    |
