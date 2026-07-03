# Stage87B - External Entitlement Recovery Design Report

## Decision

`STAGE87B_EXTERNAL_ENTITLEMENT_RECOVERY_DESIGN_READY`

## Summary

| stage    | decision                                            | basis_stage85_decision                       | basis_stage86_next_direction                              | basis_stage87a_next_direction                                   | stage71_primary_policy   | stage83c_policy   | main_failure_mode                                                                                      |   stage73_external_acc |   stage73_external_macro_f1_all3 |   stage84f_external_acc |   stage84f_external_macro_f1_all3 |   stage84f_minus_stage73_false_NE_total |   stage84f_minus_stage73_false_entitlement_total |   stage84f_minus_stage73_polarity_error_total |   stage73_correct_stage84f_wrong_total |   stage73_wrong_stage84f_correct_total |   stage73_correct_stage84f_false_ne_total |   stage73_entitled_stage84f_ne_total | recommended_next_stage                                                      |
|:---------|:----------------------------------------------------|:---------------------------------------------|:----------------------------------------------------------|:----------------------------------------------------------------|:-------------------------|:------------------|:-------------------------------------------------------------------------------------------------------|-----------------------:|---------------------------------:|------------------------:|----------------------------------:|----------------------------------------:|-------------------------------------------------:|----------------------------------------------:|---------------------------------------:|---------------------------------------:|------------------------------------------:|-------------------------------------:|:----------------------------------------------------------------------------|
| Stage87B | STAGE87B_EXTERNAL_ENTITLEMENT_RECOVERY_DESIGN_READY | STAGE85_REJECT_STAGE83C_KEEP_STAGE71_PRIMARY | external_entitlement_recovery_without_NE_safety_expansion | Stage87B_design_external_entitlement_recovery_without_NE_safety | KEEP_STAGE71_PRIMARY     | REJECT_AS_PRIMARY | Stage83C/Stage84F increased false_NE external errors and suppressed SUPPORT/REFUTE entitlement recall. |                  0.353 |                         0.326179 |                   0.326 |                          0.306672 |                                      44 |                                               -7 |                                           -10 |                                     83 |                                     56 |                                        48 |                                   89 | Stage88A non-leaking balanced entitlement recovery bridge design/generation |

## Error delta

| error_type              |   stage73 |   stage84f |   delta_stage84f_minus_stage73 |
|:------------------------|----------:|-----------:|-------------------------------:|
| false_NE_on_REFUTE      |       145 |        162 |                             17 |
| false_NE_on_SUPPORT     |       178 |        205 |                             27 |
| false_NE_total          |       323 |        367 |                             44 |
| false_REFUTE_on_NE      |        41 |         31 |                            -10 |
| false_REFUTE_on_SUPPORT |       106 |        106 |                              0 |
| false_SUPPORT_on_NE     |        39 |         42 |                              3 |
| false_SUPPORT_on_REFUTE |       138 |        128 |                            -10 |
| false_entitlement_total |        80 |         73 |                             -7 |
| polarity_error_total    |       244 |        234 |                            -10 |

## Design constraints

| constraint                           | rule                                                                                                                            | status    |
|:-------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|:----------|
| external_no_train_leakage            | Do not use VitaminC external claim/evidence examples from Stage73/84F/87A as training bridge rows.                              | MANDATORY |
| keep_stage71_primary                 | Stage71 retry2 remains the primary checkpoint until a candidate beats it on clean-dev preservation and external diagnostic.     | MANDATORY |
| no_more_ne_safety_expansion          | Do not add additional NE-safety-only bridge rows by default; Stage83A increased false_NE external errors.                       | MANDATORY |
| target_entitlement_recall            | Next bridge must target SUPPORT/REFUTE recall recovery under sufficient evidence, not generic NOT_ENTITLED conservatism.        | MANDATORY |
| preserve_false_entitlement_guardrail | Any recovery bridge must track false_SUPPORT_on_NE and false_REFUTE_on_NE so entitlement recovery does not simply over-entitle. | MANDATORY |

## Candidate options

| option                                        | description                                                                                                                          | expected_effect                                                                                   | risk                                                                                                             |   priority |
|:----------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|-----------:|
| Stage88A_balanced_entitlement_recovery_bridge | Create a small non-leaking synthetic/internal bridge emphasizing SUPPORT and REFUTE entitlement recovery with matched NE guardrails. | Reduce false_NE_on_SUPPORT and false_NE_on_REFUTE without expanding NE prediction mass.           | May increase false_SUPPORT_on_NE or false_REFUTE_on_NE if recovery pressure is too strong.                       |          1 |
| Stage88B_threshold_or_logit_calibration_only  | Evaluate post-hoc final-head threshold/logit calibration on existing Stage71/Stage84F predictions without retraining.                | Diagnose whether NE over-selection is a decision-boundary issue rather than representation issue. | May not generalize if calibration relies on external labels; must remain diagnostic-only unless using clean-dev. |          2 |
| Stage88C_polarity_balanced_recovery           | Add paired SUPPORT/REFUTE templates where evidence is sufficient and polarity is explicit, with no extra NE-only rows.               | Recover REFUTE recall and SUPPORT recall while not increasing polarity flips.                     | Can worsen false_SUPPORT_on_REFUTE / false_REFUTE_on_SUPPORT if templates are too shallow.                       |          3 |

## Promotion criteria

| criterion                   | threshold                                                                       |
|:----------------------------|:--------------------------------------------------------------------------------|
| clean_dev_preservation      | candidate clean-dev macro_f1 >= Stage71 macro_f1 - 0.003                        |
| external_macro              | candidate VitaminC external macro_f1_all3 >= Stage73 macro_f1_all3              |
| external_accuracy           | candidate VitaminC external accuracy >= Stage73 accuracy                        |
| false_ne_reduction          | false_NE_total < Stage73 false_NE_total or at minimum < Stage84F false_NE_total |
| false_entitlement_guardrail | false_entitlement_total <= Stage73 false_entitlement_total + small tolerance    |
