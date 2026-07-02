# Stage54 — VitaminC External Collapse Postmortem

## Decision

`STAGE54_VITAMINC_EXTERNAL_COLLAPSE_POSTMORTEM_READY`

## Source

- Stage53A run dir: `results/stage53a_vitaminc_frozen_external_eval_20260702_045300`
- External output dir: `results/stage53a_vitaminc_frozen_external_eval_20260702_045300/external`

## Compact summary

| decision                                            | source_run_dir                                                 | source_external_dir                                                     | stage53a_status            | primary_failure_mode                             |   vitaminc_accuracy |   macro_f1_base |   macro_f1_composed |   unsafe_support_count |   recovery_fired_count |   blocker_fired_count | stage45c_recovery_implicated   | next_stage                                |
|:----------------------------------------------------|:---------------------------------------------------------------|:------------------------------------------------------------------------|:---------------------------|:-------------------------------------------------|--------------------:|----------------:|--------------------:|-----------------------:|-----------------------:|----------------------:|:-------------------------------|:------------------------------------------|
| STAGE54_VITAMINC_EXTERNAL_COLLAPSE_POSTMORTEM_READY | results/stage53a_vitaminc_frozen_external_eval_20260702_045300 | results/stage53a_vitaminc_frozen_external_eval_20260702_045300/external | failed_external_diagnostic | external_domain_not_entitled_prediction_collapse |               0.156 |        0.115986 |            0.116717 |                      1 |                      0 |                     0 | False                          | Stage55 external collapse mechanism audit |

## Distribution summary

| kind                |   SUPPORT |   REFUTE |   NOT_ENTITLED |
|:--------------------|----------:|---------:|---------------:|
| gold                |       500 |      355 |            145 |
| base_prediction     |        21 |       33 |            946 |
| composed_prediction |        23 |       33 |            944 |

## Postmortem conclusion

Stage53A is treated as a failed external diagnostic. The dominant observed failure mode is **external-domain NOT_ENTITLED prediction collapse**, not broad unsafe SUPPORT explosion.

The Stage47 frozen recovery configuration is not identified as the cause of the failure:
- recovery fired count: `0`
- blocker fired count: `0`
- unsafe SUPPORT count: `1`

## Allowed claim

Stage53A shows that the Stage47 frozen recovery configuration does not transfer to the VitaminC sample1000 external diagnostic setting. The dominant observed failure is NOT_ENTITLED overprediction/collapse, not broad unsafe SUPPORT explosion.

## Not allowed claims

- Do not claim VitaminC external robustness.
- Do not claim broad naturalistic fact-verification generalization.
- Do not claim Stage45C recovery caused the VitaminC failure.
- Do not tune composer thresholds on this external set and report it as held-out performance.

## Leakage policy

- External data used for training: `False`
- External data used for checkpoint selection: `False`
- External data used for threshold tuning: `False`
- External data used for postmortem only: `True`

## Next stage

Stage55 should inspect external collapse mechanisms: label mapping/template mismatch, input truncation/max_length, domain lexical shift, internal prior mismatch, and base-logit NE bias.
