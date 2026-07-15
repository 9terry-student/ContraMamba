# Stage182-A data-contamination and clean-failure-set closure

**Decision:** `STAGE182A_DATA_CONTAMINATION_CONFIRMED_AND_CLEAN_MODEL_FAILURE_SET_READY`

Stage182-A exactly reconstructed all 3,600 controlled rows and rejected row-identity misalignment as an explanation. Among 39 unique hard items and their 39 matched controls, the deterministic audit found 56 clean items, 22 contaminated items, and no unresolved schema rows. The detected construction defects comprise one invalid do-support item and 21 non-polarity interventions that also changed polarity.

The 21 clean hard/control pairs divide into 14 native-frame model-failure candidates and seven clean hard rows for which the native frame head is correct. Every candidate has a valid grammar audit, exact intervention contract, valid canonical control, resolved schema, and a native-frame label/prediction mismatch. All 14 have a matched clean control.

Candidate composition is fixed as follows:

| Intervention family | Count |
|---|---:|
| `none` | 6 |
| `polarity_flip` | 6 |
| `paraphrase` | 1 |
| `location_swap` | 1 |

Thirteen candidates are Stage176 `harmful_regression` rows and one is a `beneficial_correction` row.

## Authorization

The authorized next stage is `STAGE182B_CLEAN_FRAME_MODEL_FAILURE_LOCALIZATION`, restricted to artifact-only, evaluation-only localization. The secondary authorized route is `STAGE182C_CONTROLLED_INTERVENTION_GENERATOR_REPAIR_SPEC`.

The 14-row set is a diagnostic queue, not causal proof and not a training set. This closure authorizes no annotation, relabeling, dataset edit, generator change, model import, checkpoint load, forward pass, learned probe, fitted classifier, threshold search, calibration, or training.
