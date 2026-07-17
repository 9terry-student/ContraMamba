# Stage188-B single-seed compatible-positive margin closure

**Decision:** `STAGE188B_SINGLE_SEED_MARGIN_MIXED_NO_REPLICATION_YET`

Stage188-B completed without blockers or a clear regression at seed 174. It is a single-seed internal diagnostic, not independent external validation.

## Clean-dev result

| Metric | Baseline | Intervention | Delta |
|---|---:|---:|---:|
| Accuracy | 0.880556 | 0.887500 | +0.006944 |
| Macro-F1 | 0.834878 | 0.843875 | +0.008997 |
| SUPPORT recall | 0.685393 | 0.707865 | +0.022472 |
| REFUTE recall | 1.000000 | 1.000000 | 0.000000 |
| NOT_ENTITLED recall | 0.892593 | 0.898148 | +0.005555 |
| False NOT_ENTITLED | 28 | 26 | -2 |
| False entitlement | 58 | 55 | -3 |
| Polarity errors | 0 | 0 | 0 |

## Mechanism signal

The intervention used 605 eligible rows and produced 12,100 eligible observations. Active rate decreased from 0.4628099174 in the first epoch to 0.1768595041 at the selected/final epoch. Mean eligible frame logit increased from 0.0383366482 to 1.4243969121, while raw margin loss decreased from 0.0975080065 to 0.0638179117.

The baseline sidecar non-access proof passed. Because baseline training was default-off, its training loop could not read the sidecar; therefore the baseline eligible-train reference is `not_evaluable_by_design`, not zero and not a clean-dev substitute.

## Prior-selected Stage182-B diagnostics

- Compatible FN: 13/13 positive deltas, median +0.372391, two corrections, no harm.
- Incompatible FP: error count 1 to 0, one correction, no harm.
- Matched controls: 14/14 positive deltas, median +0.265313.
- Clean-model failures: 14/14 positive deltas, median +0.372015, three corrections, no harm.

These cohorts were selected from prior internal evidence and are not independent validation.

## Why replication is required

The clean and mechanism directions are positive, but one seed cannot establish replication. The matched-control increase also leaves a non-selective or global frame-logit shift as a live explanation. Stage189 must run fresh paired baseline/intervention training at seeds 174, 175, and 176, persist the internally selected checkpoint, and then apply each checkpoint evaluation-only to the identical train integrity topology. Posthoc train-row results are mechanism diagnostics only and must never alter checkpoint selection.
