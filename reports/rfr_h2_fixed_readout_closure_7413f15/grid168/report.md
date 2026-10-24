# RFR-H2 fixed-grid diagnostic (EXPLORATORY ONLY)

Run: `rfr_h2_p6_grid168_mamba130m_cpu_6148207_20261025`; commit: `6148207713a59de49325bc7d56541ff7e31dc6d6`.

Unique contexts: 168; CASE joint: 0/8.

## Eight linked Boolean strata

| Stratum | PRE joint | POST joint | DELAYED joint | CASE joint |
|---|---|---|---|---|
| A0C0U0 | False | False | False | False |
| A0C0U1 | False | False | False | False |
| A0C1U0 | False | False | False | False |
| A0C1U1 | False | False | False | False |
| A1C0U0 | False | False | False | False |
| A1C0U1 | False | False | False | False |
| A1C1U0 | False | False | False | False |
| A1C1U1 | True | False | False | False |

## Conditional contrasts

| Condition | Correct | Total | All correct |
|---|---:|---:|---|
| A_REVISION | 24 | 48 | False |
| B_BLOCKED_C_FALSE | 0 | 24 | False |
| B_DEPENDENCY_C_TRUE | 12 | 24 | False |
| INVALID_RESISTANCE | 20 | 48 | False |
| POST_DELAYED | 60 | 144 | False |
| U_RETENTION | 28 | 56 | False |

## Fixed deterministic baselines

| Baseline | Oracle matches / 168 |
|---|---:|
| ALWAYS_NOT_ENTITLED | 0 / 168 |
| ALWAYS_REFUTE | 98 / 168 |
| ALWAYS_SUPPORT | 70 / 168 |
| AUTH_FLAG_ONLY | 82 / 168 |
| IGNORE_C | 154 / 168 |
| LATEST_ASSERTION | 144 / 168 |
| QUERY_ONLY | 98 / 168 |
| T0_ONLY | 144 / 168 |

## Error and limitation summary

Correct contexts: 70/168; failed contexts: 98.

No independent roots, confidence intervals or population claim.
No NOT_ENTITLED gold; source alias and authentication confounded.
No independent token-prior control or H3 state mediation.
Full condition/label-level errors, raw logits and candidate-only softmax are in predictions.jsonl and summary.json.
