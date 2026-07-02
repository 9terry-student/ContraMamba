# Stage44-D Internal Selection External Re-Evaluation Synthesis

## Decision

`STAGE44D_INTERNAL_SELECTION_NOT_CAUSE_EXTERNAL_COLLAPSE`

Stage44-D synthesizes the Stage44-B2 internal prior-aware checkpoint selection result with the Stage44-C one-shot read-only external re-evaluation. The conclusion is negative but useful: internal checkpoint selection collapse is not the proximate cause of the external NOT_ENTITLED collapse.

## Stage44-B2 Internal Selection Result

Stage44-B2 decision: `STAGE44B2_PRIOR_AWARE_SELECTION_READY`

| Field | Value |
|---|---:|
| Selected epoch | 47 |
| Original best metric epoch | 47 |
| Internal clean-dev accuracy | 0.973611 |
| Internal clean-dev macro-F1 | 0.961893 |
| NOT_ENTITLED prediction rate | 0.726389 |
| NE pred minus gold NE rate | -0.023611 |
| SUPPORT precision | 0.831776 |
| SUPPORT recall | 0.988889 |
| REFUTE precision | 1.0 |
| REFUTE recall | 1.0 |

Internal clean-dev gold label rates:

- NOT_ENTITLED: 0.75
- REFUTE: 0.125
- SUPPORT: 0.125

The original best checkpoint already satisfies the prior-aware anti-collapse constraints. Stage44-B's earlier fixed NOT_ENTITLED cap was too aggressive because the internal clean-dev gold NOT_ENTITLED prior was 0.75; Stage44-B2 correctly compares predicted NOT_ENTITLED rate against the internal gold prior plus delta. Internal clean-dev checkpoint selection is therefore not the source of the external NOT_ENTITLED collapse.

## Stage44-C External Read-Only Result

| Dataset | Rows | Base prediction counts | Composed prediction counts | Base macro-F1 | Composed macro-F1 | Delta macro-F1 | Changed rows | Decision |
|---|---:|---|---|---:|---:|---:|---:|---|
| VitaminC validation sample1000 | 1000 | NOT_ENTITLED 980; REFUTE 14; SUPPORT 6 | NOT_ENTITLED 979; REFUTE 14; SUPPORT 7 | 0.093653 | 0.095031 | +0.001378 | 1 | `STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE` |
| Climate-FEVER test sample1000 | 903 | NOT_ENTITLED 900; REFUTE 3 | NOT_ENTITLED 898; REFUTE 5 | 0.165973 | 0.174139 | +0.008166 | 2 | `STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE` |

The one-shot external re-evaluation remains collapsed toward NOT_ENTITLED. The composed path produces only tiny macro-F1 changes and both external decisions remain incomplete.

## Composer Path Result

`safe_structured_v2` composer output was available for all external rows:

- VitaminC: 1000/1000
- Climate-FEVER: 903/903

Stage43-C2 shadow export remained available. The composer path changed only a few rows: 1 VitaminC row and 2 Climate-FEVER rows. The composer path is operational, but it does not solve the base external collapse.

## Safety Counters

- introduced_unsafe_SUPPORT_count: 0
- introduced_REFUTE_to_SUPPORT_count: 0
- introduced_SUPPORT_to_REFUTE_count: 0

No newly introduced unsafe SUPPORT or REFUTE/SUPPORT flips were observed. This is not an external PASS because performance remains collapsed and both external decisions remain `INCOMPLETE`.

## Leakage Summary

- Stage43-B1 was not used for training.
- Stage43-B1 was not used for checkpoint selection.
- Stage43-B1 was not used for threshold selection.
- Stage43-B1 was evaluated only after post-training best-state restoration.
- Stage44-B2 selection used only internal clean-dev statistics.

## Main Conclusion

External collapse is not caused by internal checkpoint selection collapse. External collapse is not caused by missing composer fields. External collapse is not explained by the label mapping or truncation diagnostics from Stage43-C1.

The remaining explanation is distribution/generalization failure from controlled training to naturalistic fact-verification evidence. Stage44-C does not support an external validation PASS.

## Allowed Claims

- Stage44-B2 prior-aware internal selection works.
- The selected internal checkpoint satisfies internal anti-collapse constraints.
- Stage44-C preserved external-evaluation-only leakage boundaries.
- `safe_structured_v2` was available on all external rows.
- No introduced unsafe SUPPORT / REFUTE-to-SUPPORT / SUPPORT-to-REFUTE transitions were observed.
- Internal checkpoint selection is not the proximate cause of external NOT_ENTITLED collapse.

## Disallowed Claims

- Do not claim external validation PASS.
- Do not claim VitaminC transfer success.
- Do not claim Climate-FEVER robustness.
- Do not claim naturalistic fact-verification generalization.
- Do not claim Stage44 improves external performance.
- Do not use Stage43-B1 for tuning, calibration, thresholds, or future model selection.

## Recommendation

Freeze Stage43/44 as honest negative external validation. The next stage should be Stage45-A: internal-only generalization redesign plan.

Stage45 must not tune on Stage43-B1. Any future model redesign should be pre-registered using internal/controlled data only. If the model is redesigned after observing Stage43-B1 results, Stage43-B1 should be treated as no longer pristine for final external validation unless a new held-out external dataset is acquired.

## Next Stage

`Stage45-A: internal-only generalization redesign plan`
