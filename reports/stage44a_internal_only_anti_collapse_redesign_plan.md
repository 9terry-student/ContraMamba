# Stage44-A Internal-Only Anti-Collapse Redesign Plan

## Decision

`STAGE44A_INTERNAL_ONLY_ANTI_COLLAPSE_PLAN_READY`

Stage44-A defines a pre-registered, internal-only redesign plan for addressing NOT_ENTITLED collapse after the Stage43 external fact-verification audit. Stage43-B1 remains locked as external-evaluation-only evidence. Stage44 must not use Stage43-B1 labels, metrics, prediction distributions, examples, or per-dataset outcomes for optimization, checkpoint/model selection, calibration, threshold tuning, loss design, or composer behavior changes.

## Diagnosis Summary

Stage43-C0/C1/C2 established that the external fact-verification path now works as infrastructure: external evaluation runs after best-state restoration, the Stage43-B1 leakage boundary is preserved, and Stage43-C2 makes the Stage32/36/37/39 shadow/composer fields available for external rows. Stage43-C2 also resolved `missing_source_shadow_label`, and the safety counters were clean: no introduced unsafe SUPPORT, no REFUTE-to-SUPPORT, and no SUPPORT-to-REFUTE transitions.

The external result remains incomplete. VitaminC composed macro-F1 is around 0.095, Climate-FEVER composed macro-F1 is around 0.174, and predictions remain dominated by NOT_ENTITLED. This supports an infrastructure/composer-path claim, not an external generalization claim.

The next valid action is not tuning on Stage43-B1. The next valid action is an internal-only redesign to improve SUPPORT/REFUTE entitlement robustness using controlled train/dev data and internal diagnostics only.

## Allowed Inputs

- `data/controlled_v5_v3_without_time_swap.jsonl`
- Existing internal controlled dev split
- Existing internal synthetic/controlled diagnostics, only as internal diagnostics
- Existing Stage30/32/33/36/37/39/40/41/42 reports as historical internal reports
- Stage43 high-level conclusion only: external evaluation remains incomplete due to severe NOT_ENTITLED collapse; composer path is now available; do not tune on external data

## Disallowed Inputs

- Stage43-B1 external labels for any design choice
- Stage43-B1 per-example predictions
- Stage43-B1 threshold search
- Stage43-B1 calibration
- Stage43-B1 checkpoint/model selection
- Stage43-B1 examples as training templates
- Stage34/35 synthetic probes as naturalistic external validation

## Internal Hypotheses

- H1: NOT_ENTITLED over-prediction is caused by overly conservative final entitlement gating.
- H2: SUPPORT/REFUTE recall is under-protected during checkpoint selection.
- H3: Class-balanced CE or inverse-frequency loss alone may be insufficient because collapse happens through final entitlement gating, not only label imbalance.
- H4: `safe_structured_v2` changes too few rows because its source shadow labels inherit the base collapse.
- H5: Improving internal SUPPORT/REFUTE recall must not increase unsafe SUPPORT.

## Candidate Stage44-B Options

### Option A: Internal Selection Constraint

Add internal clean-dev selection constraints while preserving clean-dev macro-F1 as the primary metric:

- minimum SUPPORT recall
- minimum REFUTE recall
- maximum NOT_ENTITLED prediction rate
- no unacceptable clean-dev accuracy degradation

This option uses only internal clean-dev metrics and makes no use of Stage43-B1.

### Option B: Entitled-Class Anti-Collapse Auxiliary Loss

Add an internal auxiliary loss that protects SUPPORT/REFUTE entitlement separation using only labels and fields from controlled clean train/dev data. This may address collapse more directly than selection constraints, but it changes training behavior and therefore has a larger audit surface.

### Option C: Final Entitlement Gate Temperature/Bias Parameter

Learn or select a final entitlement gate temperature/bias parameter only on internal controlled train/dev. This is allowed only if every parameter, threshold, and selection decision is derived from internal controlled data, never Stage43-B1.

## Recommended Path

Recommend Option A first. It changes checkpoint selection only within internal clean-dev constraints, is easiest to audit for leakage, and directly tests whether SUPPORT/REFUTE recall is being under-protected during selection. If Option A fails to improve internal anti-collapse metrics without unsafe SUPPORT regressions, proceed to Option B as a separately pre-registered training change.

## Stage44-B Pre-Registered Selection Rule

Stage44-B should select checkpoints using internal clean-dev only:

- primary metric: clean-dev macro-F1
- constraints:
  - SUPPORT recall must be at least a pre-registered fixed floor or an internal-baseline-derived floor
  - REFUTE recall must be at least a pre-registered fixed floor or an internal-baseline-derived floor
  - NOT_ENTITLED prediction rate must be below a fixed maximum
  - clean-dev accuracy must not degrade beyond a small pre-registered tolerance
  - unsafe SUPPORT counters on internal diagnostics must not increase beyond a pre-registered tolerance

No floor, tolerance, cap, or model comparison may be derived from Stage43-B1. If historical internal baseline values are unavailable, Stage44-B must first compute them on internal dev only and write them to a report before any model comparison.

## Stage44-C External Re-Evaluation Rule

After Stage44-B selects a model/checkpoint using internal-only criteria, run Stage43-B1 exactly once as read-only external evaluation. Do not iterate using Stage43-B1 results. If Stage44-C improves external metrics, report that as a post-hoc external result, not as a training signal. If Stage44-C fails, freeze the result honestly.

## Leakage Guardrails

Stage43-B1 is locked. No thresholds, calibration, checkpoints, model changes, loss designs, or composer behavior changes may be selected using Stage43-B1.

Any future redesign after seeing Stage44-C external results must either acquire a new external holdout or clearly label Stage43-B1 as no longer pristine for subsequent claims.

## Allowed Claims

- Stage44-A defines an internal-only redesign plan.
- Stage43-B1 remains locked as external-evaluation-only.
- Stage44-B may use only internal controlled train/dev diagnostics.

## Disallowed Claims

- Do not claim external PASS.
- Do not claim naturalistic generalization.
- Do not claim Stage44 will fix VitaminC or Climate-FEVER.
- Do not claim Stage43-B1 can be used for tuning.

## Next Stage

Proceed to Stage44-B with Option A: implement an internal clean-dev constrained checkpoint selection audit. If internal baseline floors are not already available, compute and freeze them from internal dev only before comparing any candidate model or checkpoint.
