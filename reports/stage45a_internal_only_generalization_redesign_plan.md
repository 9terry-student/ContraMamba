# Stage45-A Internal-Only Generalization Redesign Plan

## Decision

`STAGE45A_INTERNAL_ONLY_GENERALIZATION_REDESIGN_PLAN_READY`

Stage45-A defines a pre-registered redesign plan for improving controlled-to-naturalistic generalization using only internal/controlled data and internal diagnostics. Stage43-B1 is locked as a diagnostic-only, already-observed external result and must not guide future optimization.

## Diagnosis Summary

Stage43 and Stage44 did not produce an external PASS. Stage43-B1 provided read-only external evaluation on VitaminC validation sample1000 and Climate-FEVER test sample1000. Stage43-C2 fixed the external composer path: Stage32/36/37/39 shadow fields became available for all external rows, `safe_structured_v2` became available for all external rows, and `missing_source_shadow_label` was resolved.

Stage44-B2 ruled out internal checkpoint selection collapse as the proximate cause. The selected checkpoint was epoch 47, which was also the original best metric epoch. Internal clean-dev macro-F1 was about 0.9619, internal NOT_ENTITLED gold rate was 0.75, internal NOT_ENTITLED prediction rate was about 0.7264, SUPPORT precision/recall was about 0.8318/0.9889, and REFUTE precision/recall was 1.0/1.0.

Stage44-C preserved the external-evaluation-only boundary and still found external predictions dominated by NOT_ENTITLED. No introduced unsafe SUPPORT, REFUTE-to-SUPPORT, or SUPPORT-to-REFUTE transitions were observed, but external decisions remained `INCOMPLETE`.

The remaining failure mode is not checkpoint selection, composer availability, label mapping, or truncation. The remaining failure mode is controlled-to-naturalistic generalization failure, especially SUPPORT/REFUTE entitlement collapse under external-style evidence. Stage43-B1 must not be used for further optimization.

## Leakage Boundary

Stage43-B1 disallowed uses:

- training
- synthetic data generation
- prompt/example mining
- threshold selection
- calibration
- checkpoint selection
- loss design
- data mixture design
- model selection
- composer rule changes
- repeated iterative evaluation for development

Stage43-B1 allowed uses:

- historical diagnostic summary only
- reporting the already-observed negative/incomplete result
- one clearly labeled diagnostic comparison only if absolutely necessary, never as final validation and never as a development signal

Stage43-B1 must now be treated as diagnostic-only and no longer pristine for final model-development claims. If future final external validation is needed, acquire a new held-out external set after the Stage45 redesign is frozen.

## Candidate Redesign Options

### Option A: Internal Controlled Naturalization Augmentation

Generate more naturalistic claim/evidence phrasing from internal controlled examples only. Split before augmentation to avoid train/dev leakage. Preserve labels from controlled transformations. Include paraphrase, sentence-length variation, distractor sentence insertion, entity aliasing, and evidence-style variation. Do not copy or imitate Stage43-B1 examples.

### Option B: Entitlement Balance Loss / Auxiliary Recall Protection

Add an internal-only auxiliary objective to protect SUPPORT/REFUTE entitlement against over-conservative NOT_ENTITLED gating. Use only controlled train labels and internal dev metrics. Treat this as a higher-risk intervention because loss changes can hide overfitting to internal dev if selected too aggressively.

### Option C: Internal Leave-Transformation-Family-Out Validation

Hold out transformation families internally and select checkpoints by robustness across held-out internal transformation types. This tests generalization pressure without touching Stage43-B1 and provides a better proxy than a single random controlled dev split.

### Option D: Evidence Sufficiency Contrast Curriculum

Construct internal contrast pairs where minimal evidence changes flip SUPPORT, REFUTE, and NOT_ENTITLED labels. Emphasize sufficiency boundary learning so the model learns when evidence is enough to support/refute rather than defaulting to NOT_ENTITLED. Use controlled data only.

### Option E: New External Holdout Acquisition Plan

For final claims only, acquire a new external validation set after the Stage45 design is frozen. The new holdout must be read-only and must not be used for iteration, thresholding, calibration, checkpoint selection, data generation, or model selection.

## Recommended Path

Primary recommendation: pursue Option C plus Option D first. Internal leave-family-out validation gives an internal generalization stress test, and evidence sufficiency contrast training directly targets SUPPORT/REFUTE entitlement collapse without using Stage43-B1.

Secondary recommendation: add Option A only if implemented with strict split-before-augmentation and audit fields proving no train/dev leakage. Avoid starting with Option B unless internal diagnostics show gating or loss imbalance is the main bottleneck, because loss changes can hide leakage-like overfitting to internal dev.

## Stage45-B Proposed Implementation

Stage45-B should implement an internal leave-family-out validation scaffold:

- no external files
- no Stage43-B1 access
- split controlled examples by transformation family
- evaluate robustness on held-out internal transformation families
- report transformation-family macro-F1
- report SUPPORT recall
- report REFUTE recall
- report NOT_ENTITLED prediction rate
- report unsafe SUPPORT proxy on internal controlled families

Stage45-B should not change training behavior unless the implementation is limited to diagnostics/scaffolding and selection remains internal-only.

## Stage45-C Proposed Implementation

Stage45-C should implement an evidence sufficiency contrast curriculum or auxiliary diagnostic using controlled internal data only. It should train only on internal controlled examples and select only by Stage45-B internal leave-family-out criteria.

The curriculum should focus on minimal evidence changes that cross SUPPORT/REFUTE/NOT_ENTITLED boundaries, preserving traceable labels and family metadata so improvements can be audited per transformation family.

## Stage45-D Future Evaluation Policy

Stage43-B1 should not be used as the final external claim set anymore if Stage45 redesign was informed by its failure. For a final external claim, acquire a new held-out external fact-verification set after the design is frozen.

If Stage43-B1 is reused, label it explicitly as `previously observed diagnostic set`, not clean validation. It may be reported as a historical diagnostic comparison, but not as evidence for a pristine external validation claim.

## Success Criteria

Internal-only success criteria:

- improved leave-family-out macro-F1
- no SUPPORT/REFUTE recall collapse
- NOT_ENTITLED prediction rate consistent with internal gold priors
- no increase in unsafe SUPPORT proxy
- stable performance across transformation families

External success criteria:

- not evaluated in Stage45-A/B/C
- final external success requires a new held-out dataset after design freeze

## Allowed Claims

- Stage45-A defines an internal-only generalization redesign plan.
- Stage43-B1 remains locked and cannot guide future optimization.
- Stage44 ruled out internal checkpoint selection as the proximate cause.
- Future redesign must be validated internally first.

## Disallowed Claims

- Do not claim external PASS.
- Do not claim VitaminC transfer success.
- Do not claim Climate-FEVER robustness.
- Do not claim naturalistic generalization.
- Do not claim Stage45 will solve the external failure.
- Do not use Stage43-B1 as a clean final validation set after redesign.

## Next Stage

`Stage45-B: internal leave-transformation-family-out validation scaffold`
