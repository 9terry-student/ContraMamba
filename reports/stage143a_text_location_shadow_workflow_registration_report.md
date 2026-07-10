# Stage143-A Text Location Shadow Workflow Registration Report

## 1. Summary decision

Decision: `STAGE143A_TEXT_LOCATION_SHADOW_WORKFLOW_REGISTERED_OPTIONAL_DIAGNOSTIC`

The Stage142 analyzer is now an official optional shadow diagnostic. It should be run after major prediction exports when false SUPPORT/location mismatch risk is relevant.

The workflow does not mutate source predictions. It must not be used for model selection or threshold tuning, and it must not be routed into final logits or final predictions yet.

## 2. Registered workflow

- Name: Stage142 Text Location Guard Shadow Workflow
- Script: `scripts/analyze_stage142_text_location_guard_shadow.py`
- Documentation: `docs/stage142_text_location_guard_shadow_workflow.md`
- Status: `official_optional_shadow_diagnostic`
- Final integration status: `blocked`

## 3. Evidence basis

Stage142-B reproduced the broader Stage141-A stress result:

- Stage142-B decision: `STAGE142_TEXT_LOCATION_GUARD_SHADOW_CANDIDATE_ROBUST`
- Files: 7
- Rows: 33,000
- Changed rows: 53
- Delta false SUPPORT: -53
- Delta false NOT_ENTITLED: +3
- Delta macro F1: +0.0017308314825622562
- Feature correct SUPPORT false positives: 0
- Feature precision for false SUPPORT among SUPPORT predictions: 1.0

This evidence supports optional shadow diagnostic registration only. It does not support automatic final prediction override or final-logit integration.

## 4. Policy input safety

Policy inputs are limited to claim text, evidence text, and the original prediction label.

The policy does not use `intervention_type`, `slot_mismatch_target`, gold labels, diagnostic family fields, file path heuristics, or row id heuristics. Gold/intervention/family metadata may be used only for post-hoc evaluation and audit when present.

## 5. Usage policy

Run this workflow:

- after major prediction exports that include claim/evidence/prediction fields
- after external fact-verification diagnostic exports
- for false SUPPORT audits
- when location-like entity mismatch risk is suspected

The workflow is optional and shadow-only. It is not required for model selection and is not allowed to modify source predictions.

## 6. Safety boundaries

The workflow is diagnostic-only and audit-oriented.

It does not modify final logits, final predictions, training, checkpoint selection, Stage128 guard behavior, or thresholds used for model selection. It does not use Stage15 and does not use external data for training.

## 7. Known limitations

- Regex-based location extraction can miss lower-case, non-English, ambiguous, or complex locations.
- The heuristic may confuse location, organization, and person/title spans in harder text.
- External evidence remains limited.
- Stage63 showed a small false-NE tradeoff of +3.
- The workflow is not evidence for final-logit integration.

## 8. Stage144 recommendation

Stage144 should focus on negative controls and adversarial text-location cases before any final integration discussion.

Recommended actions:

- Generate or collect same-location SUPPORT controls with varied surface forms.
- Test multi-location evidence where one location matches and another distractor is present.
- Test organization/person names that look location-like.
- Keep all analysis shadow-only.

Avoid automatic final prediction override, training loss integration, checkpoint selection based on Stage142 outputs, and using `intervention_type` or gold labels as policy inputs.
