# Stage141-B Text Location Disjoint Stress Synthesis

## 1. Summary decision

Decision: `STAGE141B_TEXT_LOC_DISJOINT_STRESS_ROBUST_SHADOW_CANDIDATE`.

Stage141-A supports promoting `text_loc_disjoint` from ad hoc analysis into reusable shadow tooling. Across 7 evaluated prediction files and 33,000 rows, `text_loc_disjoint` reduced false SUPPORT by 53 and introduced only +3 false NE in aggregate.

The next action is a reusable shadow-only analyzer, not model integration.

## 2. Policy definition and input safety

Policy name: `text_loc_disjoint`.

Policy behavior: for SUPPORT predictions only, extract location-like spans from claim and evidence text. If both sides contain non-empty disjoint location sets, shadow the SUPPORT prediction to NOT_ENTITLED.

Policy inputs:

- Claim text.
- Evidence text.
- Existing prediction label.

Policy non-inputs:

- `intervention_type`.
- `slot_mismatch_target`.
- Gold labels.
- Diagnostic family metadata.

Gold labels and intervention metadata may be used only for post-hoc evaluation or audit. They are not policy inputs.

## 3. Aggregate Stage141-A result

Stage141-A decision: `STAGE141A_TEXT_LOC_DISJOINT_STRESS_CANDIDATE_ROBUST`.

Aggregate result:

- Files evaluated: 7.
- Total rows: 33,000.
- Changed predictions: 53.
- False SUPPORT: 229 before, 176 after.
- Delta false SUPPORT: -53.
- False NE: 2,879 before, 2,882 after.
- Delta false NE: +3.
- Feature false-SUPPORT true positives: 53.
- Feature correct-SUPPORT false positives: 0.
- Minimum file-level macro-F1 delta: 0.0.
- Maximum file-level false-NE increase: +3.

## 4. Per-file behavior

Stage123-B diagnostic variants:

- Evaluated files: 5.
- Total delta false SUPPORT: -49.
- Total delta false NE: 0.
- All macro-F1 changes were non-decreasing.
- In Stage123-B diagnostic variants, false NE increase was 0.

This is strong controlled diagnostic robustness across evidence interfaces.

External VitaminC files:

- Evaluated files: 2.
- Stage53 external: no effect.
- Stage63 external: false SUPPORT -4, false NE +3.
- Stage63 macro-F1 change: slightly positive.

External evidence is still limited. Stage63 shows a small benefit with a small false-NE cost, so the signal needs more external stress testing before any final integration.

## 5. Audit observations

In Stage123-B diagnostic variants, changed rows audited as `intervention_type=location_swap`.

Example location contrasts included:

- Beacon Harbor vs Elm Valley.
- Falcon Ridge vs Aurora City.
- Dublin vs Cork.

This audit metadata was not used by the policy. It was used only to interpret behavior after the shadow changes were computed from text and predictions.

## 6. Why this is robust enough for shadow tooling

The Stage141-A stress test reduces the concern that `text_loc_disjoint` is only a Stage140 artifact. It generalized across multiple existing prediction exports, reduced false SUPPORT consistently in the controlled diagnostic files, and did not create correct-SUPPORT false positives in feature accounting.

The policy is also deployable as a shadow analyzer because it uses only claim text, evidence text, and the existing prediction label. It does not depend on diagnostic-only metadata or gold labels.

## 7. Why this is not ready for final integration

This result does not justify final-logit or final-prediction integration.

Main reasons:

- External evidence is still limited.
- Stage63 produced a small label-level false-NE increase of +3.
- The location extractor remains a simple regex heuristic.
- Current evidence supports observability and audit, not model behavior changes.

No final logits, final predictions, training behavior, checkpoint selection, Stage128 guard behavior, or Stage15 path should be modified based on Stage141-B.

## 8. Stage142 recommendation

Stage142 should implement a reusable shadow-only `text_loc_disjoint` analyzer for arbitrary prediction JSONLs.

Requirements:

- No model behavior changes.
- No final-logit integration.
- No final-prediction integration.
- Policy must use only claim/evidence text and existing prediction labels.
- Gold, intervention, and family fields may be used only for evaluation or audit when present.
- Export per-file metrics, aggregate metrics, group audit metrics, and changed examples.

Avoid:

- Turning the heuristic into a training loss.
- Using `intervention_type` as a rule.
- Using `slot_mismatch_target` as a rule.
- Claiming external generalization from the current limited external sample.
