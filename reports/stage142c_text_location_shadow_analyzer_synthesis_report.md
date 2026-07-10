# Stage142-C Text Location Shadow Analyzer Synthesis

## 1. Summary decision

Decision: `STAGE142C_TEXT_LOCATION_SHADOW_ANALYZER_REPRODUCED_ROBUST`.

Stage142-B confirms that the reusable Stage142-A script reproduced the Stage141-A broader stress result. Across 7 files and 33,000 valid rows, the analyzer reduced false SUPPORT by 53 while aggregate false NE increased by 3.

The next step is official diagnostic workflow registration, not final model integration.

## 2. What Stage142-A implemented

Stage142-A implemented `scripts/analyze_stage142_text_location_guard_shadow.py` as a reusable shadow-only analyzer for the `text_loc_disjoint` policy.

Policy behavior: for SUPPORT predictions only, extract location-like spans from claim and evidence text. If both sides contain non-empty disjoint location sets, shadow SUPPORT to NOT_ENTITLED.

The analyzer is shadow-only and diagnostic-only. It does not mutate source predictions, model logits, training behavior, checkpoint selection, export behavior, or Stage128 behavior.

## 3. Stage142-B reproduction result

Stage142-B output directory: `reports/stage142b_text_location_guard_shadow_20260710_034736`.

Stage142-B decision: `STAGE142_TEXT_LOCATION_GUARD_SHADOW_CANDIDATE_ROBUST`.

Aggregate result:

- Files evaluated: 7.
- Valid rows: 33,000.
- Changed predictions: 53.
- SUPPORT to NOT_ENTITLED shadows: 53.
- False SUPPORT: 229 before, 176 after.
- Delta false SUPPORT: -53.
- False NE: 2,879 before, 2,882 after.
- Delta false NE: +3.
- Macro F1: 0.7075607576618963 before, 0.7092915891444586 after.
- Delta macro F1: +0.0017308314825622562.
- Minimum per-file macro-F1 delta: 0.0.
- Feature false-SUPPORT true positives: 53.
- Feature correct-SUPPORT false positives: 0.
- Feature false-SUPPORT false negatives: 176.
- Feature precision for false SUPPORT among SUPPORT predictions: 1.0.
- Feature recall for false SUPPORT among SUPPORT predictions: 0.2314410480349345.

Location extraction coverage:

- Claim locations non-empty: 31,286 rows.
- Evidence locations non-empty: 23,462 rows.
- Both sides non-empty: 23,206 rows.
- Location sets disjoint: 3,062 rows.
- Only 53 SUPPORT rows triggered the policy because the policy applies only to SUPPORT predictions.

## 4. Per-file behavior

Stage123-B diagnostic variants:

- Evaluated files: 5.
- Total delta false SUPPORT: -49.
- Total delta false NE: 0.
- All macro-F1 changes were non-decreasing.
- Stage123-B diagnostic variants had zero false-NE increase.

Individual diagnostic behavior:

- `context_prefix_core`: false SUPPORT 14 to 7, false NE 357 to 357.
- `core_first_context_suffix`: false SUPPORT 14 to 7, false NE 378 to 378.
- `core_marker_context_suffix`: false SUPPORT 19 to 13, false NE 517 to 517.
- `core_only`: false SUPPORT 14 to 0, false NE 0 to 0.
- `full_evidence`: false SUPPORT 27 to 12, false NE 402 to 402.

External VitaminC files:

- Stage53 frozen external: no effect, false SUPPORT 6 to 6, false NE 814 to 814.
- Stage63 bridge-enabled external: false SUPPORT 135 to 131, false NE 411 to 414.
- Aggregate false NE increased by 3, all from the Stage63 external file.

Audit observations:

- In Stage123-B diagnostic variants, changes were concentrated in audited `location_swap` groups.
- Non-location intervention groups generally had no change.
- Example changed location pairs included Beacon Harbor vs Elm Valley, Dublin vs Cork, and Falcon Ridge vs Aurora City.
- The policy did not use `intervention_type`; that field was audit-only.

## 5. Policy input safety

The policy uses:

- Claim text.
- Evidence text.
- Original prediction label.

The policy does not use:

- `intervention_type`.
- `slot_mismatch_target`.
- Gold labels.
- Diagnostic family metadata.
- File path heuristics.
- Row id heuristics.

Gold labels and group fields are valid only for post-hoc reporting and audit. They are not policy inputs.

## 6. Why the analyzer is ready as shadow tooling

The script reproduced the Stage141-A broader stress result with the same core behavior: 53 false SUPPORT removals across 7 files, no correct-SUPPORT feature false positives, and non-negative per-file macro-F1 deltas.

The controlled Stage123-B diagnostic variants are especially strong for shadow tooling: they contributed -49 false SUPPORT with zero false-NE increase across evidence interfaces.

The analyzer is also operationally suitable as optional diagnostic tooling because it supports arbitrary prediction JSONL inputs, gold-missing count-only operation, group audits, changed-example exports, and report-only shadow predictions.

## 7. Why final integration remains blocked

This result does not justify final-logit integration or automatic final-prediction override.

Blocking reasons:

- External evidence remains limited.
- Stage63 produced a small false-NE increase of +3.
- The location extractor is regex-based.
- Current evidence supports observability and audit, not model behavior changes.

The analyzer is shadow-only and does not mutate source predictions.

## 8. Stage143 recommendation

Stage143 should register the Stage142 analyzer as an official optional diagnostic workflow and define when it should be run.

Recommended actions:

- Add documentation describing the Stage142 shadow analyzer.
- Add a report-only workflow note for applying it to future prediction JSONLs.
- Keep it optional and shadow-only.
- Do not route the policy into final logits or final predictions.

Avoid:

- Automatic final prediction override.
- Training loss integration.
- Using `intervention_type` or `slot_mismatch_target` as policy inputs.
- Claiming broad external generalization from the current limited external evidence.
