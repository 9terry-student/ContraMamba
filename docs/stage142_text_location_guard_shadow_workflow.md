# Stage142 Text Location Guard Shadow Workflow

## 1. Purpose

The Stage142 Text Location Guard Shadow Workflow documents `scripts/analyze_stage142_text_location_guard_shadow.py` as an official optional shadow-only diagnostic workflow.

The analyzer is intended for prediction JSONL files that contain claim text, evidence text, and prediction labels. It audits false SUPPORT risk when claim/evidence text appears to contain disjoint location-like spans. It does not mutate source predictions and does not provide automatic final prediction overrides.

## 2. Policy definition

The policy name is `text_loc_disjoint`.

For each row:

1. Only apply the policy to rows whose original prediction is `SUPPORT`.
2. Extract location-like spans from claim text and evidence text.
3. If both sides contain non-empty disjoint location sets, produce `shadow_prediction = NOT_ENTITLED`.
4. Otherwise, preserve the original prediction as the shadow prediction.

This workflow is shadow-only. The source prediction remains unchanged.

## 3. Policy input safety

The policy inputs are only:

- claim text
- evidence text
- original prediction label

The policy must not use:

- gold labels
- `intervention_type`
- `stage122_family`
- `family`
- `slot_mismatch_target`
- diagnostic metadata
- file path heuristics
- row id heuristics

Gold, intervention, family, and other diagnostic fields may be used only for post-hoc evaluation or audit when present. They must not be used to decide whether the policy fires.

## 4. When to run this diagnostic

Run this optional diagnostic:

- after any major prediction export that contains claim/evidence/prediction fields
- after external fact-verification diagnostic exports
- when false SUPPORT risk is being audited
- when location-like entity mismatch is suspected

Do not run this workflow:

- as training
- as model selection
- as threshold tuning
- as a final prediction override
- as proof of broad external generalization

## 5. Inputs and outputs

The analyzer accepts arbitrary prediction JSONL files with claim/evidence text and prediction labels. It supports inputs with gold labels and inputs without gold labels.

Compact CLI usage template:

```bash
python scripts/analyze_stage142_text_location_guard_shadow.py \
  --input-jsonl path/to/predictions.jsonl \
  --output-dir reports/stage142_text_location_guard_shadow \
  --write-shadow-jsonl
```

Expected output artifacts:

- `stage142_text_location_guard_shadow_report.json`
- `stage142_text_location_guard_shadow_report.md`
- `stage142_file_metrics.csv`
- `stage142_aggregate_metrics.json`
- `stage142_group_metrics.csv`
- `stage142_changed_examples.jsonl`
- optional `stage142_shadow_predictions.jsonl`

## 6. Recommended interpretation

Treat changed rows as audit candidates where a SUPPORT prediction may be vulnerable to a location mismatch. Aggregate and per-file metrics can estimate whether the shadow rule reduces false SUPPORT errors and whether it introduces false NOT_ENTITLED errors.

The Stage142-B reproduction registered the candidate as robust for optional shadow diagnostics, not for final integration. The observed result reduced false SUPPORT by 53 and increased false NOT_ENTITLED by 3, with a macro F1 delta of `+0.0017308314825622562` across 33,000 valid rows.

## 7. Safety boundaries

This workflow is:

- optional
- shadow-only
- diagnostic-only
- audit-oriented

It must not:

- modify final logits
- modify final predictions
- modify training
- affect checkpoint selection
- enable or replace the Stage128 guard
- use Stage15
- use external data for training
- use thresholds for model selection

## 8. Known limitations

- Location extraction is regex-based.
- The extractor can miss lower-case, non-English, multiword, ambiguous, or complex locations.
- The heuristic can confuse locations with organizations, people, titles, or other capitalized spans.
- External evidence remains limited.
- Stage63 showed a small false-NE tradeoff of `+3`.
- The workflow is not evidence for final-logit integration.

## 9. Relation to Stages 140-143

- Stage140-A found a candidate deployable text signal.
- Stage141-A stress-tested it across broader existing JSONLs.
- Stage142-A implemented the reusable analyzer.
- Stage142-B reproduced the stress result.
- Stage143-A registers the analyzer as an official optional diagnostic workflow.

## 10. Non-goals

This workflow does not provide:

- final-logit integration
- automatic prediction mutation
- training loss
- checkpoint selection
- Stage128 guard replacement
