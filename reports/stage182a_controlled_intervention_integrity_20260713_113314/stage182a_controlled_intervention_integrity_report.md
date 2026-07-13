# Stage182-A controlled-intervention integrity audit

**Decision:** `STAGE182A_DATA_CONTAMINATION_CONFIRMED_AND_CLEAN_MODEL_FAILURE_SET_READY`

The deterministic audit covered all 78 unique review items:
39 hard rows and 39 matched controls.

## Result

- Clean constructions: 56
- Contaminated constructions: 22
- Grammar anomalies: 1
- Non-polarity polarity leaks: 21
- Clean hard/control pairs: 21 of 39
- Final clean model-failure candidates: 14
- Deterministic contaminated items: 22
- Candidate intervention families: 4
- Schema unresolved rate: 0.0

## Decision evidence

- Criterion: `minimum_clean_hard_candidates`
- Observed clean hard candidates: 14
- Required minimum: 8
- Candidate intervention families: 4
- Matched clean controls available: 14
- Native mismatch invariant passed: True
- Passed: True

Generator equality is provenance rather than a cleanliness verdict. A row can
exactly reproduce the generator and still be excluded for a deterministic
grammar or multi-axis construction defect. The final set is a model-failure
*candidate* set, not a causal diagnosis.

No data, annotation, model, checkpoint, or training state was modified.
