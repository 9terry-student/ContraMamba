# Stage17 Temporal Comparator Notes

## Purpose

Stage17 tests whether the temporal failures observed in Stage15/Stage16 can be corrected by an explicit temporal comparison signal at evaluation time. This is a diagnostic post-processing prototype, not a trained model and not a new architecture.

## Extraction rules

The comparator extracts conservative temporal expressions from claim and evidence text.

Supported expressions include:

- weekdays: Monday through Sunday;
- months: January through December;
- relative/eventive markers: before, after, during, previously, later, earlier, upcoming, planned, completed, launched, will, had.

If both claim and evidence contain temporal expressions and the normalized expression sets differ, the comparator sets `temporal_mismatch_flag = 1`. If one or both sides lack temporal expressions, it sets `temporal_mismatch_flag = 0` and `temporal_comparator_status = insufficient_temporal_info`.

Every adjusted prediction records the extracted claim/evidence temporal expressions for audit.

## Comparator modes

- `hard_override`: if a temporal mismatch is detected and the original prediction is SUPPORT or REFUTE, downgrade to NOT_ENTITLED. This is an oracle-style diagnostic upper bound.
- `prob_penalty`: if a mismatch is detected, reduce SUPPORT probability and entitlement probability when available.
- `logit_penalty`: if logits are available, subtract a penalty from SUPPORT logits. If logits are absent, the script raises a clear error.

## Interpretation

If `hard_override` reduces temporal_mismatch false-entitlement while preserving temporal_erased and surface_control examples, then explicit temporal comparison is sufficient at diagnostic upper-bound level. If it harms temporal_erased or surface_control, the extractor is too broad. If it fails to trigger on temporal_mismatch, the extractor is too narrow.

## Caveats

Stage17 does not prove OOD robustness. It tests whether a simple external temporal comparator can correct a known controlled failure. Any successful result should be described as evidence for an explicit temporal-comparison mechanism, not as a trained model improvement.

