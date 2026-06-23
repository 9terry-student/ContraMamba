# Stage15 Slot Sensitivity Diagnostics

## Purpose

Stage15 creates a controlled diagnostic probe for testing whether ContraMamba models perform slot-resolved entitlement checking. It is motivated by Stage14-v2 results where surface and sufficiency controls were robust, but temporal and frame-slot mismatches often remained falsely entitled.

The goal is to distinguish three failure modes:

1. encoder or representation-level slot insensitivity;
2. auxiliary head failure, such as high frame probability on a frame-slot mismatch;
3. final aggregation failure, such as SUPPORT/REFUTE predictions despite low auxiliary predicate coverage.

## Probe types

- `temporal_mismatch`: reuses Stage14 `temporality_shift` examples.
- `temporal_erased`: removes the supported temporal phrase from both claim and evidence. If the only mismatch was time, the expected label becomes `SUPPORT`.
- `surface_control`: reuses Stage14 `surface_distractor` examples.
- `sufficiency_control`: reuses Stage14 `sufficiency_drop` examples.
- `frame_location_mismatch`: reuses Stage14 frame swaps whose source subtype is `location_swap`.
- `frame_role_mismatch`: reuses Stage14 frame swaps whose source subtype is `role_swap`.
- `predicate_mismatch`: reuses Stage14 `predicate_swap` examples.

## Temporal erasure rule

The generator removes simple explicit temporal phrases, including weekday and month expressions with `during`, `in`, or `on`. Examples include:

- `during Monday`
- `on Friday`
- `in March`
- `during October`

If no supported temporal phrase is found in either claim or evidence, the temporal-erasure example is skipped and counted.

## Label assumptions

Temporal-erased examples are labeled `SUPPORT` because entity, role, predicate, object, and location are intended to remain matched after temporal information is removed. This is a diagnostic assumption for controlled examples, not a general natural-language rule.

Temporal mismatch, frame location mismatch, frame role mismatch, predicate mismatch, and sufficiency control examples retain their Stage14-style expected behavior.

## Analysis outputs

`scripts/analyze_stage15_slot_sensitivity.py` writes:

- `results/stage15_slot_sensitivity_group_metrics.csv`
- `results/stage15_slot_sensitivity_examples.csv`
- `results/stage15_temporal_erasure_pairs.csv`
- `results/stage15_slot_sensitivity_summary.md`

The markdown summary highlights:

- whether temporal mismatch and temporal erasure have similar SUPPORT probabilities;
- whether frame probabilities remain high for location or role mismatches;
- whether predicate mismatch has low predicate coverage but still produces entitled final labels.

## Caution

Stage15 is a diagnostic probe suite, not full OOD validation and not a new model. It should be used to characterize failure anatomy before architecture or training changes are proposed.

