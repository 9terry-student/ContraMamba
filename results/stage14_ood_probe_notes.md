# Stage14 Controlled OOD Probe Notes

## Probe groups

Stage14 creates a controlled OOD-lite probe suite with five groups:

- `surface_distractor`
- `temporality_shift`
- `predicate_swap`
- `frame_swap`
- `sufficiency_drop`

The default output path is `data/stage14_ood_probe.jsonl`. The smoke output path used during local validation is `data/stage14_ood_probe_smoke.jsonl`.

## Generation rules

The generator reads the controlled dataset used by Stage13, defaulting to `data/controlled_v5_v3.jsonl`, and excludes `time_swap` examples before generation by default. This follows the Stage13 clean-training convention that treats `time_swap` as corrupted/confounded.

Generation is deterministic under `--seed`.

The groups are constructed as follows:

- `surface_distractor`: starts from original SUPPORT `none` examples and appends harmless punctuation/formatting noise that should not change the expected final label.
- `temporality_shift`: starts from original `none` examples and changes a `during <day>` temporal phrase in evidence while leaving the claim unchanged.
- `predicate_swap`: copies existing controlled `predicate_swap` examples into a Stage14 probe group.
- `frame_swap`: copies existing controlled frame-failure variants, including entity, event, location, role, and title/name swaps.
- `sufficiency_drop`: copies existing controlled evidence deletion/truncation examples.

Each output row preserves the fields required by the Stage13 runner:

- `id`
- `pair_id`
- `claim`
- `evidence`
- `final_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `polarity_label`
- `primary_failure_type`
- `intervention_type`

Each output row also adds:

- `stage14_probe_type`
- `source_intervention_type`
- `source_pair_id`
- `source_id`
- `stage14_expected_behavior`

## Label assumptions

The probe is designed to avoid ambiguous labels where possible:

- `surface_distractor` preserves the source final label.
- `temporality_shift` is labeled `NOT_ENTITLED` under the controlled ontology because the evidence no longer licenses the claim's temporal frame.
- `predicate_swap` is labeled according to the existing controlled predicate-failure label.
- `frame_swap` is labeled according to the existing controlled frame-failure label.
- `sufficiency_drop` is labeled according to the existing controlled sufficiency-failure label.

The suite intentionally does not reinterpret slot substitutions as ordinary open-domain fact-verification refutations. It follows the controlled-v5/v6 entitlement ontology.

## Skipped-example policy

The generator skips examples when a transformation cannot be performed safely. For example, `temporality_shift` only edits evidence containing a recognizable `during <day>` phrase. If no safe candidate exists for a group, the summary reports a skipped reason.

The script prints:

- total generated
- count by `stage14_probe_type`
- count by expected final label
- skipped counts by reason

## Why this is not full OOD validation

Stage14 is a controlled probe suite, not a full OOD benchmark. It is intended to stress specific entitlement behaviors under deterministic transformations. Passing Stage14 would not establish real-world OOD robustness, hallucination reduction, or deployment readiness. It should be used as a diagnostic step before any broader benchmark or human-data evaluation.
