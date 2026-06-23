# Stage14 Probe Expansion Notes

## Why temporality_shift was expanded

The first Stage14 probe contained only six `temporality_shift` examples because the generator only transformed weekday expressions such as `during Monday`. The controlled source data also contains many month expressions such as `during March`, so the generator now handles both weekday and month temporal slots.

The expanded generator keeps the transformation conservative: weekdays are replaced with another weekday, and months are replaced with another month. This avoids ambiguous edits such as converting a weekday into a month. The expected label remains `NOT_ENTITLED` only when the evidence temporal slot no longer matches the claim temporal slot under the controlled ontology.

If source-derived temporal examples are still below the requested target, the generator now adds deterministic synthetic temporal-slot mismatch examples. These synthetic examples keep entity, role, predicate, object, and location constant while changing only the temporal slot. They are marked with `source_intervention_type = synthetic_temporality`, and the generator prints an explicit warning when the fallback is used.

## Why n=6 was insufficient

The initial `temporality_shift` false-entitlement count was too small to support a strong temporality-specific claim. Six examples can show a warning sign, but they are not enough to distinguish a robust temporal failure from sampling noise or a narrow template artifact. Stage14 v2 targets 100 temporal examples by default when the source data permits it.

## Current strongest finding

The robust Stage14 finding before expansion was frame-slot mismatch, especially `location_swap` and `role_swap`. These subtypes produced many false-entitled outputs with high frame, predicate, and sufficiency probabilities. That pattern is stronger than the preliminary temporal observation because it appeared across more examples and across explicit frame-substitution subtypes.

The generator now balances the `frame_swap` probe across:

- `entity_swap`
- `event_swap`
- `location_swap`
- `role_swap`
- `title_name_swap`

This makes subtype reporting cleaner while avoiding synthetic low-quality examples.

## Predicate_swap interpretation

`predicate_swap` should be interpreted separately from frame failures. In the observed Stage14 seed1 analysis, predicate-swap false-entitled cases had lower predicate coverage than the dominant frame failures. This suggests an auxiliary-signal/final-decision mismatch: the predicate head partly detects the relation mismatch, but the final decision can still become entitled.

## Controlled-probe caveat

Stage14 remains a controlled OOD probe suite, not full OOD validation. The examples are deterministic transformations of the controlled dataset and should be used to diagnose model behavior under targeted perturbations. They should not be presented as evidence of broad real-world robustness.

## Stage14 v2 generation summary

Using:

```bash
python scripts/create_stage14_ood_probe.py --max-per-group 100 --output data/stage14_ood_probe_v2.jsonl
```

the generator produced:

- total records: 500
- `surface_distractor`: 100
- `temporality_shift`: 100
- `predicate_swap`: 100
- `frame_swap`: 100
- `sufficiency_drop`: 100
- expected labels: 400 `NOT_ENTITLED`, 100 `SUPPORT`
- frame subtypes: 20 each for `entity_swap`, `event_swap`, `location_swap`, `role_swap`, and `title_name_swap`
