# Stage 10A Number-Swap Single-Axis Probe

## Question

`time_swap` is a catastrophic confidently inverted case in the current v3 predictions, but entity, event, and predicate swaps are mostly rejected. The sharper question is whether the failure is temporal-specific or reflects a broader inability to compare values within the same semantic slot type.

Stage 10A changes exactly one numeric quantity while preserving the subject, predicate, object type, time expression, and sentence form. Example:

```text
Claim:    Aster Company sold 100 units in April.
Evidence: Aster Company sold 300 units in April.
```

## Label ontology

The probe assigns `number_swap` the label `NOT_ENTITLED`. This follows the existing `time_swap` ontology: a same-frame slot-value mismatch is treated as evidence that does not license the claim, rather than as an explicit polarity refutation. The auxiliary labels likewise match the time-swap convention: frame compatibility is 0, predicate coverage is 1, and sufficiency is 1.

If a future benchmark instead defines numeric mismatch as `REFUTE`, it must be evaluated separately as a value contradiction. The probe should not silently force `NOT_ENTITLED` under a different ontology.

## Decision rule

- If `number_swap` and `time_swap` both have high false-entitled rates and high frame, predicate, sufficiency, and entitlement pass probabilities, the result supports a same-type low-surface-change or slot-value comparison failure.
- If `number_swap` is correctly rejected while `time_swap` fails, the result supports a temporal-specific failure.
- Otherwise, the result is inconclusive or mixed.

## Inference prerequisite

The repository currently contains neither saved model checkpoints nor prediction JSON exports. The probe can be generated immediately, but it cannot be evaluated from the existing source tree alone. Evaluation requires one of:

1. re-running balanced-auditor inference from the saved v3 checkpoints; or
2. retraining/exporting predictions for the existing v3 model if checkpoints were not retained.

No retraining should occur without explicit authorization. Once `number_swap` predictions exist, `scripts/evaluate_number_swap_probe.py` compares them directly with the balanced-auditor `time_swap` predictions.

## Claim constraints

- Do not generalize the time-swap result into broad evidence-presence versus claim-match blindness; entity, event, and predicate swaps already contradict that broad framing.
- Do not call temporal mismatch the root cause until the number-swap contrast is evaluated.
- Do not infer a shared gate mechanism from global correlation alone; use Stage 9D.
