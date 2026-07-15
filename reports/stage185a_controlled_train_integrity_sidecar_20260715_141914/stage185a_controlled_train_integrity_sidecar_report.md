# Stage185-A controlled-train integrity sidecar report

## Decision

`STAGE185A_INTEGRITY_SIDECAR_BUILT_AND_POSITIVE_ELIGIBILITY_MATERIALIZED`

Authorized next: `STAGE186_COMPATIBLE_POSITIVE_MARGIN_FIXED_SPEC_AUDIT`. Loss implementation and training remain unauthorized.

## Stage184-A closure and authoritative identity

Stage184-A authorized deterministic sidecar construction only. The source is `data/controlled_v5_v3_without_time_swap.jsonl`, SHA-256 `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`, with 3600 rows, 300 pairs, and 12 families.

## Complete sidecar and statuses

Complete means one exact sidecar row for every source row, not that every row is clean. `ELIGIBLE` requires every integrity criterion to pass; any deterministic failure yields `INELIGIBLE`; otherwise unresolved evidence yields `UNRESOLVED`. Unresolved rows never enter positive-margin eligibility.

Integrity status and loss eligibility are distinct. Dev and frame-incompatible rows may be integrity-eligible while loss-ineligible. Positive eligibility additionally requires train split, compatible frame label, and passing time/source gates.

## Criterion coverage

```json
{
  "canonical_status": {
    "PASS": 1950,
    "UNRESOLVED": 1650
  },
  "dataset_source_status": {
    "PASS": 3600
  },
  "grammar_status": {
    "FAIL": 450,
    "PASS": 3150
  },
  "intervention_contract_status": {
    "FAIL": 1350,
    "PASS": 2250
  },
  "polarity_contamination_status": {
    "FAIL": 1350,
    "PASS": 2250
  },
  "schema_status": {
    "PASS": 3600
  },
  "time_swap_status": {
    "PASS": 3600
  }
}
```

## Stage182 regression

Overlap rows: 78; deterministic contaminated: 22; recovered polarity/grammar: 21/1; passed: True.

The Stage182 subset is a regression oracle only, never a whitelist.

## Positive and family coverage

Eligible train-compatible positives: 605 / 1440 (0.420139). Eligible families: 5; largest family share: 0.200000. Unresolved positives: 119.

Warnings: none.

## Join and hashes

The sidecar is an exact source-order, one-to-one row-ID join. JSONL SHA-256: `fdaf57ab99742f042acecc0daa2c046263a8e326af1758ebdcab6d6bddf4ecfa`; CSV SHA-256: `e978798c51d5865652653d4e6c58e7167f5a88dbedf381586637431fe62a284a`; semantic SHA-256: `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`.

## Safety

No source JSONL, generator, trainer, model, loss, checkpoint, or annotation was modified. No model/Torch/checkpoint/forward/training, LLM labeling, grammar model, text classifier, learned probe, threshold fitting, calibration, or external evaluation is used.
