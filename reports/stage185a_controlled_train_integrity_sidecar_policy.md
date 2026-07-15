# Stage185-A deterministic controlled-train integrity sidecar policy

## Authorization

Stage184-A closed with `STAGE184A_DETERMINISTIC_FAIL_CLOSED_INTEGRITY_SIDECAR_SPEC_READY` and authorizes `STAGE185_CONTROLLED_TRAIN_INTEGRITY_SIDECAR_BUILDER` only. Stage185-A may materialize and statically audit a fail-closed sidecar. It may not modify the authoritative JSONL or generator, implement a loss, choose a target margin or nonzero weight, change checkpoint selection, or train.

## Inputs and identity

The only authoritative dataset is `data/controlled_v5_v3_without_time_swap.jsonl`, SHA-256 `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`. The builder requires the complete Stage184 specification package and the final Stage182 integrity package. It verifies their decisions, authorized route, file presence, hashes where recorded, dataset topology, family set, split topology, frame-label topology, and zero `time_swap` rows before emitting a sidecar.

The frozen split is pair-level: sorted pair IDs are shuffled with seed 174, the first rounded 20% are dev pairs, and all rows of a pair remain together. Row-level random splitting is forbidden.

## Generator provenance

The builder loads `scripts/build_controlled_v5.py` only through an import-isolated label-schema shim, calls pure deterministic helpers in memory, and requires reconstructed rows to equal the authoritative JSONL exactly. It never writes reconstructed data. The generator retains original/alternate structured fact arguments and per-family branches; Stage182's `INTENDED_AXES`, `changed_axes`, polarity, and morphology contracts are reproduced explicitly in the builder.

Exact generator equality proves identity and provenance, not cleanliness. Grammar, structured-axis, polarity, schema, and canonical checks remain independent.

## Criterion statuses

Every source row receives `PASS`, `FAIL`, `UNRESOLVED`, or policy-authorized `NOT_APPLICABLE` for:

- `grammar_status`
- `intervention_contract_status`
- `polarity_contamination_status`
- `schema_status`
- `canonical_status`
- `time_swap_status`
- `dataset_source_status`

Known deterministic template/morphology defects fail grammar. General language quality is never inferred. Missing structured provenance is unresolved. Contract passage requires exact observed changed axes, preserved axes, labels, canonical linkage, evidence relation, and generator branch—not merely a family name. Polarity checks use structured canonical/rendered polarity only; no sentiment classifier or lexicon is allowed.

Canonical validity is structural: exactly one same-pair `none`, unique identity, same-pair claim/linkage, expected generator labels and facts, no cross-pair leakage, and no known canonical generator defect. It does not claim real-world truth or human-level semantic quality.

## Integrity composition

Fixed precedence:

1. Dataset/SHA/identity/join/split/topology/provenance contradictions block the whole build.
2. Any deterministic criterion `FAIL` makes the row `INELIGIBLE`.
3. With no failure, any required `UNRESOLVED` makes the row `UNRESOLVED`.
4. Only all required `PASS` makes the row `ELIGIBLE`.

`UNRESOLVED` is never promoted. Reason codes are sorted and unique; JSONL stores an array and CSV stores a `|`-delimited string.

Positive-margin eligibility is independently defined as:

```text
integrity_status == ELIGIBLE
and split == train
and frame_compatible_label == 1
and time_swap_status == PASS
and dataset_source_status == PASS
```

Thus a clean dev row may be integrity-eligible but loss-ineligible with `DEV_SPLIT_EXCLUDED`; a clean train-incompatible row receives `FRAME_LABEL_NOT_COMPATIBLE`.

## Stage182 regression oracle

The Stage182 selected 78-row subset is never a whitelist or training target. It is used only as a deterministic regression oracle. All overlap IDs must join exactly once. The 22 deterministic contaminated rows must be `INELIGIBLE`; 21 `NON_POLARITY_INTERVENTION_POLARITY_CHANGE` and one `DID_NOT_INFLECTED_PREDICATE` must be recovered exactly; no contaminated overlap may be eligible; and no Stage182-clean row may gain either contamination-specific code. A clean overlap may remain `UNRESOLVED` under stricter Stage185 gates.

## Sidecar and hashes

The sidecar contains exactly one row per source row in source order. It records identity, split, labels, seven criterion statuses, integrity status, positive eligibility, stable reason codes, canonical/family/rule IDs, source/generator/report/builder hashes, creation timestamp, and `audit_` evidence fields. Claim/evidence text is not duplicated.

The builder computes JSONL and CSV file SHA-256 plus a semantic SHA over stable-key, source-order row representations excluding `created_at` and output absolute paths. Identical dataset, generator, builder, and rule version therefore produce the same semantic SHA.

## Coverage and decision

Coverage is reported overall, by split, frame label, family, reason code, and pair. Scientific safety fields include the exact eligible-positive count/rate, eligible family count, largest family share, zero-eligible families, unresolved-positive count, and leading unresolved/ineligible reasons.

Fixed descriptive warnings are `ZERO_ELIGIBLE_POSITIVES`, `SINGLE_FAMILY_ELIGIBILITY`, and `FAMILY_CONCENTRATION_WARNING` when the largest share exceeds 0.80. No fitted coverage threshold is introduced.

With a complete valid sidecar, passing Stage182 regression, a semantic SHA, and at least one eligible train-compatible positive, the decision is `STAGE185A_INTEGRITY_SIDECAR_BUILT_AND_POSITIVE_ELIGIBILITY_MATERIALIZED`, authorizing only `STAGE186_COMPATIBLE_POSITIVE_MARGIN_FIXED_SPEC_AUDIT`. Zero eligible positives routes to `STAGE185A_INTEGRITY_SIDECAR_BUILT_WITH_ZERO_POSITIVE_ELIGIBILITY` and `STAGE186_GENERATOR_PROVENANCE_RECOVERY_SPEC`. Loss implementation and training remain unauthorized.

## Safety

Deterministic sidecar generation and static audit only: no JSONL, generator, trainer, model, loss, checkpoint, or annotation modification; no model/Torch/checkpoint/forward/training; no LLM/grammar model/text classifier/learned probe; no threshold fitting, calibration, external evaluation, `time_swap`, multi-seed, hyperparameter sweep, or unrelated refactor.
