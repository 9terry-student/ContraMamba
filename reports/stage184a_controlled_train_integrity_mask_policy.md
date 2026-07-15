# Stage184-A controlled-train integrity sidecar specification and feasibility policy

## Decision

`STAGE184A_DETERMINISTIC_FAIL_CLOSED_INTEGRITY_SIDECAR_SPEC_READY`

Authorized next stage:

`STAGE185_CONTROLLED_TRAIN_INTEGRITY_SIDECAR_BUILDER`

Stage185 may build and statically audit a row-identity sidecar only. It may not rewrite the main JSONL, implement the positive-margin loss, run a model, or train.

## Stage183-A closure

Stage183-A closed with `STAGE183A_CONTROLLED_TRAIN_INTEGRITY_MASK_REQUIRED_FIRST`. The frozen seed-174 train topology is balanced at 1,440 compatible and 1,440 incompatible rows; global positive reweighting therefore has no imbalance basis. An absolute compatible-positive frame-logit hinge remains the contingent best scientific fit because it targets the native frame head, unlike Stage175's final SUPPORT reference anchor and Stage177's relative frame ordering. The hinge remains unimplemented: the target margin and nonzero weight are unset, and loss/training authorization has not been granted.

## Complete-mask meaning

A complete mask does not assert that every row is clean. It means every authoritative row receives exactly one top-level integrity status:

- `ELIGIBLE`: every required criterion deterministically passes.
- `INELIGIBLE`: at least one required criterion deterministically fails or the row is outside loss scope.
- `UNRESOLVED`: no deterministic pass or fail is licensed from authoritative structured evidence.

`UNRESOLVED` is never treated as clean. Future positive-margin eligibility is exactly:

```text
frame_compatible_label == 1
and integrity_status == ELIGIBLE
and split == train
and intervention_type != time_swap
and dataset_source_status == PASS
```

`INELIGIBLE`, `UNRESOLVED`, frame-incompatible, dev, `time_swap`, external, and Stage34/35 diagnostic rows are excluded.

## Dataset identity contract

The sole authoritative source is `data/controlled_v5_v3_without_time_swap.jsonl`. Stage184-A records its SHA-256 and audits row/pair counts, intervention families, unique `id`, rectangular pair-family topology, rows-per-pair distribution, one `none` row per pair, base schema, labels, time-swap absence, deterministic pair split, and semantic-identity duplicates.

Expected topology is loaded from `reports/stage183a_controlled_train_integrity_mask_required_closure.json`; it is not silently substituted by analyzer constants. Any mismatch in 3,600 rows / 300 pairs, 2,880/720 train/dev rows, 240/60 train/dev pairs, or the 1,440/1,440 train frame topology blocks Stage184 completion.

The row identity key is the JSONL `id` field, represented as `row_id` in the sidecar. `pair_id + intervention_type` is the structural semantic identity. Exact duplicate row IDs, duplicate pair-family identities, missing canonical rows, or nonrectangular family coverage are blocking conditions.

## Generator provenance

`scripts/build_controlled_v5.py` preserves structured canonical facts in `FACT_TEMPLATES`, `_ADDITIONAL_FACT_ROWS`, and deterministic `_generated_fact_template` arguments. Each fact includes original and alternate title, name, role, predicate, object, time, and location. `_build_records` declares the exact branch used for every intervention and `_record` renders the stable row ID and labels.

The generator does not copy those arguments into each JSONL row. Nevertheless, Stage185 can recover them deterministically by loading only the canonical generator/label schema, reconstructing the expected rows for the observed pair count, and requiring exact row equality before using the reconstructed fact arguments. Stage182-A already implements this pattern and then compares each row with its same-pair `none` row.

Generator equality is only an identity/provenance gate. It cannot be the cleanliness verdict: the generator itself renders negative forms as `did not` followed by an already-inflected predicate, and generated non-polarity rows can inherit an unintended polarity change. Grammar, axis-contract, and polarity checks therefore remain independent criteria.

## Criterion derivability

### `grammar_valid`

The deterministic template layer can verify nonempty required slots, exact branch rendering, and known morphology rules. The Stage182-A rule detects `did not` followed by the generator's already-inflected original or alternate predicate and emits `DID_NOT_INFLECTED_PREDICATE` / `FAIL_KNOWN_TEMPLATE_GRAMMAR`.

The builder must not infer general English naturalness. A row that passes known template rules but requires a general grammaticality judgment receives `UNRESOLVED_GENERAL_GRAMMAR`, unless the exact deterministic template contract is explicitly defined as the complete grammar scope for that family. No LLM, parser, classifier, lexicon, or annotation may resolve it.

Allowed criterion states are `PASS`, `FAIL_KNOWN_TEMPLATE_GRAMMAR`, and `UNRESOLVED_GENERAL_GRAMMAR` (with `NOT_APPLICABLE` only where a criterion truly does not apply).

### `intervention_contract_exact`

The authoritative contract combines generator branch identity, original/alternate structured arguments, the same-pair canonical `none` row, and Stage182-A's `INTENDED_AXES`. For each family the builder must compare observed structured state with canonical state, validate allowed changed axes, required preserved axes, expected label transition, canonical counterpart, and evidence relation, and emit stable reason codes.

Family name alone never passes the contract. Exact generator reconstruction alone never passes the contract. If structured arguments cannot be recovered with an exact source/hash contract, status is `UNRESOLVED` with `GENERATOR_PROVENANCE_MISSING` or `INTERVENTION_CONTRACT_UNRESOLVED`.

### `polarity_contamination_absent`

`polarity_flip` permits the declared polarity change. Every other non-`none` family forbids an undeclared polarity change. The builder compares canonical structured polarity, intervention structured label, expected label transition, and deterministic rendered polarity. It generalizes Stage182-A's `NON_POLARITY_INTERVENTION_POLARITY_CHANGE` rule without adding sentiment inference.

A mismatch between structured polarity and the deterministic template realization emits `POLARITY_REALIZATION_MISMATCH`. Free-text sentiment classifiers and heuristic lexicons are prohibited.

### `schema_resolved`

This is structural, not semantic. It validates required fields; exact types and nullability; binary labels; allowed enums; `row_id`, `pair_id`, and intervention identity; row/pair uniqueness; and canonical linkage. Missing/invalid fields fail or remain unresolved according to the fail-closed table.

### `canonical_row_valid`

Structural canonical validity requires exactly one same-pair `none` row, unique canonical row identity, expected base labels, claim linkage, no cross-pair linkage, self-consistent generator arguments, and no known generator-defect reason code. It does not assert real-world truth, factuality, or complete human semantic quality.

### Scope criteria

`time_swap_absent`, `authoritative_main_dataset`, `not_external_or_stage34_35`, `required_base_schema_present`, and `frame_label_compatible` are deterministically available from exact path/hash, row fields, and split contract. `frame_label_compatible` is a loss-scope criterion, not a cleanliness claim.

## Family contract matrix

The analyzer enumerates actual families from the JSONL and joins them to the generator/source contract map. Unknown or source-unmatched families are retained as unresolved; they are never dropped or passed by name.

| Family | Intended change | Required preservation | Expected label transition from canonical |
|---|---|---|---|
| `none` | none | all axes | canonical labels retained |
| `paraphrase` | realization | title, name, role, predicate, object, time, location, polarity | labels retained |
| `entity_swap` | name | all other axes and polarity | `NOT_ENTITLED`, frame 0, predicate 0, sufficiency 1, polarity `NONE` |
| `event_swap` | object | all other axes and polarity | `NOT_ENTITLED`, frame 0, predicate 0, sufficiency 1, polarity `NONE` |
| `location_swap` | location | all other axes and polarity | `NOT_ENTITLED`, frame 0, predicate 1, sufficiency 1, polarity `NONE` |
| `role_swap` | role | all other axes and polarity | `NOT_ENTITLED`, frame 0, predicate 1, sufficiency 1, polarity `NONE` |
| `title_name_swap` | title and name | remaining axes and polarity | `NOT_ENTITLED`, frame 0, predicate 0, sufficiency 1, polarity `NONE` |
| `predicate_swap` | predicate | remaining axes and polarity | `NOT_ENTITLED`, frame 1, predicate 0, sufficiency 1, polarity `NONE` |
| `evidence_deletion` | content deletion | claim and canonical linkage; no unintended polarity | `NOT_ENTITLED`, frame 1, predicate 1, sufficiency 0, polarity `NONE` |
| `evidence_truncation` | content truncation | claim and canonical linkage; no unintended polarity | `NOT_ENTITLED`, frame 1, predicate 1, sufficiency 0, polarity `NONE` |
| `irrelevant_evidence` | content replacement | claim and canonical linkage; no unintended polarity | `NOT_ENTITLED`, frame 0, predicate 0, sufficiency 0, polarity `NONE` |
| `polarity_flip` | polarity | all non-polarity axes | canonical SUPPORT/REFUTE flipped; frame/predicate/sufficiency remain 1 |

`time_swap` exists in generator schema but must have zero rows in the authoritative no-time-swap dataset.

## Pair invariants and propagation

- Dataset/identity failures (SHA mismatch, extra/missing sidecar row, duplicate row ID): entire artifact `BLOCKED`.
- Duplicate pair-family identity, missing/duplicate canonical `none`, cross-pair canonical leakage, inconsistent split, or nonrectangular expected family topology: related pair `UNRESOLVED`; duplicate row identity additionally blocks artifact generation because a one-to-one join is impossible.
- Canonical linkage or canonical structural failure: entire related pair `UNRESOLVED`.
- Single-row intervention contract, grammar, polarity, label, or schema failure: that row `INELIGIBLE`; the pair remains usable only if canonical and pair identity invariants pass.
- Missing evidence for a row-level criterion: that row `UNRESOLVED`.
- `time_swap`, external, Stage34/35, dev, and frame-incompatible rows are `INELIGIBLE` for the positive-margin loss regardless of semantic integrity.

No policy may silently downgrade `UNRESOLVED` to `ELIGIBLE`.

## Sidecar schema

The Stage185 canonical sidecar contains one row per authoritative JSONL row:

```text
row_id, pair_id, split, intervention_type, frame_compatible_label,
grammar_status, intervention_contract_status,
polarity_contamination_status, schema_status, canonical_status,
time_swap_status, dataset_source_status,
integrity_status, eligible_for_positive_margin, reason_codes,
canonical_row_id, family_contract_id, rule_version,
source_dataset_path, source_dataset_sha256, generator_source_sha256,
integrity_builder_sha256, created_at
```

Criterion statuses are enums, never nullable booleans. The general enum surface is `PASS`, `FAIL`, `UNRESOLVED`, `NOT_APPLICABLE`; criterion-specific reason codes preserve finer meaning. `integrity_status` is exactly `ELIGIBLE`, `INELIGIBLE`, or `UNRESOLVED`. `reason_codes` is a sorted, unique JSON array in JSON artifacts and a stable `;`-delimited sorted string in CSV.

`created_at` is UTC ISO-8601 metadata and is excluded from semantic content hashes. `rule_version` is a fixed contract version, initially `stage184a_v1`.

## Join and SHA contract

- Exact one-to-one `row_id` join; duplicate sidecar IDs are forbidden.
- Missing or extra sidecar IDs block use; missing rows are never assumed eligible.
- Source dataset SHA mismatch blocks the entire sidecar.
- Source path must resolve to the authoritative main dataset; a copied file with the same hash may be audited but cannot silently change the recorded canonical path.
- Generator SHA mismatch blocks rebuilding or extending decisions. An already materialized sidecar remains interpretable only with its recorded generator SHA and builder SHA; using it with a different generator is blocked, not merely warned.
- Split is the frozen deterministic pair split recorded by the sidecar; it is recomputed from pair IDs, seed 174, and dev ratio 0.2 and must not be independently reassigned.

## Decision coverage, positive coverage, and usability

Decision coverage is feasible: every row can deterministically receive `ELIGIBLE`, `INELIGIBLE`, or `UNRESOLVED` because every missing-evidence branch has an explicit fail-closed rule.

Positive eligible coverage is not estimated in Stage184-A. The exact count among the 1,440 train compatible rows is `unavailable_before_builder`; Stage185 must compute it from the complete sidecar. The `--minimum-safe-positive-rate` default 0.0 is reporting-only and cannot turn unresolved rows into eligible rows or tune a training threshold.

Scientific usability remains uncertain even if the sidecar is technically complete. Eligible positives may be few or concentrated in `none`, paraphrase, sufficiency, or polarity families. Stage185 must report eligible counts and rates by split, frame label, family, and reason code. No coverage threshold, family weighting, or training authorization follows automatically.

## Why generator augmentation is not first

Generator provenance augmentation is not required before the sidecar builder because the current source retains reconstructible canonical/original/alternate arguments and exact intervention branches, and Stage182-A demonstrates a deterministic reconstruction/axis-delta route. The sidecar can mark evidence gaps `UNRESOLVED` rather than fabricate a pass.

This decision does not authorize regeneration, JSONL mutation, or changing generator outputs. If Stage185 cannot reproduce the exact row/fact contract under recorded hashes, it must stop with `GENERATOR_PROVENANCE_MISSING`; it must not fall back to family names or free-text semantics.

## Safety policy

Stage184-A is a static specification and feasibility audit only: no dataset modification, JSONL rewrite, generator modification, model or Torch import, checkpoint load, forward, loss implementation, training, smoke, annotation, LLM labeling, text classifier, learned parser/probe, threshold fitting, calibration, external evaluation, `time_swap`, multi-seed execution, hyperparameter sweep, or unrelated refactor is authorized.
