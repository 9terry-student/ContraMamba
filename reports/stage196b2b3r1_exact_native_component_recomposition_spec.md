# Stage196-B2-B3-R1 Exact Native Component Recomposition specification

## Authority and scope

Stage196-B2-B3-R1 is an artifact-only, inference-only intervention probe over the frozen-Mamba Stage196-B2-B3P0 epoch observations. Its scientific label is **Exact Native Component Recomposition**. It neither imports nor loads a model, creates an optimizer, trains, selects a checkpoint, performs OOD evaluation, nor changes an existing artifact. Its only writable location is the explicitly supplied fresh `--output-dir`.

The B2-B3 authority must close on its exact nine files, completed observability-required decision, empty blockers, 193/193 passed contract, identified composer graph, normalized `support_logit - not_entitled_logit` margin, seven planned variants, and two directions. The B2-B2 authority must close on its exact nine files, completed seed-specific multipath decision, 155/155 contract, frozen 16-row row-path summary, and 320-row paired-epoch table. `path_class` is read only from the static row-path authority; it is never inferred from epoch trajectories.

The old analyzer at `scripts/analyze_stage196b2b3_exact_component_recomposition.py` remains unchanged. R1 records its path and SHA-256. It preserves its fixed decision names, next-stage mapping, tail-status definition, epochs 18–20, recovery/harm population, seed roles, subtype rules, bidirectional discipline, and promotion prohibitions. The predecessor intentionally left the component predicates unevaluated because native recomposition was unavailable. R1 replaces that unavailable-data layer with exact recomposition rows and uses strict Boolean closure: in the forward direction every recovery identity must reproduce the donor tail, every harm identity must preserve the recipient tail, and a cross-seed claim requires the predicate in both seeds 184 and 185. There is no fitted threshold, score search, or optimized cutoff.

## Required command line

The analyzer has exactly these seven required arguments:

```text
--repo-root
--stage196b2b3-analysis-json
--stage196b2b2-analysis-json
--stage196b2b3p0-run-root
--stage196b2b3p0-runtime-git-commit
--current-git-commit
--output-dir
```

All paths are explicit. Companion files are resolved only as exact filenames in the two supplied analysis directories or in the six exact P0 run directories. There is no recursive search, timestamp selection, sibling guessing, modification-time selection, substring fallback, positional recovery, or row-order recovery.

## Exact source closure

The B2-B3 directory must contain only:

```text
stage196b2b3_analysis.json
stage196b2b3_report.md
stage196b2b3_composer_graph.csv
stage196b2b3_native_reconstruction.csv
stage196b2b3_component_swap_rows.csv
stage196b2b3_row_swap_summary.csv
stage196b2b3_group_swap_summary.csv
stage196b2b3_subtype_summary.csv
stage196b2b3_contract.csv
```

The B2-B2 directory must contain only its existing analysis, report, row-path summary, epoch-paired paths, group-path summary, event-order summary, intervention-type paths, contrast summary, and contract filenames. The frozen counts remain seed184 recovery 5/harm 6 and seed185 recovery 2/harm 3. Seed183 is contrast-only.

The P0 root must contain exactly:

```text
seed183_joint
seed183_frame_local_only
seed184_joint
seed184_frame_local_only
seed185_joint
seed185_frame_local_only
```

Each run must contain `composer_inputs/` and `trajectory/`. The composer namespace is the exact manifest plus files `001` through `020`; the trajectory namespace is the exact Stage196-B2-P0 channel filenames `001` through `020`. Every sidecar has 720 rows, each run has 14,400 rows in each namespace, and each global namespace has 86,400 rows. Manifest sidecar hashes are verified. All six manifests must expose the same nonempty source-hash mapping. The manifest and every composer row must carry runtime commit `fa16787efa84bb15d832b6d9382fafd77016c4e2`, which must also equal the explicit runtime-commit argument.

The directory name, recorded as `manifest_run_id`, is authoritative for seed and treatment role. An internal exporter `run_name`, including `single`, is not used as identity.

## Identity and exact pairs

Within every run and epoch, composer rows are independently unique on `stable_row_id`, `id`, `source_row_id`, and integer `dev_position`. Composer-to-trajectory validation uses exactly `(id, source_row_id, dev_position)` and requires prediction agreement. Pairs use exactly `(seed, epoch, id, source_row_id, dev_position)` and must contain one `joint` and one `frame_local_only` row. Closure is 3 seeds × 20 epochs × 720 identities = 43,200 pairs.

B2-B2-to-composer linkage uses exactly `(seed, id, source_row_id, dev_position)`. When linked, its `stable_row_id` must equal the composer `stable_row_id`. The resulting primary population is exactly 16 identities × 20 epochs = 320 identity-epochs.

## Independent native reconstruction

R1 does not accept the P0 diagnostic reconstruction as its causal calculation. It parses exported primitive and raw branch state and independently computes:

```text
E = frame_prob * predicate_coverage_prob * sufficiency_prob
REFUTE_decision = E * negative_energy
NOT_ENTITLED_decision = not_entitled_bias + softplus(raw_alpha) * (1 - E)
SUPPORT_decision = E * positive_energy
```

It independently checks `softplus_alpha = softplus(raw_alpha)`. It then reconstructs all four exhaustive final-modulation branches from raw causal state:

| Branch | Raw causal inputs | Transformation | Output delta in `[REFUTE, NOT_ENTITLED, SUPPORT]` |
|---|---|---|---|
| temporal comparator | availability, condition mask, active mask, raw temporal alpha | `m = active × softplus(raw)` | `[-m, +m, -m]` |
| predicate comparator | availability, condition mask, active mask, raw predicate alpha | `m = active × softplus(raw)` | `[-m, +m, -m]` |
| temporal adapter | availability, adapter logit, scale, active state | `m = sigmoid(logit) × scale` when active | `[-m, +m, -m]` |
| temporal channel | availability, channel logit, preservation probability, gated scale, active state | `m = sigmoid(logit) × (1-preservation) × scale` when active | `[-m, +m, -m]` |

Raw/transformed parameter agreement, gate activity, structural-null behavior, exported gate probability, and exported effective scale are independently checked. Exported branch deltas and total deltas are validation targets. They are not used as the causal formula. Branch contributions are summed, added to the reconstructed decision head, and used to derive the final margin and argmax prediction.

Across all 86,400 rows, maximum entitlement, decision-head, branch/total-delta, final-logit, and margin error must be at most `1e-6`, and prediction equality must be 1.0. Failure prevents the positive control and every scientific component row.

## Intervention dependency closure

Every counterfactual starts with the recipient's actual causal state. Subset swaps replace only the named donor primitives. Entitlement, decision logits, branch deltas, total deltas, final logits, margins, probabilities, and predictions are never directly copied. The diagnostic `polarity_support_margin` is never a causal input.

- `FRAME_ONLY` replaces only `frame_prob`.
- `PREDICATE_ONLY` replaces only `predicate_coverage_prob`.
- `SUFFICIENCY_ONLY` replaces only `sufficiency_prob`.
- `ENTITLEMENT_PRIMITIVES` replaces the three entitlement primitives.
- `POLARITY_ONLY` replaces `positive_energy` and `negative_energy`.
- `ENTITLEMENT_PLUS_POLARITY` replaces the three entitlement primitives and both polarity energies.

The four outer branches do not consume any of those subset-swapped primitives in the frozen graph. They are nevertheless recomputed from recipient raw state for every subset counterfactual. No recipient branch delta is retained as an output shortcut.

The full positive control replaces every actual composer causal input with donor state: entitlement primitives, polarity energies, decision-head raw parameters and bias, comparator availability/masks/raw parameters, adapter/channel logits and gate state, preservation state, and configured scales. Donor derived logits and deltas are not copied. All deterministic values are recomputed.

## Full-composer donor control

The positive control covers all 43,200 paired states in both directions, for 86,400 directional rows. Recomposition must reproduce all three donor logits, the donor SUPPORT-vs-NOT_ENTITLED margin, and donor prediction within `1e-6` with prediction equality 1.0.

If it fails, the decision is `STAGE196B2B3R1_POSITIVE_CONTROL_FAILED`, the next stage is `STAGE196B2B3R1_REPAIR_RECOMPOSITION`, exact failing identities and fields are retained in analysis JSON, component CSVs contain no scientifically interpreted rows, and no component decision is emitted.

## Scientific rows and summaries

After and only after the donor control passes, R1 produces:

```text
16 identities × 20 epochs × 2 directions × 6 variants = 3,840 rows
```

The two directions remain separate. A zero donor-minus-recipient margin denominator yields a JSON/CSV null closure fraction and null toward-donor indicator; it never emits NaN or infinity. Row summaries use all 20 epochs for trajectory integrity and epochs 18–20 for the frozen decision predicates. Group and subtype summaries preserve direction, variant, seed, recovery/harm role, path class, and intervention type.

The frozen subtype populations are seed184 recovery multi-channel conflict, seed185 recovery frame-entitlement gain, polarity override, composition residual, and frame/entitlement-loss-like. The seed185 harm multi-channel subtype still requires negative terminal Frame, predicate, entitlement, and margin deltas from the static B2-B2 authority.

### CSV numeric representation boundary

Historical B2-B2 row-summary values are read with `csv.DictReader`, so numeric cells remain strings. R1 keeps `number()` strict for JSON/JSONL native numeric values and uses the separate `csv_number()` parser only for numeric CSV metadata. The CSV parser accepts one whitespace-trimmed finite decimal or scientific-notation number, plus native integer/float values for internal compatibility. It rejects nulls, booleans, empty or whitespace-only cells, `NaN`, infinities, and nonnumeric text; errors include the field name and original value.

During B2-B2 primary-source validation, the analyzer selects exactly the three rows with seed185, `transition_role=harm`, and `path_class=MULTI_CHANNEL_CONFLICT`. It parses `tail3_delta_frame`, `tail3_delta_predicate`, `tail3_delta_entitlement`, and `tail3_delta_margin` from the frozen CSV authority and requires every value to be finite and strictly `< 0.0`. The `b2b2_seed185_harm_multichannel_sign_closure` contract row records the exact identities and parsed values. These values are not recomputed from P0 sidecars. The later subtype classification uses the same strict CSV parser and retains `FRAME_ENTITLEMENT_LOSS_LIKE_ROWS` unchanged.

The full typed-parser boundary audit finds no other CSV numeric calls to `number()` and no CSV boolean calls to `boolean()`. CSV seed, epoch, and position fields continue through `integer()`, whose existing contract explicitly accepts strict integer strings. Manifest and sidecar numeric/boolean fields remain JSON-native and continue through `number()` and `boolean()` unchanged.

## Decision discipline

Cross-seed component success requires both positive seeds. The evaluation order preserves the existing decision vocabulary:

1. entitlement-primitives closure across both seeds → `STAGE196B2B3_ENTITLEMENT_COMPONENT_DOMINANT`;
2. entitlement-plus-polarity closure across both seeds when neither component alone closes → `STAGE196B2B3_ENTITLEMENT_POLARITY_DISJUNCTIVE_EFFECT`;
3. polarity-only closure across both seeds → `STAGE196B2B3_POLARITY_OVERRIDE_COMPONENT_CONFIRMED`;
4. differing seed-level closures → `STAGE196B2B3_SEED_SPECIFIC_COMPONENT_EFFECT`;
5. otherwise → `STAGE196B2B3_FINAL_COMPOSER_RESIDUAL_REQUIRED`.

Reverse-direction results are retained as causal controls and never merged into forward denominators. Seed183 remains contrast-only and cannot vote in a primary component decision.

## Outputs and atomic publication

Exactly nine files are rendered in memory and written under their fixed R1 names. A fresh output directory is required. Each file is written to a unique temporary file, flushed, fsynced, and atomically replaced into its final name. Final directory closure must equal the nine-name set.

The analysis JSON records source paths and hashes, runtime and current commits, frozen decision-rule provenance, reconstruction and control evidence, population/count closure, variants/directions, seed/role/path/subtype summaries, cross-seed consistency, authorized interpretation, uncertainty, and prohibited claims. Contract `required` and `observed` evidence are serialized as valid JSON when embedded in CSV; booleans remain booleans before CSV rendering.

The composer graph copies every B2-B3 authority row without changing its source columns, adds the P0/recomposition/dependency/control columns, and appends an explicit exhaustive four-branch dependency audit. Native reconstruction is summarized per run, seed, mode, and epoch, with a global row.

## Scientific prohibitions

R1 does not authorize formal mediation, architectural sufficiency beyond the frozen composer, external/OOD validity, unfrozen-Mamba validity, training improvement, promotion, a new decision mechanism, polarity causality from the diagnostic margin, or any component claim if the positive control fails. All interpretations remain within-model interventions on the observed frozen-Mamba composer.
