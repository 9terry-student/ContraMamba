# Stage196-B2-B4 Exact Primitive-Coalition and Final-Composer Residual Localization

## Scope and execution boundary

This is an artifact-only frozen-composer analysis. It performs no training, model or checkpoint loading, fitted modeling, threshold optimization, promotion, or external/OOD evaluation. Its implementation is `scripts/analyze_stage196b2b4_final_composer_residual_localization.py`. Inputs are explicit; latest-directory selection and recursive artifact discovery are forbidden.

## Required CLI

All arguments are required:

```text
--repo-root
--stage196b2b3r1-analysis-json
--stage196b2b2-analysis-json
--stage196b2b3p0-run-root
--stage196b2b3p0-runtime-git-commit
--current-git-commit
--output-dir
```

The output directory must be fresh and disjoint from all sources. P0 commit authority is `fa16787efa84bb15d832b6d9382fafd77016c4e2`. The R1 analyzer source commit is read from the supplied R1 analysis artifact, never guessed or hardcoded.

## Exact source closure

R1 must preserve the final-composer-residual decision, the B2-B4 localization recommendation, no blockers, exact nine-file closure, 55/55 contracts, the stated exact native reconstruction and zero-error positive control, and counts 3,840 / 43,200 / 86,400 / 192 / 48 / 72.

B2-B2 must preserve the seed-specific multipath decision, its B2-B3 recommendation, exact nine-file closure, 155/155 contracts, 16 identities, 320 epoch rows, and tail epochs 18–20. Frozen seed counts are seed184 recovery 5/harm 6 and seed185 recovery 2/harm 3. Frozen `transition_role`, `intervention_type`, `path_class`, and subtype remain authoritative. Seed183 is contrast-only.

P0 must contain exactly six named runs, 120 composer and 120 trajectory sidecars, and 86,400 rows of each kind. Manifests, hashes, runtime commit, row provenance, prediction agreement, and all tolerances must close at `1e-6`.

The P0 `trajectory/` directory is mixed-purpose. Its `trajectory_namespace` gate therefore inspects only files matching the exact Stage196-B2-P0 form `stage196b2p0_epoch_channels_[0-9][0-9][0-9].jsonl`, while separately treating every file beginning with `stage196b2p0_epoch_channels_` but not equal to an expected filename as malformed. The exact observed namespace must be the 20 files numbered `001` through `020`, and the malformed list must be empty. Stage191 artifacts, checkpoints, reports, provenance files, and other unrelated filenames are retained and reported as ignored; they do not affect the gate result.

The P0 `composer_inputs/` directory remains exclusive by contract. Its exact-directory closure is intentionally unchanged: exactly 20 `stage196b2b3p0_epoch_composer_inputs_001.jsonl` through `_020.jsonl` sidecars and `stage196b2b3p0_composer_input_manifest.json`, with no other files, plus existing manifest and hash closure.

Pairs use exact `(seed, epoch, id, source_row_id, dev_position)` keys and exactly one state per mode. Strict identity additionally checks `stable_row_id`. Both directions remain separate, yielding 16 × 20 × 2 = 640 primary directional states.

## Native recomposition

The analyzer independently computes:

```text
entitlement = frame_prob * predicate_coverage_prob * sufficiency_prob
REFUTE = entitlement * negative_energy
NOT_ENTITLED = not_entitled_bias + softplus(raw_alpha) * (1 - entitlement)
SUPPORT = entitlement * positive_energy
```

It reconstructs predicate and temporal mismatch comparators, temporal adapter, and temporal channel from raw causal inputs. Each branch contributes `[-m,+m,-m]`. Transforms, effective scales, branch deltas, total deltas, logits, margin, and prediction are validation targets rather than swapped derived values. Native logit and margin errors must be at most `1e-6`, with prediction equality 1.0.

## Primitive lattice

Canonical order is `FRAME`, `PREDICATE`, `SUFFICIENCY`, `POSITIVE_ENERGY`, `NEGATIVE_ENERGY`. All 32 five-bit masks are enumerated; polarity energies remain separate. `00000` is recipient-native and `11111` installs all donor primitives.

Legacy masks are exact: `FRAME_ONLY=10000`, `PREDICATE_ONLY=01000`, `SUFFICIENCY_ONLY=00100`, `ENTITLEMENT_PRIMITIVES=11100`, `POLARITY_ONLY=00011`, and `ENTITLEMENT_PLUS_POLARITY=11111`. All 3,840 R1 rows must match lattice logits, margin, and prediction within `1e-6`. Output count is exactly 20,480.

For each state, metric, and mask, signed Möbius terms use:

```text
interaction(S) = sum(T subset S) (-1)^(|S|-|T|) value(T)
```

The 31 nonempty terms must reconstruct full-minus-empty within `1e-6`. There are exactly 20,480 Möbius rows. No absolute-value decomposition, fit, percentage normalization, score, or threshold search is used.

## Tail localization and selector search

Epochs 18–20 are aggregated for each identity, direction, and mask. A minimum reproduces donor tail status and has no reproducing proper subset; all incomparable minima are exported. Tail count is exactly 1,024.

Only `JOINT_RECIPIENT_FRAME_LOCAL_ONLY_DONOR` enters selection. Each mask passes a seed only when it reproduces every recovery donor status and preserves every harm recipient status. The same mask must pass both primary seeds for cross-seed localization. Every inclusion-minimal passing mask is reported.

## Residual lattice

Every residual state first installs all donor primitives. Canonical groups and raw causal inputs are:

```text
DECISION_HEAD_CALIBRATION: not_entitled_bias, raw_alpha
PREDICATE_COMPARATOR: predicate availability/condition, raw_alpha_predicate
TEMPORAL_COMPARATOR: temporal availability/condition, raw_alpha_temporal
TEMPORAL_ADAPTER: availability, adapter_logit, final_penalty_scale
TEMPORAL_CHANNEL: availability, channel_logit, preservation_entitlement_prob,
                  gated_penalty_scale
```

Derived transformed parameters, active flags, gate probabilities, effective contributions, deltas, logits, margins, and predictions are recomputed. All 32 masks over 640 states produce exactly 20,480 residual rows.

Residual `00000` must equal primitive `11111` state-by-state. Residual `11111` must equal donor native logits and margin within `1e-6` and prediction exactly over all 640 states. Every fixed residual mask is also tested for donor closure.

Residual Möbius terms use all donor primitives plus recipient residual groups as baseline and all donor primitives plus donor residual groups as full value. Exactly 20,480 rows reconstruct within `1e-6`. Singleton exact logit changes, exact-zero groups, epoch prediction changes, and coalition tail-status changes distinguish continuous from categorical effects.

## Decisions

Rules are evaluated in order: blocked contract failure; cross-seed primitive coalition; seed-specific primitive coalitions; row-specific primitive interaction; then distributed residual when only full residual `11111` closes all donor outputs. No result authorizes training or promotion.

## Outputs and atomic writes

Exactly nine outputs are written:

```text
stage196b2b4_analysis.json
stage196b2b4_report.md
stage196b2b4_primitive_coalition_rows.csv
stage196b2b4_primitive_mobius_terms.csv
stage196b2b4_primitive_tail_summary.csv
stage196b2b4_residual_coalition_rows.csv
stage196b2b4_residual_mobius_terms.csv
stage196b2b4_localization_summary.csv
stage196b2b4_contract.csv
```

Each payload is exclusively created in a temporary file, flushed, synchronized, and atomically renamed. Existing outputs, non-nine-file payloads, and a nonexact final namespace are refused. Contract observations remain structured values in analysis logic; booleans are not stringified before CSV rendering.

## Scientific prohibitions

Do not claim formal causal mediation, external/OOD validity, unfrozen-Mamba validity, training improvement, promotion, a universal selector from seed-specific evidence, a causal role for `polarity_support_margin`, architectural sufficiency beyond the frozen composer, validity from interaction magnitude alone, or categorical selectivity from continuous donor-logit closure.

## Static review policy

Implementation review is static only. No Python execution, compilation, analyzer run, smoke test, training, model/checkpoint loading, Kaggle command, OOD evaluation, commit, or push is authorized.
