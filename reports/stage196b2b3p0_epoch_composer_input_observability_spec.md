# Stage196-B2-B3P0 epoch composer-input observability specification

## Authority and scope

The predecessor decision is `STAGE196B2B3_ADDITIONAL_COMPOSER_OBSERVABILITY_REQUIRED`, with next stage `STAGE196B2B3P0_EPOCH_COMPOSER_INPUT_OBSERVABILITY_DESIGN`, no blockers, and 193/193 contract gates passed. The frozen source graph is authoritative. Existing Stage196-B2-P0 sidecars preserve useful channel probabilities and final SUPPORT/NOT_ENTITLED values, but omit the consumed polarity energies, REFUTE, decision-head parameters and pre-modulation logits, actual mismatch masks, and every outer modulation input/delta. They therefore cannot support exact native recomposition.

This stage adds observations only for the frozen-Mamba Stage196-B1 clean-dev epoch trajectory. It does not generalize to unfrozen Mamba and performs no component swap, counterfactual, donor analysis, fitted recovery, interpolation, or scientific effect estimation.

## Activation and observation point

The only new public trainer arguments are `--stage196b2b3p0-export-epoch-composer-inputs` (store-true, default false) and `--stage196b2b3p0-composer-input-dir` (no default location). Enabling export without the directory is rejected. Supplying the directory without the flag is rejected. The directory must differ from the existing trajectory/B2-P0 directory; existing B2-B3P0 sidecars or manifest are never overwritten.

The observation is requested only on the already-existing clean-dev evaluation forward inside each epoch, after that epoch's optimizer update and in the same evaluation snapshot used by Stage196-B2-P0 and Stage191 trajectory export. No training forward, second dev evaluation, checkpoint reload, optimizer-state change, or post-selection model state is observed.

## Exact source graph

The native graph is:

1. `src/contramamba/modeling_v6b_minimal.py`, `ContraMambaV6BMinimal.forward`, calls `FinalEntitlementDecisionHead.forward` with the actual `frame_prob`, `predicate_coverage_prob`, `sufficiency_prob`, `positive_energy`, and `negative_energy` tensors.
2. `src/contramamba/heads/entitlement_decision.py`, `FinalEntitlementDecisionHead.forward` (lines 36-77 at design time), computes entitlement, learned-alpha NOT_ENTITLED, and tensor order `[REFUTE, NOT_ENTITLED, SUPPORT]`.
3. `ContraMambaV6BMinimal.forward` (decision call near lines 432-443; modulation near lines 494-588 at design time) applies all outer logit mutations before `argmax` prediction.
4. `scripts/train_controlled_v6b_minimal.py`, the epoch clean-dev evaluation and `_stage191_export_epoch` call site, owns exact 720-row identity/order and epoch provenance.

Line spans are audit anchors for this implementation revision; callable and equations are the durable authority.

## Final-modulation source inventory

| Source file | Callable / span | Branch condition | Per-row input | Learned/config parameter | Transformation | Affected logits and sign |
|---|---|---|---|---|---|---|
| `modeling_v6b_minimal.py` | `ContraMambaV6BMinimal.forward`, temporal comparator, ~494-511 | `use_temporal_comparator` and flags not null; row flag boolean | actual `temporal_mismatch_flags` | `alpha_temporal_raw` | `softplus(raw)` | REFUTE `-`, NOT_ENTITLED `+`, SUPPORT `-` on active rows |
| same | predicate comparator, ~512-529 | `use_predicate_comparator` and flags not null; row flag boolean | actual `predicate_mismatch_flags` | `alpha_predicate_raw` | `softplus(raw)` | REFUTE `-`, NOT_ENTITLED `+`, SUPPORT `-` on active rows |
| same | temporal adapter, ~531-554 | adapter logit exists and final scale `> 0` | `sigmoid(temporal_adapter_logit.detach())` | configured final penalty scale | probability times scale | REFUTE `-`, NOT_ENTITLED `+`, SUPPORT `-` |
| same | temporal channel, ~556-588 | channel logit exists and gated scale `> 0`; preservation head required | `sigmoid(channel_logit.detach())`, `1-preservation_entitlement_prob.detach()` | configured gated penalty scale | product of both gates and scale | REFUTE `-`, NOT_ENTITLED `+`, SUPPORT `-` |

The inventory is exhaustive for mutations between `decision["logits"]` and `final_logits.argmax`. Boundary, frame-violation, predicate-isolation, preservation, and temporal diagnostic heads are diagnostic/loss sources unless explicitly consumed by the temporal-channel branch above.

## Row schema and semantics

Every row uses schema `stage196b2b3p0_epoch_composer_inputs_v1`, native logit order `REFUTE`, `NOT_ENTITLED`, `SUPPORT`, and the Stage196-B2-P0 exact normalized string IDs plus integer `dev_position`, `epoch`, and `seed`. Provenance includes run, ownership mode, current commit, trainer SHA, and FrameGate implementation-origin commit.

Causal fields are the consumed primitives, actual native entitlement, epoch decision-head scalars, pre-modulation native logits, actual branch condition inputs/masks, raw and transformed comparator parameters, adapter/channel inputs and effective scales when structurally available, branch-specific three-class deltas, total deltas, and final native logits. Epoch-level learned scalars are duplicated without rounding into every row so a record is independently reconstructable.

Diagnostic fields are entitlement recomputation/error, reconstructed decision/final logits, error maxima, margin error, reconstructed prediction, and equality flag. `polarity_support_margin` remains a predecessor diagnostic only and is never used as `positive_energy` or `negative_energy`. Existing Stage196-B2-P0 names and sidecars are unchanged.

Structurally available but inactive branches emit their actual state, false activity, and exact zero deltas. A structurally unavailable adapter/channel may expose null source logits/probabilities, is marked unavailable in every row and manifest, remains inactive, and emits zero effective contribution. Comparator raw values are never invented when absent.

## Reconstruction equations

For the frozen `explicit_product` decision mode:

```text
E = frame_prob * predicate_coverage_prob * sufficiency_prob
decision_refute = E * negative_energy
decision_not_entitled = not_entitled_bias + softplus(raw_alpha) * (1 - E)
decision_support = E * positive_energy

final_c = decision_c
        + temporal_mismatch_delta_c
        + predicate_mismatch_delta_c
        + temporal_adapter_delta_c
        + temporal_channel_delta_c

final_support_vs_not_entitled_margin = final_support - final_not_entitled
prediction = argmax([final_refute, final_not_entitled, final_support])
```

The adapter magnitude is `sigmoid(adapter_logit) * final_penalty_scale`. The temporal-channel magnitude is `sigmoid(channel_logit) * (1-preservation_entitlement_prob) * gated_penalty_scale`. Each magnitude maps to deltas `[-magnitude, +magnitude, -magnitude]` in native order. Comparator active rows use the same sign vector with their transformed alpha.

## Invariants and atomic publication

All export-only tensors are detached before CPU/scalar extraction and no diagnostic is used by a loss or returned as replacement logits. Each row must satisfy decision-head max absolute error, final max absolute error, and margin absolute error `<= 1e-6`, with identical argmax prediction. Entitlement product error is also fail-closed at `1e-6`. Failure reports run, seed, epoch, stable row ID, field, native value, reconstruction, and error.

Each sidecar is serialized to a unique temporary file in the destination, flushed and fsynced, parsed back, and checked for 720 rows, schema equality, unique identities, and reconstruction invariants before atomic rename. Final names are `stage196b2b3p0_epoch_composer_inputs_001.jsonl` through `_020.jsonl`. Partial temporary files are removed; final artifacts are not overwritten.

Before manifest publication, the implementation reopens exact files 001-020 and requires 720 rows each, one schema, unique identity per epoch, the identical identity set across epochs, 14,400 rows, non-null required causal values, valid structural-null semantics, all error bounds, and 100% prediction equality.

## Run-level manifest

Only after closure, `stage196b2b3p0_composer_input_manifest.json` is atomically published with schema/run/seed/ownership provenance; trainer, model-forward, and decision-head paths and SHA-256 values; FrameGate origin; namespace; expected/observed epochs, rows per epoch, and totals; field inventory; native order; maximum errors; prediction equality rate; branch availability; ordered sidecars and hashes; and `completed: true`. No incomplete manifest is represented as completed.

## Non-interference guarantees

With the flag false, no composer row structures or model-side delta tensors are allocated and the optional model keyword is not passed. With it true, observations reuse the no-grad clean-dev pass. Hooks are not introduced. No random operation, parameter initialization, dropout/mode transition, training forward, backward, optimizer/scheduler action, gradient owner, loss input, model prediction, postprocessing, checkpoint selection, dataset order, epoch count, seed, backbone freeze, or evaluation meaning changes.

## Future six-run execution plan (not executed here)

Future runs are `seed183_joint`, `seed183_frame_local_only`, `seed184_joint`, `seed184_frame_local_only`, `seed185_joint`, and `seed185_frame_local_only`. Their root is `reports/stage196b2b3p0_epoch_composer_input_observability_runs`. Each preserves its corresponding Stage196-B2-P0 configuration exactly, adding only the new export flag and a dedicated run-specific composer directory. The historical root `reports/stage196b2p0_epoch_channel_observability_runs` is never reused or overwritten. Expected closure is six runs, 120 sidecars, and 86,400 rows.

## Prohibited interpretations

These artifacts establish exact native recomposability and observation fidelity only. They do not establish causal component effects, donor validity, mediation, necessity, sufficiency, generalization, calibration, robustness, or behavior under unfrozen Mamba. They must not be interpreted as component swaps, counterfactual outputs, or evidence that the diagnostic predecessor polarity margin is a composer input.
