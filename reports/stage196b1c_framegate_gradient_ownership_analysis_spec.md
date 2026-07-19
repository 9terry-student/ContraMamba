# Stage196-B1-C paired-seed FrameGate gradient-ownership analysis

## Scope

Stage196-B1-C answers one question under the frozen-Mamba envelope: did blocking direct non-frame gradients through FrameGate outputs improve FrameGate-owned trainable parameters and reduce persistent SUPPORT-to-NOT_ENTITLED failure?

The only authorized mechanism is direct downstream gradient ownership through FrameGate outputs while Mamba remains frozen. The analysis does not test unfrozen shared-encoder interference, representation quality outside this envelope, external/OOD behavior, calibration, or production readiness.

## Invocation and execution policy

The analyzer is `scripts/analyze_stage196b1_framegate_gradient_ownership.py`. All five arguments are required:

```text
python scripts/analyze_stage196b1_framegate_gradient_ownership.py \
  --repo-root <repository> \
  --run-root <repository>/reports/stage196b1_framegate_gradient_ownership_runs \
  --stage196a-report-json <authoritative-stage196a-report.json> \
  --current-git-commit <40-character-analysis-commit> \
  --output-dir <empty-output-directory>
```

This document specifies argv-style execution. The analyzer uses argument-vector subprocess execution with `shell=False` only for read-only Git identity inspection. It does not train, load a model or checkpoint, mutate a checkpoint, evaluate external data, search thresholds, calibrate, average parameters, or write outside the requested output directory.

## Exact source closure

The run root is required to resolve exactly to:

```text
reports/stage196b1_framegate_gradient_ownership_runs
```

Its directory entries must be exactly, in the analyzer's canonical order:

1. `seed183_joint`
2. `seed183_frame_local_only`
3. `seed184_joint`
4. `seed184_frame_local_only`
5. `seed185_joint`
6. `seed185_frame_local_only`

No substituted seed, missing arm, additional directory, or line-order-only pairing is accepted. Runtime provenance must close to commit `9835cbbf86d83aca0964821669e63f7f6deb1c59`, Mamba, `state-spaces/mamba-130m-hf`, `v6b_minimal`, CUDA, 20 epochs, split seed 174, frozen encoder and A_log, non-trainable fully isolated shared encoder, and isolation source `frozen_runtime_configuration`. The intervention must not change encoder freeze state.

Every run must declare direct frame loss active at weight 1.0, unchanged forward values, and the exact mode-specific blocking contract:

| Mode | `framegate_nonframe_output_gradient_blocked` |
|---|---|
| `joint` | `false` |
| `frame_local_only` | `true` |

Compatible-positive margin, Stage195 parameter SWA, state capsules, external evaluation, external training, bridge training, time swap, and external threshold metrics must be absent or false as appropriate. Each run requires exactly 720 selected predictions, 720 scalar rows, 20 epoch prediction exports of 720 rows each, and 20 trajectory-ledger rows. The selected best epochs are provenance invariants (20, 20, 18, 13, 20, 20 in canonical run order), but best-epoch movement does not enter the causal rule.

## Stage196-A source closure

`--stage196a-report-json` supplies the authoritative prior localization; no Stage196-A report path is hard-coded. The analyzer requires its runnable local-channel decision, empty blockers, 127 persistent rows, and native channel threshold 0.5. It verifies the reported 72 `MULTI_LOCAL_CHANNEL_FAILURE` plus 55 `FRAME_ONLY_FAILURE` rows and the fact that those buckets close to all 127 persistent SUPPORT negatives.

The analyzer resolves the report's sibling source-closure and recurrence companions. Every source-closure gate must pass. Recurrent sets are reconstructed from the authoritative recurrence rows, never replaced with embedded position lists. Required recovered set cardinalities are:

| Set | Positions |
|---|---:|
| baseline recurrent | 22 |
| intervention recurrent | 19 |
| common recurrent | 19 |
| universal all-six | 10 |

The known universal positions `24, 25, 72, 108, 276, 324, 359, 479, 503, 635` are used only as a verification invariant after report-based recovery. Failure to recover any set unambiguously produces `STAGE196B1C_ANALYSIS_INCOMPLETE`.

## Row alignment and resolved schema

The primary alignment key is the explicit original row identifier:

```text
clean_dev_predictions.json: predictions[].id
clean_dev_scalars.jsonl:    [].id
trajectory JSONL:           [].source_row_id
```

`trajectory[].dev_position` is a certified stable position and is used for Stage196-A position membership only after the identifier mapping is validated. Incidental JSON/JSONL order is never a join key.

Every selected export and every epoch must contain 720 unique identifiers. Every epoch must contain every certified position 0 through 719 exactly once. The ID-to-position map may not drift across epochs or runs. All six runs must have identical ID populations, claims, evidence, gold labels, pair IDs, and intervention metadata. Prediction/scalar duplicates are compared within normal serialization tolerance.

The statically resolved semantic fields are:

| Meaning | Selected export | Epoch export |
|---|---|---|
| gold final label | `gold_final_label` | `gold_final_label` |
| predicted final label | `pred_final_label` | `predicted_final_label` |
| frame probability | `frame_prob` | `sigmoid(frame_logit)` |
| predicate coverage probability | `predicate_coverage_prob` | unavailable |
| sufficiency probability | `sufficiency_prob` | unavailable |
| SUPPORT score | `final_probs[2]` | `final_logits[2]` |
| NOT_ENTITLED score | `final_probs[1]` | `final_logits[1]` |
| intervention type | `intervention_type` | joined by stable ID |

Selected `final_probs` and epoch `final_logits` use canonical order `REFUTE, NOT_ENTITLED, SUPPORT`. Canonical argmax labels are recomputed. The selected export must agree with its trainer-selected trajectory epoch; it is not silently replaced by epoch 20.

## Selected checkpoint and tail-three separation

Selected-checkpoint metrics use `clean_dev_predictions.json`, whose metadata and row values must match the trainer-selected epoch. They include accuracy, macro-F1, per-label recall, SUPPORT precision, false-NOT_ENTITLED, false entitlement, polarity errors, prediction counts, and mean frame probability by gold label, intervention type, and each Stage196-A recurrent set.

Tail-three behavior always uses exact epochs 18, 19, and 20. A persistent stable SUPPORT negative is:

```text
gold == SUPPORT
and prediction(epoch 18) == NOT_ENTITLED
and prediction(epoch 19) == NOT_ENTITLED
and prediction(epoch 20) == NOT_ENTITLED
```

Stable SUPPORT-correct rows predict SUPPORT at all three epochs. Remaining gold SUPPORT rows are unstable. Persistent REFUTE polarity errors are gold REFUTE predicted SUPPORT at all three epochs. Persistent false entitlement is definable as gold NOT_ENTITLED predicted an entitled label (`REFUTE` or `SUPPORT`) at every tail epoch.

## Baseline-defined stable-correct controls

For each seed, controls are defined only from the joint arm: gold SUPPORT and SUPPORT predictions at epochs 18, 19, and 20. The paired intervention is evaluated on those exact identifiers. Outcomes are mutually exclusive: preserved SUPPORT at all three epochs, changed to NOT_ENTITLED at all three, changed to REFUTE at all three, or unstable/mixed. Preservation rate is preserved divided by the baseline-defined population. Intervention outcomes never redefine the population.

## Frozen local-channel localization

Persistent SUPPORT negatives are localized using the selected-checkpoint scalar snapshot and Stage196-A's native threshold 0.5 for frame, predicate, sufficiency, and entitlement aggregation. Polarity uses the native exported SUPPORT-facing margin boundary (`polarity_margin >= 0`), not a searched threshold. Final composition passes only when the selected final prediction equals gold.

Bucket precedence preserves Stage196-A terminology while adding the required polarity/unresolved distinctions:

1. two or more failed local channels: `MULTI_LOCAL_CHANNEL_FAILURE`
2. exactly one failed local channel: `FRAME_ONLY_FAILURE`, `PREDICATE_ONLY_FAILURE`, or `SUFFICIENCY_ONLY_FAILURE`
3. local channels pass but entitlement aggregation fails: `AGGREGATION_FAILURE`
4. preceding channels pass but polarity fails: `POLARITY_ONLY_FAILURE`
5. preceding channels pass but final label fails: `FINAL_COMPOSITION_FAILURE`
6. otherwise: `UNRESOLVED`

FrameGate failure count is the number of persistent SUPPORT negatives with failed `frame_prob`, independent of the mutually exclusive bucket label.

## Recurrent-position effects and trajectories

The baseline recurrent, intervention recurrent, common recurrent, and universal all-six sets remain separate. For every set, seed, and position the analyzer emits selected joint/intervention frame probabilities and delta; tail-three mean joint/intervention frame probabilities and delta; both tail prediction patterns; rescue; and previously-correct harm.

The epoch trajectory contains exactly 120 rows (`3 × 2 × 20`). Each row reports accuracy, macro-F1, SUPPORT recall, three error counts, mean frame probability on gold SUPPORT, mean frame probability on the union of persistent joint-arm positions, mean frame probability on Stage196-A common recurrent positions, and a tail-membership precursor. The precursor is not meaningful before epoch 18; at epochs 18–20 it means gold SUPPORT and NOT_ENTITLED at every available member of the fixed tail prefix.

## Paired estimands

Every delta is `frame_local_only - joint` and remains visible for seeds 183, 184, and 185. The CSV also includes mean, median, and positive/zero/negative direction counts. Error-count rows state that a negative delta is improvement. Metrics are selected accuracy, macro-F1, SUPPORT recall, false-NOT_ENTITLED, false entitlement, polarity error, persistent stable SUPPORT negatives, FrameGate failures among those negatives, stable-correct SUPPORT preservation, and selected mean frame probability on Stage196-A common and universal sets.

## Precommitted decision logic

The analyzer can select exactly one of:

```text
STAGE196B1C_SUPPORTS_DIRECT_FRAMEGATE_GRADIENT_INTERFERENCE
STAGE196B1C_DOES_NOT_SUPPORT_DIRECT_FRAMEGATE_GRADIENT_INTERFERENCE
STAGE196B1C_MIXED_GRADIENT_OWNERSHIP_EFFECT
STAGE196B1C_ANALYSIS_INCOMPLETE
```

Support requires all six gates: persistent SUPPORT negatives lower in at least two seeds and never higher; FrameGate failures lower in at least two and never higher; common-recurrent frame probability higher in at least two and never lower; full stable-correct preservation in at least two seeds; false entitlement not increased in at least two; and polarity error not increased in at least two.

If support is not met, mixed is selected for material paired-direction conflict, persistent rescue without coherent recurrent frame-probability improvement, multi-seed FrameGate reduction with multi-seed false-entitlement or polarity degradation, selected/tail direction disagreement, or one-seed dominance of the three-part primary signature. Integrity-passing results with neither the complete support signature nor a mixed condition select does-not-support. Missing artifacts, provenance failure, alignment ambiguity, unresolved schema, Stage196-A recovery failure, or contract failure select incomplete and authorize no scientific inference.

## Output closure

The output directory must be absent or empty. The analyzer creates exactly:

1. `stage196b1c_analysis.json`
2. `stage196b1c_report.md`
3. `stage196b1c_run_summary.csv`
4. `stage196b1c_paired_seed_deltas.csv`
5. `stage196b1c_tail3_persistent_rows.csv`
6. `stage196b1c_recurrent_position_effects.csv`
7. `stage196b1c_epoch_trajectory.csv`
8. `stage196b1c_contract.csv`

An analysis failure still writes this exact closure: an incomplete JSON/Markdown report, empty data CSVs with headers, and the accumulated failing contract ledger. Runtime or analysis failure is never converted into scientific evidence.

## Interpretation and next stage

A support decision authorizes only: under frozen Mamba, direct non-frame gradients through FrameGate outputs interfered with FrameGate-owned trainable parameters. It recommends `STAGE196B2_PROMOTE_FRAME_LOCAL_GRADIENT_OWNERSHIP` for architectural/training-contract promotion analysis only.

A does-not-support decision authorizes only: under frozen Mamba, direct FrameGate-output gradient interference was not supported as the primary cause. It recommends `STAGE196B2_RETURN_TO_FRAME_REPRESENTATION_HYPOTHESIS`; this permits design of a separately testable representation intervention but does not implement one.

A mixed decision recommends `STAGE196B2_NO_PROMOTION_TARGETED_CAUSAL_FOLLOWUP` and records the smallest unresolved causal question. An incomplete decision recommends `STAGE196B1C_REPAIR_ANALYSIS_INPUTS`.

No decision automatically authorizes retraining, a new loss, a model or trainer change, threshold search, calibration, SWA, data mutation, external/OOD evaluation, or production advancement.
