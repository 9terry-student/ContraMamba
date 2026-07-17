# Stage191-D late trajectory phase-flip and state-path diagnostic specification

## Scope and immutable inputs

Stage191-D is an existing-artifact diagnostic over the six completed Stage191-B deterministic replays. It never trains, evaluates external or OOD data, constructs a model, changes a checkpoint, selects a replacement model, or authorizes model advancement.

The frozen Stage191-B source commit is `0872e66ccb05ae8a166f5cabf4e084272dc49500`. The input directory must resolve exactly to `<repo-root>/reports/stage191b_deterministic_replay_manifest_20260717_153524`; matching only its basename is insufficient. The required Stage191-B decision is `STAGE191B_DETERMINISTIC_REPLAY_MANIFEST_READY`. The only accepted runs are baseline and intervention for training seeds 174, 175, and 176, all with split seed 174.

The analyzer takes explicit `--repo-root`, `--stage191b-dir`, `--current-diagnostic-git-commit`, and `--output-dir` arguments. It does not discover a timestamped input or output directory and does not import project modules.

## Safe output establishment

Before creating or writing the output directory, the analyzer requires its resolved parent to be exactly `<repo-root>/reports`, its basename to start with `stage191d_trajectory_phase_flip_`, and the directory to be absent or empty. It rejects the Stage191-B directory, descendants of Stage191-B, and paths in or below `scripts`, `data`, or `src`. Because an accepted output is an immediate child of `<repo-root>/reports`, it cannot be inside any nested frozen Stage189/190/191 input directory. A nonempty output is never overwritten. Failure to establish a safe output produces no blocked report in the unsafe location, prints the exception traceback or an explicit diagnostic to standard error, and returns nonzero.

After a safe output directory is established, any analysis exception produces the fail-closed blocked report and all fixed-header ledgers in that directory.

## Diagnostic source-commit identity

Git is used only for read-only provenance identity and never for mutation. The supplied diagnostic commit must be exact 40-character lowercase hexadecimal, repository `HEAD` must equal it, and the current bytes of both `scripts/analyze_stage191d_trajectory_phase_flip.py` and `reports/stage191d_trajectory_phase_flip_spec.md` must equal their blobs at that commit. Both files must have neither staged nor unstaged differences. Subprocess calls use argument arrays, `shell=False`, captured output, and fail closed. No globally clean worktree is required. The report records current-file and commit-blob SHA256 for both files.

## Stage191-B independent identity and Stage191-C closure

The Stage191-B main report must have empty blockers, diagnostic replay only, runnable and replay-authorized state, no training-for-advancement authorization, no model-advancement decision, no external data, authorized seeds `[174, 175, 176]`, the frozen Stage191-B commit, and exactly these six run/seed/arm/split identities. Its selected prediction reference identity keys must be exactly the six run names.

The frozen selected-checkpoint SHA256 map is:

- `seed174_baseline`: `8e31dbd1459a67e65571ea1926a6e1a5f49f1ae2e57deb8455b41617f9ed972c`
- `seed174_intervention`: `66cfb4fd91c29dfc6d4f243e701103e6b82e6ccac810a3b7a17d5b05310a57b3`
- `seed175_baseline`: `5baa306161f204dbc984681a0b18c22484e3724aea4bfbdc9858e6f434ea1c0a`
- `seed175_intervention`: `c00700d170e11fcc0376e2fe0ca7bc8037a76330330ed1b99fa21a35775a8018`
- `seed176_baseline`: `8bcae6880e68cf8f34b9fb86f3f987e26872511636af33a64f6cc012857e51ea`
- `seed176_intervention`: `539ff0c226e6f862abf99c9b4ceaf883989cee9985c5a5cf438801ecb87620a5`

The main report map must equal this map, and every per-run manifest `original_selected_checkpoint_sha256` must equal its frozen value. For each run, `original_stage189_run_directory` must be an exact existing directory; its selected checkpoint must resolve exactly to `<original_stage189_run_directory>/selected_checkpoint.pt`, be a regular file, and be rehashed. That current artifact SHA256 must equal the frozen map, the per-run manifest value, and the Stage191-B main-report map. The dedicated gate is `original_selected_checkpoint_artifact_exact`; equality between JSON strings is not artifact proof. Each per-run original selected prediction path and SHA must equal the corresponding `validated_prediction_path` and `validated_prediction_sha256` in the main report identity, whose `passed` flag must be exactly true. Equality between mutable per-run fields alone is not identity proof.

Each per-run manifest must have exact run, seed, arm, split, selected epoch, commit, authorization, internal-only arguments, and expected cardinalities. Each replay directory contains exactly one contract, one 20-row trajectory ledger, 20 enumerated epoch prediction exports, 20 enumerated capsules, `training_report.json`, and `clean_dev_predictions.json`. No fuzzy matching or fallback path is allowed.

The contract must show canonical label order `REFUTE`, `NOT_ENTITLED`, `SUPPORT`, source commit `0872e66ccb05ae8a166f5cabf4e084272dc49500`, authorized seeds, exact identity, both observability flags, unchanged training semantics, no extra forward, no `loss_logits`, and no external data. Original and replay argv are rejected for external/OOD/bridge paths, enable flags, or non-`none` Stage57/66/75/80A bridge modes.

The intervention sidecar must resolve exactly to `<repo-root>/reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl`, exist as a regular file, and independently recompute to semantic SHA256 `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc` using the Stage185 canonical JSON contract that excludes `created_at`. Configured and runtime expected/observed semantic hashes must equal the same value. Baseline requires exact disabled, unaccessed, pathless, hashless state.

Replay clean history must equal the original Stage189 `runs.single.v7_epoch_diagnostic_history` exactly. Compatible-positive-margin `epoch_metrics` must also be exactly equal. Historical macro-F1 and normalized canonical prediction counts must equal every trajectory epoch. Selected and final epochs are exact non-bool integers.

### Exact selected prediction artifact

The complete original and replay `clean_dev_predictions.json` values are read and must satisfy exact Python structural equality:

`original_prediction_artifact == replay_prediction_artifact`

This includes metadata, list order, every row field, and every value. Both artifacts must still contain exactly 720 prediction objects with exact canonical `gold_final_label` and `pred_final_label`. The selected trajectory export label pairs must equal the clean selected prediction pairs in fixed row order. The dedicated closure gate is `selected_clean_prediction_artifact_exact`; subset-field equality cannot satisfy it.

Any failed Stage191-C or provenance gate blocks all Stage191-D interpretation.

## Epoch export reconstruction

Every Stage191 epoch export has exactly 720 rows with exact non-bool integer epoch and `dev_position` 0 through 719, canonical gold/prediction labels, `final_logits` as an exact length-three list of finite non-bool numeric values, finite non-bool numeric `final_ce`, and a present `frame_logit` that is either null or finite non-bool numeric. `source_row_id` remains optional auxiliary metadata and is never required or synthesized.

From all 720 rows the analyzer reconstructs mean `final_ce`, dense canonical prediction counts, clean accuracy, macro-F1, SUPPORT recall, false entitlement, false not-entitled, and polarity error. Reconstructed mean CE and other floating metrics use the same fixed `1e-7` absolute/relative tolerance. Reconstructed count/error fields are exact. Trajectory false-entitlement, false-not-entitled, polarity-error, and all three prediction counts must be exact non-bool integers. At every epoch the gold total is exactly 720 and gold SUPPORT is exactly 89. The equivalence ledger records per-epoch reconstructed CE evidence under `reconstructed_clean_dev_ce_matches`.

## Output trajectory analyses

Per-run epoch rows retain clean CE, accuracy, macro-F1, SUPPORT recall, both entitlement error totals, polarity errors, all prediction counts, selected epoch, and replacement status. Paired rows are intervention minus baseline. At epochs 19 and 20 sign is exactly `-1`, `0`, or `1`; zero is unsigned.

Within-run transitions use exact aligned `dev_position` for 18-to-19, 19-to-20, and selected-to-final. Paired transitions use baseline as previous and intervention as next for all 20 epochs. Every summary contains the full canonical 3-by-3 matrix, unchanged/changed rows, both NOT_ENTITLED/SUPPORT directions, REFUTE-involved changes, exclusive boundary fraction, corrections, regressions, and wrong-to-different-wrong changes. A correction changes incorrect to gold; a regression changes gold to incorrect; wrong-to-different-wrong changes between two distinct incorrect labels. Gold-conditioned summaries use identical definitions.

For epochs 15 through 20, total variation is the sum of absolute consecutive changes, amplitude is max minus min, and direction reversals compare consecutive nonzero directions after ignoring zero steps. Selected-to-final drift is final minus selected. Statistics cover SUPPORT count, NOT_ENTITLED count, SUPPORT recall, and false-entitlement total, with paired arm comparisons.

## Authoritative Stage190 source and parameter-inventory contract

The authoritative Stage190 diagnostic commit is exactly `ac0b9032b94436ce8ac8134c650d389134faebd4`. Using read-only subprocess calls with argument arrays and `shell=False`, the analyzer reads `ac0b9032b94436ce8ac8134c650d389134faebd4:scripts/run_stage190b_gradient_conflict_diagnostic.py` through `git show`. It computes the frozen commit-blob SHA256 and current working-file SHA256, requires exact byte equality, and requires no staged or unstaged difference for that source file. It also verifies the exact grouping-source snippets. A manually entered SHA is never accepted without proving it is the frozen commit blob. The report records `authoritative_stage190_diagnostic_commit`, `authoritative_stage190_grouping_source_path`, `authoritative_stage190_grouping_commit_blob_sha256`, `observed_stage190_grouping_source_sha256`, `current_source_equals_frozen_commit_blob`, `unstaged_clean`, `staged_clean`, and `source_contract_passed`.

The authoritative Stage190-A directory is exactly `<repo-root>/reports/stage190a_gradient_conflict_manifest_20260717_113644`; the analyzer uses this frozen relative path and never discovers a timestamped directory. For each exact Stage191-D run it reads only `stage190a_<run>_manifest.json`, validates the exact manifest filename/run identity, seed, training seed, arm, split seed, frozen Stage190 commit, selected-checkpoint SHA, `runnable=true`, and `blocking_reasons=[]`, and parses a separately tokenized `--output-dir` from the manifest argv. That directory must be the exact `stage190b_<run>` child named by the manifest.

Each resolved Stage190-B directory must contain `stage190b_gradient_report.json` and `stage190b_parameter_inventory.csv`. The report must have decision `STAGE190B_GRADIENT_DIAGNOSTIC_EXPORTED`, empty blockers, exact run identity fields, and frozen diagnostic commit and checkpoint identity. Its `parameter_group_contract` must be a dict with the exact ordered five conceptual groups, `disjoint=true`, `exhaustive=true`, and `zero_size_justifications` as a dict with exactly those five group keys and string values. Where Stage190 exposes selected-checkpoint identity, it must equal the frozen run checkpoint SHA.

The inventory contains only actual `requires_grad=True` parameters. Its header, row order, unique parameter-name set, shape, numel, dtype, group, report ordering SHA, parameter count, and total numel are validated exactly. Every row group must belong to the five conceptual groups and receives exactly one fixed Stage191-D alias; unknown groups are rejected. A conceptual group may have zero inventory rows. For each conceptual group, the analyzer computes its exact inventory row count. A nonzero group must have justification `nonzero module-owned group`; a zero-row group must have justification `zero-size because the selected checkpoint has no trainable parameters owned by this conceptual module set`. Missing, extra, non-string, or row-count-inconsistent justifications block.

All six inventories must expose the same ordered canonical parameter-name, topology, and group mapping, which may legitimately omit a zero-size conceptual group. They must also have identical five-group row-count dictionaries and identical zero-size group sets. Inventory paths, file SHA256 values, mapping SHA256 values, conceptual-contract results, row counts, nonzero and zero-size group sets, justifications, topology results, and per-run mapping status are recorded.

The exact Stage190 conceptual groups and Stage191-D reporting aliases are:

- `frame_head` -> `frame_head`
- `decision_head` -> `decision_head`
- `router_and_epistemic_heads` -> `router_or_epistemic`
- `backbone` -> `backbone`
- `other_trainable` -> `other`

For every Stage191 capsule, its ordered trainable parameter names must equal its authoritative Stage190 inventory exactly, and shape, numel, and dtype metadata must be compatible. Reporting groups are assigned only from actual inventory rows. No ownership is synthesized for a zero-size conceptual group, no parameter-name prefix inference exists, and an unknown parameter is never silently assigned to `other`. The exhaustive/disjoint ownership gate is grounded in equality with the canonical Stage190 inventory mapping. Any source, inventory, topology, alias, overlap, omission, justification, row-count, zero-size-set, or six-run mapping mismatch blocks.

Every state step still emits all five Stage191-D reporting groups. A zero-size group emits squared and ordinary L2 step norm `0.0`, parameter count `0`, tensor count `0`, and squared-step fraction `0.0` whenever the total step norm is nonzero. State displacement remains descriptive and cannot establish causal responsibility.
## Capsule integrity and state path

Capsules are loaded only by `torch.load(path, map_location="cpu", weights_only=True)`. Absence of tensor-only support blocks; there is no unsafe fallback. No model is constructed.

Every capsule must match epoch, training seed, split seed, arm, ordered parameter and buffer names, disjoint key sets, dtype/shape metadata, trainer-compatible scalar-safe state hashes, and trajectory file SHA. Consecutive steps produce float64-accumulated trainable squared/L2 norm, normalization by prior state norm, buffer norm, and cosine to the preceding step where defined. Epochs 15 through 20 produce path length, direct displacement, and ratio. The exact five reporting groups retain squared norm, norm, total-squared fraction, parameter numel, and tensor count per step.

## Precommitted decisions and restrictions

The decision is exactly one of:

1. `STAGE191D_TRAJECTORY_DIAGNOSTIC_BLOCKED`
2. `STAGE191D_LATE_SUPPORT_NE_PHASE_FLIP_CONFIRMED`
3. `STAGE191D_LATE_CLASS_REDISTRIBUTION_WITHOUT_PHASE_FLIP`
4. `STAGE191D_TRAJECTORY_EFFECT_INCONCLUSIVE`

Confirmed phase flip requires all six closure passes; nonzero opposite SUPPORT-count delta signs at epochs 19 and 20 for all seeds; nonzero opposite false-entitlement delta signs for all seeds; absolute REFUTE-count and polarity-error deltas at most one in all late cells; and at least 95% exclusive NOT_ENTITLED/SUPPORT changed rows from epoch 19 to 20 in exactly the two intervention runs selected at epoch 19. A zero denominator fails.

Late redistribution without phase flip requires closure, failure of the full phase-flip conjunction, and for every seed at epochs 19 and 20 at least one paired change, at least 95% exclusive NOT_ENTITLED/SUPPORT changes, absolute REFUTE-count delta at most one, and absolute polarity-error delta at most one. Otherwise the result is inconclusive.

The phase-flip and redistribution conjunctions are mutually exclusive decision alternatives, not simultaneous universal gates. The decision ledger records them as `phase_flip_condition_observed` and `redistribution_condition_observed` with `required=decision_alternative`; a normal non-selected alternative is not reported as a failed required gate. The universal gate `selected_decision_matches_precommitted_taxonomy` passes only when the selected decision exactly matches the observed alternative: confirmed for phase flip, redistribution only when phase flip is false and redistribution is true, and inconclusive only when both are false.

The analyzer may describe checkpoint-phase sensitivity, late SUPPORT/NOT_ENTITLED redistribution, trajectory instability, and descriptive state-path concentration. It may not claim causal group responsibility, statistical significance, generalization beyond three paired seeds, external performance, or model advancement. Confirmed phase flip recommends only Stage192 design of a trajectory-stabilized entitlement mechanism or selection rule. Stage192 training is never authorized.

## Outputs and fail-closed behavior

The exact JSON/Markdown reports and twelve prescribed CSV ledgers are retained. On an analysis exception after safe output establishment, the report preserves exception type, message, and traceback, emits only `STAGE191D_TRAJECTORY_DIAGNOSTIC_BLOCKED`, and writes available or empty fixed-header ledgers. No fallback input/output directory, inferred row, substituted metric, unsafe capsule load, training authorization, or positive decision is permitted.
