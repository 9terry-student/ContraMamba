# Stage193 fresh-seed tail3 replication manifest and analysis specification

## Scope and frozen hypothesis

Stage193 is a diagnostic-only fresh-seed replication of the descriptive Stage192-A temporal logit-smoothing signal. Stage192-A closed checkpoint selection with `STAGE192A_NO_TRAJECTORY_STABLE_SELECTOR`; its old seeds are frozen historical evidence and are not reinterpreted as independent validation. Stage193 tests the precommitted tail3 hypothesis on training seeds 177, 178, and 179 with split seed 174. This is a hypothesis test, not hyperparameter tuning.

The historical independent comparator had mean pair macro-F1 0.832347, mean pair accuracy 0.880556, SUPPORT-delta range 119.333333, and false-entitlement-delta range 76.666667. The descriptive tail3 comparator had mean pair macro-F1 0.821919, mean pair accuracy 0.883565, SUPPORT-delta range 44, false-entitlement-delta range 25, maximum absolute REFUTE delta 0, and maximum absolute polarity delta 0.

No Stage193 decision authorizes model advancement, subsequent training, or any training outside the exact six-run Stage193-B diagnostic matrix.

## Frozen identities

- Frozen trainer blob commit (`TRAINER_BLOB_COMMIT`): `e83d8af756fa84b7a91c14e0910ae388b07b5f02`. This identity proves only the bytes of `scripts/train_controlled_v6b_minimal.py`; it is not the Stage193 runtime repository commit.
- Stage192-A implementation commit: `a768d848256f88a7a1a15cc02a058f4d7d0a35f7`.
- Stage191-B replay implementation commit: `0872e66ccb05ae8a166f5cabf4e084272dc49500`.
- Stage185 semantic sidecar SHA256: `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`.
- Stage191-B directory basename: `stage191b_deterministic_replay_manifest_20260717_153524`.
- Stage185 sidecar directory basename: `stage185a_controlled_train_integrity_sidecar_20260715_141914`.

Every timestamped input directory is supplied explicitly. Neither program performs fuzzy or latest-directory discovery.

## Stage193-A inputs and path safety

The builder requires `--repo-root`, `--stage191b-dir`, `--stage192a-dir`, `--stage185-sidecar-dir`, `--stage193b-run-root`, `--current-diagnostic-git-commit`, and `--output-dir`.

The output is an absent or empty immediate child of `<repo-root>/reports` whose basename begins `stage193a_tail3_fresh_seed_manifest_`. The planned Stage193-B root is an absent or empty immediate reports child whose basename begins `stage193b_tail3_fresh_seed_runs_`. The two paths differ from one another and from every frozen input. The builder records planned paths but does not create the Stage193-B root or any run directory.

The supplied diagnostic commit is lowercase hexadecimal length 40, equals HEAD, contains byte-identical blobs for this specification, the builder, and the analyzer, and has no staged or unstaged differences for those files. It is frozen as `stage193_runtime_repository_commit`: the future commit containing the committed Stage193 specification, builder, and analyzer, and the exact commit at which Stage193-B must run. Git inspection is read-only, argument-array based, and `shell=False`; global worktree cleanliness is not required. Independently, current trainer bytes and their SHA256 must equal `git show e83d8af756fa84b7a91c14e0910ae388b07b5f02:scripts/train_controlled_v6b_minimal.py`. Runtime Git provenance is therefore expected to equal the Stage193-A/C implementation commit, while trainer bytes and SHA are expected to equal the frozen e83d8af blob.

## Frozen input closure

The builder validates the exact Stage191-B basename, READY closure, frozen implementation commit, six historical manifests, ordered identities, and template-bearing argv. It validates the explicitly supplied Stage192-A immediate reports child, its exact fourteen outputs, no-selector closure, empty selector sets, null winner, all-true global closure, and frozen implementation commit. It validates the exact Stage185 directory basename and recomputes the semantic SHA over canonical JSONL rows excluding `created_at`, using sorted keys and compact UTF-8 JSON.

## Frozen six-run matrix and argv derivation

The run order is exactly:

1. `seed177_baseline`
2. `seed177_intervention`
3. `seed178_baseline`
4. `seed178_intervention`
5. `seed179_baseline`
6. `seed179_intervention`

Canonical labels are `REFUTE`, `NOT_ENTITLED`, `SUPPORT`. The trainer path is `scripts/train_controlled_v6b_minimal.py`; its bytes are frozen by `TRAINER_BLOB_COMMIT`, while execution provenance is frozen separately by `stage193_runtime_repository_commit`.

Baseline and intervention argv templates are derived from all six authoritative Stage191-B replay manifests. Within each arm, normalization may remove only source seed, source output paths, and the two Stage191 observability flags; every normalized template must be identical. Generated argv preserve all other tokens and order. Allowed mutations are limited to the new training seed, Stage193 output/run identity paths, removal of `--stage191-trajectory-replay-observability` and `--stage191-save-trajectory-state-capsules`, and addition of `--stage193-tail3-fresh-seed-observability`. The frozen `v6b_minimal` temporal comparator remains enabled and is not a Stage193 treatment.

Every argv preserves `v6b_minimal`, Mamba, CUDA, `state-spaces/mamba-130m-hf`, `data/controlled_v5_v3_without_time_swap.jsonl`, explicit split seed 174, 20 epochs, `final_macro_f1` selection, and all frozen optimizer, scheduler, batching, maximum-length, loss, margin, reporting, and checkpoint semantics. Smoke, truncation, loss sweep, external/OOD/bridge/synthetic data, temporal auxiliary data/loss/adapter/channel/cap/constrained selection, and state capsules are absent. Final CE uses `output["logits"]` only.

Baseline has compatible-positive margin weight 0 and no sidecar options. Intervention has weight 0.05, margin logit 0, the exact Stage185 JSONL path, and the frozen semantic SHA.

A READY manifest authorizes only execution of these exact six diagnostic runs. It does not authorize model advancement or further training.

## Stage193-A outputs and failure behavior

The builder writes exactly six files: the JSON and Markdown reports, `stage193a_run_manifest.jsonl`, `stage193a_run_command_matrix.csv`, `stage193a_source_and_template_gate.csv`, and `stage193a_precommitted_gate.csv`. JSONL contains exactly six ordered rows when READY. The report and every JSONL row separately record `trainer_blob_commit`, `trainer_blob_sha256`, and `stage193_runtime_repository_commit`; the command matrix exposes `trainer_blob_commit`, `trainer_blob_sha256`, and `runtime_repository_commit` and has no ambiguous `trainer_source_commit` column. Each row also freezes argv, planned output and trajectory paths, twenty prediction paths, zero capsules, arm contract, and diagnostic-only authorization. Report and manifest integer identities/cardinalities use strict `type(value) is int` closure, so booleans never satisfy zero or any other integer contract.

Before safe output establishment, failure writes nothing. Afterwards, any exception produces `STAGE193A_TAIL3_FRESH_SEED_MANIFEST_BLOCKED`, all fixed-header outputs, and a nonzero return. Otherwise the decision is `STAGE193A_TAIL3_FRESH_SEED_MANIFEST_READY`.

## Stage193-C inputs and artifact-only validation

The analyzer requires explicit `--repo-root`, `--stage193a-dir`, `--stage193b-run-root`, `--stage192a-dir`, `--current-diagnostic-git-commit`, and `--output-dir`. Its output is an absent or empty immediate reports child whose basename begins `stage193c_tail3_fresh_seed_replication_`.

It validates the exact six Stage193-A outputs and READY closure, including the three distinct Stage193-A identity fields, then independently validates each planned run. The current trainer must remain byte-identical to the blob at `TRAINER_BLOB_COMMIT`, and the current Stage193-C source commit must equal the Stage193-A `stage193_runtime_repository_commit`. Runtime provenance must contain argv exactly equal to the manifest; `source_provenance.git_commit` and the trajectory contract `trainer_source_commit` must equal `stage193_runtime_repository_commit`, never `TRAINER_BLOB_COMMIT`, while both runtime trainer SHA records must equal `trainer_blob_sha256`. Contract identity must also show mode `stage193_tail3_fresh_seed_replication`, Stage191 observability false, authorized seeds `[177,178,179]`, split seed 174, 20 epochs, 720 dev rows, zero capsules, `output["logits"]`, reused Stage191 implementation, no extra forward, no loss logits, unchanged training semantics, and no external data.

Each run has exactly twenty ledger rows for epochs 1 through 20 and exactly twenty enumerated prediction exports. Export paths and SHA256 values, 720 rows, positions 0 through 719, canonical labels, finite three-logit vectors, finite row CE, canonical argmax predictions, and gold alignment across every epoch and all runs are exact. Metrics are reconstructed independently. Clean CE is the exact CPU `torch.float32` mean of ordered row CE values; count metrics are exact and accuracy, macro-F1, and SUPPORT recall use tolerance `1e-7`. Every authorized seed, training seed, split seed, epoch, dev position, selected epoch, row/cardinality field, prediction count, gold count, and confusion-matrix cell is accepted only when `type(value) is int`; booleans never satisfy an integer contract, including zero. The authoritative normal training-report paths are `training_report.runs.single.best_epoch` and `training_report.runs.single.final_epoch`; the report root must be an object, `runs` must be an object with the exact key set `{"single"}`, and `runs.single` must be an object. `training_report.runs.single.best_epoch` and `run_provenance.finalization.selected_epoch` must be exact non-bool integers in 1 through 20 and equal one another. `training_report.runs.single.final_epoch` must be an exact non-bool integer equal to 20. When `run_provenance.finalization.completed_epochs` is present in the frozen provenance schema, it must also be an exact non-bool integer equal to 20. No root-level selected-epoch fallback or fuzzy schema discovery is permitted. No checkpoint, model, or capsule is loaded, and zero capsule files may exist.

## Comparators and metrics

`independent_selected` reads each run's export at its trainer-selected epoch without loading a checkpoint. `tail2_mean_logits` averages epochs 19 and 20 row logits in float64. `tail3_mean_logits` averages epochs 18, 19, and 20 row logits in float64. Canonical label order resolves exact argmax ties. Tail2 is descriptive and cannot produce a positive decision.

For every comparator and seed, the analyzer records baseline and intervention clean CE, accuracy, macro-F1, SUPPORT recall, false entitlement, false not-entitled, polarity error, and three prediction counts, plus intervention-minus-baseline deltas. It records full canonical transition matrices, changed rows, corrections, regressions, wrong-to-different-wrong, REFUTE-involved changes, exclusive SUPPORT/NOT_ENTITLED transition fraction, and gold-conditioned summaries.

Fresh aggregates over seeds 177-179 record mean pair CE, accuracy, macro-F1, SUPPORT recall; SUPPORT-delta minimum, maximum, and range; false-entitlement-delta minimum, maximum, and range; maximum absolute REFUTE delta; and maximum absolute polarity delta.

Pooled descriptive aggregates combine frozen Stage192-A rows for seeds 174-176 with fresh rows for seeds 177-179. Historical independent evidence maps to `independent_selected`; tail2 and tail3 names remain exact. Old predictions and checkpoints are not reopened. Pooled output is explicitly not a statistical-significance claim.

## Stage193-C exact outputs and fail-closed behavior

The analyzer writes exactly:

1. `stage193c_tail3_fresh_seed_replication_report.json`
2. `stage193c_tail3_fresh_seed_replication_report.md`
3. `stage193c_stage192a_closure_gate.csv`
4. `stage193c_run_identity_gate.csv`
5. `stage193c_epoch_metric_reconstruction.csv`
6. `stage193c_comparator_metrics_by_seed.csv`
7. `stage193c_comparator_aggregate_fresh.csv`
8. `stage193c_comparator_aggregate_pooled.csv`
9. `stage193c_pair_transition_summary.csv`
10. `stage193c_pair_transition_by_gold.csv`
11. `stage193c_primary_criterion_gate.csv`
12. `stage193c_precommitted_decision_gate.csv`

Before safe output establishment, failure writes nothing. Afterwards, every failure writes all twelve fixed-header outputs, a blocked report, and exits nonzero. Both Stage193-A and Stage193-C reports explicitly state diagnostic-only scope, no checkpoint/model/capsule loading, no external data, no statistical-significance claim, no model advancement, and no subsequent training authorization.

## Precommitted criteria and decisions

Fresh tail3 quality requires macro-F1 no more than 0.015 below fresh independent, accuracy no more than 0.01 below, maximum absolute REFUTE delta at most 1, and maximum absolute polarity delta at most 1. Fresh smoothing requires SUPPORT range at most 0.65 of independent and false-entitlement range at most 0.65 of independent. At least two fresh seeds must have absolute SUPPORT delta at most 40 and absolute false-entitlement delta at most 25.

Pooled tail3 requires macro-F1 no more than 0.015 below pooled independent, accuracy no more than 0.01 below, both delta ranges at most 0.70 of their independent ranges, and maximum absolute REFUTE and polarity deltas at most 1.

Exactly one decision is emitted:

- `STAGE193C_TAIL3_FRESH_SEED_REPLICATION_BLOCKED` for provenance, frozen-input, schema, hash, cardinality, alignment, reconstruction, or analysis failure.
- `STAGE193C_TAIL3_SMOOTHING_REPLICATED` only when every fresh quality, fresh smoothing, seed-level, and pooled criterion passes.
- `STAGE193C_TAIL3_SMOOTHING_PARTIAL_SIGNAL` only when all fresh and pooled quality criteria pass, the full positive conjunction fails, and at least three of the four fresh/pooled SUPPORT/false-entitlement range criteria pass.
- `STAGE193C_TAIL3_SMOOTHING_NOT_REPLICATED` when integrity passes but neither positive nor partial conditions pass.

A replicated result recommends designing, but not authorizing, one interpretable EMA/SWA-style mechanism. A partial result recommends diagnosing the failing seed and transition direction. A negative result closes post-hoc smoothing and recommends a distinct interpretable SUPPORT/NOT_ENTITLED boundary mechanism.

The analyzer writes exactly twelve named outputs with fixed headers and fail-closed completion after safe output establishment. Every report states: diagnostic only; no checkpoint/model/capsule loading; no external data; no statistical-significance claim; no model advancement; and no subsequent training authorization.
