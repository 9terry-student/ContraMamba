# Stage189-A three-seed margin replication and posthoc reference specification

**Ready decision:** `STAGE189A_THREE_SEED_MARGIN_REPLICATION_AND_POSTHOC_REFERENCE_SPEC_READY`
**Blocked decision:** `STAGE189A_THREE_SEED_MARGIN_REPLICATION_SPEC_BLOCKED`
**Authorized next:** `STAGE189B_THREE_SEED_PAIRED_TRAINING`

## Frozen experiment

Stage189-B consists of six fresh runs: baseline and intervention at model/training seeds 174, 175, and 176. Every run uses the fixed clean main train/dev split seed 174 through explicit `--split-seed 174`, so all six runs share the same frozen 2,880/720 row-ID split and only initialization/training randomness varies by model seed. Split seed is common across arms and is not an allowed arm difference. Stage188 seed-174 checkpoints are not reusable. All runs use `data/controlled_v5_v3_without_time_swap.jsonl`, `v6b_minimal`, Mamba `state-spaces/mamba-130m-hf`, CUDA, 20 epochs, and `final_macro_f1` internal clean-dev selection. Final classifier CE remains sourced from `output["logits"]`; loss logits, external data/evaluation, Stage15 OOD, time-swap main training, sweeps, threshold tuning, and selection changes are forbidden.

The baseline uses weight 0.0 and must omit both sidecar options. The intervention uses weight 0.05, margin logit 0.0, and the authoritative Stage185-A sidecar with semantic SHA `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`. Stage174-C, Stage175-B, Stage177-C, and unrelated interventions remain off.

Artifacts from commit `21c733533317a5d5aff447a98cb4efeeaec4ee49` or any pre-split-seed-contract Stage189-A/Stage189-B run are not official replication artifacts and must not be mixed with new runs. The trainer SHA, Git commit, and explicit split provenance must all match the new manifest contract. No code path auto-discovers or reuses those old artifacts.

## Selected checkpoint

The trainer already has default-off `--save-selected-checkpoint`. It saves `_ood_best_state`, not the final epoch state, into the provenance run directory and reload-verifies the artifact. Stage189 extends only its opt-in metadata with selection metric name/value, parsed args, dataset identity, Git commit, trainer path/SHA, compatible-positive margin configuration, and run/timestamp identity. Optimizer and scheduler state remain excluded. Omitting the flag preserves existing behavior.

## Stage189-C evaluation-only reference

After training and internal checkpoint selection are complete, Stage189-C loads exactly one selected checkpoint and its run provenance, validates all identities, and evaluates the 1,440 `split=train` compatible rows joined exactly to the authoritative sidecar: 605 ELIGIBLE, 716 INELIGIBLE, and 119 UNRESOLVED. It exports direct finite `output["frame_logit"]` and model probability without inversion or substitution. `model.eval()` plus inference mode is mandatory; optimizer, loss, backward, tuning, and selection are absent. Existing clean-dev scalar exports are reused for prior-selected clean-dev cohorts.

Training-row outputs are posthoc mechanism diagnostics, never generalization evidence.

## Stage189-D precommitment

Every seed must pass exact arm identity, row-ID, checkpoint, provenance, dataset, trainer, sidecar, and direct-score gates. Hard clean guardrails are macro-F1 delta >= -0.01, SUPPORT recall delta >= -0.02, false-entitlement delta <= +2, no polarity increase, and nonzero predictions for every label. Aggregate mean directions require macro-F1 >= 0, SUPPORT recall >= 0, and false-entitlement <= 0.

Mechanism replication requires, for all three seeds, eligible mean and median frame-logit delta > 0, eligible positive-delta fraction >= 0.80, and selected/final active rate below first-epoch active rate.

Selectivity requires at least two of three seeds for each comparison: eligible mean delta above ineligible mean delta; eligible mean delta above matched-control mean delta; and compatible-FN median delta above matched-control median delta. Matched controls remain prior-selected diagnostics.

The decision taxonomy is fail-closed. Identity/runtime failure is `STAGE189D_THREE_SEED_MARGIN_REPLICATION_BLOCKED`. Any hard clean guardrail failure, aggregate clean direction failure, two or more mechanism failures, class collapse, polarity regression, or a newly harmed critical Stage182-B row is `STAGE189D_THREE_SEED_MARGIN_NEGATIVE_OR_REGRESSIVE`. Exactly one mechanism failure, with every hard clean and aggregate clean gate passing, is `STAGE189D_THREE_SEED_MARGIN_PARTIAL_REPLICATION_NO_ADVANCE`; one failed seed is not automatically negative. With 3/3 mechanism replication and aggregate clean direction passing, failure of one or more selectivity tests is `STAGE189D_THREE_SEED_MARGIN_REPLICATED_BENEFICIAL_BUT_NONSELECTIVE`. Only every identity/runtime and hard clean gate, 3/3 mechanism replication, aggregate clean direction, every precommitted 2/3 selectivity test, and zero newly harmed critical Stage182-B rows yields `STAGE189D_THREE_SEED_MARGIN_POSITIVE_SELECTIVE_REPLICATION`.
