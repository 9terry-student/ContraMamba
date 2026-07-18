# Stage192-A paired trajectory-stable checkpoint-selection diagnostic

## Scope

Stage192-A is a diagnostic over existing Stage191-B trajectory exports and an explicitly supplied authoritative Stage191-D output directory. It does not train, construct a model, load a checkpoint or state capsule, use external data, authorize Stage192-B, or make a model-advancement decision.

The frozen Stage191-B replay commit is `0872e66ccb05ae8a166f5cabf4e084272dc49500`; its directory is exactly `<repo-root>/reports/stage191b_deterministic_replay_manifest_20260717_153524`. The frozen Stage191-D implementation commit is `08ae49a79148ca448340c1948b5c9991b6919f04`. The Stage191-D directory is always supplied explicitly and is never discovered by timestamp.

The accepted runs are baseline and intervention for training seeds 174, 175, and 176, with split seed 174 and canonical label order `REFUTE`, `NOT_ENTITLED`, `SUPPORT`. Historical selected epochs are 20/20 for seed 174 and 20/19 for seeds 175 and 176 (baseline/intervention). Final epoch is 20 for every run.

## Inputs and provenance

The analyzer requires `--repo-root`, `--stage191b-dir`, `--stage191d-dir`, `--current-diagnostic-git-commit`, and `--output-dir`. It accepts no fallback input paths. The supplied Stage191-D directory must be an existing immediate child of `<repo-root>/reports`, its basename must start exactly with `stage191d_trajectory_phase_flip_`, and it must differ from Stage191-B and the Stage192-A output. It is supplied explicitly; no fuzzy or timestamp discovery is permitted. The supplied Stage192-A commit is lowercase hexadecimal of length 40, equals repository HEAD, contains byte-identical blobs for this specification and analyzer, and has no staged or unstaged differences for those two files. Git subprocess calls are read-only, use argument arrays and `shell=False`, and do not require a globally clean worktree.

The supplied Stage191-D directory must contain exactly its authoritative fourteen outputs. Its report must carry the confirmed late SUPPORT/NOT_ENTITLED phase-flip decision, runnable closure, no blockers, passed Stage191-C equivalence and source contract, the frozen Stage191-D commit, the exact float32 CPU clean-CE reduction contract, and all diagnostic-only/no-training/no-advancement flags. Every universally required Stage191-D precommit gate and every Stage191-C equivalence gate must pass.

### Frozen Stage191-D CSV closure

The Stage191-C equivalence CSV header is exactly `gate,run,required,observed,passed,blocking_reason`; it is nonempty and every row has a strictly parsed boolean `passed=true`. The Stage191-D precommit CSV header is exactly `gate,required,observed,passed` and its gate names, in exact order, are:

1. `stage190_grouping_source_contract`
2. `stage190_exact_parameter_inventory_mapping`
3. `parameter_ownership_exhaustive_disjoint`
4. `stage191c_equivalence_all_six`
5. `support_delta_nonzero_opposite_sign_all_seeds`
6. `false_entitlement_delta_nonzero_opposite_sign_all_seeds`
7. `refute_absolute_delta_at_most_one_all_late_cells`
8. `polarity_absolute_delta_at_most_one_all_late_cells`
9. `selected_epoch19_intervention_transition_concentration_at_least_095`
10. `phase_flip_condition_observed`
11. `redistribution_condition_observed`
12. `selected_decision_matches_precommitted_taxonomy`

The first four and final gates require and pass true. The seven middle evidence rows require exactly `decision_alternative`; their passed cell may be empty. The `phase_flip_condition_observed` and `redistribution_condition_observed` cells remain strict Booleans and must parse as true and false respectively.

The `selected_decision_matches_precommitted_taxonomy` observed cell is strict JSON and must be exactly an object with key set `phase_flip_pass`, `redistribution_pass`, and `decision`. The first value must be exactly true, the second exactly false, and decision exactly `STAGE191D_LATE_SUPPORT_NE_PHASE_FLIP_CONFIRMED`. Its required and passed cells must each parse strictly as true. Malformed JSON, extra or missing keys, non-Boolean condition values, or another decision blocks. The taxonomy object is never passed to the Boolean CSV parser.

## Replay validation

The Stage191-B main report must have the ready decision, `runnable=true`, empty blockers, diagnostic replay only, replay execution authorized, training-for-advancement authorization false, model advancement false, external data false, authorized seeds `[174,175,176]`, and frozen replay commit `0872e66ccb05ae8a166f5cabf4e084272dc49500`. Its ordered identity matrix is exactly seed174 baseline/intervention, seed175 baseline/intervention, and seed176 baseline/intervention, with exact run, non-bool integer seed, arm, and non-bool integer split seed 174. It also requires `stage == "Stage191-B"`, exact non-bool integer cardinalities `expected_trajectory_rows_per_run == 20`, `expected_prediction_rows_per_epoch == 720`, and `expected_state_capsules_per_run == 20`, plus `logits_source == 'output["logits"]'`.

Each replay manifest additionally requires `stage == "Stage191-B"`, replay execution authorization, logits source `output["logits"]`, exact selected epoch and prediction cardinalities, and unconditional exact non-bool integer `expected_state_capsules == 20`. Each trajectory contract requires authorized seeds `[174,175,176]`, 20 epochs, 720 dev rows, exact canonical labels, `logits_source == 'output["logits"]'`, both Stage191 observability flags true, and training-semantics-changed, extra-forward, loss-logits, and external-data flags false. Stage192-A validates capsule cardinality metadata only; physical state capsules remain unused and need not be present or opened.

Each exact Stage191-B replay child must contain one trajectory ledger with epochs 1 through 20 and exactly twenty enumerated prediction exports. The analyzer validates run, seed, arm, split, canonical labels, internal-only authorization, selected and final epochs, exact export paths and SHA256 values, positions 0 through 719, aligned gold labels, finite three-logit vectors, finite row CE, normalized prediction counts, and reconstructed clean metrics. Clean CE is reproduced exactly as the mean of the 720 ordered `final_ce` values in `torch.float32` on CPU. Accuracy, macro-F1, and SUPPORT recall use a fixed `1e-7` comparison tolerance; count metrics are exact. Checkpoints and capsules are never opened.

## Frozen selection rules

All candidates operate only on epochs 15 through 20. `historical_independent` is the non-winning reference. The six synchronous candidates are `sync_epoch19`, `sync_epoch20`, `sync_mean_macro`, `sync_min_macro`, `sync_mean_ce`, and `sync_stability_constrained`.

The mean-macro rule maximizes pair mean macro-F1, then minimum-arm macro-F1, then minimizes pair mean CE, then prefers the later epoch. The min-macro rule maximizes minimum-arm macro-F1, then pair mean macro-F1, then minimizes pair mean CE, then prefers later. The mean-CE rule minimizes pair mean CE, then maximizes pair mean and minimum-arm macro-F1, then prefers later.

For `sync_stability_constrained`, quality eligibility requires pair mean macro-F1 within 0.005 of the best synchronous epoch and each arm within 0.01 of that arm's historical selected macro-F1. In the clipped one-epoch neighborhood of every eligible epoch, the rule minimizes lexicographically: the number of SUPPORT-delta and false-entitlement-delta series containing both nonzero signs; SUPPORT-delta amplitude; false-entitlement-delta amplitude; consecutive NOT_ENTITLED/SUPPORT prediction churn summed across both arms and edges; negative pair mean macro-F1; pair mean CE; and negative epoch. It is unavailable rather than substituted when eligibility is empty.

## Evaluation

Every selector records selected arm metrics and intervention-minus-baseline deltas. Its perturbation grid is the Cartesian product of each arm's selected epoch plus or minus one, clipped to 15 through 20. Zero is unsigned. Summaries record delta ranges and signs, phase flips, REFUTE and polarity bounds, and worst clean quality.

Selected baseline-to-intervention transitions include the full canonical 3-by-3 matrix, changed-row boundary concentration, corrections, regressions, wrong-to-different-wrong changes, and gold-conditioned versions.

`tail2_mean_logits` and `tail3_mean_logits` average row logits in float64 over epochs 19-20 and 18-20 respectively. Canonical order resolves exact argmax ties. They are descriptive only and never affect the winning rule or decision.

### Quality-preserving criteria

A non-reference selector qualifies only when all conditions hold:

1. available for all three seeds;
2. synchronous for all three seeds;
3. mean macro-F1 is at least the historical reference minus 0.005;
4. mean accuracy is at least the historical reference minus 0.01;
5. total false entitlement is at most the historical reference plus 30;
6. total false not-entitled is at most the historical reference plus 30;
7. total polarity error is at most the historical reference plus 1;
8. joint phase flip is present in at most one seed;
9. mean SUPPORT-delta perturbation range is at most 0.75 times the historical reference;
10. mean false-entitlement perturbation range is at most 0.75 times the historical reference;
11. the base selected pair has absolute REFUTE prediction-count delta at most 1 and absolute polarity-error delta at most 1 for every seed.

### Quality-tradeoff criteria

A selector has a stability gain with quality tradeoff only when the full quality-preserving criteria fail; it is available and synchronous for all seeds; joint phase flip is present in at most one seed; both mean perturbation ranges are at most 0.75 times their historical references; mean macro-F1 is at least the reference minus 0.015; and mean accuracy is at least the reference minus 0.02.

### Exact winner order

Among quality-preserving selectors, the winner is selected in this exact lexicographic order:

1. fewer joint-phase-flip seeds;
2. lower mean SUPPORT-delta perturbation range;
3. lower mean false-entitlement perturbation range;
4. higher aggregate mean macro-F1;
5. higher aggregate mean accuracy;
6. lower aggregate mean clean CE;
7. selector name ascending.

## Decisions and restrictions

Exactly one decision is emitted:

1. `STAGE192A_TRAJECTORY_SELECTION_DIAGNOSTIC_BLOCKED`
2. `STAGE192A_TRAJECTORY_STABLE_PAIR_SELECTOR_IDENTIFIED`
3. `STAGE192A_STABILITY_GAIN_WITH_QUALITY_TRADEOFF_ONLY`
4. `STAGE192A_NO_TRAJECTORY_STABLE_SELECTOR`

The exact decision conjunctions are:

- `STAGE192A_TRAJECTORY_SELECTION_DIAGNOSTIC_BLOCKED` only through fail-closed exception handling after an integrity or analysis failure.
- `STAGE192A_TRAJECTORY_STABLE_PAIR_SELECTOR_IDENTIFIED` only when the quality-preserving selector set is nonempty, the winner is non-null, and it equals the exact lexicographic winner.
- `STAGE192A_STABILITY_GAIN_WITH_QUALITY_TRADEOFF_ONLY` only when the quality-preserving set is empty, the tradeoff set is nonempty, and the winner is null.
- `STAGE192A_NO_TRAJECTORY_STABLE_SELECTOR` only when both selector sets are empty and the winner is null.

A normal nonselected decision alternative is not a failed required gate.

The Stage192-A universal closure ledger appends global rows for `stage191d_closure_passed`, `stage191b_replay_validation_passed`, `temporal_ensembles_excluded_from_decision`, `winning_selector_matches_exact_lexicographic_order`, and `selected_decision_matches_precommitted_taxonomy`. Their results are also recorded in the JSON report.

A positive result supports only a Stage192-B fresh-seed validation design for the frozen winner; it does not authorize that run. Tradeoff-only recommends refining the objective without training. No-selector recommends trajectory-level optimization or regularization design. No statistical significance, broader generalization, external/OOD performance, causal parameter-group conclusion, deployment validation, training authorization, or model advancement may be claimed.

## Exact outputs

The analyzer writes exactly these fourteen files:

1. `stage192a_trajectory_stable_selection_report.json`
2. `stage192a_trajectory_stable_selection_report.md`
3. `stage192a_stage191d_closure_gate.csv`
4. `stage192a_selector_definition.csv`
5. `stage192a_selector_choice_by_seed.csv`
6. `stage192a_selected_arm_metrics.csv`
7. `stage192a_selector_aggregate_metrics.csv`
8. `stage192a_pair_delta_by_selector.csv`
9. `stage192a_perturbation_grid.csv`
10. `stage192a_perturbation_summary.csv`
11. `stage192a_selected_pair_transition_summary.csv`
12. `stage192a_selected_pair_transition_by_gold.csv`
13. `stage192a_temporal_ensemble_comparator.csv`
14. `stage192a_precommitted_gate.csv`

## Output and fail-closed contract

The output is an absent or empty immediate child of `<repo-root>/reports` whose basename starts `stage192a_trajectory_stable_selection_`; it cannot equal or descend from either frozen input. A nonempty directory is never overwritten. Before safe establishment, failure writes nothing there and returns nonzero.

After establishment, every exception writes the JSON and Markdown blocked reports plus all twelve fixed-header CSV ledgers (fourteen files total), preserving exception type, message, and traceback. Blocked output always has `runnable=false`, and training, model construction, external-data use, model advancement, and Stage192-B authorization are all false.
