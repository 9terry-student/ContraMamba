# Stage192-A paired trajectory-stable checkpoint-selection diagnostic

## Scope

Stage192-A is a diagnostic over existing Stage191-B trajectory exports and an explicitly supplied authoritative Stage191-D output directory. It does not train, construct a model, load a checkpoint or state capsule, use external data, authorize Stage192-B, or make a model-advancement decision.

The frozen Stage191-B replay commit is `0872e66ccb05ae8a166f5cabf4e084272dc49500`; its directory is exactly `<repo-root>/reports/stage191b_deterministic_replay_manifest_20260717_153524`. The frozen Stage191-D implementation commit is `08ae49a79148ca448340c1948b5c9991b6919f04`. The Stage191-D directory is always supplied explicitly and is never discovered by timestamp.

The accepted runs are baseline and intervention for training seeds 174, 175, and 176, with split seed 174 and canonical label order `REFUTE`, `NOT_ENTITLED`, `SUPPORT`. Historical selected epochs are 20/20 for seed 174 and 20/19 for seeds 175 and 176 (baseline/intervention). Final epoch is 20 for every run.

## Inputs and provenance

The analyzer requires `--repo-root`, `--stage191b-dir`, `--stage191d-dir`, `--current-diagnostic-git-commit`, and `--output-dir`. It accepts no fallback input paths. The supplied Stage192-A commit is lowercase hexadecimal of length 40, equals repository HEAD, contains byte-identical blobs for this specification and analyzer, and has no staged or unstaged differences for those two files. Git subprocess calls are read-only, use argument arrays and `shell=False`, and do not require a globally clean worktree.

The supplied Stage191-D directory must contain exactly its authoritative fourteen outputs. Its report must carry the confirmed late SUPPORT/NOT_ENTITLED phase-flip decision, runnable closure, no blockers, passed Stage191-C equivalence and source contract, the frozen Stage191-D commit, the exact float32 CPU clean-CE reduction contract, and all diagnostic-only/no-training/no-advancement flags. Every universally required Stage191-D precommit gate and every Stage191-C equivalence gate must pass.

## Replay validation

Each exact Stage191-B replay child must contain one trajectory ledger with epochs 1 through 20 and exactly twenty enumerated prediction exports. The analyzer validates run, seed, arm, split, canonical labels, internal-only authorization, selected and final epochs, exact export paths and SHA256 values, positions 0 through 719, aligned gold labels, finite three-logit vectors, finite row CE, normalized prediction counts, and reconstructed clean metrics. Clean CE is reproduced exactly as the mean of the 720 ordered `final_ce` values in `torch.float32` on CPU. Accuracy, macro-F1, and SUPPORT recall use a fixed `1e-7` comparison tolerance; count metrics are exact. Checkpoints and capsules are never opened.

## Frozen selection rules

All candidates operate only on epochs 15 through 20. `historical_independent` is the non-winning reference. The six synchronous candidates are `sync_epoch19`, `sync_epoch20`, `sync_mean_macro`, `sync_min_macro`, `sync_mean_ce`, and `sync_stability_constrained`.

The mean-macro rule maximizes pair mean macro-F1, then minimum-arm macro-F1, then minimizes pair mean CE, then prefers the later epoch. The min-macro rule maximizes minimum-arm macro-F1, then pair mean macro-F1, then minimizes pair mean CE, then prefers later. The mean-CE rule minimizes pair mean CE, then maximizes pair mean and minimum-arm macro-F1, then prefers later.

For `sync_stability_constrained`, quality eligibility requires pair mean macro-F1 within 0.005 of the best synchronous epoch and each arm within 0.01 of that arm's historical selected macro-F1. In the clipped one-epoch neighborhood of every eligible epoch, the rule minimizes lexicographically: the number of SUPPORT-delta and false-entitlement-delta series containing both nonzero signs; SUPPORT-delta amplitude; false-entitlement-delta amplitude; consecutive NOT_ENTITLED/SUPPORT prediction churn summed across both arms and edges; negative pair mean macro-F1; pair mean CE; and negative epoch. It is unavailable rather than substituted when eligibility is empty.

## Evaluation

Every selector records selected arm metrics and intervention-minus-baseline deltas. Its perturbation grid is the Cartesian product of each arm's selected epoch plus or minus one, clipped to 15 through 20. Zero is unsigned. Summaries record delta ranges and signs, phase flips, REFUTE and polarity bounds, and worst clean quality.

Selected baseline-to-intervention transitions include the full canonical 3-by-3 matrix, changed-row boundary concentration, corrections, regressions, wrong-to-different-wrong changes, and gold-conditioned versions.

`tail2_mean_logits` and `tail3_mean_logits` average row logits in float64 over epochs 19–20 and 18–20 respectively. Canonical order resolves exact argmax ties. They are descriptive only and never affect the winning rule or decision.

Aggregate quality-preserving and quality-tradeoff gates are exactly those in the Stage192-A implementation request. A quality-preserving winner is selected by fewer joint phase-flip seeds; lower mean SUPPORT-delta range; lower mean false-entitlement-delta range; higher mean macro-F1; higher mean accuracy; lower mean CE; and selector name ascending.

## Decisions and restrictions

Exactly one decision is emitted:

1. `STAGE192A_TRAJECTORY_SELECTION_DIAGNOSTIC_BLOCKED`
2. `STAGE192A_TRAJECTORY_STABLE_PAIR_SELECTOR_IDENTIFIED`
3. `STAGE192A_STABILITY_GAIN_WITH_QUALITY_TRADEOFF_ONLY`
4. `STAGE192A_NO_TRAJECTORY_STABLE_SELECTOR`

A positive result supports only a Stage192-B fresh-seed validation design for the frozen winner; it does not authorize that run. Tradeoff-only recommends refining the objective without training. No-selector recommends trajectory-level optimization or regularization design. No statistical significance, broader generalization, external/OOD performance, causal parameter-group conclusion, deployment validation, training authorization, or model advancement may be claimed.

## Output and fail-closed contract

The output is an absent or empty immediate child of `<repo-root>/reports` whose basename starts `stage192a_trajectory_stable_selection_`; it cannot equal or descend from either frozen input. A nonempty directory is never overwritten. Before safe establishment, failure writes nothing there and returns nonzero.

After establishment, every exception writes the JSON and Markdown blocked reports plus all twelve fixed-header CSV ledgers (fourteen files total), preserving exception type, message, and traceback. Blocked output always has `runnable=false`, and training, model construction, external-data use, model advancement, and Stage192-B authorization are all false.
