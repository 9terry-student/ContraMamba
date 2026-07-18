# Stage194-A artifact-only entitlement-boundary localization specification

## Scope and frozen question

Stage194-A is a diagnostic-only, artifact-only localization of the replicated
Stage193-C tail3 signal.  Stage193-C emitted
`STAGE193C_TAIL3_SMOOTHING_REPLICATED`: tail3 mean logits reduced fresh-seed
SUPPORT-delta range from 33 to 6 and false-entitlement-delta range from 29 to
1, with maximum absolute REFUTE delta 1 and polarity delta 0, but mean pair
SUPPORT recall fell from 0.679775 to 0.516854.

This stage distinguishes four explanations: trainer-selected checkpoint
temporal-consensus outlier removal, mean-logit magnitude domination by one
extreme epoch, persistent SUPPORT versus NOT_ENTITLED boundary bias, or a
mixture.  It is not a temporal-aggregation search, does not select a production
inference rule, and does not authorize model advancement, training, EMA, SWA,
calibration, or any other intervention.  Median aggregation is a diagnostic
counterfactual only.

## Frozen identities and inputs

- Stage193 runtime repository commit:
  `89a9805d0e9c877774f9ce4b356297d31645b74b`
- Trainer blob commit: `e83d8af756fa84b7a91c14e0910ae388b07b5f02`
- Trainer SHA256:
  `25d42bdcd204219a2b2e5e7bf2a8b14459eafb4945c05c61ab3611bc9e7365bc`
- Expected Stage193-C decision: `STAGE193C_TAIL3_SMOOTHING_REPLICATED`
- Training seeds: 177, 178, 179; split seed: 174
- Canonical labels, in tie-breaking order: `REFUTE`, `NOT_ENTITLED`, `SUPPORT`
- Frozen run order: `seed177_baseline`, `seed177_intervention`,
  `seed178_baseline`, `seed178_intervention`, `seed179_baseline`,
  `seed179_intervention`

The analyzer requires explicit `--repo-root`, `--stage193a-dir`,
`--stage193b-run-root`, `--stage193c-dir`,
`--current-diagnostic-git-commit`, and `--output-dir`.  It performs no fuzzy,
latest, glob-based, or timestamp discovery.  Every supplied path is distinct.
The output is absent or an empty immediate child of `<repo-root>/reports`, and
its basename begins `stage194a_entitlement_boundary_localization_`.

## Source and frozen-byte closure

The supplied Stage194 diagnostic commit must be lowercase hexadecimal length
40, equal repository HEAD, contain byte-identical blobs for this specification
and `scripts/analyze_stage194a_entitlement_boundary_localization.py`, and have
no staged or unstaged changes for those two files.  Global worktree cleanliness
is not required.

Independently, current bytes for the Stage193-A specification, Stage193-A
builder, Stage193-C analyzer, and frozen trainer must equal their blobs at the
Stage193 runtime commit.  Trainer bytes must additionally equal their blob at
the trainer blob commit and have the frozen SHA256.  Every Git invocation is a
read-only argument array passed to `subprocess.run` with `shell=False`.

## Frozen Stage193 closure

Stage193-A must contain exactly its six named outputs.  Its report must be the
READY, runnable, diagnostic-only six-run manifest with no blocking reasons,
the frozen runtime commit, trainer blob identity, exact run order, 20 ledger
rows, 20 prediction exports, 720 rows per export, zero state capsules, and no
model-advancement or subsequent-training authorization.  Its two gate CSVs
must have their exact header and all rows pass.  Its command matrix and JSONL
manifest must have their exact schemas, six-row order, strict integer
identities, exact planned paths, argv, and arm contracts.

Stage193-C must contain exactly its twelve named outputs.  Its report must have
stage `Stage193-C`, the replicated decision, runnable true, no blocking reasons,
diagnostic-only true, no checkpoint/model/capsule loading, no external data, no
statistical-significance claim, no model advancement, no subsequent training,
the frozen runtime and trainer identities, primary candidate
`tail3_mean_logits`, and descriptive-only tail2.  Every primary criterion row
must pass.  Its four-row decision gate must select only the replicated decision.
The reported fresh aggregates must reproduce, to six decimals, the frozen
independent-selected values 0.581720/0.875694/0.828554/0.679775 with ranges
33/29, and tail3 values 0.570286/0.890509/0.820833/0.516854 with ranges 6/1,
maximum absolute REFUTE delta 1, and maximum absolute polarity delta 0.

The supplied Stage193-B root must equal the root frozen by Stage193-A and have
exactly the six run directories.  Each run independently validates exact run,
seed, split seed, arm, invoked argv, Stage193 observability mode, disabled
Stage191 observability, runtime commit, trainer SHA256, epochs 1 through 20,
all 20 enumerated exports and ledger hashes, 720 ordered rows per export,
canonical labels, finite three-logit vectors, canonical argmax predictions,
gold alignment across epochs and runs, the authoritative selected/final epoch
schema, zero capsules, and inactive external and auxiliary data.  All integer
contracts use `type(value) is int`; booleans never satisfy them.

All twenty exports and their hashes are validated.  Localization retains in
memory only the trainer-selected export and epochs 18, 19, and 20.  It never
loads a `.pt` file, checkpoint, capsule, model, project module, or external,
OOD, bridge, synthetic, or time-swap data.

## Float64 margins and aggregations

Python `float` values are used as IEEE-754 binary64.  For each row and source:

```
entitlement_margin = logits[SUPPORT] - logits[NOT_ENTITLED]
refute_margin = logits[REFUTE] - max(logits[NOT_ENTITLED], logits[SUPPORT])
```

Tail3 mean logits are classwise `math.fsum` over epochs 18--20 divided by 3.
Tail3 median logits are the classwise middle values of those epochs.  Exact
argmax ties use canonical label order.  The row table records selected, epoch
18, epoch 19, epoch 20, mean, and median logits, predictions, both margins;
population standard deviation and range of the three late entitlement margins;
consecutive sign changes; positive, negative, and exact-zero vote counts; and
the three-symbol sign pattern.

## Strict mechanism definitions

Primary SUPPORT/NOT_ENTITLED mechanism flags are false for a REFUTE-involved
row.  A row is REFUTE-involved when any of its selected, epoch18, epoch19,
epoch20, mean, or median predictions is REFUTE; such rows remain explicitly
counted separately.

`selected_consensus_outlier` applies only to gold SUPPORT when the selected
epoch is one of 18--20, selected prediction is SUPPORT, mean prediction is
NOT_ENTITLED, the selected late margin is positive, the other two late margins
are negative, and none of selected/late/mean entitlement margins is zero.  A
selected epoch outside 18--20 cannot satisfy the strict two-other-late-epoch
definition.

`mean_magnitude_override` applies to gold SUPPORT with selected SUPPORT, mean
NOT_ENTITLED, exactly two positive late margins, exactly one negative late
margin, negative arithmetic-mean entitlement margin, and no zero among the
selected/late/mean margins.  `median_rescue` is a mean-magnitude override whose
median-logit prediction is SUPPORT.

`persistent_stable_negative` applies to gold SUPPORT with mean NOT_ENTITLED,
three negative late margins, three late NOT_ENTITLED predictions, and no zero
among the late and mean margins.  `temporally_mixed_negative` applies to a
non-REFUTE gold SUPPORT row with mean NOT_ENTITLED that is not persistent and
has no exact zero among selected, late, mean, and median entitlement margins.
An exact-tie row is kept outside that category.

## Aggregates

Support-recall decomposition is emitted for each run, each arm pooled across
three seeds, and all six runs pooled.  It records the frozen count, TP, recall,
loss/gain direction, mechanism, and false-negative fields.  The same nine
scopes form the boundary-mechanism and counterfactual summaries.  Primary
pooled ratios are:

```
consensus_outlier_share = selected_consensus_outlier / losses_to_NOT_ENTITLED
magnitude_override_share = mean_magnitude_override / losses_to_NOT_ENTITLED
median_rescue_rate = median_rescue / mean_magnitude_override
persistent_bias_share = persistent_stable_negative / mean_NOT_ENTITLED_FN
```

Every denominator is recorded.  A zero denominator yields JSON/CSV null, not
zero.  Gold-conditioned margin summaries cover all nine scopes, three labels,
and six sources.  Temporal-pattern summaries cover every observed sign pattern
within each of the nine scopes and three gold labels.

## Precommitted evidence and decision

Strong temporal consensus requires losses to NOT_ENTITLED >= 12,
consensus share >= 0.60, and magnitude share <= 0.25.  Strong mean magnitude
requires losses >= 12, magnitude share >= 0.40, at least 8 magnitude overrides,
and median rescue rate >= 0.70.  Strong persistent bias requires at least 30
mean NOT_ENTITLED false negatives, at least 20 persistent negatives, and
persistent share >= 0.60.

Moderate evidence requires respectively consensus share >= 0.40; magnitude
share >= 0.25 and rescue rate >= 0.50; or persistent share >= 0.40.  Null never
satisfies a threshold.

Exactly one decision is selected:

1. `STAGE194A_ENTITLEMENT_LOCALIZATION_BLOCKED` only on integrity or analysis
   failure.
2. `STAGE194A_TEMPORAL_CONSENSUS_OUTLIER_DOMINANT` only for strong temporal
   consensus with neither other mechanism strong or moderate.
3. `STAGE194A_MEAN_MAGNITUDE_OUTLIER_DOMINANT` only for strong mean magnitude
   with neither other mechanism strong or moderate.
4. `STAGE194A_PERSISTENT_ENTITLEMENT_BIAS_DOMINANT` only for strong persistent
   bias with neither other mechanism strong or moderate.
5. `STAGE194A_MIXED_TEMPORAL_AND_BOUNDARY_MECHANISMS` when at least two strong
   mechanisms pass, one strong plus another moderate passes, or no strong and
   at least two moderate mechanisms pass.
6. `STAGE194A_ENTITLEMENT_MECHANISM_INCONCLUSIVE` otherwise after integrity
   passes.

Dominant temporal consensus recommends designing one interpretable EMA/SWA
style mechanism without authorizing it.  Dominant mean magnitude recommends a
separate robust-aggregation diagnostic, not EMA/SWA training.  Dominant
persistent bias recommends designing an explicit entitlement-boundary
mechanism or calibration without authorizing either.  Mixed evidence requires
one mechanism at a time, beginning with the largest evidence share.

## Exact outputs and centrally frozen schemas

The analyzer writes exactly these twelve files:

1. `stage194a_entitlement_boundary_localization_report.json`
2. `stage194a_entitlement_boundary_localization_report.md`
3. `stage194a_stage193_closure_gate.csv`
4. `stage194a_run_identity_gate.csv`
5. `stage194a_row_margin_decomposition.csv`
6. `stage194a_support_recall_decomposition.csv`
7. `stage194a_boundary_mechanism_summary.csv`
8. `stage194a_gold_conditioned_margin_summary.csv`
9. `stage194a_temporal_pattern_summary.csv`
10. `stage194a_diagnostic_counterfactual_summary.csv`
11. `stage194a_mechanism_criterion_gate.csv`
12. `stage194a_precommitted_decision_gate.csv`

Every CSV header is declared once in `CSV_HEADERS`.  The schemas are:

- closure: `gate, required, observed, passed, blocking_reason`
- identity: `run, gate, required, observed, passed, blocking_reason`
- row decomposition: `run, training_seed, split_seed, arm, dev_position,
  gold_label, selected_epoch, selected_logits, epoch18_logits, epoch19_logits,
  epoch20_logits, tail3_mean_logits, tail3_median_logits, selected_prediction,
  epoch18_prediction, epoch19_prediction, epoch20_prediction,
  tail3_mean_prediction, tail3_median_prediction, selected_entitlement_margin,
  epoch18_entitlement_margin, epoch19_entitlement_margin,
  epoch20_entitlement_margin, tail3_mean_entitlement_margin,
  tail3_median_entitlement_margin, selected_refute_margin,
  epoch18_refute_margin, epoch19_refute_margin, epoch20_refute_margin,
  tail3_mean_refute_margin, tail3_median_refute_margin,
  late_entitlement_margin_population_stddev, late_entitlement_margin_range,
  consecutive_sign_change_count, support_sign_vote_count,
  not_entitled_sign_vote_count, exact_zero_tie_count, sign_pattern,
  selected_to_mean_transition, selected_to_median_transition, refute_involved,
  selected_consensus_outlier, mean_magnitude_override, median_rescue,
  persistent_stable_negative, temporally_mixed_negative`
- support decomposition: `scope, aggregate, run_count, gold_support_count,
  selected_support_true_positives, tail3_mean_support_true_positives,
  tail3_median_support_true_positives, selected_support_recall,
  tail3_mean_support_recall, tail3_median_support_recall,
  selected_to_mean_support_losses, selected_to_mean_support_gains,
  net_support_true_positive_change, losses_to_NOT_ENTITLED, losses_to_REFUTE,
  gains_from_NOT_ENTITLED, gains_from_REFUTE, selected_consensus_outlier_count,
  mean_magnitude_override_count, median_rescue_count,
  persistent_stable_negative_count, temporally_mixed_negative_count,
  tail3_mean_NOT_ENTITLED_false_negative_count,
  tail3_mean_REFUTE_false_negative_count, refute_involved_gold_support_count`
- mechanism summary: `scope, aggregate, run_count,
  selected_to_mean_support_losses_to_NOT_ENTITLED,
  selected_consensus_outlier_count, mean_magnitude_override_count,
  median_rescue_count, tail3_mean_NOT_ENTITLED_false_negative_count,
  persistent_stable_negative_count, temporally_mixed_negative_count,
  consensus_outlier_denominator, consensus_outlier_share,
  magnitude_override_denominator, magnitude_override_share,
  median_rescue_denominator, median_rescue_rate, persistent_bias_denominator,
  persistent_bias_share, refute_involved_support_loss_count`
- gold-conditioned margin: `scope, aggregate, run_count, gold_label, source,
  count, mean_entitlement_margin, median_entitlement_margin,
  population_stddev, minimum, maximum, negative_margin_fraction,
  positive_margin_fraction, exact_zero_fraction, pred_REFUTE,
  pred_NOT_ENTITLED, pred_SUPPORT`
- temporal pattern: `scope, aggregate, run_count, gold_label, sign_pattern,
  count, fraction, mean_margin_range, mean_margin_population_stddev,
  selected_to_mean_transition_counts, selected_to_median_transition_counts`
- counterfactual: `scope, aggregate, run_count, gold_support_count,
  tail3_mean_support_true_positives, tail3_median_support_true_positives,
  tail3_mean_support_recall, tail3_median_support_recall,
  mean_to_median_support_gains, mean_to_median_support_losses,
  mean_NOT_ENTITLED_to_median_SUPPORT, mean_REFUTE_to_median_SUPPORT,
  mean_SUPPORT_to_median_NOT_ENTITLED, mean_SUPPORT_to_median_REFUTE,
  mean_magnitude_override_count, median_rescue_count, median_rescue_rate,
  diagnostic_only`
- mechanism criteria: `mechanism, evidence_level, criterion, required,
  observed, passed`
- decision gate: `decision, taxonomy_condition, required, observed, passed`

The row decomposition contains exactly 4,320 rows in frozen run order and
ascending dev position.  Reports explicitly state artifact-only diagnostic
scope, no checkpoint/model/capsule loading, no external data, diagnostic-only
median, no statistical-significance claim, no model advancement, no training,
no EMA/SWA authorization, and no calibration authorization.

Before safe output establishment a failure writes nothing and exits nonzero.
After safe establishment every failure emits all twelve fixed-header files, a
blocked report, and the blocked return code.
