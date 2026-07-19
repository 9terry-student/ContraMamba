# Stage196-A persistent SUPPORT boundary localization specification

## Scope

Stage196-A is an artifact-only localization of the 127 frozen Stage195-C
persistent stable SUPPORT negatives.  It asks whether native local epistemic
channels fail or whether rows that pass those channels are stopped at the
entitlement aggregation or final composition boundary.  It performs no
training, model/tokenizer/checkpoint/state-capsule loading, GPU work, threshold
search, calibration, statistical testing, intervention, architecture or loss
selection, or production advancement.

Only explicitly named JSON, JSONL, CSV, Markdown and read-only Git blobs/status
are read.  Python binary64 is the calculation type.  The only probability-like
channel threshold is the precommitted native threshold 0.5.

## Frozen upstream closure

- Stage195-C runtime commit: `b258328d4984160429217d18127b924ec0561415`.
- Stage195-C decision: `STAGE195C_PARAMETER_SWA_MIXED_OR_INCONCLUSIVE`.
- Stage195-C is runnable, blocker-free, and not BLOCKED.
- run order: `seed180_baseline`, `seed180_intervention`,
  `seed181_baseline`, `seed181_intervention`, `seed182_baseline`,
  `seed182_intervention`.
- training seeds: exact non-bool integers 180, 181, 182; split seed 174.
- canonical logit order: `REFUTE`, `NOT_ENTITLED`, `SUPPORT`.
- 720 aligned dev rows per run and 4,320 Stage195-C transition rows.
- frozen persistent counts in run order: 22, 14, 30, 20, 18, 23; pooled 127.
- persistent parameter-SWA SUPPORT rescue count: zero per run and pooled.

The supplied Stage195-C directory must contain exactly its nine published
files.  Its JSON report, row-transition exact schema, source-closure exact CSV
header, decision, runtime identity, cardinalities, run order and blockers are
validated.  Every required Stage195-C source-closure row must pass with an
empty blocking reason.

The Stage195-C precommitted decision gate keeps exact header
`decision,taxonomy_condition,required,observed,passed` and exactly six decision
rows in its frozen taxonomy order.  `required` is the precommitted expected
Boolean for that decision condition.  `passed` states whether the observed
condition equals that expected Boolean.  `observed` is a nonempty serialized
JSON decision-evidence object; it is not a Boolean CSV field.  Consequently the
selected decision has `required=True` and `passed=True`, while every
non-selected decision has `required=False` and still has `passed=True`.  All
six observed values must parse as dictionaries.  Their decision evidence must
be exactly equal after excluding the row-local `condition`; that field must
exactly equal the row's `required` Boolean, so it is true for the selected row
and false for the other five.  The selected gate decision must equal the
Stage195-C report decision.  Stage196-A never compares `observed` to
the strings `"True"` or `"False"`.

This typed source-schema correction changes no Stage196-A source population,
127-row persistent cohort, mechanism bucket, native threshold 0.5, recurrence
logic, decision taxonomy, or exact nine-output schema.

## CLI and path safety

The analyzer accepts exactly `--repo-root`, `--stage195c-dir`,
`--stage195b-run-root`, `--current-diagnostic-git-commit`, and `--output-dir`.
There is no latest, fuzzy, glob or timestamp discovery.

The Stage195-C and Stage195-B paths are existing, distinct immediate children
of `<repo-root>/reports`, respectively prefixed
`stage195c_parameter_swa_causal_analysis_` and
`stage195b_tail3_parameter_swa_runs_`.  The output is a distinct absent or
entirely empty immediate reports child prefixed
`stage196a_persistent_support_boundary_localization_`.  All supplied resolved
paths are distinct.

Before any output is written, the diagnostic commit must be lowercase 40-hex,
equal repository HEAD, and the committed blobs of this specification and
analyzer must be byte-identical to current bytes.  Both files must have no
staged or unstaged modification.  Git is invoked only as a read-only argument
array with `shell=False`; global worktree cleanliness is not required.

## Stage195-B scalar and identity closure

Each exact six-run immediate child must contain, at minimum,
`clean_dev_scalars.jsonl`, `clean_dev_predictions.json`,
`training_report.json`, `stage191_dev_predictions_epoch_020.jsonl`, and
`stage195_tail3_parameter_swa_contract.json`.  Stage195-C itself supplies the
authoritative epoch-18/19/20 and SWA row evidence; the repeated epoch-20 and SWA
contract artifacts are independently aligned to it.

The trainer implementation freezes these source schemas:

- Stage191 epoch prediction rows have exactly
  `epoch,dev_position,source_row_id,gold_final_label,predicted_final_label,final_logits,final_ce,frame_logit`.
- `clean_dev_predictions.json` has exact top-level keys `metadata,predictions`.
  Prediction rows are the additive Stage28-E export.  Fields consumed when
  actually present are `id,stable_id,source_id,pair_id,example_index,claim,evidence,intervention,intervention_type,normalized_intervention,gold_label,gold_final_label,pred_label,pred_final_label,final_logits,frame_logit,frame_prob,predicate_coverage_prob,sufficiency_prob,entitlement_prob,polarity_margin,positive_energy,negative_energy`.
- The Stage115 scalar exporter always emits base keys
  `id,claim,evidence,gold_label,prediction,frame_logit,frame_prob,score_source`
  and additively copies present Stage113 fields.  Stage196-A requires the
  present numeric subset `predicate_coverage_prob,sufficiency_prob,
  entitlement_prob,polarity_margin,positive_energy,negative_energy`, plus the
  base `frame_logit,frame_prob`.  No other optional field is invented.

The physical Stage115 row does not serialize `dev_position`.  Its trainer
contract preserves prediction-row order and unique `id`; Stage196-A therefore
defines a joined scalar record whose `dev_position` is the already-validated
Stage191/Stage195-C position at the same list index, after exact ID/gold/
prediction alignment with `clean_dev_predictions.json`.  This is positional
joining, not synthesized source metadata.  The joined records are exactly 720
and positions are exactly 0--719.  Every required scalar is finite; probabilities
are also in [0,1].

Stage115 scalar export is not intrinsically a final-epoch export.
`clean_dev_predictions.json` and `clean_dev_scalars.jsonl` both come from the
trainer-selected `_best_dev_predictions` after loading the selected best state.
Their authoritative `scalar_source_epoch` is `runs["single"]["best_epoch"]`.
Stage191 epoch 20 and Stage195-C epoch-20 fields are the distinct final-epoch
snapshot.  Stage196-A loads the exact
`stage191_dev_predictions_epoch_NNN.jsonl` named by the selected epoch and
validates the selected trajectory's canonical label/logit alignment, then clean
predictions and all eight full-channel scalars against that single selected
snapshot.  Independently, it validates the epoch-20 trajectory
against Stage195-C epoch-20 labels and logits.  A selected prediction may differ
from epoch 20 without source corruption.

Localization is scientifically runnable only when the selected scalar epoch is
one of the frozen tail3 source epochs 18, 19, or 20.  A selected epoch outside
that set BLOCKS instead of mixing its full-channel scalars with an epoch-20
frame logit or persistent-tail evidence.  Selected and epoch-20 trajectories
must retain identical gold/source-row identity.  Stage115's six-decimal native
frame-logit serialization must equal the selected trajectory frame logit under
the trainer's exact `round(value, 6)` rule.

The Stage195-B training report does not use top-level `training_seed` or
`resolved_split_seed` as authoritative identity fields.  Its authoritative
training/split identity is the exact `split_seed_contract` and `configuration`
dictionaries.  The split contract freezes the training seed, configured and
resolved split seed 174, explicit fixed-split policy, and 2,880/720 clean-main
train/dev cardinalities.  Configuration independently freezes the training and
deterministic seed fields, configured/resolved split, architecture, backbone,
model, CUDA device, and external/time-swap denials.  These identities must agree
with the manifest-derived seed, clean-prediction metadata, Stage195-C rows, and
SWA contract.

Top-level `runs` is a separate dictionary with exact key `"single"`.
`runs["single"]` is the trainer-internal namespace and records
`run_name="single"`, final epoch 20, and a valid best epoch.  Manifest run IDs
such as `seed180_baseline` remain external run-directory identities and are
never required as trainer run names.  The training report's Stage115 scalar
path must still resolve exactly to that external run directory's
`clean_dev_scalars.jsonl`.  No external/OOD use or `time_swap` main-clean use is
allowed.

Stable identity uses an actually serialized `id`/`stable_id`/`source_id`; pair
ID and intervention type use actually serialized fields.  Missing metadata is
output as null, never guessed.  Pair analysis is unavailable, not blocking,
unless both pair ID and intervention type are present and usable.

This training-report namespace correction changes no 127-row persistent
population, scalar join, native threshold 0.5, mechanism bucket, recurrence
logic, Stage196-A decision taxonomy, or exact nine-output schema.

## Populations and native channel localization

For each run, gold SUPPORT rows are partitioned, in this order, into mutually
exclusive populations:

1. `persistent_stable_support_negative`: epochs 18, 19 and 20 all predict
   NOT_ENTITLED.
2. `stable_correct_support_control`: all three predict SUPPORT.
3. `temporal_support_consensus_outlier`: the Stage195-C
   `target_support_consensus_outlier` flag is true and the row is not above.
4. `other_gold_support`: all remaining gold SUPPORT rows.

Native passes are `frame_prob >= 0.5`, `predicate_coverage_prob >= 0.5`,
`sufficiency_prob >= 0.5`, and `entitlement_prob >= 0.5`.
`all_local_channels_pass` is the conjunction of the first three.

Persistent rows are classified in exact evaluation order into one and only one
bucket:

1. `MULTI_LOCAL_CHANNEL_FAILURE`: at least two local channels fail.
2. `FRAME_ONLY_FAILURE`: only frame fails.
3. `PREDICATE_ONLY_FAILURE`: only predicate coverage fails.
4. `SUFFICIENCY_ONLY_FAILURE`: only sufficiency fails.
5. `ENTITLEMENT_AGGREGATION_FAILURE`: all local channels pass and entitlement
   probability is below 0.5.
6. `FINAL_COMPOSITION_BOUNDARY_FAILURE`: all local channels and entitlement
   pass while epoch 20 still predicts NOT_ENTITLED.

Missing, multiply assigned or unassigned persistent rows BLOCK the analysis.
These are descriptive localization labels, not causal proof or architecture
prescriptions.

## Logit, recurrence, pair and arm diagnostics

For epoch 20 and parameter SWA, finite three-class logits produce SUPPORT minus
NOT_ENTITLED, SUPPORT minus REFUTE, top-1/top-2 margin, and 1-based
NOT_ENTITLED/SUPPORT ranks using canonical column order for ties.  Count, mean,
median, minimum and maximum are reported for persistent negatives, stable
correct controls and temporal SUPPORT outliers.  No p-value, interval or
significance test is computed.

Within each arm, every gold SUPPORT dev position is aligned across the three
seeds.  Output records persistent and stable-correct seed counts (0--3), bucket
distribution and scalar mean/min/max.  Persistent count at least two is
`recurrent_persistent_within_arm`; count three is
`universal_persistent_within_arm`.  Cross-arm summaries count baseline-only,
intervention-only and shared recurrence plus positions persistent in all six
runs.  Repeated seed rows are not treated as independent samples.

When frozen metadata supports it, gold SUPPORT `none` and `paraphrase` rows are
paired within the actual pair ID and summarized as both/one/neither persistent,
both stable correct, and one-stable/one-persistent.  Otherwise status is exactly
`not_available_from_frozen_source_schema` and no values are inferred.

Baseline and intervention are paired by exact seed and dev position.  The four
persistent transitions are baseline persistent to intervention persistent,
baseline persistent to intervention stable correct, baseline stable correct to
intervention persistent, and neither persistent.  Their mechanism buckets and
intervention-minus-baseline scalar deltas are descriptive only.

## Primary estimands and decision taxonomy

Primary estimands are pooled persistent bucket counts by arm, recurrent and
universal position counts by arm, all-local-pass share, entitlement-aggregation
share, final-composition share, dominant local failure channel, and recurrence
overlap between arms.

Decision order is:

1. `STAGE196A_PERSISTENT_SUPPORT_BOUNDARY_LOCALIZATION_BLOCKED` for any
   provenance, schema, alignment, cardinality, finite-value, bucket or
   calculation failure.
2. `STAGE196A_RECURRENT_LOCAL_CHANNEL_FAILURE` when recurrence exists in at
   least one arm, local-failure buckets are at least two thirds of pooled
   persistent rows, and some specific local channel fails on at least half.
3. `STAGE196A_RECURRENT_ENTITLEMENT_AGGREGATION_FAILURE` when recurrence exists
   and that bucket is at least two thirds.
4. `STAGE196A_RECURRENT_FINAL_COMPOSITION_BOUNDARY_FAILURE` when recurrence
   exists and that bucket is at least two thirds.
5. `STAGE196A_SEED_SPECIFIC_BOUNDARY_VARIANCE` when persistent rows exist,
   neither arm has recurrence and a strict majority of unique persistent
   arm-position occurrences occur in exactly one seed.
6. `STAGE196A_MIXED_PERSISTENT_MECHANISMS` for every other integrity-passing
   result.

If more than one dominant condition is true, the selected decision is mixed.

## Outputs, failure and publication

Exactly nine files are published:

1. `stage196a_persistent_support_boundary_report.json`
2. `stage196a_persistent_support_boundary_report.md`
3. `stage196a_run_population_summary.csv`
4. `stage196a_persistent_row_localization.jsonl`
5. `stage196a_mechanism_bucket_summary.csv`
6. `stage196a_cross_seed_recurrence.csv`
7. `stage196a_paired_arm_transition.csv`
8. `stage196a_source_closure.csv`
9. `stage196a_precommitted_decision_gate.csv`

READY persistent JSONL has exactly 127 rows and the minimum fields requested by
the Stage196-A contract, including complete epoch-20/SWA logits, margins and
recurrence flags.  The run summary records the four mutually exclusive SUPPORT
population counts.  Mechanism summary records per-run, per-arm and pooled
buckets.  Cross-seed rows contain one record per arm and gold SUPPORT position.
Paired-arm rows contain one record per seed and dev position.  Source closure
uses exact header `scope,run,gate,required,observed,passed,blocking_reason` and
decision gate uses exact header
`decision,taxonomy_condition,required,observed,passed`.

Before safe output-path establishment, failure writes nothing and exits
nonzero.  Afterwards every exception publishes all nine fixed-schema/header
BLOCKED artifacts, with runnable false, a blocking reason and exception type,
message and traceback, then exits nonzero.  BLOCKED JSONL is empty.  READY data
is calculated and checked in memory.  Exclusive private files are fsynced and
atomically renamed, with the report JSON last; publication failure removes only
the nine exact targets/private files.  Existing output is never overwritten.

Every integrity-passing result keeps architecture/loss selection, calibration,
threshold tuning, automatic Stage196-B authorization, production advancement,
statistical significance and external generalization false.
