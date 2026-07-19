# Stage195-C parameter-SWA causal analysis specification

## Scope

Stage195-C is an artifact-only, diagnostic-only analysis of the completed frozen
Stage195-B six-run matrix. It asks whether the epoch-18/19/20 trainable-parameter
SWA corrects late-epoch temporal-consensus outliers relative to epoch 20. It
does not train, import project model code, load a model, tokenizer, checkpoint,
or state capsule, search aggregation rules, calibrate logits, move an
entitlement boundary, select a production model, authorize later training, or
claim statistical significance or external generalization.

Only JSON, JSONL, CSV, Markdown, Git blobs, and Git status are read. Python
`float` is the binary64 calculation type. The analyzer never uses a GPU.

## Frozen identities and matrix

- `stage195a_runtime_repository_commit`:
  `daddd0eb6f21514ed074f63defa0313323cef555`
- `trainer_blob_commit`:
  `bd27e46daf218a57da9a3142c9e4bc5cc44ad53a`
- `trainer_blob_sha256`:
  `4fe903c9f3aa21ee6365a0297c27e4a333d295dbb851384efc7bc8d3f7607954`
- training seeds: exact non-bool integers 180, 181, 182
- split seed: exact non-bool integer 174
- source epochs: exact non-bool integers 18, 19, 20
- canonical label/logit/tie order: `REFUTE`, `NOT_ENTITLED`, `SUPPORT`
- prediction rows per export: 720; gold SUPPORT rows: 89
- run order: `seed180_baseline`, `seed180_intervention`,
  `seed181_baseline`, `seed181_intervention`, `seed182_baseline`,
  `seed182_intervention`

The analyzer requires only `--repo-root`, `--stage195a-dir`,
`--stage195b-run-root`, `--current-diagnostic-git-commit`, and `--output-dir`.
It performs no latest, fuzzy, glob, or timestamp discovery.

### Run-identity namespaces

`manifest_run_id` and `trainer_run_name` are distinct namespaces and are never
compared as if they were the same value.

- `manifest_run_id` is one of the six `seedNNN_baseline|intervention` values. It
  identifies the Stage195-A manifest row, the Stage195-B immediate child
  directory, and the `run` column/field in every Stage195-C row-transition and
  summary output.
- `trainer_run_name` is exactly `"single"` for the frozen one-run trainer
  invocation. The `run` field in Stage195-P0 SWA predictions, metrics, and
  contract belongs to this namespace and must equal `"single"`; it is not the
  manifest run ID.
- The frozen Stage191 trajectory prediction eight-key schema and trajectory
  contract do not currently serialize a `run` field. Their manifest association
  is therefore proven by the exact run directory, manifest paths, hashes, and
  exact seed/split/arm contract. If a Stage191 contract exposes a `run` field,
  it belongs to the trainer namespace and must equal `"single"`; no
  manifest-run comparison is introduced.

This namespace correction repairs source-schema interpretation only. It changes
no causal estimand, metric, aggregation, decision taxonomy, or nine-file output
schema.

## Path and source-code provenance closure

`stage195a-dir` and `stage195b-run-root` are existing immediate children of
`<repo-root>/reports` with prefixes
`stage195a_tail3_parameter_swa_manifest_` and
`stage195b_tail3_parameter_swa_runs_`. The output is an absent or entirely
empty immediate reports child with prefix
`stage195c_parameter_swa_causal_analysis_`. All five supplied resolved paths
are distinct.

`stage195c_runtime_repository_commit` is the supplied lowercase 40-hex commit.
Read-only Git argument-array calls with `shell=False` prove that it equals HEAD,
that the committed blobs for this specification and the analyzer are
byte-identical to current bytes, and that neither file has staged or unstaged
changes. Global worktree cleanliness is not required.

The report keeps these fields separate:
`stage195c_runtime_repository_commit`,
`stage195a_runtime_repository_commit`, `trainer_blob_commit`, and
`trainer_blob_sha256`. It defines no ambiguous `trainer_source_commit` field.

## Stage195-A closure

The supplied Stage195-A directory contains exactly its six specified files.
Its report and six JSONL rows have the exact schemas frozen by Stage195-A. The
report is `STAGE195A_TAIL3_PARAMETER_SWA_MANIFEST_READY`, runnable,
diagnostic-only, blocker-free, and records the exact identities, matrix,
cardinalities, labels, and denial flags. Every source/template and precommitted
gate has exact header `gate,required,observed,passed,blocking_reason`, passes,
and has an empty reason. The command matrix has its exact Stage195-A header.

Each manifest row has the frozen seed, split seed, arm contract, runtime and
trainer identities, cardinalities and denial flags. Baseline has margin weight
0.0, margin logit 0.0, and null sidecar fields. Intervention has weight 0.05,
margin logit 0.0, the frozen sidecar semantic SHA, and a non-null sidecar path.
Each planned run directory is exactly the corresponding immediate child of the
supplied Stage195-B root; every expected artifact path is that child's frozen
filename.

## Per-run artifact closure

The Stage195-B root contains exactly the six run directories. Extra ordinary
training outputs inside a run are permitted, but these files are required:
`training_report.json`, `stage191_trajectory_contract.json`,
`stage191_trajectory_epoch_metrics.jsonl`, all twenty
`stage191_dev_predictions_epoch_NNN.jsonl` files, and the three Stage195 SWA
files. Exactly twenty prediction exports are enumerated; exactly twenty ledger
rows cover epochs 1--20 and bind every export path and SHA256.

Every trajectory export has the exact Stage191 eight-key row schema:
`epoch,dev_position,source_row_id,gold_final_label,predicted_final_label,final_logits,final_ce,frame_logit`.
It contains 720 rows at positions 0--719, canonical labels, finite length-three
logits, canonical argmax predictions, and finite nonnegative CE. Epoch,
position, identity, and cardinality integers reject booleans. Gold labels align
across all epochs and all runs and contain exactly 89 SUPPORT rows.

The trajectory contract proves Stage195 mode, exact run identity, canonical
labels, 20 epochs, 720 rows, 89 SUPPORT rows, source epochs `[18,19,20]`, zero
capsules, no external data, no calibration or boundary shift, and the frozen
runtime/trainer identity. Training report seed/split/run/arm and provenance are
cross-checked when those identities are recorded by the frozen runtime.

SWA prediction rows have their exact twelve-key Stage195-P0 schema. SWA metrics
have their exact twenty-four-key schema. The final contract proves exact source
epochs and capture count, CPU float64 accumulation, original-dtype casting,
epoch-20 restoration and identical epoch-20/restored fingerprints, one
post-training clean-dev forward, no optimizer/scheduler averaging, checkpoint
selection, SWA checkpoint, calibration, entitlement shift, external data, or
state capsules. Its trainer blob identity is frozen. Contract prediction and
metric paths are exact, and both SHA256 values are recomputed from file bytes.
No run directory contains a Stage195/SWA-named `.pt`, `.pth`, `.bin`, or
`.safetensors` file, nor any Stage191 state capsule.

## Frozen reconstructed predictors

For each aligned row only epoch 18, 19, and 20 final logits are retained.

1. Epoch 20 is the exported epoch-20 predictor and primary control.
2. Tail-three mean logits are classwise `math.fsum` divided by 3, followed by
   canonical argmax.
3. Tail-three majority is the label occurring at least twice. If all labels
   differ, `majority_available` is false and its prediction is JSON null. No
   tie-break or mean substitution is used.
4. Parameter SWA is read verbatim from its prediction artifact; its logits are
   never recomputed.

No median, trimmed mean, robust mean, EMA, alternative window, calibration, or
logit shift is calculated.

## Causal estimands and mechanisms

`temporal_consensus_outlier` means majority is available and epoch 20 differs
from that majority. Subtype is exactly one of `consensus_correct`,
`epoch20_correct`, or `both_wrong`; non-outliers use `not_temporal_outlier`.

On an outlier, SWA response is `swa_rescue` for epoch20-wrong/SWA-correct,
`swa_harm` for epoch20-correct/SWA-wrong,
`swa_wrong_label_change` for both wrong with a changed label, and otherwise
`swa_unchanged`. The primary per-run endpoint is
`swa_rescue_count - swa_harm_count`. Alignment with majority and mean logits is
reported independently.

A target SUPPORT consensus outlier has gold SUPPORT, epoch20 NOT_ENTITLED,
available majority SUPPORT. Its SWA SUPPORT rescue, unchanged NOT_ENTITLED,
other-label, and mean-logit SUPPORT counts and rates are reported. A persistent
stable SUPPORT negative has gold SUPPORT and NOT_ENTITLED at all three epochs.
Its SWA rescue/remain and mean-remain counts and rates are a separate
boundary-bias endpoint and are never merged with the temporal endpoint. A zero
denominator produces null.

Metrics are reconstructed from artifact logits/predictions: mean clean CE,
accuracy, three-class macro F1, SUPPORT recall, false entitlement (gold
NOT_ENTITLED predicted REFUTE or SUPPORT), false NOT_ENTITLED (gold REFUTE or
SUPPORT predicted NOT_ENTITLED), and polarity error (REFUTE/SUPPORT cross-error).
Counts, confusion matrices, total corrected/harmed/net rows, the six requested
label transitions, prediction agreement geometry, and mean/median L1/L2
artifact-logit distances are emitted. Distance median is a summary statistic,
not a predictor.

Seed pairs are exact baseline/intervention pairs. Arm summaries contain
three-run arithmetic mean, median, minimum, maximum, positive/zero/negative
seed direction counts, and pooled row-count totals. Pooled rows are descriptive
totals only because dev positions repeat across seeds. No p-value, confidence
interval, or significance test is calculated.

## Decision taxonomy and order

Evaluation order is BLOCKED, replicated harm, replicated support or its
trade-off variant, no support, then mixed/inconclusive.

1. `STAGE195C_PARAMETER_SWA_CAUSAL_ANALYSIS_BLOCKED`: any provenance, schema,
   hash, cardinality, alignment, or calculation failure.
2. `STAGE195C_PARAMETER_SWA_REPLICATED_CAUSAL_HARM`: both arm pooled endpoints
   are negative, at least two of three seeds per arm are negative, and pooled
   harms exceed rescues in both arms.
3. Temporal primary support requires both arm pooled endpoints positive, at
   least two positive seeds per arm, and harms strictly below rescues in both
   arms. It is
   `STAGE195C_PARAMETER_SWA_REPLICATED_TEMPORAL_CAUSAL_SUPPORT` when both arms'
   pooled false-entitlement and polarity deltas are nonpositive and there is no
   trade-off trigger. It is
   `STAGE195C_PARAMETER_SWA_TEMPORAL_SUPPORT_WITH_BOUNDARY_TRADEOFF` if any arm
   has positive pooled false-entitlement or false-NOT_ENTITLED delta, negative
   three-run mean macro-F1 delta, or non-uniform seed direction in any of those
   safety deltas.
4. `STAGE195C_PARAMETER_SWA_NO_TEMPORAL_CAUSAL_SUPPORT`: neither replicated
   support nor harm, and overall pooled endpoint is nonpositive, or there is no
   positive outlier response because SWA mostly preserves epoch 20. “Mostly” is
   frozen as `pooled_swa_unchanged_count >= pooled_temporal_outlier_count / 2`
   with overall pooled endpoint nonpositive.
5. `STAGE195C_PARAMETER_SWA_MIXED_OR_INCONCLUSIVE`: every other
   integrity-passing result, including arm/seed disagreement, positive pooled
   effect without replication, mixed temporal rescue and persistent boundary
   harm, or fewer than 12 target SUPPORT consensus-outlier rows overall. The
   last threshold is a descriptive mechanistic-stability rule, not inference.

Every taxonomy row records the textual condition, required Boolean, structured
observed values, and whether the selected decision matches that condition.
Exactly one row has `required=true` and `observed=true` in a READY report.

## Exact outputs and schemas

Exactly nine files are published:

1. `stage195c_parameter_swa_causal_report.json`
2. `stage195c_parameter_swa_causal_report.md`
3. `stage195c_run_summary.csv`
4. `stage195c_row_transition.jsonl`
5. `stage195c_temporal_outlier_transition.csv`
6. `stage195c_support_mechanism_summary.csv`
7. `stage195c_paired_seed_arm_delta.csv`
8. `stage195c_source_closure.csv`
9. `stage195c_precommitted_decision_gate.csv`

The report JSON exact keys are:
`stage,decision,runnable,blocking_reasons,diagnostic_only,artifact_only_analysis,training_performed,model_loaded,tokenizer_loaded,checkpoint_loaded,state_capsule_loaded,external_data_used,stage195c_runtime_repository_commit,stage195a_runtime_repository_commit,trainer_blob_commit,trainer_blob_sha256,canonical_labels,source_epochs,ordered_runs,stage195a_directory,stage195b_run_root,row_transition_count,all_twenty_exports_validated_per_run,run_summaries,arm_aggregates,paired_seed_deltas,overall_pooled,decision_taxonomy,production_swa_selected,model_advancement_authorized,subsequent_training_authorized,entitlement_correction_implemented,calibration_authorized,ema_authorized,statistical_significance_claimed,external_generalization_claimed,parameter_averaging_adopted_as_final_architecture,interpretation_restrictions,exception`.

`stage195c_row_transition.jsonl` has exactly 4,320 rows and exact keys in this
order:
`stage,run,seed,arm,split_seed,dev_position,gold_label,epoch18_prediction,epoch19_prediction,epoch20_prediction,mean_logit_prediction,majority_available,majority_prediction,swa_prediction,epoch20_correct,mean_logit_correct,majority_correct,swa_correct,temporal_consensus_outlier,temporal_outlier_subtype,swa_transition_type,swa_aligns_majority,swa_aligns_mean_logit,target_support_consensus_outlier,persistent_stable_support_negative,epoch18_logits,epoch19_logits,epoch20_logits,tail3_mean_logits,swa_logits,swa_vs_epoch20_l1,swa_vs_epoch20_l2,swa_vs_mean_logit_l1,swa_vs_mean_logit_l2`.
Unavailable majority prediction and correctness are null.

CSV headers, each in exact order, are:

- run summary:
  `run,seed,arm,row_count,temporal_outlier_count,consensus_correct_outlier_count,epoch20_correct_outlier_count,both_wrong_outlier_count,swa_rescue_count,swa_harm_count,swa_wrong_label_change_count,swa_unchanged_count,temporal_outlier_net_correctness_change,epoch20_clean_ce,swa_clean_ce,clean_ce_delta,epoch20_accuracy,swa_accuracy,accuracy_delta,epoch20_macro_f1,swa_macro_f1,macro_f1_delta,epoch20_support_recall,swa_support_recall,support_recall_delta,epoch20_false_entitlement_total,swa_false_entitlement_total,false_entitlement_delta,epoch20_false_not_entitled_total,swa_false_not_entitled_total,false_not_entitled_delta,epoch20_polarity_error_total,swa_polarity_error_total,polarity_error_delta,epoch20_pred_counts,swa_pred_counts,total_corrected_rows,total_harmed_rows,net_correctness_change,correct_not_entitled_to_false_entitlement,false_entitlement_to_correct_not_entitled,epoch20_SUPPORT_to_swa_NOT_ENTITLED,epoch20_NOT_ENTITLED_to_swa_SUPPORT,epoch20_REFUTE_to_swa_NOT_ENTITLED,epoch20_NOT_ENTITLED_to_swa_REFUTE,swa_epoch20_agreement_rate,swa_mean_logit_agreement_rate,swa_majority_agreement_rate,swa_majority_temporal_outlier_agreement_rate,swa_mean_temporal_outlier_agreement_rate,swa_and_mean_both_correct,only_swa_correct,only_mean_logit_correct,swa_and_mean_both_wrong,swa_vs_epoch20_l1_mean,swa_vs_epoch20_l1_median,swa_vs_epoch20_l2_mean,swa_vs_epoch20_l2_median,swa_vs_mean_logit_l1_mean,swa_vs_mean_logit_l1_median,swa_vs_mean_logit_l2_mean,swa_vs_mean_logit_l2_median`.
- temporal outliers:
  `run,seed,arm,temporal_outlier_count,consensus_correct_count,epoch20_correct_count,both_wrong_count,swa_rescue_count,swa_rescue_rate,swa_harm_count,swa_harm_rate,swa_wrong_label_change_count,swa_unchanged_count,swa_aligns_majority_count,swa_aligns_majority_rate,swa_aligns_mean_logit_count,swa_aligns_mean_logit_rate,net_correctness_change`.
- SUPPORT mechanisms:
  `run,seed,arm,target_support_consensus_outlier_count,target_swa_support_rescue_count,target_swa_support_rescue_rate,target_swa_remains_not_entitled_count,target_swa_remains_not_entitled_rate,target_swa_other_label_count,target_swa_other_label_rate,target_mean_logit_support_count,target_mean_logit_support_rate,persistent_stable_support_negative_count,persistent_swa_support_rescue_count,persistent_swa_support_rescue_rate,persistent_swa_remains_not_entitled_count,persistent_swa_remains_not_entitled_rate,persistent_mean_logit_remains_not_entitled_count,persistent_mean_logit_remains_not_entitled_rate`.
- paired seeds:
  `seed,metric,baseline,intervention,intervention_minus_baseline`.
- source closure:
  `scope,run,gate,required,observed,passed,blocking_reason`.
- decision gate:
  `decision,taxonomy_condition,required,observed,passed`.

## Failure and publication

Before the safe output directory is established, failure writes nothing and
returns nonzero. Afterwards every exception produces all nine fixed-schema or
fixed-header BLOCKED outputs, including exception type, message, traceback and
a blocking reason, then returns nonzero. BLOCKED row JSONL is empty. READY data
is fully constructed and schema/cardinality checked in memory. All outputs are
written to exclusive private temporary files, fsynced, and atomically renamed;
the JSON report is renamed last. Cleanup is restricted to the nine exact target
and private temporary names. No partial READY-looking report is retained.

All integrity-passing decisions keep production selection, model advancement,
later training, entitlement correction, calibration, EMA, statistical
significance, external generalization, and final-architecture adoption false.
