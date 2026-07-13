# Stage181-A stratified frame roadmap policy

## Identity and unit of analysis

The analysis unit is a unique source item, never a review instance. The
Stage180-A hidden item key is authoritative for hard/control, match, and repeat
identity. `row_id` is the only cross-artifact identity key. `stable_row_index`
is diagnostic metadata and must not be used as identity. Repeat instances are
used only to validate consistency and never increase item counts.

## Provisional status and action boundary

All Stage180 annotations are `single_reviewer_provisional`. High confidence is
not a confirmed label or causal explanation. Data rewrite, relabel, exclusion,
or filtering proposals remain proposals until human adjudication. Model-side
failure loci are diagnostic taxonomy, not causal proof.

Stage181-A performs no dataset edit, automatic relabeling, intervention-generator
change, training-subset construction, filtering, loss or model modification,
training, fitting, calibration, threshold selection, checkpoint change,
external evaluation, time-swap experiment, or multi-seed experiment.

## Primary strata

Each unique source item receives exactly one primary stratum, using conservative
priority: consistency/schema holds; control classification; conflicting evidence;
then data, model, or semantic evidence.

- `DATA_INTERVENTION_REPAIR_CANDIDATE`: high-confidence hard item with a
  questionable/likely incorrect gold assessment or multi-axis, weak, or broken
  intervention, and a repair/adjudication/diagnostic-only recommended action.
  This is a human-adjudication queue, not edit authorization.
- `CLEAN_LABEL_MODEL_FAILURE_CANDIDATE`: high-confidence hard item with
  `gold_consistent`, a clean/canonical intervention, and representation,
  head/readout, or downstream-boundary locus, without a data-repair action.
- `GENUINELY_HARD_SEMANTIC_CANDIDATE`: hard item with consistent or
  indeterminate gold, no broken/weak intervention, and a hard/cannot-determine
  locus or an ambiguous/insufficient independent judgment.
- `ADJUDICATION_HOLD`: low confidence, repeat inconsistency, incomplete schema,
  unclear questionable-gold action, or structurally incompatible axes.
- `CLEAN_CONTROL_REFERENCE`: correct native frame prediction, consistent gold,
  clean/canonical validity, sufficient Pass 1 and Pass 2 confidence, and no
  data/model flag.
- `CONTROL_ANOMALY`: a control with questionable gold, broken/weak/multi-axis
  construction, ambiguity/insufficient context, an incorrect native frame head,
  or a data/model issue flag. It cannot be used as hard/model evidence.
- `MIXED_EVIDENCE`: multiple substantive hard-item strata match, or strong
  data-side and model-side evidence coexist.

## Secondary tags

Items may receive multiple tags: `DATA_GRAMMAR_ARTIFACT`,
`DATA_POLARITY_LEAK`, `DATA_MULTI_AXIS_EDIT`, `DATA_WEAK_EDIT`,
`LABEL_ADJUDICATION_REQUIRED`, `MODEL_REPRESENTATION_CANDIDATE`,
`MODEL_READOUT_CANDIDATE`, `MODEL_FINAL_BOUNDARY_CANDIDATE`,
`HARD_REFERENT_RESOLUTION`, `HARD_PREDICATE_SCOPE`, `HARD_POLARITY`,
`HARD_TEMPORAL`, `HARD_EVENT_IDENTITY`, `CONTROL_CONTAMINATION`,
`LOW_CONFIDENCE`, and `PROVISIONAL_AI_REVIEW`. Every item must include
`PROVISIONAL_AI_REVIEW`.

## Matched-control interpretation

Each hard item is compared only with the control linked by the hidden key.
Hard-only construction artifacts support intervention-construction review;
shared artifacts support pair/template-level review. A hard-only model-side
failure with a clean control supports a clean model-failure candidate. An
anomalous control weakens the contrast and never becomes hard/model evidence.

## Descriptive aggregation and priority

Beneficial-25 and harmful-14 cohorts are summarized separately. Risk differences
and two-sided Fisher exact tests are descriptive only. Intervention-family
summaries are descriptive at every support size; support below three is not a
priority gate.

Data and model priority scores are fixed rule-based sorting aids, not fitted
statistics, probabilities, scientific metrics, or causal effect estimates.

## Decision gate

The analyzer emits exactly one of:

1. `STAGE181A_PROVISIONAL_DUAL_TRACK_ROADMAP_READY` when both data and model
   candidate counts are at least five, each hard-item rate is at least 0.15,
   and controls do not account for the contrast.
2. `STAGE181A_PROVISIONAL_DATA_REPAIR_ROADMAP_READY` when data candidates are at
   least eight, exceed the model rate by at least 0.20, and high-confidence data
   issues are clearly higher than in controls.
3. `STAGE181A_PROVISIONAL_MODEL_FAILURE_ROADMAP_READY` when model candidates are
   at least eight, exceed the data rate by at least 0.20, matched controls are
   clean, and label/data issues are at most three.
4. `STAGE181A_SECOND_REVIEW_OR_HUMAN_ADJUDICATION_REQUIRED` when hold/mixed is at
   least 0.30, control anomaly is at least 0.20, confidence or axes are
   inadequate, or no provisional roadmap gate is stable.

The adjudication conditions are a safety override. Stage182 is specification or
adjudication only; no Stage181 decision authorizes execution.

