# Native State Kinematics
# Cohort and Measurement Feasibility Audit Authority Specification

STATUS = CANDIDATE

PHASE =
COHORT_AND_MEASUREMENT_FEASIBILITY_AUDIT

PARENT_DESIGN_AUTHORITY =
c4286a4d8af9ae31b7e44de2a3e79b560efa2355

PARENT_HYPOTHESIS =
12c86088f68482870dd53cbcf6c363499b248f81

O0C_VALIDATED_NATIVE_STATE_RESULT =
ff2fb076f6e66a34a632515bb8502d8b1c90ad7f


## 1. Authority boundary

THIS_AUTHORITY_CREATES_NEW_SCIENTIFIC_TRAJECTORY_EVIDENCE =
NO

FEASIBILITY_AUDIT_ALLOWED =
YES

TRAINING_ALLOWED =
NO

MODEL_PARAMETER_UPDATE_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_INFERENCE_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

PRIMARY_KINEMATIC_ENDPOINT_COMPUTATION_ALLOWED =
NO

POST4_SPEED_ALLOWED =
NO

POST4_TURNING_ALLOWED =
NO

POST4_PATH_EFFICIENCY_ALLOWED =
NO

CORRECT_VS_WRONG_NATIVE_STATE_COMPARISON_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

GPU_REQUIRED =
NO

PROMOTION_ALLOWED =
NO


## 2. Purpose

This authority permits only the non-outcome-leaking feasibility audit
required by the frozen first confident-error design.

The audit asks whether the first confirmatory experiment can be run
without weakening its preregistered scientific question.

The audit may establish:

1. exact source and provenance identity;
2. exact confidence availability;
3. exact input-token coordinate semantics;
4. whether a defensible decisive-evidence annotation rule can exist;
5. whether the required prefix window can exist;
6. whether an adequate confident-correct/confident-wrong cohort can exist;
7. whether prospective matched-pair sample size can meet the frozen
   planning requirement;
8. whether native recurrent-state measurement is technically provisionable
   under separately authorized future execution.

The audit must not calculate the scientific kinematic effect.


## 3. Frozen confirmatory source population

PRIMARY_CONFIRMATORY_SOURCE =
A0_REPLACEMENT_R1_SEED180_CLEAN_DEV

A0_ARM =
A0

A0_TRAINING_SEED =
180

A0_SPLIT_SEED =
8192

A0_CLEAN_DEV_ROW_COUNT =
720

A0_DATASET_FULL_ROW_COUNT =
3600

A0_TRAIN_ROW_COUNT =
2880

A0_BACKBONE =
mamba

A0_MODEL_NAME =
state-spaces/mamba-130m-hf

A0_ARCHITECTURE =
v6b_minimal

A0_ENCODER_FROZEN =
TRUE

The 720 frozen clean-dev prediction rows are the only candidate
confirmatory population under this authority.

Grouped-factorial, single-edge, pairwise, D1, and other Gen3 prediction
runs must not be substituted for the A0 source population.


## 4. Frozen source identities

### 4.1 Main dataset

A0_MAIN_DATASET_PATH =
reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl

A0_MAIN_DATASET_SHA256 =
eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3

A0_MAIN_DATASET_BYTES =
1879593

A0_MAIN_DATASET_ROWS =
3600


### 4.2 Frozen A0 prediction export

A0_PREDICTION_LOGICAL_PATH =
reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0/clean_dev_predictions.json

A0_PREDICTION_SHA256 =
5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d

A0_PREDICTION_BYTES =
4840320

A0_PREDICTION_ROWS =
720


### 4.3 Frozen A0 JSONL prediction export

A0_REPORT_PREDICTION_LOGICAL_PATH =
reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0/training_report_predictions.jsonl

A0_REPORT_PREDICTION_SHA256 =
80fef1e7fa1df6b99c797ef61dcc79bd552a65f79126f231dce47d5971ecd334


### 4.4 A0 provenance

A0_RUN_PROVENANCE_LOGICAL_PATH =
reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0/run_provenance.json

A0_RUN_PROVENANCE_SHA256 =
a758538e93e6e52ca261cb593285c298344808a3626eed7d9b9664e29a6c1a3d

A0_TRAINING_REPORT_LOGICAL_PATH =
reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0/training_report.json

A0_TRAINING_REPORT_SHA256 =
2cdf0925e3a0ef1b925f6b00ac4b2095d18a113896a437ded77622f5134b2013


### 4.5 A0 checkpoint identity

A0_SELECTED_CHECKPOINT_LOGICAL_PATH =
reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0/selected_checkpoint.pt

A0_SELECTED_CHECKPOINT_SHA256 =
4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c

A0_SELECTED_CHECKPOINT_BYTES =
518269943

The checkpoint may be byte-hashed.

It may not be instantiated or used for inference under this authority.


## 5. Prediction/source binding

The frozen A0 prediction export has already established:

A0_FINAL_OUTPUT_SCHEMA =
PASS_720_OF_720

A0_PREDICTION_STABLE_ID_UNIQUENESS =
PASS_720_OF_720

A0_STABLE_IDS_PRESENT_IN_MAIN_DATASET =
PASS_720_OF_720

Required per-row final-output fields are:

stable_id
gold_label
pred_label
external_class_order
final_logits
final_probs

The audit must fail if any frozen source byte identity differs.


## 6. Confidence statistic lock

The confidence statistic is fixed before its distribution is inspected.

CONFIDENCE_STATISTIC =
PREDICTED_CLASS_FINAL_PROBABILITY

For row i:

confidence_i =
final_probs_i[argmax(final_probs_i)]

Equivalently, because schema integrity requires argmax(final_probs)
to equal pred_label:

confidence_i =
max(final_probs_i)

CONFIDENCE_THRESHOLD =
0.5

CONFIDENT =
confidence_i >= 0.5

NOT_CONFIDENT =
confidence_i < 0.5

Rationale:

0.5 is an absolute-majority probability criterion.

It is:

- independent of native-state outcomes;
- independent of correct/wrong cohort counts;
- not selected from an observed confidence distribution;
- not a quantile tuned on the confirmatory set.

CONFIDENCE_THRESHOLD_SWEEP =
PROHIBITED

CONFIDENCE_THRESHOLD_RELAXATION_AFTER_COUNT_INSPECTION =
PROHIBITED

If the frozen threshold produces insufficient confident-wrong cases,
the feasibility audit fails for this design.

It must not silently lower the threshold.


## 7. Primary decisive-commitment population

Before tau_e and prefix eligibility are applied:

DECISIVE_PREDICTION =
pred_label in {SUPPORT, REFUTE}

CORRECT =
pred_label == gold_label

WRONG =
pred_label != gold_label

The first experiment excludes final predicted NOT_ENTITLED rows.

The feasibility audit may report counts after the confidence threshold
has been frozen by this authority.

It may report counts by:

- CORRECT versus WRONG;
- predicted SUPPORT versus predicted REFUTE;
- gold label;
- intervention type;
- primary failure type;
- pair/template identifier.

These are cohort metadata.

They are not native-state trajectory outcomes.


## 8. A0 token-coordinate lock

The A0 Mamba input coordinate follows the frozen trainer contract:

claim_ids =
tokenizer.encode(
    claim,
    add_special_tokens=False,
    truncation=True,
    max_length=63
)

evidence_ids =
tokenizer.encode(
    evidence,
    add_special_tokens=False,
    truncation=True,
    max_length=64
)

serialized input:

claim_ids
+
[separator_id]
+
evidence_ids

with:

separator_id =
tokenizer.eos_token_id

falling back to tokenizer.pad_token_id only if eos_token_id is absent.

Evidence begins at:

EVIDENCE_START =
len(claim_ids) + 1

The full model input is padded to maximum length 128.

The audit must not substitute the O0b serialization:

"Claim: ...\nEvidence: ..."

for the A0 coordinate.

O0B_ABSOLUTE_TOKEN_INDEX_DIRECT_REUSE =
PROHIBITED


## 9. Exact tokenizer identity prerequisite

A0_TOKENIZER_NAME =
state-spaces/mamba-130m-hf

However, name alone is insufficient provenance.

Before token-level tau_e annotation or prefix eligibility is accepted,
the audit must recover and bind an exact tokenizer identity sufficient
to reproduce the original A0 tokenization.

The audit may inspect:

- run_provenance.json;
- training_report.json;
- frozen trainer source;
- source provenance;
- local cached tokenizer metadata;
- repository validation artifacts.

TOKENIZER_ONLY_CPU_EXECUTION =
CONDITIONAL

Tokenizer-only execution is permitted only after exact tokenizer
identity/revision is recovered.

MODEL_FORWARD_DURING_TOKENIZER_AUDIT =
NO

NETWORK_DOWNLOAD =
NO

If exact tokenizer identity cannot be recovered from frozen/local
provenance, then:

TOKEN_COORDINATE_PROVENANCE =
BLOCKED

and the audit stops before scientific execution authority.


## 10. O0b/O0c relationship

O0b/O0c provide methodological precedent for:

- token-aligned state indexing;
- evidence-boundary validation;
- divergence-index validation;
- state-tensor serialization;
- full token-time native-state instrumentation;
- recurrent-state tensor provenance.

They do not provide the A0 tau_e annotations.

O0b used a different serialization coordinate.

Therefore:

O0B_EVIDENCE_START_AS_A0_TAU_E =
PROHIBITED

O0B_FIRST_DIVERGENCE_AS_A0_TAU_E =
PROHIBITED

O0C_ANCHOR_INDEX_AS_A0_TAU_E =
PROHIBITED

O0c artifacts may be inspected only for measurement feasibility.

No O0c result may become the confirmatory confident-error effect.


## 11. Decisive-evidence annotation problem

The design authority requires one defensible token index:

tau_e

for each admitted confirmatory example.

tau_e means:

the token-time location of a prespecified
conclusion-critical evidence event

not merely:

- evidence segment start;
- evidence segment end;
- arbitrary midpoint;
- first available token;
- final token;
- a native-state change point.

The main dataset does not directly contain:

evidence_start
evidence_end
evidence_span
decisive_evidence_index
tau_e

Therefore:

DIRECT_TAU_E_METADATA =
NO

TAU_E_REQUIRES_DETERMINISTIC_SEMANTIC_ANNOTATION =
YES


## 12. Tau-e anti-leakage phases

The feasibility audit must separate tau_e construction from model outcome
inspection.

### Phase A — semantic annotation feasibility

Phase A may read only:

- dataset rows;
- pair_id;
- claim;
- evidence;
- intervention_type;
- gold semantic labels;
- generator/source code;
- validated regeneration metadata;
- tokenizer identity and tokenization after identity binding.

Phase A must not read:

- pred_label;
- final_probs;
- final_logits;
- is_correct;
- model native states;
- P1/P2/P3 measurements.

Phase A must determine whether an exact deterministic semantic rule can map
eligible examples to a defensible conclusion-critical event.

A valid rule may depend on known dataset semantics such as:

- intervention type;
- structured fact slot;
- polarity/negation operator;
- localized substituted semantic slot;
- validated omission boundary where representable in consumed token time.

It may not depend on whether the model was correct or wrong.


### Phase B — annotation freeze candidate

Before prediction rows are joined, Phase A must produce:

TAU_E_RULE_CANDIDATE

and:

TAU_E_ANNOTATION_MANIFEST_CANDIDATE

with exact SHA256 identities.

The annotation manifest must contain, at minimum:

stable_id
tau_e_status
tau_e_absolute_token_index_or_null
event_type
annotation_basis
evidence_start
terminal_index
post4_prefix_eligible
exclusion_code

The rule candidate and manifest candidate must be treated as immutable
for the subsequent cohort-count portion of the audit.


### Phase C — cohort join

Only after Phase B identities exist may the audit join annotations to the
frozen A0 prediction rows and inspect confident correct/wrong counts.

No tau_e rule may be changed in response to those counts.


## 13. Tau-e admissibility

A tau_e annotation is admissible only if:

1. it is deterministic from allowed semantic inputs;
2. it maps to the exact A0 consumed-token coordinate;
3. it is conclusion-critical rather than merely segment-boundary metadata;
4. it is independent of model correctness;
5. it is independent of confidence;
6. it is independent of native-state values;
7. its semantic basis can be audited;
8. the event token is actually consumed by the frozen input.

When no defensible consumed token represents the semantic event:

tau_e_status =
UNANNOTATABLE

The row is excluded from the primary confirmatory cohort.

The audit must not invent a convenient surrogate.


## 14. Special omission/truncation rule

Evidence deletion or truncation may create a conclusion-critical
absence rather than a consumed decisive token.

An absent token is not automatically a valid tau_e.

For:

evidence_deletion
evidence_truncation

the annotation system must fail closed unless a defensible consumed-token
boundary can satisfy the design's event semantics and prefix requirement.

If not:

tau_e_status =
UNANNOTATABLE_ABSENCE_EVENT

No synthetic absent-token position may be inserted into model time.


## 15. Prefix feasibility

For each annotated example define:

T =
last non-padding consumed token index

The frozen primary prefix requirement is:

tau_e + 4 <= T - 1

Therefore:

POST4_PREFIX_ELIGIBLE =
TRUE

iff the above inequality holds.

Rows that fail it are excluded from the primary confirmatory cohort.

The audit may report:

- annotated count;
- unannotatable count;
- prefix-eligible count;
- prefix-ineligible count;
- counts by semantic event/intervention family.

It may not calculate any native-state endpoint.


## 16. Native recurrent-state measurement feasibility

The audit may establish whether the required native state can in principle
be captured.

Existing O0c evidence establishes methodological feasibility for direct
native selective-SSM recurrent-state instrumentation.

However, the first confident-error experiment requires provenance binding
to the frozen A0 encoder/input coordinate.

Therefore the audit must distinguish:

GENERAL_NATIVE_STATE_INSTRUMENTATION_FEASIBILITY

from:

A0_EXACT_BACKBONE_BINDING

The audit may inspect:

- frozen encoder declaration;
- model/checkpoint metadata;
- checkpoint file identity;
- source provenance;
- O0c instrumentation contracts;
- model/tokenizer identities.

CHECKPOINT_BYTE_HASHING =
YES

CHECKPOINT_MODEL_INSTANTIATION =
NO

CHECKPOINT_FORWARD =
NO

NATIVE_STATE_FORWARD =
NO

If exact A0 encoder identity cannot be bound sufficiently for later
reproducible native-state extraction:

A0_NATIVE_STATE_BINDING_FEASIBILITY =
BLOCKED

A later narrow metadata/weight-identity authority may then be required.


## 17. Prospective sample-size rule

The sample-size gate is fixed before any native-state effect is observed.

PRIMARY_PLANNING_TEST =
PAIRED_TWO_SIDED

PRIMARY_FAMILY_SIZE =
3

FAMILY_WISE_ALPHA =
0.05

CONSERVATIVE_PLANNING_ALPHA_PER_ENDPOINT =
0.016666666666666666

TARGET_POWER =
0.80

SMALLEST_EFFECT_OF_INTEREST_STANDARDIZED_PAIRED =
0.50

MINIMUM_MATCHED_PAIR_COUNT =
45

The 45-pair planning gate is a conservative approximate requirement for
80% power to detect a paired standardized effect of 0.5 under a two-sided
alpha approximately equal to 0.05 / 3.

The eventual confirmatory analysis remains the frozen matched-pair
permutation/Holm design.

This power gate must not be recomputed from observed native-state effects.


## 18. Predicted-class representation gate

The primary design preserves predicted SUPPORT and predicted REFUTE strata.

For feasibility:

TOTAL_MATCHED_PAIR_CAPACITY_REQUIRED =
AT_LEAST_45

and:

MINIMUM_POTENTIAL_MATCHED_PAIRS_PER_REPRESENTED_PREDICTED_CLASS =
10

If one decisive predicted class has fewer than 10 feasible correct/wrong
pairings, that stratum is insufficient for the planned cross-stratum
directional consistency check.

The audit must report this explicitly.

It must not merge SUPPORT and REFUTE to hide stratum failure.


## 19. Matching feasibility only

This audit does not freeze the final matching algorithm.

It may compute deterministic feasibility summaries using only:

- predicted class;
- fixed confidence;
- input token length;
- tau_e;
- intervention/event family;
- pair/template identity.

It may report:

- maximum possible one-to-one pair count under exact predicted-class strata;
- confidence overlap;
- input-length overlap;
- tau_e overlap;
- intervention/event-family overlap.

It must not inspect native-state outcomes while choosing a matching rule.

The final exact matching algorithm requires a later authority before
scientific execution.


## 20. Allowed cohort outputs

After the tau_e rule/manifest candidate has been byte-identified,
the audit may report:

- total 720 source rows;
- decisive prediction count;
- confident decisive prediction count;
- confident-correct count;
- confident-wrong count;
- counts by predicted class;
- counts by gold class;
- counts by intervention type;
- tau_e annotation status counts;
- POST4 prefix eligibility counts;
- potential matched-pair capacity;
- balance feasibility;
- prospective power gate result.

These outputs are feasibility evidence only.

They are not evidence for or against the native-state precursor hypothesis.


## 21. Explicitly forbidden outputs

The audit must not produce or inspect:

- native-state correct-vs-wrong speed differences;
- native-state correct-vs-wrong turning differences;
- native-state correct-vs-wrong path-efficiency differences;
- token-wise state separation scans;
- best-layer scans;
- best-token scans;
- kinematic p-values;
- kinematic effect sizes;
- trajectory clustering by correctness;
- learned trajectory classifiers;
- decision-space projections;
- terminal-state rescue analyses.

No output of this audit may be described as:

NATIVE_KINEMATICS_PRECURSOR_SUPPORTED

or:

NATIVE_KINEMATICS_PRECURSOR_NOT_SUPPORTED


## 22. Feasibility verdict dimensions

The audit must report each dimension independently:

SOURCE_PROVENANCE_FEASIBILITY

CONFIDENCE_FEASIBILITY

TOKENIZER_PROVENANCE_FEASIBILITY

TAU_E_ANNOTATION_FEASIBILITY

POST4_PREFIX_FEASIBILITY

CONFIDENT_ERROR_COHORT_FEASIBILITY

MATCHING_FEASIBILITY

PROSPECTIVE_POWER_FEASIBILITY

A0_NATIVE_STATE_BINDING_FEASIBILITY


## 23. Overall feasibility PASS

OVERALL_FEASIBILITY =
PASS

requires all of:

1. frozen source identities pass;
2. exact tokenizer provenance is recoverable;
3. tau_e deterministic annotation rule is defensible;
4. tau_e manifest is complete for enough candidate rows;
5. enough rows satisfy the POST4 prefix requirement;
6. confident-correct and confident-wrong decisive cases coexist;
7. potential matching can produce at least 45 pairs;
8. predicted-class representation gate is satisfied where required;
9. A0 native-state binding is reproducibly provisionable;
10. no native-state scientific outcome has been accessed.

PASS authorizes only preparation of a later implementation/execution
authority.

It does not itself authorize native-state extraction.


## 24. Overall feasibility BLOCKED / FAIL

The audit must stop or return a bounded failure if any critical prerequisite
cannot be established.

Examples include:

- exact tokenizer identity unavailable;
- no defensible tau_e rule;
- decisive event occurs too near terminal position;
- too few confident wrong decisive commitments;
- confidence overlap inadequate;
- fewer than 45 potential matched pairs;
- exact A0 native encoder identity cannot be reproduced.

A failure is a scientific/design result about feasibility.

It must not be repaired by:

- lowering confidence threshold;
- redefining tau_e as evidence start;
- moving tau_e earlier for convenience;
- reducing POST4 to POST1/POST2;
- changing primary layer;
- adding new seeds;
- using grouped-factorial predictions;
- using a different dataset;
- scanning native-state outcomes first.

Any such reformulation requires new authority.


## 25. Audit implementation boundary

A later implementation of this audit may:

- read JSON/JSONL/text provenance;
- hash files;
- inspect frozen source code;
- inspect local tokenizer metadata;
- run tokenizer-only CPU operations after exact tokenizer binding;
- produce deterministic annotation and feasibility artifacts;
- compute cohort metadata and count summaries.

It may not:

- train;
- evaluate a model forward;
- instantiate the checkpoint;
- run GPU code;
- run Kaggle;
- extract recurrent states;
- calculate P1/P2/P3.


## 26. Required audit artifacts

A completed audit should produce, at minimum:

1. tau_e semantic rule candidate;
2. tau_e annotation manifest candidate;
3. cohort feasibility manifest;
4. validated feasibility report candidate;
5. SHA256 identities for all generated artifacts.

All artifacts must bind this authority commit after it is frozen.

No generated artifact may be treated as confirmatory scientific evidence.


## 27. Next stage after a PASS

If and only if the audit passes, the controller may prepare:

NATIVE_STATE_KINEMATICS_FIRST_MEASUREMENT_IMPLEMENTATION_AUTHORITY

or a narrower prerequisite authority if exact encoder/state binding still
requires bounded implementation work.

Scientific execution remains separately gated.


## 28. Current authority verdict

AUTHORITY_VERDICT =
READY_FOR_FREEZE_REVIEW

CONFIDENCE_STATISTIC =
PREDICTED_CLASS_FINAL_PROBABILITY

CONFIDENCE_THRESHOLD =
0.5

TAU_E =
NOT_PRETENDED_RESOLVED

TAU_E_RESOLUTION_METHOD =
OUTCOME_BLIND_SEMANTIC_ANNOTATION_FEASIBILITY_FIRST

MINIMUM_MATCHED_PAIR_COUNT =
45

SCIENTIFIC_TRAJECTORY_OUTCOMES_ALLOWED =
NO

TRAINING_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

END_OF_NATIVE_STATE_KINEMATICS_COHORT_MEASUREMENT_FEASIBILITY_AUDIT_AUTHORITY
