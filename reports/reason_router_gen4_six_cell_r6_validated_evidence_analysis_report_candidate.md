# ContraMamba Gen4 Six-Cell R6 Validated Evidence Analysis Report - Candidate

## 1. Status

STATUS =
VALIDATED_EVIDENCE_ANALYSIS_CANDIDATE

PHASE =
GEN4_R6_POST_STATISTICAL_INTERPRETATION

THIS_REPORT_PERFORMS_NEW_STATISTICAL_TESTING =
NO

THIS_REPORT_PERFORMS_MODEL_EXECUTION =
NO

## 2. Frozen provenance

R6_VALIDATED_RESULT_FREEZE =
5896d4740cd390a56ff5e2a3459c68ce65e2bc84

R6_EXECUTION_AUTHORITY =
f437ec356e448a1240a8f9c895e779fccaff1a95

R6_STATISTICAL_ANALYSIS_IMPLEMENTATION =
32209bead3915e233ae3f4eca0090860cb245a68

R6_IMPLEMENTATION_AUTHORITY =
5ae6d7cfd11f9d2b64617a145c7c248becd97aad

R5_VALIDATED_SCIENTIFIC_INFERENCE_ARTIFACT_FREEZE =
a3b5bcf2ded8dc0e86e859bbba12b5601a2fdea0

GEN4_OUTCOME_STATISTICAL_TESTING_SPECIFICATION =
4dc5bacd10a254b5ecd339ac1fe78bad9def5c47

PRIMARY_RESULT_ARTIFACT =
reports/reason_router_gen4_six_cell_r6_statistical_analysis_a3b5bcf/r6_primary_confirmatory_results.csv

PRIMARY_RESULT_SHA256 =
00681ccd1ec3ea11ab152d41ab60a326bb6a9cd2a33e2551fbb3ac528f93ac52

STATISTICAL_SUMMARY_ARTIFACT =
reports/reason_router_gen4_six_cell_r6_statistical_analysis_a3b5bcf/r6_statistical_analysis_summary.json

STATISTICAL_SUMMARY_SHA256 =
fe9b309023786418485bf9ac2cce59a78ac074a7488e1eff92e994e65462887c

## 3. Frozen inferential scope

PRIMARY_Y =
q_authorized

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

PRIMARY_PAIR_COUNT =
300

EVALUATOR_POPULATION =
FIXED_PRESPECIFIED_18_EVALUATORS

EVALUATOR_AGGREGATION =
EQUAL_WEIGHT_FIXED_POPULATION_MEAN

CONFIRMATORY_FAMILY_SIZE =
6

MULTIPLICITY_METHOD =
HOLM_BONFERRONI

FAMILYWISE_ALPHA =
0.05

The conclusions below apply only to the frozen 300 source-pair experiment
under the fixed prespecified 18-evaluator population.

## 4. Confirmatory result disposition

### 4.1 Title

ESTIMAND =
delta_title

MEAN_Q_AUTHORIZED_CONTRAST =
-0.004618828506295189

HOLM_ADJUSTED_P =
0.23921930738175834

D_Z =
-0.06808647387431682

DECISION =
SYSTEMATIC_EFFECT_NOT_ESTABLISHED

The frozen experiment does not establish a systematic title-only effect on
q_authorized.

This is not evidence that the exact title effect is zero.

### 4.2 Name

ESTIMAND =
delta_name

MEAN_Q_AUTHORIZED_CONTRAST =
-0.019670098963779983

HOLM_ADJUSTED_P =
8.49906010087348e-24

D_Z =
-0.64710274906839

DECISION =
SYSTEMATIC_WITHIN_MECHANISM_EFFECT_SUPPORTED

DIRECTION =
NEGATIVE

Within the frozen experiment, name-axis substitution systematically reduces
q_authorized relative to sham.

### 4.3 Role

ESTIMAND =
delta_role

MEAN_Q_AUTHORIZED_CONTRAST =
0.010517086642794311

HOLM_ADJUSTED_P =
0.04146926577397731

D_Z =
0.13424097933026438

DECISION =
SYSTEMATIC_WITHIN_MECHANISM_EFFECT_SUPPORTED

DIRECTION =
POSITIVE

Within the frozen experiment, role-axis substitution systematically increases
q_authorized relative to sham.

### 4.4 Predicate

ESTIMAND =
delta_predicate

MEAN_Q_AUTHORIZED_CONTRAST =
0.04420943002363115

HOLM_ADJUSTED_P =
1.0117344686735611e-13

D_Z =
0.4645547785489984

DECISION =
SYSTEMATIC_WITHIN_MECHANISM_EFFECT_SUPPORTED

DIRECTION =
POSITIVE

Within the frozen experiment, predicate-axis substitution systematically
increases q_authorized relative to sham.

### 4.5 Title-name interaction

ESTIMAND =
interaction_title_name

MEAN_Q_AUTHORIZED_CONTRAST =
0.00919857732080682

HOLM_ADJUSTED_P =
8.008447528207001e-05

D_Z =
0.2502265343877293

DECISION =
TITLE_NAME_NONADDITIVITY_SUPPORTED

The joint title-name condition is not adequately described by simple additive
combination of the frozen title and name main effects.

### 4.6 Title versus name

ESTIMAND =
title_minus_name

MEAN_Q_AUTHORIZED_CONTRAST =
0.015051270457484793

HOLM_ADJUSTED_P =
0.00042573195229326166

D_Z =
0.2225467764404304

DECISION =
TITLE_NAME_EFFECT_DIFFERENCE_SUPPORTED

The title-axis and name-axis effects differ systematically.

Because the title-only effect itself is not established while the name effect
is negative and strongly established, this result must not be rewritten as a
claim that title has a positive systematic effect.

## 5. Overall validated scientific conclusion

SUPPORTED_CONFIRMATORY_ESTIMANDS =
5_OF_6

UNSUPPORTED_CONFIRMATORY_ESTIMANDS =
delta_title

The frozen Gen4 experiment establishes that q_authorized is systematically
sensitive to multiple controlled semantic-axis substitutions.

The response is axis-dependent rather than uniform:

- name produces a supported negative displacement;
- role produces a supported positive displacement;
- predicate produces a supported positive displacement;
- title alone is not established as a systematic effect;
- title and name differ;
- title-name composition exhibits supported nonadditivity.

Therefore the experiment supports:

SEMANTIC_AXIS_DEPENDENT_AUTHORIZATION_RESPONSE =
ESTABLISHED_WITHIN_FROZEN_SCOPE

and:

TITLE_NAME_NONADDITIVE_RESPONSE =
ESTABLISHED_WITHIN_FROZEN_SCOPE

It does not support a claim that all semantic axes have systematic effects.

## 6. Secondary descriptive consistency

The secondary entitlement-probability and
support-vs-best-nonsupport-margin summaries are descriptive only.

They may be used to describe the direction and scale of observed model-output
behavior, but they do not create additional confirmatory hypotheses and do
not alter the primary q_authorized decisions.

No secondary inferential claim is promoted by this report.

## 7. Claims explicitly not established

The validated R6 evidence does not establish:

NATIVE_MAMBA_STATE_CAUSALITY

ARBITRARY_MODEL_GENERALIZATION

ARBITRARY_DATASET_GENERALIZATION

MECHANISM_INVARIANT_GENERALIZATION

TRAINING_BENEFIT

TASK_PERFORMANCE_IMPROVEMENT

FEATURE_USEFULNESS_FOR_TRAINING

TITLE_ONLY_SYSTEMATIC_EFFECT

GENERAL_SEMANTIC_UNDERSTANDING

The experiment establishes controlled output sensitivity under the frozen
mechanism and evaluator population, not the internal state mechanism producing
that sensitivity.

## 8. Mechanistic implication

The Gen4 intervention now provides a validated behavioral contrast family with
known outcome asymmetry.

This changes the next scientific question.

The next question is no longer:

DO_CONTROLLED_SEMANTIC_AXES_AFFECT_Q_AUTHORIZED

That question has been answered within the frozen scope.

The next question is:

WHERE_AND_HOW_DO_THE_SUPPORTED_AXIS_EFFECTS_APPEAR_IN_NATIVE_MAMBA_STATE_DYNAMICS

The most informative frozen contrasts for a mechanistic bridge are:

NAME =
supported negative effect

PREDICATE =
supported positive effect

TITLE =
non-established main effect control

TITLE_NAME =
supported nonadditive composition

ROLE =
supported smaller positive effect

These are scientific contrast roles, not a permission to select or discard
data post hoc.

## 9. Next research object

NEXT_RESEARCH_OBJECT =
GEN4_NATIVE_MAMBA_STATE_MECHANISTIC_BRIDGE_SPECIFICATION

The next object must be specification-only.

It must define before any state extraction or execution:

- exact native Mamba state object or objects of interest;
- layer/time/token coordinates;
- state-space comparison geometry;
- how the six frozen structural cells map into state contrasts;
- how output-level supported and unsupported effects constrain predictions;
- falsification conditions;
- independence from downstream router-only representations;
- provenance and serialization requirements;
- minimal analysis before any complex architecture or training.

It must not begin training.

It must not add a learned probe merely because one is convenient.

It must not tune hyperparameters for predictive performance.

It must not infer native-state causality from the R6 output effect alone.

## 10. Current project boundary

CODE_CORRECTNESS =
PASS

R5_SCIENTIFIC_INFERENCE_EXECUTION =
PASS

R5_ARTIFACT_PROVENANCE =
PASS

R6_STATISTICAL_EXECUTION =
PASS

R6_RESULT_ARTIFACT_PROVENANCE =
PASS

R6_CONFIRMATORY_SCIENTIFIC_CONCLUSION =
ESTABLISHED_WITHIN_FROZEN_SCOPE

NATIVE_STATE_MECHANISM =
NOT_YET_ESTABLISHED

TRAINING =
NOT_AUTHORIZED

MODEL_EXECUTION_FOR_NEXT_STAGE =
NOT_AUTHORIZED

KAGGLE =
NOT_AUTHORIZED

## 11. Stop condition

Stop after this validated evidence analysis candidate is created and reviewed.

Do not run another statistical test.

Do not run model inference.

Do not extract native Mamba states.

Do not train.

Do not use Kaggle.

A later frozen mechanistic-bridge specification is required before any new
scientific execution.
