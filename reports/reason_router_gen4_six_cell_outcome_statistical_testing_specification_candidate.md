# ContraMamba Gen4 Six-Cell Outcome and Statistical Testing Specification - Candidate

## 1. Status

STATUS = CANDIDATE

PHASE =
GEN4_SIX_CELL_OUTCOME_STATISTICAL_TESTING_SPECIFICATION

OUTCOME_EVIDENCE_INVENTORY_AUTHORITY =
2708a873be01dcf6709df242c0611b84671261ab

OUTCOME_DESIGN_AUTHORITY =
92895d709de851e5167cf2d02d47604b7cdf90d1

SIX_CELL_CANONICAL_ARTIFACT_COMMIT =
5b79d8585b20cf6fa4cfe52bbdbdce52374653aa

GEN3_GROUPED_EXECUTION_AUTHORITY =
dbd2746cef80b6d3e1c2c428afb96004887827b8

GEN3_GROUPED_VALIDATED_EVIDENCE =
c97fd33dd8aa9c116f45071ee4545093ebba8f1c

This document prospectively defines the Gen4 primary outcome, evaluator
aggregation, pair-blocked estimands, statistical testing family, row-identity
contract, missingness rules, and future outcome artifact contract.

This document does not authorize evaluator execution.

## 2. Frozen structural input

Canonical six-cell artifact:

reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

SHA256:

b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

Bytes:

1465573

Rows:

1800

Source pairs:

300

Cells per source pair:

C0_SHAM
C1_TITLE
C2_NAME
C3_ROLE
C4_PREDICATE
C5_TITLE_NAME

The structural intervention design is immutable.

## 3. Frozen evaluator source

Tier-1 direct outcome reuse is unavailable.

The admissible Tier-2 evaluator source is:

FULL_PRESPECIFIED_GEN3_GROUPED_18_RUN_MATRIX

The matrix consists of:

3 frozen training seeds:

180
181
182

times

6 topology-prespecified grouped arms:

G3-GROUP-U-HALF
G3-GROUP-Q-HALF
G3-GROUP-D-HALF
G3-GROUP-U-Q-HALF
G3-GROUP-U-D-HALF
G3-GROUP-Q-D-HALF

for exactly 18 evaluator checkpoints.

The matrix is used in full.

No checkpoint is promoted, dropped, ranked, or selected by historical
performance.

Therefore:

EVALUATOR_POPULATION =
FULL_PRESPECIFIED_GEN3_GROUPED_18_RUN_MATRIX

EVALUATOR_COUNT =
18

SINGLE_CHECKPOINT_PRESELECTION =
FORBIDDEN

PERFORMANCE_WEIGHTING =
FORBIDDEN

EVALUATOR_DROPOUT =
FORBIDDEN_FOR_PRIMARY_ANALYSIS

## 4. Common evaluator contract

All 18 evaluators share the frozen contract:

split_seed =
8192

source_commit =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

model_name =
state-spaces/mamba-130m-hf

backbone =
mamba

reason_router_mode =
explicit_product

gradient_ownership_mode =
edge_specific

freeze_encoder =
true

The grouped arm and training seed are the only prespecified evaluator
coordinates that vary within this population.

## 5. Exact checkpoint identities

Every future execution must authenticate the following exact checkpoint bytes
before any model load.

seed180 / G3-GROUP-D-HALF:
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

seed180 / G3-GROUP-Q-D-HALF:
2e51f64702a3ebf21d5d8e8aa84745b62b3faa01112b5b8f10525ba6435dbc8c

seed180 / G3-GROUP-Q-HALF:
eb349aefca6d992df42f6239e7cf642d560755c0b1819397dba1d746b33bd8e3

seed180 / G3-GROUP-U-D-HALF:
08654abb9c1ec67d42fa1b3464f19298f21ff79b866fb0cf8b7a97d59a45ff86

seed180 / G3-GROUP-U-HALF:
a8cd296136816f806394ca98d6433bfa560f5691ab37e661347c2db838966708

seed180 / G3-GROUP-U-Q-HALF:
0701ce934ae3ef34cd9f9d229c9321599b4ca150db8dabc3c8a740668b8f0aad

seed181 / G3-GROUP-D-HALF:
afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f

seed181 / G3-GROUP-Q-D-HALF:
390b4fe3266d8eddebe74d9732321d1f96e2a7095ecae67b6155a2d535b655ba

seed181 / G3-GROUP-Q-HALF:
3b5044fddb7f542c9e06a318a5a81a731d94475f7f67b7e5c5a7787ab3af0ba6

seed181 / G3-GROUP-U-D-HALF:
7adffc577e00b9a9150bca28ed83b35eb5574458f71d5bc276ebd8f557b00e4d

seed181 / G3-GROUP-U-HALF:
e2a9fd1ca6e50856b2349fc5bc915c54e6a71848aaa8c59aaa1f8c8647e89699

seed181 / G3-GROUP-U-Q-HALF:
1be3be2ddd13762d36c69ef16ccbdd0ee4bd5ad732eff46e7a66cab703c2db50

seed182 / G3-GROUP-D-HALF:
f9db48a3b3b9fdc6df4e2bb2086d11fd80fd595e6096c0095d1992f6c7d777f2

seed182 / G3-GROUP-Q-D-HALF:
cb1f4812d11643089bb87064c436b2e890554435254c961e5ed3f766b61b412b

seed182 / G3-GROUP-Q-HALF:
67d0cbf855b24a291f55ce87425dcd4d77b5f7a59fb119c57c261c6378a4342e

seed182 / G3-GROUP-U-D-HALF:
f1d84bab31f9080f0f3cfc6d0ee49cdc2743ad7c8c3a620bee7f576ca32ebef1

seed182 / G3-GROUP-U-HALF:
47b43899119a0a450e0b5cf8134ade521d8cea9ca568110b32223de6109ef5a4

seed182 / G3-GROUP-U-Q-HALF:
129be6e930b5f7e6737ad671a9646c867d150e8751907bb1abfd3fe64570669f

Each real checkpoint had byte count:

518270455

Any SHA256 mismatch is a hard execution blocker.

No checkpoint substitution is permitted.

## 6. Primary scientific outcome

The primary Gen4 outcome is:

PRIMARY_Y =
q_authorized

For structural row r and evaluator e:

Y(r,e) =
q_authorized(r,e)

The primary outcome is scalar and is required for every Gen4 row under every
evaluator.

PRIMARY_Y_COUNT =
1

## 7. Why q_authorized is primary

q_authorized is selected prospectively because:

1. it is present in the common historical output schema of all 18 evaluators;
2. it is defined without requiring a new Gen4 gold-label assignment;
3. it is applicable to every six-cell row under one common definition;
4. it does not privilege only the SUPPORT-versus-NOT_ENTITLED boundary;
5. it is directly aligned with the frozen historical authorization-side
   geometry;
6. it allows continuous pair-blocked contrasts rather than thresholded
   prediction changes;
7. it was selected before any Gen4 six-cell model outcome was inspected.

Therefore:

PRIMARY_Y_POST_HOC_SELECTION =
NO

## 8. Outcomes explicitly not selected as primary

The following are not primary outcomes:

aggregate accuracy
is_correct
unreduced_final_ce_loss
prediction
support_ne_margin_active
entitlement_prob

Accuracy, correctness, and final CE would require a separate valid Gen4 gold
outcome contract.

support_ne_margin_active privileges one final-class boundary.

entitlement_prob remains mechanistically relevant but is not needed as a second
co-primary endpoint.

No primary endpoint switching is allowed after future Gen4 outcome inspection.

## 9. Secondary diagnostics

Exactly two secondary continuous diagnostics are prospectively retained.

### 9.1 Entitlement diagnostic

SECONDARY_Y_1 =
entitlement_prob

This is descriptive and mechanistic.

It does not replace q_authorized.

### 9.2 Universal final-support boundary diagnostic

Define:

SECONDARY_Y_2 =
support_vs_best_nonsupport_logit_margin

where:

support_vs_best_nonsupport_logit_margin =
support_logit - max(ne_logit, refute_logit)

This diagnostic uses the common final-logit schema and avoids privileging only
the NOT_ENTITLED branch.

It is descriptive/secondary.

No confirmatory p-value family is defined for secondary diagnostics in this
specification.

## 10. Categorical diagnostic

Final prediction may be retained as a descriptive categorical diagnostic.

It is not a primary or secondary inferential endpoint.

Prediction-transition counts may be reported but cannot replace the primary
q_authorized analysis.

## 11. Evaluator aggregation

The 18 evaluators are a fixed prespecified evaluator population.

They are not treated as 18 independent statistical samples.

For source pair p, cell c, and evaluator e define:

Y_pce =
q_authorized for pair p, cell c, evaluator e.

Define the evaluator-averaged cell outcome:

Ybar_pc =
(1 / 18) * sum_e Y_pce

Every evaluator receives equal weight.

No weighting by:

accuracy
macro-F1
historical residual behavior
arm
seed
checkpoint performance

is permitted.

Thus:

EVALUATOR_AGGREGATION =
EQUAL_WEIGHT_FIXED_POPULATION_MEAN

## 12. Primary inferential unit

The inferential unit remains:

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

PRIMARY_PAIR_COUNT =
300

The 18 evaluator observations for a structural row are repeated fixed-evaluator
measurements, not additional independent source-pair samples.

Therefore:

PSEUDOREPLICATION_BY_EVALUATOR =
FORBIDDEN

ROW_LEVEL_N_5400_INTERPRETATION =
FORBIDDEN

The primary analysis has 300 pair-level observations for each structural
estimand.

## 13. Frozen primary estimands

Using evaluator-averaged Ybar:

Delta_title(p) =
Ybar(p,C1_TITLE) - Ybar(p,C0_SHAM)

Delta_name(p) =
Ybar(p,C2_NAME) - Ybar(p,C0_SHAM)

Delta_role(p) =
Ybar(p,C3_ROLE) - Ybar(p,C0_SHAM)

Delta_predicate(p) =
Ybar(p,C4_PREDICATE) - Ybar(p,C0_SHAM)

Interaction_title_name(p) =
Ybar(p,C5_TITLE_NAME)
- Ybar(p,C1_TITLE)
- Ybar(p,C2_NAME)
+ Ybar(p,C0_SHAM)

Title_minus_name(p) =
Delta_title(p) - Delta_name(p)

These definitions are immutable before execution.

## 14. Confirmatory hypothesis family

The confirmatory family contains exactly six two-sided null hypotheses.

H0_TITLE:

mean_p Delta_title(p) = 0

H0_NAME:

mean_p Delta_name(p) = 0

H0_ROLE:

mean_p Delta_role(p) = 0

H0_PREDICATE:

mean_p Delta_predicate(p) = 0

H0_TITLE_NAME_INTERACTION:

mean_p Interaction_title_name(p) = 0

H0_TITLE_EQUALS_NAME:

mean_p Title_minus_name(p) = 0

No directional alternative is assumed.

SIDEDNESS =
TWO_SIDED

CONFIRMATORY_HYPOTHESIS_COUNT =
6

## 15. Point estimates

For every confirmatory estimand d(p), report:

N =
300

mean =
arithmetic mean over the 300 source-pair values

standard deviation =
sample standard deviation over the 300 source-pair values

standard error =
SD / sqrt(300)

median =
empirical median over the 300 source-pair values

minimum

maximum

No row-level or evaluator-level observation count may replace N=300 in the
primary table.

## 16. Primary statistical test

For each of the six confirmatory pair-level estimands, use a two-sided
one-sample Student t test of the mean against zero.

For estimand d:

t =
mean(d) / (sd(d) / sqrt(300))

degrees of freedom:

299

The test is applied only after evaluator averaging and pair-level contrast
construction.

STATISTICAL_TEST =
TWO_SIDED_ONE_SAMPLE_T_TEST_ON_PAIR_LEVEL_CONTRASTS

DF =
299

This test targets systematic mean displacement over the frozen 300-pair
source population under the fixed 18-evaluator population.

It does not establish generalization to arbitrary datasets or arbitrary
models.

## 17. Confidence intervals

For each raw confirmatory mean effect, report the ordinary two-sided 95 percent
Student-t confidence interval using df=299.

The confidence interval is an uncertainty summary for the raw effect estimate.

Multiplicity decisions are governed by the separately specified Holm procedure
below.

No confidence interval may be interpreted as model-universal generalization.

## 18. Multiplicity control

The six confirmatory p-values form one family.

Family-wise alpha:

0.05

Multiplicity method:

HOLM_BONFERRONI

Procedure:

1. sort the six raw two-sided p-values from smallest to largest;
2. compare the smallest with 0.05/6;
3. continue sequentially using 0.05/(6-k+1);
4. after the first non-rejection, all remaining hypotheses are non-rejected;
5. report raw and Holm-adjusted p-values.

Therefore:

CONFIRMATORY_FAMILYWISE_ALPHA =
0.05

MULTIPLICITY_METHOD =
HOLM_BONFERRONI

No uncorrected p-value may be used as the confirmatory decision criterion.

## 19. Effect-size reporting

For every confirmatory estimand report:

raw mean q_authorized contrast

and

paired standardized effect:

d_z =
mean(d) / sd(d)

if sd(d) > 0.

If sd(d) = 0:

d_z =
UNDEFINED_ZERO_VARIANCE

No arbitrary small/medium/large verbal threshold is assigned to d_z.

No minimum effect-size threshold is introduced post hoc.

## 20. Confirmatory decision rule

For an individual estimand:

SYSTEMATIC_WITHIN_MECHANISM_EFFECT_SUPPORTED

requires:

Holm-adjusted p < 0.05

for that estimand.

The direction of the effect is the sign of the raw mean contrast.

A non-rejected null means:

SYSTEMATIC_EFFECT_NOT_ESTABLISHED

It does not prove exact zero effect.

For title-versus-name:

TITLE_NAME_EFFECT_DIFFERENCE_SUPPORTED

requires Holm-adjusted p < 0.05 for H0_TITLE_EQUALS_NAME.

For the title-name interaction:

TITLE_NAME_NONADDITIVITY_SUPPORTED

requires Holm-adjusted p < 0.05 for H0_TITLE_NAME_INTERACTION.

## 21. Scientific falsification interpretation

If none of the six confirmatory hypotheses rejects after Holm correction:

PRIMARY_Q_AUTHORIZED_AXIS_EFFECT =
NOT_ESTABLISHED

This would falsify the claim that the current frozen experiment establishes a
systematic semantic-axis effect on the selected primary authorization outcome.

It would not invalidate the structural identifiability result itself.

If one or more hypotheses reject:

the supported claim is limited to the corresponding within-mechanism
q_authorized effect under the frozen evaluator population.

It does not establish:

mechanism-invariant generalization
arbitrary-model generalization
native-Mamba-state causality
feature usefulness
training benefit
task-performance improvement

## 22. Evaluator heterogeneity reporting

For transparency, the same six raw pair-level contrasts may be summarized
separately for each evaluator coordinate.

Allowed descriptive grouping:

training seed
grouped arm
exact evaluator checkpoint

These evaluator-specific summaries are secondary heterogeneity diagnostics.

They are not additional confirmatory hypothesis families.

They may not be used to drop, promote, reweight, or select evaluators after
outcome inspection.

## 23. Gen4 row-identity contract

Future evaluator output must preserve directly from the canonical input:

row_id
source_pair_id
contrast_cell_id

These fields must be echoed verbatim.

The output must not derive scientific identity from:

claim text
evidence text
tokenized text
fuzzy matching
semantic matching
implicit row order

Therefore:

ROW_IDENTITY_MODE =
DIRECT_METADATA_PRESERVATION

TEXT_SEMANTIC_JOIN =
FORBIDDEN

ROW_ORDER_ONLY_JOIN =
FORBIDDEN

## 24. Expected future execution cardinality

Canonical structural rows:

1800

Evaluator count:

18

Expected evaluator-row observations:

32400

Every evaluator must produce exactly one outcome row for every structural row.

Required unique composite key:

(evaluator_seed, evaluator_arm, row_id)

Expected unique composite-key count:

32400

Each row_id must occur exactly:

18

times in the complete evaluator outcome artifact.

Each source_pair_id must contribute:

6 cells x 18 evaluators =
108 evaluator-row observations.

## 25. Missingness rule

Primary analysis requires the complete fixed evaluator population and complete
six-cell source-pair blocks.

No imputation is permitted.

No partial-evaluator primary analysis is permitted.

No incomplete six-cell source pair is permitted.

If any required checkpoint, structural row, or outcome field is missing:

PRIMARY_ANALYSIS =
BLOCKED_INCOMPLETE_MATRIX

A failed evaluator execution must be repaired under a later explicit authority
before confirmatory testing.

## 26. Future evaluator output schema

A future outcome artifact must contain at minimum:

schema_version
structural_artifact_commit
statistical_specification_commit
evaluator_source_commit
evaluator_seed
evaluator_arm
checkpoint_sha256
source_pair_id
row_id
contrast_cell_id
q_authorized
entitlement_prob
support_logit
ne_logit
refute_logit
support_vs_best_nonsupport_logit_margin
prediction

The exact final serialization schema must be frozen before execution.

No gold label is required by the primary q_authorized analysis.

## 27. Checkpoint byte authentication

Before any checkpoint is deserialized:

1. locate the exact expected checkpoint artifact;
2. compute SHA256 over raw bytes;
3. compare against the frozen seed/arm identity;
4. verify byte count where available;
5. reject on any mismatch.

Therefore:

CHECKPOINT_BYTE_REAUTHENTICATION_BEFORE_LOAD =
REQUIRED

CHECKPOINT_SUBSTITUTION =
FORBIDDEN

CHECKPOINT_LOADABILITY =
NOT_YET_ESTABLISHED

## 28. Model identity

The expected model family remains:

state-spaces/mamba-130m-hf

backbone:

mamba

The historical evaluator source commit remains:

3e0e9a435068c552abf20f3a74e0c3eccca344a3

No architecture change is authorized by this specification.

## 29. Tokenizer identity blocker

The inventory established evaluator and checkpoint identity but did not freeze a
byte-exact tokenizer package/revision for future Gen4 execution.

Therefore:

TOKENIZER_IDENTITY =
NOT_YET_FROZEN_FOR_GEN4_EXECUTION

TOKENIZER_EXECUTION =
BLOCKED_PENDING_IDENTITY_FREEZE

Before scientific evaluator execution, a later read-only capability/provenance
audit must establish the exact tokenizer/model-loading contract inherited from
the frozen evaluator lineage.

No tokenizer may be executed merely to discover its identity.

## 30. Checkpoint loadability blocker

The recovered checkpoint bytes were hashed but never deserialized by the
inventory audit.

Therefore:

CHECKPOINT_LOADABILITY =
NOT_ESTABLISHED

A later explicitly authorized evaluator capability/preflight stage must establish
loadability and output compatibility before scientific execution.

This statistical specification does not authorize that load.

## 31. Execution implementation capability blocker

The inventory established historical output schema compatibility.

It did not establish that the currently frozen repository contains a
Gen4-capable inference path that:

- consumes the canonical six-cell artifact;
- loads all 18 exact checkpoints;
- preserves row_id/source_pair_id directly;
- emits q_authorized;
- emits the two secondary diagnostics;
- serializes exactly one row per structural row/evaluator coordinate;
- fails closed on checkpoint/tokenizer/provenance mismatch.

Therefore:

GEN4_TIER2_EVALUATOR_EXECUTION_CAPABILITY =
NOT_YET_ESTABLISHED

## 32. Current execution authority

This specification authorizes no execution.

MODEL_INFERENCE =
NOT_AUTHORIZED

CHECKPOINT_LOADING =
NOT_AUTHORIZED

TOKENIZER_EXECUTION =
NOT_AUTHORIZED

EVALUATOR_EXECUTION =
NOT_AUTHORIZED

STATISTICAL_TESTING =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

KAGGLE_EXECUTION =
NOT_AUTHORIZED

## 33. Next required object

After this specification is frozen, the next object is:

GEN4_SIX_CELL_TIER2_EVALUATOR_EXECUTION_CAPABILITY_AUDIT

That audit must be read-only.

It must determine whether the existing frozen code and provenance can satisfy:

- exact checkpoint provisioning;
- exact checkpoint hash binding;
- checkpoint loading contract;
- tokenizer identity;
- Gen4 canonical input consumption;
- direct identity preservation;
- q_authorized extraction;
- required secondary diagnostic extraction;
- deterministic output serialization.

It must not run model inference.

It must not run a tokenizer.

It must not load checkpoints unless a separate later authority explicitly
permits a narrow loadability preflight.

## 34. Current scientific decision

PRIMARY_Y =
q_authorized

PRIMARY_Y_COUNT =
1

SECONDARY_Y_1 =
entitlement_prob

SECONDARY_Y_2 =
support_vs_best_nonsupport_logit_margin

EVALUATOR_POPULATION =
FULL_PRESPECIFIED_GEN3_GROUPED_18_RUN_MATRIX

EVALUATOR_AGGREGATION =
EQUAL_WEIGHT_FIXED_POPULATION_MEAN

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

PRIMARY_PAIR_COUNT =
300

CONFIRMATORY_HYPOTHESIS_COUNT =
6

SIDEDNESS =
TWO_SIDED

STATISTICAL_TEST =
TWO_SIDED_ONE_SAMPLE_T_TEST_ON_PAIR_LEVEL_CONTRASTS

DF =
299

CONFIRMATORY_FAMILYWISE_ALPHA =
0.05

MULTIPLICITY_METHOD =
HOLM_BONFERRONI

FUTURE_ROW_IDENTITY_CONTRACT =
DIRECT_METADATA_PRESERVATION_REQUIRED

EXPECTED_FUTURE_EVALUATOR_ROWS =
32400

MISSINGNESS_RULE =
FAIL_CLOSED_COMPLETE_MATRIX_REQUIRED

MODEL_INFERENCE =
NOT_AUTHORIZED

TOKENIZER_EXECUTION =
NOT_AUTHORIZED

STATISTICAL_TESTING =
NOT_AUTHORIZED

SCIENTIFIC_OUTCOME_CONCLUSION =
NOT_ESTABLISHED

NEXT_OBJECT =
GEN4_SIX_CELL_TIER2_EVALUATOR_EXECUTION_CAPABILITY_AUDIT

## 35. Stop condition

Stop after this specification candidate is created and reviewed.

Do not inspect Gen4 outcomes.

Do not load checkpoints.

Do not run model inference.

Do not execute the tokenizer.

Do not compute q_authorized for the six-cell rows.

Do not perform the six confirmatory tests.

Do not train.

Do not use Kaggle.

A later frozen authority is required before any execution operation.
