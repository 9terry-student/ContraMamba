# ContraMamba Gen4 Six-Cell Post-Freeze Outcome Design Specification - Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_SIX_CELL_POST_FREEZE_OUTCOME_DESIGN
- Frozen canonical artifact commit: 5b79d8585b20cf6fa4cfe52bbdbdce52374653aa
- Result/provenance authority: 66c1218ad3e911f8ead24a18944c3df276803b57
- V2 execution authority: 8593e1478e9ff366d814bb158429340143fae997
- Identifiability-breaking contrast authority: 0a0da5354782e542520fb5bba146ab1a599d17ef
- Upstream structural blocker authority: 7fb89e1a0b8ed6bc73e9f57fe22a1a8a42329d0
- Training authorized: NO
- Evaluation authorized: NO
- Model inference authorized: NO
- Tokenizer execution authorized: NO
- Kaggle authorized: NO

The structural six-cell design is now frozen and provenance-valid.

The next scientific problem is not further materialization.

The next problem is to define an admissible outcome Y and the statistical /
falsification contract for testing the already-frozen within-mechanism
contrasts.

## 2. Frozen structural evidence

Canonical artifact:

reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

SHA256:

b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

Bytes:

1465573

Rows:

1800

Source pairs:

300

Every source pair contains exactly:

C0_SHAM
C1_TITLE
C2_NAME
C3_ROLE
C4_PREDICATE
C5_TITLE_NAME

The six-cell design itself is no longer the scientific uncertainty.

## 3. Frozen estimands

The primary estimands remain exactly the pair-blocked within-mechanism
contrasts defined upstream.

For source pair p:

Delta_title(p) =
    Y(p, C1_TITLE) - Y(p, C0_SHAM)

Delta_name(p) =
    Y(p, C2_NAME) - Y(p, C0_SHAM)

Delta_role(p) =
    Y(p, C3_ROLE) - Y(p, C0_SHAM)

Delta_predicate(p) =
    Y(p, C4_PREDICATE) - Y(p, C0_SHAM)

Interaction_title_name(p) =
    Y(p, C5_TITLE_NAME)
    - Y(p, C1_TITLE)
    - Y(p, C2_NAME)
    + Y(p, C0_SHAM)

Title-versus-name contrast:

Delta_title(p) - Delta_name(p)

These estimands are structural and already fixed.

No outcome Y is selected by this document.

## 4. Immediate scientific question

The immediate question is:

Which outcome evidence, already frozen or available under a later explicit
execution authority, can instantiate Y without introducing a new confound,
post-hoc target selection, provenance ambiguity, or incompatible scoring
procedure across the six cells?

Therefore:

OUTCOME_SELECTION_REQUIRES_PREDECLARED_PROVENANCE = YES

POST_HOC_OUTCOME_SELECTION_AFTER_SIX_CELL_RESULT_INSPECTION = FORBIDDEN

## 5. Outcome admissibility requirements

A candidate Y is scientifically admissible only if all of the following can
be established before inspecting six-cell effect results.

1. The outcome has a precise mathematical definition.
2. The same outcome definition applies to all six cells.
3. The same scoring/evaluation procedure applies to all six cells.
4. The outcome is linked to rows through frozen structural identity, not
   rendered-text semantic reconstruction.
5. The outcome does not redefine contrast-cell identity.
6. Directionality is declared where meaningful.
7. Missingness handling is declared before effect inspection.
8. Any transformation or aggregation is declared before effect inspection.
9. Any model/checkpoint/evaluator identity is frozen.
10. Any tokenizer identity is frozen if tokenizer execution is later required.
11. Pair membership remains intact.
12. Outcome provenance is sufficient to reproduce every reported Y value.

If any requirement fails:

OUTCOME_CANDIDATE = INADMISSIBLE

## 6. Preferred evidence hierarchy

Outcome candidates must be considered in this order.

Tier 1:
Already-frozen outcome-bearing artifacts whose provenance and row linkage are
sufficient and whose scoring semantics are already scientifically justified.

Tier 2:
Already-existing frozen model/evaluator checkpoints for which a later,
separately authorized deterministic inference-only execution can generate Y.

Tier 3:
New training or new evaluator construction.

Tier 3 is not authorized by this design and should not be used merely because
Tier 1 or Tier 2 are inconvenient.

Therefore:

PREFER_EXISTING_FROZEN_OUTCOME_EVIDENCE = YES

NEW_TRAINING_AS_FIRST_OUTCOME_SOURCE = NO

## 7. Required read-only outcome inventory

Before selecting Y, perform a read-only repository audit for candidate
outcome-bearing evidence.

The audit must identify, for every candidate:

- artifact/report path;
- authority commit;
- physical SHA256 where applicable;
- source population;
- source-pair or row-level join key;
- outcome field(s);
- mathematical meaning;
- model/checkpoint identity if applicable;
- evaluator identity if applicable;
- tokenizer dependency if applicable;
- whether values already exist or require execution;
- whether the same procedure can score all six cells;
- whether the candidate preserves pair blocking;
- whether the evidence predates six-cell outcome inspection;
- whether the candidate can be reproduced.

The audit may inspect repository artifacts and metadata only.

It may not run inference.

It may not run a tokenizer.

It may not evaluate the six-cell rows.

## 8. Join-key boundary

A future outcome must join to the six-cell structure through explicit frozen
identity.

Preferred keys are:

source_pair_id
row_id

or a separately frozen deterministic mapping proven to be one-to-one.

The following may not be used as the primary scientific join mechanism:

- claim text matching;
- evidence text matching;
- fuzzy text similarity;
- manual semantic reconstruction;
- tokenizer-derived reconstruction;
- post-hoc interpretation.

Therefore:

TEXT_SEMANTIC_JOIN_FOR_OUTCOME_PROVENANCE = FORBIDDEN

## 9. Pair-blocking requirement

All inferential analysis remains pair-blocked.

The 300 source pairs, not the 1800 rows, define the independent experimental
blocks for the primary within-pair contrasts.

Therefore:

PRIMARY_INFERENTIAL_UNIT = SOURCE_PAIR

PRIMARY_PAIR_COUNT = 300

ROW_LEVEL_INDEPENDENCE_ASSUMPTION = FORBIDDEN

## 10. Outcome-selection discipline

The final outcome authority must select:

- exactly one primary Y;
- optionally a small predeclared set of secondary diagnostics.

Secondary diagnostics may not replace the primary endpoint after result
inspection.

Therefore:

PRIMARY_OUTCOME_COUNT = 1

POST_HOC_PRIMARY_ENDPOINT_SWITCHING = FORBIDDEN

## 11. Statistical contract boundary

This specification does not yet select a statistical test because the valid
test depends on the mathematical scale and sampling properties of the selected
Y.

However, the later outcome authority must freeze before execution:

- primary estimand;
- aggregation across 300 pair-level contrasts;
- uncertainty interval method;
- null hypothesis;
- test statistic;
- sidedness;
- significance or decision threshold, if used;
- multiplicity treatment across the four main effects and interaction;
- missing-data rule;
- falsification criterion;
- effect-size reporting rule.

No test may be selected after inspecting the six-cell outcome effects.

## 12. Frozen scientific null family

The structural nulls remain:

H0_title:
systematic Delta_title = 0

H0_name:
systematic Delta_name = 0

H0_role:
systematic Delta_role = 0

H0_predicate:
systematic Delta_predicate = 0

H0_title_name_interaction:
systematic Interaction_title_name = 0

H0_title_equals_name:
systematic Delta_title - Delta_name = 0

The exact population summary and statistical realization remain downstream of
outcome selection.

## 13. Scientific claim tiers

Tier A:

WITHIN_MECHANISM_AXIS_EFFECT

This is the only claim family targeted by the frozen six-cell minimum core.

Tier B:

TITLE_NAME_INTERACTION

This is supported structurally by the four title/name factorial cells.

Tier C:

MECHANISM_INVARIANT_AXIS_GENERALIZATION

This is not established by the current single-mechanism design and remains
deferred.

Therefore:

MECHANISM_INVARIANT_AXIS_GENERALIZATION = OUT_OF_SCOPE

## 14. No direct feature promotion

Even a positive future outcome result would not automatically authorize using
semantic-axis masks as model features.

Outcome evidence and feature intervention are separate scientific phases.

Therefore:

DIRECT_FEATURE_PROMOTION_FROM_OUTCOME_RESULT = NOT_AUTOMATIC

FEATURE_IMPLEMENTATION = NOT_AUTHORIZED

## 15. Current authority boundary

This specification authorizes no scientific execution.

It does not authorize:

- model inference;
- tokenizer execution;
- evaluator execution;
- six-cell outcome computation;
- training;
- fine-tuning;
- checkpoint creation;
- statistical testing over new outcomes;
- Kaggle.

It defines only the next read-only evidence-selection step.

## 16. Next required object

After this specification is frozen, the next object is:

GEN4_SIX_CELL_OUTCOME_EVIDENCE_INVENTORY_AUDIT

That audit is read-only.

Its purpose is to enumerate and classify existing candidate Y sources under
the admissibility rules above.

The audit must end with one of:

OUTCOME_CANDIDATE_FOUND_AND_PROVENANCE_ADEQUATE

or

NO_EXISTING_OUTCOME_CANDIDATE_IS_ADEQUATE

It must not itself execute or score the six-cell dataset.

## 17. Decision after inventory

If an adequate existing candidate is found:

the next object is a frozen outcome/statistical testing specification.

If no adequate existing candidate is found:

the next scientific decision is whether deterministic inference-only outcome
generation is justified.

That later decision must identify an exact frozen model/checkpoint/evaluator
and require a separate execution authority.

New training is not the default fallback.

## 18. Current decision

SIX_CELL_STRUCTURAL_ARTIFACT = FROZEN

SIX_CELL_STRUCTURAL_PROVENANCE = VALID

WITHIN_MECHANISM_AXIS_IDENTIFIABILITY = STRUCTURALLY_ENABLED

OUTCOME_Y = NOT_YET_SELECTED

PRIMARY_OUTCOME_COUNT = 1

PRIMARY_INFERENTIAL_UNIT = SOURCE_PAIR

PRIMARY_PAIR_COUNT = 300

TEXT_SEMANTIC_JOIN_FOR_OUTCOME_PROVENANCE = FORBIDDEN

POST_HOC_OUTCOME_SELECTION_AFTER_SIX_CELL_RESULT_INSPECTION = FORBIDDEN

PREFER_EXISTING_FROZEN_OUTCOME_EVIDENCE = YES

NEW_TRAINING_AS_FIRST_OUTCOME_SOURCE = NO

MECHANISM_INVARIANT_AXIS_GENERALIZATION = OUT_OF_SCOPE

MODEL_INFERENCE = NOT_AUTHORIZED

TOKENIZER_EXECUTION = NOT_AUTHORIZED

TRAINING_EVALUATION = NOT_AUTHORIZED

KAGGLE_EXECUTION = NOT_AUTHORIZED

NEXT_OBJECT = GEN4_SIX_CELL_OUTCOME_EVIDENCE_INVENTORY_AUDIT

## 19. Stop condition

Stop after this outcome-design specification candidate is created and reviewed.

Do not inspect six-cell model outcomes.

Do not run model inference.

Do not run an evaluator.

Do not execute a tokenizer.

Do not train.

Do not perform significance testing.

Do not use Kaggle.

A later frozen authority must explicitly authorize the read-only
GEN4_SIX_CELL_OUTCOME_EVIDENCE_INVENTORY_AUDIT.
