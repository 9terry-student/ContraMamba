# Native State Kinematics
# tau_e Semantic Rule Candidate

STATUS = CANDIDATE

SUPERSESSION_AUTHORITY_COMMIT =
46f97ed403a47f530b139987836f85381e5ae599

SOURCE_DATASET_SHA256 =
eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3

STRUCTURED_SOURCE_PRODUCER =
scripts/build_controlled_v5.py::fact_templates_for_count

STRUCTURED_SOURCE_PRODUCER_SHA256 =
9fbd94a151c4d83a5e824412d7c0837062fedd20628f4f198116b2d08b679410

CANONICAL_ANALYSIS_TOKENIZER_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

CANONICAL_TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf


## 1. Rule principle

tau_e is the final consumed evidence token overlapping the complete,
prespecified conclusion-critical semantic event span.

The event span is selected only from frozen generator semantics.

Model prediction, confidence, correctness, native state, and P1/P2/P3
are not inputs to this rule.


## 2. Predicate/polarity families

For:

- none
- paraphrase
- polarity_flip

all non-polarity factual slots are fixed by the generator contract.

The conclusion-critical event is therefore the complete predicate-polarity
realization:

- positive predicate; or
- complete "did not <predicate>" realization.

EVENT =
PREDICATE_POLARITY_REALIZATION_END


## 3. Localized swap families

entity_swap:
    end of alternate_name

event_swap:
    end of alternate_object

location_swap:
    end of alternate_location

role_swap:
    end of alternate_role

title_name_swap:
    end of the complete alternate_title + alternate_name identity span

predicate_swap:
    end of alternate_predicate

The selected slot is exactly the structured slot changed by the frozen
generator for that intervention.


## 4. Absence events

evidence_deletion and evidence_truncation do not receive synthetic tau_e.

STATUS =
UNANNOTATABLE_ABSENCE_EVENT

An absent decisive fact is not inserted into model time.


## 5. Irrelevant evidence

irrelevant_evidence has no unique claim-relative localized structured slot.

Selecting "weather", "bulletin", or another convenient token would be an
arbitrary event-time choice.

STATUS =
UNANNOTATABLE_NO_LOCALIZED_STRUCTURED_EVENT


## 6. Token-coordinate rule

Claim is tokenized with:

add_special_tokens=False
max active budget=63

Evidence is tokenized with:

add_special_tokens=False
max active budget=64

A0 absolute evidence coordinate begins at:

evidence_start = len(claim_ids) + 1

For an annotatable semantic character span, find all raw evidence tokens
whose tokenizer offset overlaps that span.

The event token is the last overlapping token.

tau_e =
evidence_start + event_evidence_token_index

If the event token is outside the first 64 evidence tokens, the example is
UNANNOTATABLE_EVENT_NOT_CONSUMED.


## 7. Prefix lock

terminal_index is the last non-padding consumed A0 token.

POST4_PREFIX_ELIGIBLE is true iff:

tau_e + 4 <= terminal_index - 1

An annotated row that fails this requirement retains tau_e but is excluded
from the primary confirmatory cohort with:

POST4_PREFIX_INELIGIBLE


## 8. Explicit prohibitions

tau_e is not:

- evidence_start;
- evidence_end;
- arbitrary midpoint;
- final token by convenience;
- first tokenizer divergence;
- O0b index;
- O0c anchor;
- native-state change point.

PREDICTION_ARTIFACT_ACCESS =
NO

CONFIDENCE_ACCESS =
NO

CORRECT_WRONG_ACCESS =
NO

NATIVE_STATE_ACCESS =
NO

END_OF_TAU_E_SEMANTIC_RULE_CANDIDATE
