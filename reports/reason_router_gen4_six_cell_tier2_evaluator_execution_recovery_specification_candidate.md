# ContraMamba Gen4 Six-Cell Tier-2 Evaluator Execution Recovery Specification - Candidate

## 1. Status

STATUS =
CANDIDATE

PHASE =
GEN4_SIX_CELL_TIER2_EVALUATOR_EXECUTION_RECOVERY_SPECIFICATION

PARENT_CAPABILITY_AUDIT_AUTHORITY =
56b9b5bfb2d417f18ca3586c74f4d18a5eb90dc2

PARENT_STATISTICAL_SPECIFICATION_AUTHORITY =
4dc5bacd10a254b5ecd339ac1fe78bad9def5c47

PRIMARY_Y =
q_authorized

EVALUATOR_POPULATION =
FULL_PRESPECIFIED_GEN3_GROUPED_18_RUN_MATRIX

This specification defines the minimum recovery path required to make
the frozen Tier-2 evaluator population scientifically executable on the
frozen Gen4 six-cell structural artifact.

This specification authorizes no execution and no implementation.

## 2. Frozen capability blocker

The parent capability audit established:

GEN4_TIER2_EVALUATOR_EXECUTION_CAPABILITY =
INCOMPLETE

TRAINER_EMITS_Q_AUTHORIZED_STATIC_SIGNAL =
YES

TRAINER_HAS_ALL_GEN4_IDENTITY_TERMS =
NO

EXISTING_FULL_STATIC_CAPABILITY_SCRIPT =
NO

EXPLICIT_EVAL_ONLY_CLI_STATIC_SIGNAL =
NO

EXACT_TOKENIZER_IDENTITY_IN_18_RUN_PROVENANCE =
NO

Therefore direct execution through the existing historical trainer is not
authorized.

## 3. Recovery objective

The recovery objective is strictly:

construct a provenance-closed, label-free, inference-only path from the
frozen Gen4 six-cell artifact through the exact frozen 18-checkpoint evaluator
population to deterministic outcome rows required by the frozen statistical
specification.

The recovery is infrastructure only.

It must not change the scientific intervention or estimator.

## 4. Scientific non-delta requirement

The following must remain unchanged:

PRIMARY_Y =
q_authorized

SECONDARY_Y_1 =
entitlement_prob

SECONDARY_Y_2 =
support_vs_best_nonsupport_logit_margin

EVALUATOR_COUNT =
18

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

PRIMARY_PAIR_COUNT =
300

CONFIRMATORY_HYPOTHESIS_COUNT =
6

MODEL_FORWARD_SEMANTICS_DELTA =
ZERO_REQUIRED

STATISTICAL_SEMANTICS_DELTA =
ZERO_REQUIRED

INTERVENTION_SEMANTICS_DELTA =
ZERO_REQUIRED

## 5. Historical evaluator implementation authority

The 18 checkpoint provenance records bind the evaluator lineage to:

HISTORICAL_EVALUATOR_SOURCE_COMMIT =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

The recovery path must not silently assume that the current HEAD trainer has
identical construction or forward semantics.

Therefore:

CURRENT_HEAD_MODEL_FORWARD_EQUIVALENCE =
NOT_ESTABLISHED

IMPLICIT_CURRENT_TRAINER_EQUIVALENCE =
FORBIDDEN

Before implementation execution is authorized, the exact model construction,
checkpoint-load contract, forward path, output keys, and relevant serializer
semantics must be statically bound to the historical evaluator source lineage.

## 6. Model implementation binding rule

A future implementation may use one of only two admissible strategies.

Strategy A:

use implementation whose relevant model construction and forward semantics are
byte- or statically-proven equivalent to the historical evaluator lineage.

Strategy B:

use an explicitly frozen historical implementation snapshot bound directly to
the evaluator source commit.

No strategy may be selected by convenience after scientific outcome
inspection.

The exact strategy must be frozen before real checkpoint load.

MODEL_IMPLEMENTATION_BINDING =
PENDING_STATIC_RECOVERY_AUDIT

## 7. Existing trainer modification boundary

The existing large training program:

scripts/train_controlled_v6b_minimal.py

must not be converted into the Gen4 execution surface by adding a large
evaluation branch.

Therefore:

EXISTING_TRAINER_MODIFICATION_FOR_GEN4 =
FORBIDDEN

A later implementation must use a dedicated minimal inference adapter/interface.

The adapter may reuse proven existing primitives.

It may not modify model equations.

## 8. Label-free requirement

The frozen Gen4 structural artifact contains no:

final_label
label

The primary q_authorized analysis requires no gold label.

Therefore the recovery path must be:

LABEL_FREE =
REQUIRED

TRAINING_LOSS =
FORBIDDEN

GOLD_LABEL_SYNTHESIS =
FORBIDDEN

PSEUDO_LABEL_CREATION =
FORBIDDEN

Any code path that requires a Gen4 final label is inadmissible.

## 9. Canonical Gen4 input identity

The only scientific structural input is:

reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

SHA256:

b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

Bytes:

1465573

Rows:

1800

Source pairs:

300

No regeneration from claim/evidence text is permitted.

## 10. Identity preservation contract

For every input row, the evaluator path must preserve verbatim:

row_id
source_pair_id
contrast_cell_id

These fields are metadata.

They must not be tokenized as identity surrogates.

They must not be reconstructed from rendered text.

Therefore:

ROW_IDENTITY_MODE =
DIRECT_METADATA_CARRIAGE

TEXT_SEMANTIC_IDENTITY_RECONSTRUCTION =
FORBIDDEN

ROW_ORDER_ONLY_IDENTITY =
FORBIDDEN

FUZZY_JOIN =
FORBIDDEN

## 11. Existing tokenizer correction evidence

Existing frozen tokenizer evidence is admitted as supporting provenance.

TOKENIZER_CORRECTION_AUTHORITY =
daa5d479f1a040cac308358db711871b0af27020

TOKENIZER_CORRECTION_VALIDATION =
c9b2d16c6b14ddc0a14a0c0e55963640c86ffee2

TOKENIZER_GATE_SUPERSESSION_AUTHORITY =
46f97ed403a47f530b139987836f85381e5ae599

Existing evidence established active-encoding equivalence for the frozen A0
3600-row use case.

That evidence is not silently generalized to Gen4.

## 12. Existing tokenizer family identities

The previously validated tokenizer families are:

FAMILY_A_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

FAMILY_B_REVISION_REFERENCE =
5708daa364c50b880e7bd92eab456e0d34492ee9

Family A exact active files:

tokenizer.json SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

tokenizer_config.json SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

special_tokens_map.json SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

Family B exact active files:

tokenizer.json SHA256 =
3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8

tokenizer_config.json SHA256 =
fcd5669efe1150240c13ee4bd863316de4f2abd14cb1806a8cdbcbea6577bc99

special_tokens_map.json SHA256 =
10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c

## 13. Scope limit of existing tokenizer evidence

The existing tokenizer correction established:

ACTIVE_ENCODING_CONTENT_EQUIVALENCE =
PASS

on the prior frozen 3600-row A0 token-coordinate contract.

It also explicitly retained:

HISTORICAL_TOKENIZER_EXACT_SNAPSHOT =
NOT_RECOVERED

and:

HISTORICAL_WRAPPER_RUNTIME_EQUIVALENCE =
UNRESOLVED_TRANSFORMERS_5_0_0_NOT_LOCALLY_AVAILABLE

Therefore:

GEN4_TOKENIZER_EQUIVALENCE_FROM_PRIOR_A0_EVIDENCE =
NOT_ESTABLISHED

The previous PASS is reusable methodology and provenance evidence, not direct
Gen4 execution authority.

## 14. Historical Gen3 encoding contract recovery

Before any tokenizer execution on Gen4 rows, a read-only static audit must
recover the exact encoding contract used by the Gen3 grouped evaluator lineage.

The audit must inspect:

- historical source commit 3e0e9a435068c552abf20f3a74e0c3eccca344a3;
- all 18 run_provenance.json files;
- checkpoint metadata already available without deserializing checkpoints;
- frozen trainer/source code only.

It must determine and freeze, where applicable:

model name
tokenizer resolution rule
claim encoding rule
evidence encoding rule
claim/evidence concatenation rule
separator identity
special-token behavior
padding rule
truncation rule
claim token budget
evidence token budget
total maximum sequence length
attention-mask construction
any token-type behavior
tensor dtype
evaluation batch assumptions that affect model input bytes

No tokenizer execution is required for this static audit.

## 15. Gen4 tokenizer conformance requirement

After the historical Gen3 encoding contract is frozen, a separately authorized
tokenizer-only preflight must apply that exact contract to all 1800 frozen Gen4
rows.

The preflight must use exact locally provisioned tokenizer-family files.

It must not download from the network.

It must not resolve a mutable model name as scientific provenance.

At minimum it must compare every relevant observed tokenizer family capable of
representing the historical ambiguity.

Required result:

GEN4_ACTIVE_ENCODING_EQUIVALENCE =
PASS_1800_OF_1800

for all fields that constitute model input.

These include, as applicable:

input_ids
attention_mask
claim/evidence boundary coordinates
separator placement
padding positions
truncation outcomes

Any mismatch blocks evaluator execution.

## 16. Canonical tokenizer selection rule

Only after Gen4 conformance is established may one exact tokenizer content
family be frozen as the canonical procedural reference.

Selection must occur before model outcome inspection.

Preferred candidate:

FAMILY_A_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

because it is already the canonical analysis reference in prior frozen
tokenizer provenance work.

However:

GEN4_CANONICAL_TOKENIZER =
NOT_YET_AUTHORIZED

The future preflight must establish admissibility under the recovered Gen3
encoding contract.

## 17. Network boundary

For all tokenizer provenance and scientific execution stages:

NETWORK_TOKENIZER_DOWNLOAD =
FORBIDDEN

MUTABLE_FROM_PRETRAINED_RESOLUTION =
FORBIDDEN_AS_SCIENTIFIC_PROVENANCE

LOCAL_EXACT_CONTENT_AUTHENTICATION =
REQUIRED

A local exact tokenizer directory must be authenticated by file SHA256 before
tokenizer construction.

## 18. Checkpoint population

The checkpoint population remains exactly the full prespecified 18-run matrix.

No checkpoint may be dropped.

No replacement checkpoint may be introduced.

No checkpoint may receive performance-based weighting.

CHECKPOINT_COUNT =
18

CHECKPOINT_SELECTION =
FORBIDDEN

CHECKPOINT_WEIGHTING =
EQUAL_FIXED_POPULATION

## 19. Checkpoint provisioning boundary

The approximately 518 MB checkpoint files are external historical run
artifacts and are not to be added to Git.

Therefore:

CHECKPOINT_BINARY_COMMIT_TO_GIT =
FORBIDDEN

A future execution environment must provision each checkpoint separately and
bind it to:

training seed
grouped arm
exact expected SHA256
expected byte count

before load.

## 20. Checkpoint authentication

Before any torch.load or equivalent deserialization:

raw checkpoint SHA256 must be computed.

The observed identity must exactly match the frozen 18-checkpoint manifest from
the statistical specification.

Mismatch behavior:

FAIL_CLOSED

No fallback checkpoint search is permitted.

No newest-checkpoint selection is permitted.

No best-checkpoint reselection is permitted.

## 21. Checkpoint loadability preflight

Checkpoint loadability remains separately unestablished.

A later narrow preflight may be authorized to:

- instantiate the proven model implementation;
- load all 18 checkpoints with map_location=cpu;
- verify expected state-dict compatibility;
- verify frozen checkpoint metadata contract;
- perform no model forward.

Required result:

CHECKPOINT_LOADABILITY =
PASS_18_OF_18

This recovery specification does not authorize that preflight.

## 22. Dedicated inference adapter

The eventual implementation must provide a dedicated inference-only adapter.

It must not train.

It must not create an optimizer.

It must not call backward.

It must not mutate parameters.

It must set the model into inference/evaluation behavior before forward use.

The implementation must fail closed if invoked without exact frozen
provenance arguments.

## 23. Adapter minimum input contract

The adapter must accept:

canonical six-cell JSONL
authenticated tokenizer directory/content identity
authenticated checkpoint root or explicit checkpoint manifest
frozen evaluator seed/arm coordinate
frozen provenance identities

It may not accept arbitrary scientific datasets under the Gen4 authority.

## 24. Adapter minimum output contract

For every evaluator-row observation the future deterministic artifact must
contain at minimum:

schema_version
structural_artifact_commit
statistical_specification_commit
recovery_implementation_commit
historical_evaluator_source_commit
evaluator_seed
evaluator_arm
checkpoint_sha256
tokenizer_identity
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

No statistical p-values belong in the evaluator artifact.

## 25. Output-semantic reuse requirement

The adapter must not invent a new interpretation of:

q_authorized
entitlement_prob
support_logit
ne_logit
refute_logit
prediction

The output mapping must reuse or prove equivalence to the frozen historical
evaluator serialization semantics.

Final-class index mappings must not be guessed.

## 26. Derived secondary diagnostic

The only newly derived numeric diagnostic authorized at serialization is:

support_vs_best_nonsupport_logit_margin =
support_logit - max(ne_logit, refute_logit)

It must be computed mechanically from the frozen output logits.

It must not alter model output.

## 27. Cardinality contract

Input rows:

1800

Evaluators:

18

Expected evaluator outcome rows:

32400

Required unique key:

(evaluator_seed, evaluator_arm, row_id)

Required unique-key count:

32400

Each row_id occurrence count:

18

Each source_pair_id occurrence count:

108

Any violation blocks scientific statistical testing.

## 28. Missingness

The scientific execution matrix must be complete.

No missing evaluator is permitted.

No missing structural row is permitted.

No missing primary q_authorized value is permitted.

No imputation is permitted.

No partial primary analysis is permitted.

MISSINGNESS_POLICY =
FAIL_CLOSED_COMPLETE_MATRIX

## 29. Determinism boundary

Scientific inference must use:

no training
no stochastic augmentation
no parameter mutation
no outcome-dependent batching
no evaluator selection

The eventual execution authority must freeze:

device
dtype policy
batch size
serialization order
serialization numeric representation
runtime library identities

before first scientific forward.

## 30. Validation before real inference

Implementation validation must occur before real checkpoint inference.

Tests must use synthetic or non-scientific fixtures where possible.

Required validation classes include:

canonical input-schema validation
identity-preservation validation
duplicate-row rejection
missing-row rejection
checkpoint-manifest validation
checkpoint-SHA mismatch rejection
tokenizer-file mismatch rejection
label-free input acceptance
gold-label independence
output-schema validation
deterministic serialization
32400-row cardinality logic
no-training/no-backward guard
historical output-mapping equivalence

Implementation validation PASS does not authorize scientific execution.

## 31. Forbidden implementation deltas

The recovery implementation must not change:

model architecture
parameter values
router equations
edge-gradient semantics
reason probabilities
q_authorized definition
entitlement definition
final-logit semantics
checkpoint selection
evaluator population
primary outcome
secondary outcomes
six-cell design
statistical estimands
multiplicity procedure

## 32. No training fallback

If the recovery cannot reproduce the historical evaluator contract:

NEW_TRAINING_FALLBACK =
NOT_AUTHORIZED

Failure to recover tokenizer/model compatibility is a provenance blocker.

It is not authorization to retrain.

## 33. Phase ordering

Required order:

PHASE_R1 =
HISTORICAL_INPUT_AND_FORWARD_CONTRACT_STATIC_RECOVERY

then:

PHASE_R2 =
TOKENIZER_ACTIVE_ENCODING_GEN4_CONFORMANCE_PREFLIGHT

then:

PHASE_R3 =
DEDICATED_INFERENCE_ADAPTER_IMPLEMENTATION_AND_STATIC_TESTS

then:

PHASE_R4 =
CHECKPOINT_LOADABILITY_PREFLIGHT

then:

PHASE_R5 =
SCIENTIFIC_INFERENCE_EXECUTION_AUTHORITY

then only after validated outcome artifact import:

PHASE_R6 =
STATISTICAL_TESTING_AUTHORITY

No phase may be skipped.

## 34. Immediate next object

The immediate next object is:

GEN4_SIX_CELL_TIER2_HISTORICAL_INPUT_FORWARD_CONTRACT_AUDIT

This audit is read-only.

It must recover exact historical evaluator input and forward/output contract
from frozen source and provenance without:

tokenizer execution
checkpoint loading
model instantiation
model forward
training
statistical testing

## 35. Current execution boundary

TOKENIZER_EXECUTION =
NOT_AUTHORIZED

CHECKPOINT_LOAD =
NOT_AUTHORIZED

MODEL_INSTANTIATION =
NOT_AUTHORIZED

MODEL_INFERENCE =
NOT_AUTHORIZED

STATISTICAL_TESTING =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

KAGGLE_EXECUTION =
NOT_AUTHORIZED

## 36. Current recovery decision

RECOVERY_MODE =
MINIMAL_DEDICATED_INFERENCE_ONLY_ADAPTER

EXISTING_TRAINER_MODIFICATION_FOR_GEN4 =
FORBIDDEN

MODEL_IMPLEMENTATION_BINDING =
PENDING_STATIC_RECOVERY_AUDIT

GEN4_TOKENIZER_EQUIVALENCE_FROM_PRIOR_A0_EVIDENCE =
NOT_ESTABLISHED

TOKENIZER_PROVENANCE_REUSE =
ADMITTED_AS_SCOPE_LIMITED_SUPPORTING_EVIDENCE

GEN4_ACTIVE_ENCODING_EQUIVALENCE =
NOT_YET_ESTABLISHED

CHECKPOINT_LOADABILITY =
NOT_YET_ESTABLISHED

SCIENTIFIC_OUTCOME_CONCLUSION =
NOT_ESTABLISHED

NEXT_OBJECT =
GEN4_SIX_CELL_TIER2_HISTORICAL_INPUT_FORWARD_CONTRACT_AUDIT

## 37. Stop condition

Stop after this recovery specification candidate is created and reviewed.

Do not modify trainer code.

Do not implement the inference adapter.

Do not run a tokenizer.

Do not load a checkpoint.

Do not instantiate a model.

Do not run a forward pass.

Do not calculate Gen4 q_authorized.

Do not perform statistical testing.

Do not train.

Do not use Kaggle.

RECOVERY_SPECIFICATION_RESULT =
READY_FOR_FREEZE_REVIEW
