# ContraMamba Gen4 Six-Cell Tier-2 Dedicated Inference Adapter Implementation Authority - Candidate

## 1. Status

STATUS =
CANDIDATE

PHASE =
R3_DEDICATED_INFERENCE_ADAPTER_IMPLEMENTATION_AND_STATIC_TESTS

PARENT_R2_RESULT_FREEZE =
17f1ddfc8286796f27c4a61716a21e14126bb836

HISTORICAL_EVALUATOR_SOURCE_COMMIT =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

IMPLEMENTATION_AUTHORITY =
NOT_ACTIVE_UNTIL_THIS_SPEC_IS_FROZEN

## 2. Purpose

R3 authorizes only the minimal implementation required to make the frozen Gen4
six-cell artifact evaluable by the previously frozen Tier-2 Gen3 grouped
checkpoint population.

R3 is not scientific inference authority.

R3 is not checkpoint-loadability execution authority.

R3 is not training authority.

The objective is to implement a dedicated label-free inference-only adapter
whose model forward semantics are bound to the historical Gen3 grouped source
lineage and whose input coordinates are bound to the validated R2 tokenizer
contract.

## 3. Frozen parent evidence

R2 establishes:

GEN4_ACTIVE_ENCODING_CONTENT_EQUIVALENCE =
PASS

GEN4_CANONICAL_TOKENIZER_CONTENT =
FAMILY_A

GEN4_CANONICAL_TOKENIZER_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

GEN4_SERIALIZED_INPUT_COORDINATE_SHA256 =
d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a

HISTORICAL_WRAPPER_RUNTIME_EQUIVALENCE =
UNRESOLVED

R3 must preserve that unresolved wrapper-runtime status.

## 4. Historical model binding

The authoritative historical model source is:

src/contramamba/modeling_v6b_minimal.py

at:

3e0e9a435068c552abf20f3a74e0c3eccca344a3

HISTORICAL_MODEL_GIT_BLOB =
e29d361d99579b0a86a626ebcdebd59ba465881c

HISTORICAL_MODEL_SHA256 =
8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5

The current repository model implementation is not an authorized substitute.

MODEL_FORWARD_SEMANTICS_DELTA =
ZERO_REQUIRED

## 5. Historical head dependency binding

The historical model imports the existing contramamba heads package.

Historical heads tree:

HISTORICAL_HEADS_TREE_GIT_OBJECT =
68d26855aa511fcd41d6f395ae5f87177a162678

At parent R2 result freeze, the current heads tree has the same Git tree
identity.

Therefore:

CURRENT_HEADS_TREE_HISTORICAL_BYTE_EQUIVALENCE =
YES

R3 may reuse the existing heads package without copying it.

Any change to src/contramamba/heads during R3 is forbidden.

## 6. Exact implementation scope

R3 implementation may create exactly these three files:

1.
src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py

2.
scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py

3.
tests/test_reason_router_gen4_six_cell_tier2_inference_adapter.py

No existing tracked file may be modified.

No fourth implementation file is authorized without a new authority decision.

## 7. Historical snapshot file requirement

The new snapshot file:

src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py

must be an exact byte copy of:

3e0e9a435068c552abf20f3a74e0c3eccca344a3:
src/contramamba/modeling_v6b_minimal.py

Required:

SNAPSHOT_SHA256 =
8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5

SNAPSHOT_SOURCE_DELTA =
ZERO_BYTES

No cleanup, formatting, renaming inside the source, comment change, or modern
API adaptation is allowed in the snapshot.

The filename may differ; file contents may not.

## 8. Existing trainer boundary

The following file must not be modified:

scripts/train_controlled_v6b_minimal.py

The following current model file must not be modified:

src/contramamba/modeling_v6b_minimal.py

R3 must not retrofit Gen4 support into the training CLI.

EXISTING_TRAINER_MODIFICATION_FOR_GEN4 =
FORBIDDEN

## 9. Canonical Gen4 input

The adapter must accept only the canonical Gen4 six-cell materialization:

CANONICAL_GEN4_ARTIFACT_SHA256 =
b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

CANONICAL_GEN4_ARTIFACT_BYTES =
1465573

CANONICAL_GEN4_ROWS =
1800

CANONICAL_GEN4_SOURCE_PAIRS =
300

Each input row must directly provide and preserve:

row_id
source_pair_id
contrast_cell_id
claim
evidence

No fuzzy identity recovery is permitted.

## 10. Label-free contract

The Gen4 adapter must not require:

final_label
frame_compatible_label
predicate_covered_label
sufficiency_label
polarity_label
primary_failure_type

Gold labels must not be synthesized.

The adapter must construct model feature tensors only:

input_ids
attention_mask
claim_mask
evidence_mask

LABEL_DEPENDENCY =
ZERO_REQUIRED

## 11. Canonical tokenizer contract

The adapter must bind tokenizer content to Family A:

REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

Required exact file SHA256:

tokenizer.json =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

tokenizer_config.json =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

special_tokens_map.json =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

Mutable model-name tokenizer resolution is forbidden for scientific use.

Network tokenizer resolution is forbidden.

## 12. Input construction contract

The adapter must implement the frozen R1/R2 procedure:

MAX_LENGTH =
128

CLAIM_BUDGET =
63

SEPARATOR =
EOS_TOKEN_ID

EVIDENCE_BUDGET =
64

ADD_SPECIAL_TOKENS =
false

PAD_TO =
128

EOS_TOKEN_ID =
0

EFFECTIVE_PAD_TOKEN_ID =
0

Masks:

claim_mask covers claim tokens only.

evidence_mask covers evidence tokens only.

separator is excluded from both span masks.

attention_mask covers all non-padding sequence positions.

R3 static tests must exercise this logic without model execution.

## 13. Model construction boundary

The adapter must import ContraMambaV6BMinimal from the historical snapshot
module, not from the current modeling_v6b_minimal module.

MODEL_NAME =
state-spaces/mamba-130m-hf

ARCHITECTURE =
v6b_minimal

REASON_ROUTER_MODE =
explicit_product

GRADIENT_OWNERSHIP_MODE =
edge_specific

FREEZE_ENCODER =
true

Any constructor or forward-affecting configuration used during later R4/R5
must be recovered from historical run provenance or historical source.

R3 must not invent new defaults to make checkpoint loading succeed.

## 14. Historical wrapper-runtime limitation

Historical run provenance records:

TRANSFORMERS_VERSION =
5.0.0

R2 did not establish equivalence between that historical wrapper runtime and
the currently installed Transformers runtime.

Therefore R3 implementation must not claim runtime equivalence.

The adapter may contain model-construction code for later authorized phases,
but R3 validation must not instantiate the model.

HISTORICAL_WRAPPER_RUNTIME_EQUIVALENCE =
UNRESOLVED

## 15. Exact evaluator population

The adapter must recognize exactly the frozen 18 evaluators:

Seeds:

180
181
182

Arms:

G3-GROUP-D-HALF
G3-GROUP-Q-D-HALF
G3-GROUP-Q-HALF
G3-GROUP-U-D-HALF
G3-GROUP-U-HALF
G3-GROUP-U-Q-HALF

No evaluator may be selected based on Gen4 outcomes.

No checkpoint weighting may be outcome-dependent.

No single-checkpoint shortcut is permitted.

## 16. Frozen checkpoint SHA registry

The implementation must encode or otherwise fail-closed against these exact
checkpoint SHA256 identities:

seed180 / G3-GROUP-D-HALF =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

seed180 / G3-GROUP-Q-D-HALF =
2e51f64702a3ebf21d5d8e8aa84745b62b3faa01112b5b8f10525ba6435dbc8c

seed180 / G3-GROUP-Q-HALF =
eb349aefca6d992df42f6239e7cf642d560755c0b1819397dba1d746b33bd8e3

seed180 / G3-GROUP-U-D-HALF =
08654abb9c1ec67d42fa1b3464f19298f21ff79b866fb0cf8b7a97d59a45ff86

seed180 / G3-GROUP-U-HALF =
a8cd296136816f806394ca98d6433bfa560f5691ab37e661347c2db838966708

seed180 / G3-GROUP-U-Q-HALF =
0701ce934ae3ef34cd9f9d229c9321599b4ca150db8dabc3c8a740668b8f0aad

seed181 / G3-GROUP-D-HALF =
afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f

seed181 / G3-GROUP-Q-D-HALF =
390b4fe3266d8eddebe74d9732321d1f96e2a7095ecae67b6155a2d535b655ba

seed181 / G3-GROUP-Q-HALF =
3b5044fddb7f542c9e06a318a5a81a731d94475f7f67b7e5c5a7787ab3af0ba6

seed181 / G3-GROUP-U-D-HALF =
7adffc577e00b9a9150bca28ed83b35eb5574458f71d5bc276ebd8f557b00e4d

seed181 / G3-GROUP-U-HALF =
e2a9fd1ca6e50856b2349fc5bc915c54e6a71848aaa8c59aaa1f8c8647e89699

seed181 / G3-GROUP-U-Q-HALF =
1be3be2ddd13762d36c69ef16ccbdd0ee4bd5ad732eff46e7a66cab703c2db50

seed182 / G3-GROUP-D-HALF =
f9db48a3b3b9fdc6df4e2bb2086d11fd80fd595e6096c0095d1992f6c7d777f2

seed182 / G3-GROUP-Q-D-HALF =
cb1f4812d11643089bb87064c436b2e890554435254c961e5ed3f766b61b412b

seed182 / G3-GROUP-Q-HALF =
67d0cbf855b24a291f55ce87425dcd4d77b5f7a59fb119c57c261c6378a4342e

seed182 / G3-GROUP-U-D-HALF =
f1d84bab31f9080f0f3cfc6d0ee49cdc2743ad7c8c3a620bee7f576ca32ebef1

seed182 / G3-GROUP-U-HALF =
47b43899119a0a450e0b5cf8134ade521d8cea9ca568110b32223de6109ef5a4

seed182 / G3-GROUP-U-Q-HALF =
129be6e930b5f7e6737ad671a9646c867d150e8751907bb1abfd3fe64570669f

## 17. Checkpoint handling contract

Before any future checkpoint deserialization, the adapter must:

1. resolve seed and arm;
2. resolve the exact frozen expected SHA256;
3. hash the checkpoint bytes;
4. fail closed on mismatch;
5. only then permit deserialization.

CHECKPOINT_SHA_AUTHENTICATION_BEFORE_DESERIALIZATION =
REQUIRED

R3 itself does not deserialize checkpoints.

## 18. Checkpoint state contract

The future load path must preserve the historical strict state-dict semantics.

STRICT_STATE_DICT_LOADING =
REQUIRED

No:

strict=False
missing-key filtering
unexpected-key filtering
automatic key rewriting
head dropping
shape coercion

is permitted merely to obtain a load PASS.

Any incompatibility is an R4 result, not an R3 repair opportunity.

## 19. Historical model outputs

The adapter must expose the historical scientific quantities needed by the
frozen Gen4 outcome specification:

q_authorized
entitlement_prob
refute_logit
ne_logit
support_logit
prediction

External class order is fixed:

0 = REFUTE
1 = NOT_ENTITLED
2 = SUPPORT

The adapter may also expose final_logits as the ordered three-element vector.

## 20. Primary and secondary outcome integrity

Primary:

q_authorized

Secondary descriptive quantities:

entitlement_prob

support_vs_best_nonsupport_logit_margin =
support_logit - max(refute_logit, ne_logit)

R3 must not introduce or substitute a different primary outcome.

## 21. Output row identity

Each future evaluator output row must directly include:

seed
arm
checkpoint_sha256
row_id
source_pair_id
contrast_cell_id

No output identity may be reconstructed from claim/evidence text.

Expected future scientific output row count:

1800 * 18 =
32400

Required future key:

(seed, arm, row_id)

must be unique.

## 22. Complete-matrix contract

Every Gen4 row must have exactly 18 evaluator observations.

Every source pair must eventually have:

6 cells * 18 evaluators =
108 observations.

Missing evaluator rows, duplicated evaluator rows, or partial checkpoint
populations fail closed.

No available-case analysis is permitted.

## 23. Serializer design

The adapter must implement a minimal dedicated serializer.

It must not invoke the full historical/current training prediction serializer.

This is necessary because:

prediction_records_v6b is historically equivalent but training-oriented;

the P2 export helper is not current/historical AST-equivalent;

Gen4 must remain label-free.

The dedicated serializer must extract only the frozen required outcome fields.

## 24. Static unit-test requirements

R3 tests must cover at minimum:

historical snapshot SHA256 exactness;

historical/current heads-tree identity expectation;

exact six-arm / three-seed / 18-checkpoint registry;

rejection of unknown seed or arm;

checkpoint SHA mismatch rejection before any loader callback;

label-free Gen4 row acceptance;

missing row_id rejection;

missing source_pair_id rejection;

missing contrast_cell_id rejection;

duplicate row_id rejection;

63/1/64 feature construction;

separator mask exclusion;

padding to length 128;

fixed external class order;

synthetic-output extraction for q_authorized;

synthetic-output extraction for entitlement_prob;

synthetic three-logit extraction;

prediction argmax mapping;

support-minus-best-nonsupport margin;

future output-key construction;

complete-matrix validation on synthetic small fixtures.

## 25. Test execution boundary

R3 static tests may use:

Python standard library
tokenizers
torch tensors for synthetic serializer fixtures

R3 tests must not:

instantiate ContraMambaV6BMinimal;
instantiate MambaModel;
call MambaConfig.from_pretrained;
call MambaModel.from_pretrained;
load a checkpoint;
run model.forward;
download any artifact;
train;
evaluate scientific outcomes.

MODEL_INSTANTIATION_DURING_R3_TESTS =
FORBIDDEN

## 26. Network boundary

R3 implementation and tests must be network-independent.

NETWORK_ACCESS =
FORBIDDEN

No model or tokenizer download may be used to make tests pass.

## 27. CLI boundary

The adapter CLI must require an explicit operation.

There must be no implicit scientific inference default.

Allowed future operation names may include:

validate-input
checkpoint-loadability
infer

But authority remains phase-specific:

R3 may execute only static/unit validation.

R4 may later authorize checkpoint-loadability.

R5 may later authorize infer.

The presence of a CLI operation does not itself authorize its execution.

## 28. Provenance outputs for future phases

Future R4/R5 outputs must record at minimum:

adapter source commit
historical source commit
historical snapshot SHA256
heads tree identity
canonical Gen4 artifact SHA256
canonical tokenizer revision reference
canonical tokenizer file SHA256 values
runtime Python version
runtime torch version
runtime transformers version
device
dtype
batch size
checkpoint SHA256
seed
arm

R3 implementation must provide a deterministic provenance-record builder for
these fields.

## 29. Runtime unresolved state

R3 must not silently resolve:

HISTORICAL_WRAPPER_RUNTIME_EQUIVALENCE =
UNRESOLVED

R4 checkpoint loadability and later runtime qualification must remain separate
from R3 code correctness.

## 30. Scientific execution boundary

After R3 implementation/static tests PASS:

CODE_CORRECTNESS =
MAY_BE_ESTABLISHED

but:

CHECKPOINT_LOADABILITY =
NOT_ESTABLISHED

SCIENTIFIC_INFERENCE_SUCCESS =
NOT_ESTABLISHED

ARTIFACT_PROVENANCE_VALIDITY_FOR_SCIENTIFIC_RUN =
NOT_ESTABLISHED

SCIENTIFIC_CONCLUSION =
NOT_ESTABLISHED

## 31. R4 transition

After R3 implementation is independently validated and frozen, the next phase
is:

R4_CHECKPOINT_LOADABILITY_PREFLIGHT

R4 must authenticate and attempt strict loading of all 18 frozen checkpoints.

R4 is not authorized by this candidate.

## 32. Explicit prohibitions

R3 must not:

modify historical checkpoints;

modify canonical Gen4 materialization;

modify frozen tokenizer snapshots;

modify existing trainer code;

modify current modeling_v6b_minimal.py;

modify heads code;

run training;

run scientific inference;

perform statistical testing;

select a checkpoint based on results;

change the primary outcome;

add post-hoc evaluator weighting;

use Kaggle.

## 33. Implementation validation target

The implementation phase must end with:

exact three-file implementation scope;

historical snapshot byte identity PASS;

static/unit test suite PASS;

no tracked files outside authorized scope modified;

no checkpoint deserialization;

no model instantiation;

no model forward;

no scientific inference.

## 34. Current authorization

Before this authority is frozen:

R3_IMPLEMENTATION =
NOT_AUTHORIZED

After this exact authority is frozen:

R3_IMPLEMENTATION =
AUTHORIZED_EXACT_THREE_FILE_SCOPE

R3_STATIC_TESTS =
AUTHORIZED

CHECKPOINT_LOADABILITY_EXECUTION =
NOT_AUTHORIZED

MODEL_INFERENCE =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

STATISTICAL_TESTING =
NOT_AUTHORIZED

KAGGLE =
NOT_AUTHORIZED

## 35. Result

R3_IMPLEMENTATION_AUTHORITY_RESULT =
READY_FOR_FREEZE_REVIEW

NEXT_ACTION_AFTER_FREEZE =
IMPLEMENT_EXACT_THREE_FILE_DEDICATED_INFERENCE_ADAPTER_AND_STATIC_TESTS
