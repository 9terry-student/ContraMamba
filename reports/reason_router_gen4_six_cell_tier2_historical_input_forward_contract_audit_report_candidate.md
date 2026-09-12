# ContraMamba Gen4 Six-Cell Tier-2 Historical Input / Forward Contract Audit Report - Candidate

## 1. Status

STATUS =
CANDIDATE

PHASE =
GEN4_SIX_CELL_TIER2_HISTORICAL_INPUT_FORWARD_CONTRACT_AUDIT

PARENT_RECOVERY_SPECIFICATION_AUTHORITY =
b71e5dc272433d4303d289f9e1cf01519eccc137

HISTORICAL_EVALUATOR_SOURCE_COMMIT =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

AUDIT_MODE =
READ_ONLY_STATIC_RECOVERY

MODEL_FORWARD_PERFORMED =
NO

TOKENIZER_EXECUTION_PERFORMED =
NO

CHECKPOINT_DESERIALIZATION_PERFORMED =
NO

TRAINING_PERFORMED =
NO

STATISTICAL_TESTING_PERFORMED =
NO

## 2. Historical source identities

Historical trainer:

scripts/train_controlled_v6b_minimal.py

HISTORICAL_TRAINER_GIT_BLOB =
ae902c06bcef92a20c012b9c35ef0ee8c8478b9f

HISTORICAL_TRAINER_SHA256 =
45ea7785648171ae61232b7f505bdce002195f39081e5d3fb8fa9980a46d3aba

Historical v5 input helper source:

scripts/train_controlled_v5.py

HISTORICAL_V5_GIT_BLOB =
b5fc93a5c2a52baaa14996ce16e656e9d92534de

HISTORICAL_V5_SHA256 =
774ddd10deb8bf494f054044bddff5705b8329c557895e25fba4ceaa5eb18978

Historical v6B model source:

src/contramamba/modeling_v6b_minimal.py

HISTORICAL_MODEL_GIT_BLOB =
e29d361d99579b0a86a626ebcdebd59ba465881c

HISTORICAL_MODEL_SHA256 =
8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5

## 3. Current whole-trainer relation

CURRENT_TRAINER_GIT_BLOB =
3dcc0864b85bde5fb8090c3b7bdbd04de02025e0

CURRENT_TRAINER_SHA256 =
b67a905bff87c3ae730e2e330bddd428dfd6b4d7ae859b7187bf07355eec3d72

CURRENT_WHOLE_TRAINER_BYTE_EQUIVALENCE =
NO

The current full trainer is not accepted as an implicit substitute for the
historical grouped evaluator implementation.

## 4. Historical runtime contract recovered from 18 run provenances

The complete 18-run grouped population agrees on:

MODEL_NAME =
state-spaces/mamba-130m-hf

ARCHITECTURE =
v6b_minimal

TRANSFORMERS_VERSION =
5.0.0

MAX_LENGTH =
128

SPLIT_SEED =
8192

TRAINING_SEEDS =
180,181,182

REASON_ROUTER_MODE =
explicit_product

GRADIENT_OWNERSHIP_MODE =
edge_specific

FREEZE_ENCODER =
true

RESOLVED_EVAL_DEV_BATCH_SIZE =
720

The six grouped arms are the exact previously frozen evaluator population.

## 5. Historical tokenizer construction

The actual grouped main path constructs the tokenizer with:

AutoTokenizer.from_pretrained(args.model_name)

The historical source does not supply a revision argument there.

Therefore:

HISTORICAL_TOKENIZER_REVISION_ARGUMENT =
ABSENT

If tokenizer.pad_token_id is missing, the grouped path requires a valid
eos_token_id and sets:

tokenizer.pad_token =
tokenizer.eos_token

Therefore the historical procedural padding normalization is explicit even
though the exact historical remote snapshot revision was not recorded.

## 6. Historical grouped input helper

The actual grouped train/dev path calls:

v5.encode_mamba_records(
    records,
    tokenizer,
    args.max_length
)

The exact historical encode helper is bound by:

ENCODE_MAMBA_RECORDS_HISTORICAL_SOURCE_SHA256 =
23ef735f6db552aa498204fc3bffa53f8169058b3aff4c790ace93a904d382ea

ENCODE_MAMBA_RECORDS_HISTORICAL_AST_SHA256 =
01f82105a2618e5ce511766e68b0c14d847b6d36deecb3def904f3761039b0e1

CURRENT_ENCODE_MAMBA_RECORDS_AST_EQUIVALENT =
YES

## 7. Exact 128-token encoding contract

For max_length = 128:

CLAIM_TOKEN_BUDGET =
63

SEPARATOR_TOKEN_COUNT =
1

EVIDENCE_TOKEN_BUDGET =
64

Claim encoding:

add_special_tokens =
false

truncation =
true

max_length =
63

Evidence encoding:

add_special_tokens =
false

truncation =
true

max_length =
64

The separator is:

tokenizer.eos_token_id

with tokenizer.pad_token_id fallback only if EOS is absent inside the helper.

Under the grouped main path EOS absence together with missing PAD fails before
normal operation; missing PAD with valid EOS is normalized to EOS.

The final model input sequence is:

truncated_claim_ids
+
one separator id
+
truncated_evidence_ids

then padded to exactly 128 positions.

## 8. Input tensor contract

The historical helper constructs:

input_ids:
torch.long

attention_mask:
torch.bool

claim_mask:
torch.bool

evidence_mask:
torch.bool

For each row:

- attention_mask is true over the consumed sequence;
- claim_mask is true only over claim tokens;
- separator belongs to neither claim_mask nor evidence_mask;
- evidence_mask begins immediately after the separator;
- padding occupies remaining positions.

The Gen4 adapter must reproduce these feature tensors without requiring label
tensors.

## 9. Historical label dependency separation

The historical v5 helper also constructs training label tensors.

Those label tensors are not model features.

Gen4 contains no final_label or auxiliary gold labels.

Therefore the future adapter must reproduce only historical model-feature
construction:

input_ids
attention_mask
claim_mask
evidence_mask

GEN4_LABEL_TENSOR_CONSTRUCTION =
FORBIDDEN

GEN4_GOLD_LABEL_REQUIREMENT =
NO

## 10. Historical model construction

For the grouped v6b_minimal branch, historical main uses:

build_mamba_model(...)

which constructs the historical:

ContraMambaV6BMinimal

from the frozen historical source lineage.

When no backbone is supplied, the historical model implementation uses:

MambaConfig.from_pretrained(model_name)

then:

config.use_mamba_kernels = True

then:

MambaModel.from_pretrained(model_name, config=config)

The backbone source model name is:

state-spaces/mamba-130m-hf

## 11. Model implementation binding decision

HISTORICAL_MODEL_CLASS_AST_SHA256 =
56b2d08d9dfaba1814b52b21617a56cdb5c4521c10a21a60bd8d39e55da1a0a4

CURRENT_MODEL_CLASS_AST_SHA256 =
712cbb65ee07a05e73e9c033d99e82bb271db762f016f9b42ee1f0527eef4242

CURRENT_MODEL_CLASS_AST_EQUIVALENT =
NO

Because the recovery contract requires zero forward-semantic delta and the
current repository cannot be assumed equivalent merely from name continuity:

MODEL_IMPLEMENTATION_BINDING =
HISTORICAL_SOURCE_SNAPSHOT_REQUIRED

The future implementation must bind model construction and forward semantics to:

3e0e9a435068c552abf20f3a74e0c3eccca344a3

It may not treat current HEAD model code as authoritative unless a later
authority explicitly proves the exact required forward subset equivalent.

## 12. Historical checkpoint format

Historical checkpoint loading accepts:

1. a dictionary containing model_state_dict plus optional metadata; or
2. a raw model state dictionary.

The 18 scientific grouped checkpoints were independently SHA-authenticated in
the frozen inventory/statistical specification.

No checkpoint was loaded in this audit.

CHECKPOINT_LOADABILITY =
NOT_YET_ESTABLISHED

## 13. Historical state-dict load contract

For the grouped G3 arms, the historical load helper reaches the strict product
path:

model.load_state_dict(state, strict=True)

Historical helper binding:

P2_LOAD_STATE_DICT_HISTORICAL_AST_SHA256 =
f227a13c73f1ee5f55916a944a6a6b474a50eb0f8aa8310d98ce104c9355f59f

CURRENT_P2_LOAD_STATE_DICT_AST_EQUIVALENT =
YES

The future checkpoint-loadability preflight must still demonstrate strict
compatibility for all 18 checkpoints.

## 14. Historical external class order

The historical source independently fixes:

P2_EXTERNAL_CLASS_ORDER =
REFUTE,NOT_ENTITLED,SUPPORT

and:

REFUTE =
0

NOT_ENTITLED =
1

SUPPORT =
2

Therefore final-logit interpretation is:

final_logits[0] =
refute_logit

final_logits[1] =
ne_logit

final_logits[2] =
support_logit

No class-index inference from observed outcomes is permitted.

## 15. Historical q_authorized serialization

The historical P2 prediction export reads:

q_frame
q_predicate
q_sufficiency
q_authorized

from the model output and serializes:

q_authorized =
fourth authorization mass

The selected primary outcome therefore has an explicit historical export
semantics.

The existing statistical specification remains:

PRIMARY_Y =
q_authorized

## 16. Historical outcome serializer relation to current code

Historical prediction_records_v6b binding:

PREDICTION_RECORDS_V6B_HISTORICAL_AST_SHA256 =
afad027c2432e6a539d9b8533173deeae58fcf771f5049ea05bf71bac917c57b

CURRENT_PREDICTION_RECORDS_V6B_AST_EQUIVALENT =
YES

Historical reason-router export helper binding:

P2_EXPORT_HELPER_HISTORICAL_AST_SHA256 =
d58c54d6e3a01403bb0330efa9ed7aa7a1510a409fe486b720a9d53f502975b7

CURRENT_P2_EXPORT_HELPER_AST_EQUIVALENT =
NO

A current helper that is not AST-equivalent must not silently redefine the
historical scientific output contract.

The dedicated Gen4 adapter should implement only the minimal historical subset
needed for the frozen outcomes.

## 17. Frozen Gen4 required outputs

Per evaluator-row observation the future adapter must expose:

q_authorized

entitlement_prob

refute_logit

ne_logit

support_logit

prediction

and derive only:

support_vs_best_nonsupport_logit_margin =
support_logit - max(ne_logit, refute_logit)

No new scientific outcome is introduced.

## 18. Metadata carriage

Historical training identity fields are not a substitute for Gen4 identities.

The future adapter must carry directly and verbatim:

row_id
source_pair_id
contrast_cell_id

from the canonical Gen4 input to every output row.

Identity must not depend on:

text matching
row order alone
historical pair_id inference
semantic reconstruction

## 19. Tokenizer provenance consequence

The historical procedural encoding contract is now recovered.

However the original 18 runs did not record an immutable tokenizer revision.

Therefore:

HISTORICAL_INPUT_PROCEDURE =
RECOVERED

HISTORICAL_EXACT_TOKENIZER_REVISION =
NOT_RECOVERED

GEN4_TOKENIZER_CONFORMANCE_PREFLIGHT =
STILL_REQUIRED

The previously frozen family A/B active-tokenizer evidence is admissible as the
basis for that preflight, but its previous 3600-row PASS does not itself prove
Gen4 1800-row equivalence.

## 20. R1 contract verdict

HISTORICAL_SOURCE_CONTRACT_AVAILABLE =
YES

HISTORICAL_GROUPED_MAIN_ENCODING_PATH =
RECOVERED

HISTORICAL_128_TOKEN_INPUT_CONTRACT =
RECOVERED

HISTORICAL_EXTERNAL_CLASS_ORDER =
RECOVERED

HISTORICAL_Q_AUTHORIZED_EXPORT_SEMANTICS =
RECOVERED

HISTORICAL_CHECKPOINT_LOAD_CONTRACT =
RECOVERED_STATICALLY

MODEL_IMPLEMENTATION_BINDING =
HISTORICAL_SOURCE_SNAPSHOT_REQUIRED

CURRENT_WHOLE_TRAINER_SUBSTITUTION =
FORBIDDEN

CHECKPOINT_LOADABILITY =
NOT_YET_ESTABLISHED

GEN4_ACTIVE_ENCODING_EQUIVALENCE =
NOT_YET_ESTABLISHED

## 21. Next phase

Under the frozen recovery phase ordering:

PHASE_R1 =
COMPLETE_CANDIDATE_PENDING_FREEZE

The next phase after this report is frozen is:

PHASE_R2 =
TOKENIZER_ACTIVE_ENCODING_GEN4_CONFORMANCE_PREFLIGHT

That preflight is tokenizer-only and CPU-only.

It must not load a model or checkpoint.

It must compare the exact recovered historical input contract over all 1800
canonical Gen4 rows using the already provisioned exact tokenizer-family
contents.

## 22. Execution boundary

TOKENIZER_EXECUTION =
NOT_AUTHORIZED_BY_THIS_REPORT

CHECKPOINT_LOAD =
NOT_AUTHORIZED

MODEL_INSTANTIATION =
NOT_AUTHORIZED

MODEL_FORWARD =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

STATISTICAL_TESTING =
NOT_AUTHORIZED

KAGGLE =
NOT_AUTHORIZED

## 23. Stop condition

Stop after this report candidate is created and validated.

Do not run the R2 tokenizer preflight until this R1 report is frozen.

Do not implement the inference adapter.

Do not load checkpoints.

Do not run model inference.

Do not perform statistical testing.

R1_AUDIT_RESULT =
READY_FOR_FREEZE_REVIEW
