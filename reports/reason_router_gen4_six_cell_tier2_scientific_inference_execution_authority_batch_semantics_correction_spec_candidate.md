# ContraMamba Gen4 R5 Scientific Inference Execution Authority
# Batch/Cache Semantics Correction Specification - Candidate

## 1. Status

STATUS =
CANDIDATE

CORRECTION_TYPE =
BOUNDED_AUTHORITY_CORRECTION

PARENT_R5_AUTHORITY_FREEZE_COMMIT =
6b0ab6e62fc670191f3921d86fae8daf27d43caf

PARENT_R5_AUTHORITY_FILE =
reports/reason_router_gen4_six_cell_tier2_scientific_inference_execution_authority_spec_candidate.md

PARENT_R5_AUTHORITY_SHA256 =
f865fa6307597b562ae18b812219bf8233c4116b53f60ee2ad057a15c217a2aa

PARENT_R4_RESULT_FREEZE_COMMIT =
ffa889d184ad4236689a690384d5268665f5bd87

HISTORICAL_EVALUATOR_SOURCE_COMMIT =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

R5_HARNESS_IMPLEMENTATION =
BLOCKED_UNTIL_THIS_CORRECTION_IS_FROZEN

R5_SYNTHETIC_FORWARD =
NOT_AUTHORIZED

R5_SCIENTIFIC_FORWARD =
NOT_AUTHORIZED

## 2. Defect

The frozen parent R5 authority interpreted:

HISTORICAL_RESOLVED_EVAL_BATCH_SIZE =
720

as if it were the batch size for raw token-to-Mamba scientific forward.

That interpretation is incorrect.

The historical frozen-encoder execution path precomputed Mamba encoder hidden
states before train/dev evaluation.

Therefore historical resolved eval batch size 720 governed downstream/head
forward over already cached encoder states, not raw Mamba encoder execution.

## 3. Historical cache evidence

Historical trainer source executes, when the Mamba encoder is frozen:

if args.backbone == "mamba" and args.freeze_encoder
and segmented dual-pass is inactive:

v5.cache_frozen_encoder_states(model, train_inputs)
v5.cache_frozen_encoder_states(model, dev_inputs)

The historical helper contract is:

def cache_frozen_encoder_states(
    model,
    inputs,
    batch_size=8,
)

It:

1. requires all Mamba parameters to be frozen;
2. calls model.mamba.eval();
3. runs under torch.no_grad();
4. slices input_ids in chunks of 8;
5. evaluates model.mamba(input_ids=ids).last_hidden_state;
6. concatenates the chunks;
7. stores the result as inputs["encoder_hidden_states"].

HISTORICAL_ENCODER_CACHE_BATCH_SIZE =
8

HISTORICAL_ENCODER_CACHE_OUTPUT_KEY =
encoder_hidden_states

## 4. Historical downstream evidence

Historical model_feature_inputs includes encoder_hidden_states whenever that key
is present.

Historical ContraMambaV6BMinimal forward behavior is:

if encoder_hidden_states is None:
    execute self.mamba(input_ids=input_ids)
else:
    use encoder_hidden_states as token_states

Therefore downstream model evaluation after the historical cache operation
bypassed a second Mamba encoder execution.

HISTORICAL_DOWNSTREAM_USES_CACHED_ENCODER_STATES =
YES

HISTORICAL_DOWNSTREAM_REEXECUTES_MAMBA =
NO

## 5. Correct interpretation of batch provenance

Historical exact-18 run provenance remains valid:

HISTORICAL_RESOLVED_EVAL_BATCH_SIZE =
720

Its corrected semantic interpretation is:

HISTORICAL_DOWNSTREAM_EVAL_BATCH_SIZE =
720

It is not evidence for:

RAW_MAMBA_BATCH_SIZE =
720

The parent authority's use of 720 as the raw Mamba execution batch is therefore
superseded.

## 6. Corrected R5 two-stage inference contract

R5 scientific execution must reproduce the historical frozen-encoder path.

Stage A: encoder cache.

For every evaluator:

1. authenticate checkpoint SHA;
2. deserialize with map_location="cpu", weights_only=True;
3. construct the exact historical model shell;
4. strict-load the checkpoint;
5. move the model to cuda:0;
6. call model.eval();
7. keep autocast disabled;
8. keep dtype float32;
9. enter torch.inference_mode();
10. compute Mamba last_hidden_state from canonical input_ids in fixed chunks of 8;
11. concatenate in exact canonical row order;
12. retain the resulting encoder_hidden_states for that evaluator.

R5_ENCODER_CACHE_BATCH_SIZE =
8

R5_ENCODER_ROWS_PER_EVALUATOR =
1800

R5_ENCODER_CACHE_CHUNKS_PER_EVALUATOR =
225

R5_ENCODER_CACHE_ADAPTIVE_BATCHING =
FORBIDDEN

R5_ENCODER_CACHE_OOM_FALLBACK =
FORBIDDEN

Stage B: cached-state downstream forward.

For every evaluator:

1. use the evaluator-specific encoder_hidden_states from Stage A;
2. preserve canonical attention_mask, claim_mask, and evidence_mask;
3. execute downstream historical model forward in fixed batches of 720;
4. ensure encoder_hidden_states is passed to the model;
5. ensure raw Mamba is not called during downstream batches;
6. serialize through the frozen R3 serializer semantics.

R5_DOWNSTREAM_BATCH_SIZE =
720

R5_DOWNSTREAM_BATCH_PARTITION =
720,720,360

R5_DOWNSTREAM_FORWARD_CALLS_PER_EVALUATOR =
3

R5_DOWNSTREAM_ADAPTIVE_BATCHING =
FORBIDDEN

R5_DOWNSTREAM_OOM_FALLBACK =
FORBIDDEN

## 7. Complete scientific forward counts

For 18 evaluators:

EXPECTED_SCIENTIFIC_MAMBA_CACHE_FORWARD_CALLS =
4050

Derivation:

225 cache chunks per evaluator
times 18 evaluators
equals 4050.

EXPECTED_SCIENTIFIC_DOWNSTREAM_FORWARD_CALLS =
54

Derivation:

3 downstream chunks per evaluator
times 18 evaluators
equals 54.

These counts exclude the non-scientific synthetic preflight.

## 8. Encoder-state reuse boundary

Encoded token/mask tensors may be reused across evaluators.

ENCODED_TOKEN_INPUT_REUSE_ACROSS_EVALUATORS =
AUTHORIZED

Cached encoder_hidden_states may not be reused across evaluator checkpoints
unless a separate authority first establishes exact encoder-state equivalence.

ENCODER_HIDDEN_STATE_REUSE_ACROSS_EVALUATORS =
FORBIDDEN

Each evaluator must compute its own cache from its own authenticated,
strict-loaded checkpoint.

## 9. R3 adapter interaction correction

The frozen R3 adapter historical_forward helper currently passes:

input_ids
attention_mask
claim_mask
evidence_mask
decision_mode
gradient_ownership_mode
edge_gradient_lambdas
return_q_diagnostics

It does not pass encoder_hidden_states.

Therefore that helper alone cannot reproduce the recovered historical
frozen-encoder evaluation path.

PARENT_AUTHORITY_REQUIREMENT =
USE_FROZEN_ADAPTER_HISTORICAL_FORWARD_INTERFACE

CORRECTED_REQUIREMENT =
DO_NOT_USE_ADAPTER_HISTORICAL_FORWARD_FOR_CACHED_DOWNSTREAM_EXECUTION

The frozen R3 adapter itself remains immutable.

R3_ADAPTER_MODIFICATION =
FORBIDDEN

The R5 execution harness is authorized to implement a thin cached-state
downstream call that reproduces the same frozen forward argument contract while
additionally passing encoder_hidden_states.

The harness must obtain:

decision_mode
gradient_ownership_mode
edge_gradient_lambdas
return_q_diagnostics

from the frozen R3 adapter constants/helpers rather than duplicating scientific
semantics independently.

## 10. Corrected cached downstream call

The harness downstream call must be semantically equivalent to:

model(
    input_ids=batch_input_ids,
    attention_mask=batch_attention_mask,
    claim_mask=batch_claim_mask,
    evidence_mask=batch_evidence_mask,
    encoder_hidden_states=batch_encoder_hidden_states,
    decision_mode=adapter.DECISION_MODE,
    gradient_ownership_mode=adapter.GRADIENT_OWNERSHIP_MODE,
    edge_gradient_lambdas=adapter.expected_edge_gradient_lambdas(arm),
    return_q_diagnostics=True,
)

The inclusion of input_ids alongside encoder_hidden_states is retained only for
the historical model's shape-consistency check.

The model must not invoke self.mamba during this downstream call.

## 11. Synthetic GPU preflight correction

The target-runtime synthetic preflight must exercise the corrected two-stage
path.

For the frozen synthetic checkpoint:

1. create only synthetic/non-Gen4 token/mask tensors;
2. execute Mamba encoder cache in batch size 8;
3. execute cached-state downstream forward;
4. repeat the complete two-stage process twice;
5. require deterministic finite q_authorized, entitlement_prob, and logits;
6. verify that downstream forward does not trigger a second Mamba execution.

SYNTHETIC_PREFLIGHT_SCIENTIFIC_EVIDENCE =
NO

CANONICAL_GEN4_FORWARD_BEFORE_SYNTHETIC_PASS =
FORBIDDEN

## 12. Harness scope

The parent authority's exact-two-file implementation scope remains unchanged:

AUTHORIZED_NEW_FILE =
scripts/reason_router_gen4_six_cell_tier2_scientific_inference.py

AUTHORIZED_NEW_TEST_FILE =
tests/test_reason_router_gen4_six_cell_tier2_scientific_inference.py

No existing R3 adapter, historical snapshot, checkpoint, or canonical Gen4
artifact may be changed.

## 13. Required static harness tests added by this correction

The future harness test file must prove statically or with synthetic mock models:

1. encoder cache batch size is exactly 8;
2. 1800 rows imply exactly 225 encoder chunks;
3. downstream batch size is exactly 720;
4. downstream partition is exactly 720,720,360;
5. encoder_hidden_states is passed downstream;
6. cached downstream path does not call Mamba;
7. adapter historical_forward is not used for the cached downstream operation;
8. edge-gradient lambdas come from the frozen adapter helper;
9. no adaptive/OOM batch fallback exists;
10. evaluator-specific encoder cache is not reused across evaluators;
11. expected scientific Mamba cache forward count is 4050;
12. expected scientific downstream forward count is 54.

These tests must not instantiate the real model, deserialize a checkpoint, or
perform scientific inference.

## 14. Unchanged R5 contracts

All parent R5 authority provisions remain unchanged except where explicitly
superseded by this correction.

Unchanged contracts include:

canonical Gen4 artifact identity
exact 18 evaluator population
checkpoint SHA authentication
weights_only=True deserialization
strict state-dict loading
historical runtime identities
Tesla T4 requirement
single-process single-device execution
float32
fp16=false
bf16=false
autocast disabled
model.eval()
torch.inference_mode()
tokenizer identities
R2 encoding contract
scientific output semantics
32400-row complete matrix
serialization order
numeric no-rounding rule
JSONL UTF-8 LF byte contract
no statistical testing in R5
no evaluator dropping
no row dropping
no checkpoint replacement
no mamba-ssm installation
no training
no backward

## 15. Authority precedence

For any conflict between the parent R5 authority and this correction:

THIS_CORRECTION =
CONTROLLING_AUTHORITY

Specifically superseded parent provisions are:

R5_BATCH_SIZE=720 as a raw Mamba batch
EXPECTED_BATCH_PARTITION_PER_EVALUATOR=720,720,360 as the complete raw model path
the requirement to use adapter.historical_forward for cached downstream
execution.

Correct replacements are:

R5_ENCODER_CACHE_BATCH_SIZE =
8

R5_DOWNSTREAM_BATCH_SIZE =
720

R5_DOWNSTREAM_BATCH_PARTITION =
720,720,360

CACHED_DOWNSTREAM_FORWARD =
HARNESS_THIN_ORCHESTRATION_USING_FROZEN_ADAPTER_SEMANTICS

## 16. Authorization state

Before this correction is reviewed, committed, and pushed:

R5_HARNESS_IMPLEMENTATION =
BLOCKED

R5_SYNTHETIC_GPU_PREFLIGHT =
NOT_AUTHORIZED

R5_SCIENTIFIC_FORWARD =
NOT_AUTHORIZED

After this exact correction is frozen:

R5_HARNESS_IMPLEMENTATION =
AUTHORIZED_EXACT_TWO_NEW_FILES_ONLY

R5_SYNTHETIC_GPU_PREFLIGHT =
NOT_AUTHORIZED_UNTIL_HARNESS_VALIDATED_AND_FROZEN

R5_SCIENTIFIC_FORWARD =
NOT_AUTHORIZED_UNTIL_HARNESS_VALIDATED_AND_FROZEN_AND_SYNTHETIC_PREFLIGHT_PASSES

## 17. Result

R5_BATCH_CACHE_SEMANTICS_CORRECTION_RESULT =
READY_FOR_FREEZE_REVIEW
