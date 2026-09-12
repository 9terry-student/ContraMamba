# ContraMamba Gen4 Six-Cell Tier-2 Scientific Inference Execution Authority Specification - Candidate

## 1. Status

STATUS =
CANDIDATE

PHASE =
R5_SCIENTIFIC_INFERENCE_EXECUTION_AUTHORITY

PARENT_R4_RESULT_FREEZE_COMMIT =
ffa889d184ad4236689a690384d5268665f5bd87

R4_CHECKPOINT_LOADABILITY =
PASS_18_OF_18

R4_ARTIFACT_PROVENANCE =
PASS

R3_IMPLEMENTATION_COMMIT =
d62a424c8d13e582ffde7e8d2f8b7e0f43b610b2

R3_RESULT_FREEZE_COMMIT =
64cc172ae1446e8ba9200d2365a08aad6f5d87ee

HISTORICAL_EVALUATOR_SOURCE_COMMIT =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

This specification governs the first Gen4 scientific model forward.

No scientific forward is authorized before this authority and the required
execution harness are independently frozen.

## 2. Scientific object

CANONICAL_GEN4_ARTIFACT =
reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

CANONICAL_GEN4_ARTIFACT_SHA256 =
b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

CANONICAL_GEN4_ARTIFACT_BYTES =
1465573

GEN4_INPUT_ROWS =
1800

GEN4_SOURCE_PAIRS =
300

GEN4_CONTRAST_CELLS =
6

EVALUATOR_COUNT =
18

EXPECTED_SCIENTIFIC_OUTPUT_ROWS =
32400

PRIMARY_OUTCOME =
q_authorized

## 3. Frozen statistical interpretation

The R5 execution produces evaluator-level observations only.

R5 must not compute:

p-values
confidence intervals
effect sizes
Holm correction
scientific significance decisions
contrast-level scientific conclusions

Those belong only to a later separately authorized R6 statistical phase.

## 4. Evaluator population

SEEDS =
180,181,182

ARM_ORDER =
G3-GROUP-D-HALF
G3-GROUP-Q-D-HALF
G3-GROUP-Q-HALF
G3-GROUP-U-D-HALF
G3-GROUP-U-HALF
G3-GROUP-U-Q-HALF

EVALUATOR_ORDER =
seed-major then arm-order above

CHECKPOINT_REGISTRY_SOURCE =
scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py

CHECKPOINT_REGISTRY_COUNT =
18

CHECKPOINT_RESELECTION =
FORBIDDEN

CHECKPOINT_REPLACEMENT =
FORBIDDEN

FAILED_EVALUATOR_DROPPING =
FORBIDDEN

PARTIAL_MATRIX_PROMOTION =
FORBIDDEN

## 5. Checkpoint authentication

Each checkpoint must be authenticated against the frozen exact-18 SHA registry
before deserialization.

CHECKPOINT_DESERIALIZATION =
torch.load(map_location="cpu",weights_only=True)

STRICT_STATE_DICT_LOAD =
REQUIRED

STRICT_FALSE =
FORBIDDEN

STATE_DICT_KEY_REWRITE =
FORBIDDEN

STATE_DICT_FILTER =
FORBIDDEN

R4_PASS may be reused as supporting evidence, but R5 must independently
authenticate the checkpoint SHA used for each scientific evaluator execution.

## 6. Historical model construction

HISTORICAL_MODEL_SNAPSHOT =
src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py

HISTORICAL_MODEL_SNAPSHOT_SHA256 =
8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5

DEDICATED_INFERENCE_ADAPTER =
scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py

DEDICATED_INFERENCE_ADAPTER_SHA256 =
83177c351f82a781586c63bd8d4ef1b40e759b5a94858dc1837d65502cbff6e5

MODEL_NAME =
state-spaces/mamba-130m-hf

MODEL_CONFIG_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

MODEL_CONFIG_SHA256 =
784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a

MUTABLE_FROM_PRETRAINED_RESOLUTION =
FORBIDDEN

The exact local config must be authenticated before model construction.

## 7. Tokenizer contract

CANONICAL_TOKENIZER_FAMILY =
A

TOKENIZER_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

TOKENIZER_CONFIG_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

SPECIAL_TOKENS_MAP_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

MAX_LENGTH =
128

CLAIM_BUDGET =
63

SEPARATOR_TOKENS =
1

EVIDENCE_BUDGET =
64

EOS_TOKEN_ID =
0

PAD_TOKEN_ID =
0

ADD_SPECIAL_TOKENS =
false

GEN4_ACTIVE_ENCODING_EQUIVALENCE =
PASS_FROM_R2

HISTORICAL_WRAPPER_RUNTIME_EQUIVALENCE =
NOT_CLAIMED

R5 binds directly to the already validated active token-ID and mask contract,
not to mutable AutoTokenizer wrapper behavior.

## 8. Historical runtime provenance

The exact 18 historical evaluator runs were homogeneous on:

HISTORICAL_PYTHON =
3.12.13

HISTORICAL_TORCH =
2.10.0+cu128

HISTORICAL_TRANSFORMERS =
5.0.0

HISTORICAL_CUDA_RUNTIME =
12.8

HISTORICAL_DEVICE =
cuda

HISTORICAL_GPU_MODEL =
Tesla T4

HISTORICAL_VISIBLE_GPU_COUNT =
2

HISTORICAL_MAMBA_SSM_VERSION =
null

HISTORICAL_FP16 =
false

HISTORICAL_AUTOCAST =
disabled

HISTORICAL_RESOLVED_EVAL_BATCH_SIZE =
720

HISTORICAL_PARALLEL_MODEL_WRAPPER =
NONE

## 9. Authorized R5 execution runtime

R5 scientific execution must use:

DEVICE =
cuda:0

GPU_MODEL =
Tesla T4

PROCESS_COUNT =
1

MODEL_DEVICE_COUNT =
1

DATAPARALLEL =
FORBIDDEN

DISTRIBUTED_DATAPARALLEL =
FORBIDDEN

DTYPE_POLICY =
FLOAT32

FP16 =
false

BF16 =
false

AUTOCAST =
disabled

MODEL_EVAL =
required

GRADIENT_TRACKING =
disabled

TORCH_INFERENCE_MODE =
required

TRAINING =
forbidden

BACKWARD =
forbidden

OPTIMIZER =
forbidden

PARAMETER_MUTATION =
forbidden

## 10. Runtime identity gate

Before any scientific Gen4 forward, the execution must record and require:

Python 3.12.13
Torch 2.10.0+cu128
Transformers 5.0.0
CUDA runtime reported by PyTorch 12.8
CUDA available
GPU 0 name Tesla T4
mamba_ssm unavailable / null

Any mismatch blocks scientific forward unless separately reviewed under a new
runtime-conformance authority.

No automatic package-version substitution is permitted.

## 11. Effective Mamba backend

The historical evaluator provenance records:

mamba_ssm_version =
null

Therefore R5 must not install mamba-ssm as an optimization.

MAMBA_SSM_INSTALLATION =
FORBIDDEN

FAST_PATH_ENABLEMENT_BY_NEW_DEPENDENCY =
FORBIDDEN

CONFIG_USE_MAMBA_KERNELS =
true

The effective Transformers backend must be recorded before scientific forward.

If the required historical runtime unexpectedly exposes an accelerated Mamba
path not supported by the historical provenance, R5 stops before Gen4 forward.

## 12. Scientific batch contract

R5_BATCH_SIZE =
720

R5_BATCH_SIZE_SOURCE =
HISTORICAL_RESOLVED_EVAL_BATCH_SIZE

GEN4_ROWS_PER_EVALUATOR =
1800

EXPECTED_BATCH_PARTITION_PER_EVALUATOR =
720,720,360

OUTCOME_DEPENDENT_BATCHING =
FORBIDDEN

ADAPTIVE_BATCH_SIZE =
FORBIDDEN

OOM_BATCH_SIZE_FALLBACK =
FORBIDDEN

If batch size 720 cannot execute under the frozen runtime, stop. Do not silently
reduce the batch size.

## 13. Pre-scientific synthetic forward gate

Before any canonical Gen4 row is forwarded, R5 must perform a non-scientific
synthetic forward gate.

The gate must:

1. use the frozen historical model construction;
2. use one authenticated frozen checkpoint;
3. use only synthetic/non-Gen4 feature tensors;
4. run in model.eval() and torch.inference_mode();
5. use float32 with autocast disabled;
6. run the identical synthetic batch twice;
7. require finite q_authorized, entitlement_prob, and logits;
8. require identical output tensor shapes;
9. require deterministic repeated outputs under the same runtime;
10. perform no statistical analysis.

SYNTHETIC_GATE_CHECKPOINT =
seed180/G3-GROUP-D-HALF

SYNTHETIC_GATE_SCIENTIFIC_EVIDENCE =
NO

If this gate fails, canonical Gen4 forward is forbidden.

## 14. Canonical Gen4 input execution

Only after the synthetic gate passes:

1. authenticate canonical Gen4 artifact SHA256 and byte count;
2. authenticate canonical tokenizer files;
3. encode all 1800 rows using the frozen R2 contract;
4. require exact canonical row order and identities;
5. require the known R2 aggregate encoded-coordinate identity where applicable;
6. reuse the immutable encoded input tensors across all 18 evaluators;
7. perform evaluator execution sequentially.

No label or gold outcome is passed to the model.

## 15. Scientific forward

For each evaluator:

1. authenticate checkpoint SHA;
2. deserialize CPU weights_only=True;
3. construct exact model shell;
4. strict-load checkpoint;
5. move model to cuda:0;
6. call model.eval();
7. enter torch.inference_mode();
8. execute exactly three batches: 720,720,360;
9. use the frozen adapter historical_forward interface;
10. serialize model outputs immediately;
11. free evaluator model and checkpoint state before next evaluator.

No training or mutation is permitted.

## 16. Output semantic contract

The frozen serializer semantics are:

q_authorized =
float(q_authorized_tensor.item())

entitlement_prob =
float(entitlement_prob_tensor.item())

REFUTE_LOGIT_INDEX =
0

NOT_ENTITLED_LOGIT_INDEX =
1

SUPPORT_LOGIT_INDEX =
2

prediction =
argmax(final_logits)

support_vs_best_nonsupport_logit_margin =
support_logit - max(refute_logit, ne_logit)

No recalibration, thresholding, clipping, rounding, temperature scaling, or
post-hoc transformation is permitted.

## 17. Scientific row schema

Every output row must contain at minimum:

schema_version
structural_artifact_commit
statistical_specification_commit
recovery_implementation_commit
r4_result_freeze_commit
r5_execution_authority_commit
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
refute_logit
ne_logit
support_logit
support_vs_best_nonsupport_logit_margin
prediction

Additional provenance fields may be included if deterministic.

No p-value or contrast statistic may be included.

## 18. Serialization order

The output artifact order is frozen as:

1. seed order 180,181,182;
2. within seed, ARM_ORDER from Section 4;
3. within evaluator, exact canonical Gen4 JSONL row order.

No sorting by prediction or numeric model outcome is permitted.

## 19. Numeric serialization representation

Numeric model outputs are converted from tensors exactly through Python scalar
conversion using float(tensor.item()).

JSON numeric values remain JSON numbers.

SCIENTIFIC_FLOAT_ROUNDING =
FORBIDDEN

SCIENTIFIC_FLOAT_STRING_FORMATTING =
FORBIDDEN

NAN =
FORBIDDEN

POSITIVE_INFINITY =
FORBIDDEN

NEGATIVE_INFINITY =
FORBIDDEN

All scientific numeric outputs must be finite.

## 20. JSONL byte serialization

Each scientific row must be serialized with:

ensure_ascii=False
sort_keys=True
separators=(",",":")

Each row is followed by exactly one LF byte.

UTF8_BOM =
FORBIDDEN

OUTPUT_FORMAT =
JSONL_UTF8_LF

## 21. Complete-matrix contract

UNIQUE_KEY =
(evaluator_seed,evaluator_arm,row_id)

REQUIRED_UNIQUE_KEY_COUNT =
32400

REQUIRED_ROW_COUNT =
32400

EACH_ROW_ID_OCCURRENCE_COUNT =
18

EACH_SOURCE_PAIR_ID_OCCURRENCE_COUNT =
108

MISSING_Q_AUTHORIZED =
FORBIDDEN

DUPLICATE_KEYS =
FORBIDDEN

IMPUTATION =
FORBIDDEN

The frozen adapter complete-matrix validator must pass before the R5 artifact is
eligible for result validation.

## 22. Execution outputs

R5 must produce:

1. one deterministic 32400-row scientific evaluator JSONL;
2. one deterministic execution summary JSON;
3. no statistical-analysis artifact.

The execution summary must record:

authority commit
HEAD commit
runtime identities
GPU identities
effective Mamba backend
batch size
canonical input SHA
tokenizer identities
18 checkpoint SHAs
scientific row count
matrix validation result
output JSONL SHA256
output JSONL byte count
model-forward count
training/backward flags

## 23. Scientific execution harness

A thin execution-only harness is required.

AUTHORIZED_NEW_FILE =
scripts/reason_router_gen4_six_cell_tier2_scientific_inference.py

AUTHORIZED_NEW_TEST_FILE =
tests/test_reason_router_gen4_six_cell_tier2_scientific_inference.py

The harness may only orchestrate frozen R2/R3/R4 primitives and the exact
contracts in this authority.

It may not modify:

scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py
src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py
existing checkpoint files
canonical Gen4 artifact

## 24. Harness validation before scientific execution

Before any scientific forward:

HARNESS_STATIC_TESTS =
REQUIRED

SYNTHETIC_FORWARD_PREFLIGHT =
REQUIRED_ON_TARGET_GPU_RUNTIME

The harness must be independently reviewed and frozen at a specific commit.

SCIENTIFIC_FORWARD_BEFORE_HARNESS_FREEZE =
FORBIDDEN

## 25. Kaggle policy

R5 scientific execution requires GPU execution.

KAGGLE =
AUTHORIZED_ONLY_AFTER_R5_AUTHORITY_AND_HARNESS_FREEZE

GPU =
T4

The run must be bound to the exact pushed harness commit.

No run from another commit may be reused.

CPU scientific execution is not the primary authorized R5 path.

## 26. Failure policy

R5 fails closed on:

runtime-version mismatch
GPU-model mismatch
unexpected Mamba backend
synthetic determinism failure
canonical artifact mismatch
tokenizer mismatch
checkpoint SHA mismatch
checkpoint strict-load failure
OOM at batch size 720
non-finite output
missing output field
duplicate evaluator-row key
incomplete matrix
serialization-contract failure

Forbidden recovery includes:

changing batch size
enabling fp16
enabling bf16
installing mamba-ssm
dropping evaluators
dropping rows
changing checkpoint
strict=False
changing tokenizer
changing model architecture
changing scientific outcome semantics

Any such correction requires new authority.

## 27. R5 interpretation boundary

Successful R5 establishes:

EXECUTION_SUCCESS =
SCIENTIFIC_INFERENCE_MATRIX_PRODUCED

ARTIFACT_PROVENANCE_VALIDITY =
PENDING_POST_EXECUTION_VALIDATION

SCIENTIFIC_STATISTICAL_CONCLUSION =
NOT_ESTABLISHED

R5 success does not itself establish any six-cell contrast.

## 28. R6 boundary

STATISTICAL_TESTING =
NOT_AUTHORIZED_BY_R5

R6 may begin only after the R5 scientific outcome artifact and execution
summary are imported and independently validated.

## 29. Authorization state

Before this candidate is frozen:

R5_HARNESS_IMPLEMENTATION =
NOT_AUTHORIZED

R5_SCIENTIFIC_FORWARD =
NOT_AUTHORIZED

After this exact authority is reviewed, committed, and pushed:

R5_HARNESS_IMPLEMENTATION =
AUTHORIZED_EXACT_TWO_NEW_FILES_ONLY

R5_SCIENTIFIC_FORWARD =
NOT_AUTHORIZED_UNTIL_HARNESS_VALIDATED_AND_FROZEN

After the exact harness is validated and frozen:

R5_SYNTHETIC_GPU_PREFLIGHT =
AUTHORIZED

R5_CANONICAL_GEN4_FORWARD =
AUTHORIZED_ONLY_IF_SYNTHETIC_PREFLIGHT_PASSES

## 30. Next object

NEXT_OBJECT_AFTER_R5_AUTHORITY_FREEZE =
R5_SCIENTIFIC_INFERENCE_HARNESS_IMPLEMENTATION_AND_STATIC_VALIDATION

## 31. Result

R5_EXECUTION_AUTHORITY_RESULT =
READY_FOR_FREEZE_REVIEW
