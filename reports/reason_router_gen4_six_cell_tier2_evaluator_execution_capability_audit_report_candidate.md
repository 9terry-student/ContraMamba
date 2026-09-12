# ContraMamba Gen4 Six-Cell Tier-2 Evaluator Execution Capability Audit Report - Candidate

## 1. Status

VERDICT =
BLOCKED_RECOVERY_REQUIRED

PHASE =
GEN4_SIX_CELL_TIER2_EVALUATOR_EXECUTION_CAPABILITY_AUDIT

STATISTICAL_SPECIFICATION_AUTHORITY =
4dc5bacd10a254b5ecd339ac1fe78bad9def5c47

PRIMARY_Y =
q_authorized

EVALUATOR_POPULATION =
FULL_PRESPECIFIED_GEN3_GROUPED_18_RUN_MATRIX

This report is a read-only capability audit.

It does not authorize checkpoint loading, tokenizer execution, model inference,
statistical testing, training, or Kaggle execution.

## 2. Frozen trainer identity

Trainer:

scripts/train_controlled_v6b_minimal.py

Git blob:

3dcc0864b85bde5fb8090c3b7bdbd04de02025e0

SHA256:

b67a905bff87c3ae730e2e330bddd428dfd6b4d7ae859b7187bf07355eec3d72

Bytes:

1335154

The trainer was inspected statically only.

## 3. Frozen Gen4 input identity

Canonical six-cell artifact SHA256:

b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

Bytes:

1465573

The canonical schema contains:

row_id
source_pair_id
contrast_cell_id
claim
evidence

The canonical schema does not contain:

final_label
label

Therefore the Gen4 scientific input is structurally unlabeled for the selected
primary q_authorized analysis.

## 4. Existing q_authorized capability

Static inspection found q_authorized in the frozen trainer.

The existing serialization logic contains a q_authorized extraction path.

Therefore:

TRAINER_EMITS_Q_AUTHORIZED_STATIC_SIGNAL =
YES

This establishes that the historical model family can expose the selected
primary scientific quantity in at least one existing code path.

It does not establish Gen4 execution compatibility.

## 5. Gen4 identity preservation capability

The frozen trainer contains many row_id references and limited source_pair_id
references.

However:

contrast_cell_id occurrences in the frozen trainer =
0

The tracked repository contains no script that simultaneously contains:

q_authorized
row_id
source_pair_id
contrast_cell_id

Therefore:

TRAINER_HAS_ALL_GEN4_IDENTITY_TERMS =
NO

TRACKED_SCRIPT_WITH_Q_AUTHORIZED_AND_ALL_GEN4_ID_FIELDS_COUNT =
0

EXISTING_FULL_STATIC_CAPABILITY_SCRIPT =
NO

The future scientific evaluator must preserve directly:

row_id
source_pair_id
contrast_cell_id

without text reconstruction or row-order inference.

Therefore the existing execution path is insufficient.

## 6. Explicit inference-only execution surface

Static CLI audit found:

EVAL_ONLY_FLAG_COUNT =
0

Checkpoint/resume-related CLI surfaces do exist.

Therefore:

EXPLICIT_EVAL_ONLY_CLI_STATIC_SIGNAL =
NO

CHECKPOINT_OR_RESUME_CLI_STATIC_SIGNAL =
YES

Existing checkpoint-loading capability alone does not authorize or establish a
label-free Gen4 inference-only execution path.

## 7. Label dependency boundary

The frozen six-cell artifact has no final_label and no label.

The historical trainer contains multiple static accesses to:

final_label
label

Therefore:

GEN4_INPUT_HAS_FINAL_LABEL =
NO

GEN4_INPUT_HAS_LABEL =
NO

HISTORICAL_TRAINER_HAS_LABEL_DEPENDENCY_SIGNALS =
YES

This audit does not claim that every trainer path requires labels.

It establishes that no already-frozen explicit Gen4 label-free inference-only
path has been demonstrated.

## 8. Existing checkpoint loading signals

Static inspection found existing:

torch.load
load_state_dict
checkpoint/resume CLI

capabilities.

Therefore:

CHECKPOINT_LOADING_PRIMITIVES_EXIST =
YES

However checkpoint loading was not executed.

CHECKPOINT_LOADABILITY =
NOT_ESTABLISHED_BY_THIS_AUDIT

No recovered checkpoint was deserialized.

## 9. Tokenizer provenance result

The 18 real Gen3 grouped run provenance files were inspected statically.

All 18 expose historical runtime/model information including:

transformers_version =
5.0.0

model_name =
state-spaces/mamba-130m-hf

However the audit found no tokenizer-specific immutable identity field such as:

tokenizer SHA256
tokenizer blob identity
tokenizer revision
tokenizer commit

Therefore:

EXACT_TOKENIZER_HASH_OR_REVISION_FIELD_COUNT =
0

EXACT_TOKENIZER_IDENTITY_IN_18_RUN_PROVENANCE =
NO

The model name alone is not sufficient to freeze byte-exact tokenizer
provenance for a new scientific execution.

## 10. Existing tokenizer code

The historical trainer uses:

AutoTokenizer.from_pretrained(args.model_name)

in existing code paths.

That behavior does not itself freeze an immutable tokenizer revision.

Therefore:

MODEL_NAME_ONLY_TOKENIZER_RESOLUTION =
INSUFFICIENT_FOR_GEN4_SCIENTIFIC_PROVENANCE

TOKENIZER_EXECUTION =
BLOCKED_PENDING_IDENTITY_RECOVERY

No tokenizer was imported or executed by this audit.

## 11. Repository-wide capability result

Repository-wide tracked Python inspection found no existing script satisfying
the complete future Gen4 evaluator contract.

Required combined capability:

- consume canonical six-cell artifact;
- preserve row_id;
- preserve source_pair_id;
- preserve contrast_cell_id;
- load one exact frozen checkpoint;
- expose q_authorized;
- expose required secondary diagnostics;
- operate without new training;
- operate without requiring Gen4 gold labels;
- serialize deterministic provenance-bound output;
- fail closed on checkpoint/tokenizer mismatch.

Observed:

EXISTING_COMPLETE_GEN4_TIER2_EVALUATOR_PATH =
NO

## 12. Existing capability classification

The current repository contains reusable primitives:

- historical model implementation;
- q_authorized extraction;
- checkpoint loading primitives;
- exact 18-checkpoint SHA identities;
- canonical Gen4 structural input;
- frozen statistical outcome contract.

But it does not contain one frozen path connecting these primitives under the
required scientific provenance and identity contract.

Therefore:

GEN4_TIER2_EVALUATOR_EXECUTION_CAPABILITY =
INCOMPLETE

DIRECT_USE_OF_EXISTING_TRAINER_FOR_GEN4 =
NOT_AUTHORIZED

## 13. Required recovery scope

The minimum recovery must remain inference-only.

It must not introduce:

new training
fine-tuning
checkpoint selection
evaluator weighting
new scientific outcome selection
new intervention cells
new gold labels

The minimum recovery must provide:

1. deterministic canonical six-cell JSONL input loading;
2. direct preservation of row_id;
3. direct preservation of source_pair_id;
4. direct preservation of contrast_cell_id;
5. exact checkpoint seed/arm/SHA binding;
6. exact tokenizer provenance recovery and freeze;
7. label-free forward-only evaluation;
8. q_authorized extraction;
9. entitlement_prob extraction;
10. support/ne/refute logit extraction;
11. derived support_vs_best_nonsupport_logit_margin;
12. deterministic output serialization;
13. fail-closed cardinality and composite-key checks;
14. no model parameter updates.

## 14. Scientific non-delta requirement

The recovery path must not alter model semantics.

For a fixed checkpoint and tokenized input:

MODEL_FORWARD_SEMANTICS_DELTA =
ZERO_REQUIRED

The recovery implementation may adapt:

input plumbing
metadata carriage
checkpoint provisioning
output serialization
provenance checking

It may not alter:

architecture
weights
router equations
reason masses
final logits
q_authorized definition
entitlement definition
statistical estimands

## 15. Tokenizer recovery requirement

Before implementation execution can be authorized, an immutable tokenizer
identity must be recovered or provisioned.

Acceptable evidence must bind the tokenizer to an immutable artifact identity
such as:

exact revision/commit
exact local snapshot identity
or byte-hashed tokenizer package/file set under a frozen provenance contract

Resolving only the mutable model name:

state-spaces/mamba-130m-hf

is insufficient.

Therefore:

TOKENIZER_PROVENANCE_RECOVERY =
REQUIRED

## 16. Future execution artifact cardinality

The future outcome artifact remains required to contain:

1800 structural rows
x
18 evaluators

for:

EXPECTED_EVALUATOR_ROW_COUNT =
32400

Required composite uniqueness:

(evaluator_seed, evaluator_arm, row_id)

Each row_id must occur exactly 18 times.

Each source_pair_id must contribute exactly 108 evaluator-row observations.

These cardinalities are unchanged by the capability recovery.

## 17. Current capability decision

TRAINER_EMITS_Q_AUTHORIZED_STATIC_SIGNAL =
YES

TRAINER_HAS_ALL_GEN4_IDENTITY_TERMS =
NO

EXISTING_FULL_STATIC_CAPABILITY_SCRIPT =
NO

EXPLICIT_EVAL_ONLY_CLI_STATIC_SIGNAL =
NO

CHECKPOINT_OR_RESUME_CLI_STATIC_SIGNAL =
YES

EXACT_TOKENIZER_IDENTITY_IN_18_RUN_PROVENANCE =
NO

GEN4_TIER2_EVALUATOR_EXECUTION_CAPABILITY =
INCOMPLETE

MINIMAL_INFERENCE_ONLY_RECOVERY =
REQUIRED

## 18. Current execution boundary

CHECKPOINT_LOAD =
NOT_AUTHORIZED

MODEL_INSTANTIATION =
NOT_AUTHORIZED

MODEL_INFERENCE =
NOT_AUTHORIZED

TOKENIZER_EXECUTION =
NOT_AUTHORIZED

STATISTICAL_TESTING =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

KAGGLE_EXECUTION =
NOT_AUTHORIZED

SCIENTIFIC_OUTCOME_CONCLUSION =
NOT_ESTABLISHED

## 19. Next required object

The next authority object is:

GEN4_SIX_CELL_TIER2_EVALUATOR_EXECUTION_RECOVERY_SPECIFICATION

That specification must prospectively define the minimum allowed delta for:

- tokenizer provenance recovery;
- inference-only adapter/interface;
- canonical Gen4 metadata preservation;
- exact checkpoint authentication;
- label-free forward execution;
- deterministic q_authorized output serialization;
- validation and stop conditions.

It must not authorize execution.

## 20. Stop condition

Stop after this capability audit report candidate is created and reviewed.

Do not modify model code.

Do not load checkpoints.

Do not run tokenizer code.

Do not run inference.

Do not perform statistical testing.

Do not train.

Do not use Kaggle.

CAPABILITY_AUDIT_RESULT =
BLOCKED_RECOVERY_REQUIRED
