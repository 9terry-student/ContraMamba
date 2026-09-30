# ContraMamba Gen5 Phase 2 Full-Batch Feasibility Recovery Amendment

## 0. Status

This document is a **minimal execution-semantics amendment** to the already-frozen
Gen5 Phase 2 training execution authority.

Parent execution authority commit:

`5ab3174cc24731f867e512c9381b9abcc3263915`

Current recovery implementation head at the triggering CUDA preflight:

`ae8fd8cae718d2f965c592e345f0ee07f325822b`

This amendment does **not** change the scientific question, model intervention,
training objective, data, split, seeds, arms, optimizer, checkpoint selection,
fresh-assay firewall, or confirmatory analysis.

It exists only because the parent authority explicitly requires an amendment when
the exact 2880-row monolithic CUDA forward does not fit the qualified Tesla T4
runtime.

## 1. Triggering evidence

The corrected Phase 2 CUDA preflight successfully passed the prior
Transformers-kernel binding blocker and reached the qualified Mamba CUDA fast path.

The exact 2880-row monolithic forward then stopped as:

`GEN5_PHASE2_FULL_BATCH_EXECUTION_FEASIBILITY_BLOCKED`

Observed CUDA OOM facts:

- GPU: Tesla T4
- total capacity: 14.56 GiB
- PyTorch allocated at failure: 13.15 GiB
- attempted additional allocation: 2.11 GiB
- failure location: qualified `mamba_inner_fn` fast path
- backward executed: false
- optimizer step executed: false
- scientific training executed: false
- fresh XG1 loaded: false
- scientific p-values: 0

The failed run does not need to be imported or committed for this amendment.
Its role is only to establish the runtime feasibility blocker already observed
during the authorized preflight.

## 2. Immutable scientific contract

All of the following remain frozen and unchanged:

- parent checkpoint SHA256:
  `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`
- R22 SHA256:
  `a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`
- C22 SHA256:
  `c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`
- target layer: zero-based layer 22
- correction rank: 2
- trainable tensors: correction A/B only
- trainable numel: 50,688
- arms: exactly `G5-C0`, `G5-C1`, `G5-M1`
- seeds: exactly `5201`, `5202`, `5203`
- train rows: exactly 2880, frozen order and encoding identity
- dev rows: exactly 720, frozen order and encoding identity
- objective: `FINAL_3WAY_CROSS_ENTROPY_ONLY`
- optimizer: AdamW
- learning rate: 0.001
- weight decay: 0.0001
- scheduler: none
- global correction-parameter gradient clip: 5.0
- epochs: 20
- logical optimizer steps: exactly 20
- checkpoint selection: final fixed step only
- no parent unfreeze
- no rank/layer/projector search
- no rescue seed expansion
- no task-metric checkpoint selection
- no fresh XG1 access during recovery or training
- no scientific ownership p-value during recovery or training

## 3. Authorized memory recovery

The monolithic 2880-row backbone kernel launch may be replaced by
**checkpointed contiguous backbone row streaming**.

Frozen recovery parameter:

`BACKBONE_STREAM_ROWS = 240`

The exact ordered 2880-row training tensor is partitioned only along its batch/row
dimension into 12 contiguous 240-row backbone chunks.

For every chunk:

1. use the exact corresponding `input_ids` rows;
2. supply the exact corresponding attention-mask rows to the Phase 2 layer-22
   wrapper through the existing `phase2_active_mask` bridge;
3. execute the same frozen Mamba backbone and layer-22 correction;
4. use PyTorch non-reentrant activation checkpointing for the chunked backbone
   call so backward recomputation does not retain the full backbone activation
   stack;
5. preserve the chunk's autograd connection to correction A/B.

The resulting chunk `last_hidden_state` tensors are concatenated in original row
order.

The historical downstream ContraMamba heads are then executed **once** on the
single concatenated 2880-row hidden-state tensor using the existing
`encoder_hidden_states` path.

The final logits must therefore still have shape `[2880, 3]`.

The final cross-entropy is computed **once** over all 2880 logits and labels.

Each epoch must still execute exactly:

- one logical 2880-row objective;
- one `loss.backward()`;
- one global correction-gradient clipping operation;
- one AdamW optimizer step.

This amendment does **not** authorize ordinary minibatch SGD or multiple
loss/backward calls accumulated into one optimizer step.

## 4. Stochastic-semantics preservation

The frozen parent downstream heads contain dropout and their stochastic training
semantics must not change.

Therefore:

- the training seed is reset exactly once at the same epoch boundary as in the
  parent authority;
- the streamed Mamba backbone must not consume CPU or CUDA RNG;
- runtime code must capture RNG state immediately before backbone streaming and
  assert exact equality immediately after the concatenated backbone output is
  produced;
- downstream heads are invoked once, on the full 2880-row concatenated hidden
  state, after that RNG-invariance assertion;
- no per-chunk downstream-head forward is authorized.

Any observed RNG-state change during backbone streaming is a hard blocker.

## 5. Authorized implementation scope

Preferred and authorized code scope is only:

- `scripts/train_reason_router_gen5_phase2_state_update_ownership.py`
- `tests/test_reason_router_gen5_phase2_state_update_ownership.py`

Do not modify:

- shared Gen5 correction implementation;
- historical parent model implementation;
- shared kernel compatibility code;
- dataset/split/tokenizer artifacts;
- R22/C22 artifacts;
- fresh XG1 artifacts.

If the recovery cannot be implemented in the two-file runner scope, stop rather
than broadening scope silently.

## 6. Mandatory equivalence gate

Before full-size execution, the runner must perform a bounded CUDA equivalence
check on a fixed contiguous subset that fits monolithically.

Use the same model parameters, arm, seed, training mode, frozen parent, and encoded
rows for both paths.

Compare:

1. monolithic historical forward;
2. checkpointed backbone-streamed forward followed by one full downstream forward.

The gate must verify:

- identical input row identities and order;
- exact same pre-forward RNG state;
- backbone-streaming RNG state unchanged;
- logits numerically equivalent;
- scalar CE numerically equivalent;
- correction A gradient numerically equivalent;
- correction B gradient numerically equivalent;
- no parent gradients;
- no R22/C22 gradients;
- no optimizer step;
- no parameter mutation.

The implementation must report the observed maximum absolute residuals.

Acceptance tolerances:

- logits max abs residual: `<= 1e-5`
- CE abs residual: `<= 1e-6`
- A-gradient max abs residual: `<= 1e-5`
- B-gradient max abs residual: `<= 1e-5`

Any failure blocks full-size feasibility execution and training.

## 7. Full-size feasibility gate

After bounded equivalence passes, execute one exact ordered 2880-row
checkpointed-backbone-streamed objective for `seed=5201`, `arm=G5-M1`.

This gate is authorized to execute:

- full forward;
- one full CE;
- one backward for memory/gradient feasibility.

It must **not** execute an optimizer step.

Required PASS conditions:

- final logits shape `[2880, 3]`;
- finite logits and CE;
- exactly one backward;
- correction A/B gradients present and finite;
- no parent gradients;
- no R22/C22 gradients;
- parent parameter fingerprint unchanged;
- R22/C22 identities unchanged;
- full objective fits the qualified T4 runtime;
- exact CUDA fast path remains active;
- fresh XG1 loaded: false;
- scientific p-value count: 0;
- optimizer step executed: false.

After the gate, correction gradients must be cleared and no correction parameter
value may have changed.

PASS label:

`PASS_GEN5_PHASE2_STREAMED_FULL_BATCH_CUDA_PREFLIGHT`

FAIL/OOM label:

`GEN5_PHASE2_STREAMED_FULL_BATCH_EXECUTION_FEASIBILITY_BLOCKED`

## 8. Conditional training authorization

If and only if both:

1. the bounded monolithic-vs-streamed equivalence gate passes; and
2. `PASS_GEN5_PHASE2_STREAMED_FULL_BATCH_CUDA_PREFLIGHT` is produced,

the already-frozen 3-arm x 3-seed Phase 2 training matrix is authorized using this
exact streamed-backbone execution path.

No additional execution-authority document is required for that training matrix.

All nine cells must retain the parent authority's:

- exact 20 epochs;
- exact 20 optimizer steps;
- same-seed cross-arm initialization;
- final fixed-step checkpoint only;
- correction-only optimizer ownership;
- parent/basis immutability checks;
- no task evaluation;
- no fresh XG1;
- no scientific p-value.

## 9. Stop conditions

Stop without scientific training if any of the following occurs:

- bounded equivalence exceeds tolerance;
- streamed backbone changes CPU or CUDA RNG state;
- full-size streamed forward/backward OOMs;
- multiple CE/backward calls per epoch would be required;
- parent or basis gradients/mutation appear;
- exact CUDA fast path is not used;
- implementation requires modifying files outside the two-file scope;
- data/split/tokenizer/seed/arm/loss/optimizer semantics would change;
- fresh XG1 would be accessed.

No automatic change to `BACKBONE_STREAM_ROWS`, batch semantics, precision,
autocast, optimizer, model, or scientific design is authorized after a failure.

## 10. Scientific boundary

Passing the recovery gates establishes only:

1. execution-path equivalence within the frozen tolerances; and
2. CUDA memory feasibility of the exact logical full-batch objective.

It does not establish the Phase 2 ownership hypothesis.

The scientific ownership claim remains untested until the authorized training
matrix completes, its artifacts are validated, and the separately frozen fresh XG1
ownership assay is executed under its later authority.
