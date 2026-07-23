# Stage196-B2-B6P8 Full Trainable-Path Replay Implementation Specification

## Scope and authority

This stage implements replay and probes it only. It adds no direction or
candidate-order stability loss, teacher, EMA, optimizer objective, checkpoint
selection rule, or combined intervention.

The implementation consumes the P7 design without changing its boundary:

```text
Mamba last_hidden_state [B,T,H]
  -> complete joint downstream production path
  -> complete frame-local-only donor downstream production path
  -> row-wise five-primitive selection
  -> production FinalEntitlementDecisionHead
  -> recipient-native final modulation
```

The probe requires the exact P7 analysis plus its ten-file companion closure.
Because those runtime artifacts are not checked into this source checkout, the
probe never synthesizes them: a missing file, failed P7 contract, ambiguous row
mapping, or changed boundary yields `STAGE196B2B6P8_BLOCKED_CONTRACT_FAILURE`.

## Replay-state schema

The opt-in native output contains only the tensors needed at the P7 boundary:

| Tensor | Shape | Purpose |
|---|---|---|
| `encoder_hidden_states` | `[B,T,H]` | authoritative shared Mamba state |
| `attention_mask` | `[B,T]` | production mask validation/pooling |
| `claim_mask` | `[B,T]` | production claim pooling |
| `evidence_mask` | `[B,T]` | production evidence pooling |
| optional mismatch flags | `[B]` | native final-modulation conditions |

These are the live CUDA tensors. The implementation does not detach or copy the
hidden state, does not move it to CPU, and does not include labels, row IDs,
seed IDs, NumPy values, or Python scalars in the replay state. Frozen backbone
parameters do not imply an explicit activation detach.

The RNG context is separate from replay state. It records CPU and all CUDA
generator states immediately before and after the native downstream path.

## Exact replay API

`ContraMambaV6BMinimal.replay_full_trainable_path` is the canonical API. It
accepts the captured state, exact ordered candidate-action tensors, the declared
gradient ownership mode, captured stochastic context, live native output, the
separately loaded counterpart model, and row-aligned P7 action keys.

Candidate identities are exactly:

```text
00100000000000
01000000000000
10000000000000
```

They are opaque selector identities, not primitive bitmasks. The model validates
their order but never derives actions from their string positions. Each row action
is an authoritative five-bit selection in this P7 order:

```text
FRAME, PREDICATE, SUFFICIENCY, POSITIVE_ENERGY, NEGATIVE_ENERGY
```

The probe loads those row actions and their keys from
`stage196b2b6p7_candidate_semantic_trace.csv`. Arbitrary candidates are rejected.

## Shared Mamba and full downstream replay

The replay-capable native call performs the only measured Mamba forward. The
counterpart call supplies `encoder_hidden_states`, so it performs zero Mamba
forwards. FrameGate, PredicateCoverageHead, SufficiencyGate,
PolarityEnergyHead, auxiliary enabled heads, and FinalEntitlementDecisionHead
are invoked as their production modules. The existing frame-local-only hooks
remain installed, preserving direct FrameGate ownership and detached downstream
aliases without changing forward values.

P7 selected two sequential stochastic downstream arm executions followed by
deterministic candidate composition. It did not select three independent donor
replays or a vectorized dropout path. The implementation follows that choice.

## Stochastic state

FrameGate, PredicateCoverageHead, SufficiencyGate, and PolarityEnergyHead contain
downstream dropout. Replay restores the pre-native RNG snapshot for the donor,
runs the donor in the recipient model mode, restores the donor's original model
mode, and restores the post-native RNG snapshot. CPU and every CUDA generator are
covered. The global RNG therefore advances exactly as the native path did once.
The implementation never forces model-wide evaluation mode.

No downstream BatchNorm, sampling, random router, or persistent stateful buffer
was introduced. The probe enumerates actual dropout modules and buffers.

## Canonical geometry

The production class order remains:

```text
REFUTE, NOT_ENTITLED, SUPPORT
```

Every candidate returns differentiable class scores, the three primary pairwise
margins, top-1/runner-up margin, and all seven response deltas relative to native.
All primary margins are derived by one geometry helper. Top-1/runner-up uses the
existing `torch.topk(..., sorted=True)` authority with no epsilon. At an exact
tie, the value is differentiable almost everywhere but the selected subgradient
is the implementation-defined `topk` tie selection; the mathematical derivative
is not unique at that point.

## Native preservation and trainer plumbing

The native forward body remains authoritative. With
`stage196b2b6p8_return_replay_state=False`, output keys and computation are the
previous path. No RNG capture, replay state, replay, loss, or additional graph is
created.

The trainer flag
`--stage196b2b6p8-enable-full-trainable-path-replay-api` defaults false. When
enabled it only asks the native training forward to expose replay state. For a
physically batched call it retains per-chunk state/RNG pairs. It does not consume
them in P8. Provenance records enabled/default-off and explicit false values for
replay execution, loss integration, optimizer/scheduler changes, checkpoint
schema changes, and checkpoint selection changes.

## Gradient probe

The probe loads the joint and frame-local-only reconstructed seed183 checkpoints,
installs their original ownership modes, and groups parameters by direct module
ownership: frozen backbone, frame, predicate, sufficiency, polarity,
entitlement/decision, selector/router, final composer, and other trainable.

It independently calls `torch.autograd.grad` for:

- every native class score and native primary margin for both arms;
- every candidate class score and primary margin;
- all four direction response coordinates;
- all three unordered candidate pairs for every primary response coordinate.

Each target/group row reports tensor counts, requires-grad counts, connected and
unused counts, finite and nonzero counts, L1/L2 norms, and maximum magnitude. A
present all-zero gradient is `CONNECTED_ZERO_AT_OBSERVED_BATCH`, never
`DISCONNECTED`. Required recipients are selected from the P7 gradient-path CSV,
not inferred only from parameter names.

## Equivalence, provenance, and mutation

The probe resets RNG before ordinary and replay-capable native forwards and uses
exact `torch.equal` for available logits, prediction, frame, predicate,
sufficiency, polarity, entitlement, and decision-head state. Candidate IDs,
row-action keys, primitive width, production arithmetic, class order, and finite
logits are audited separately. Any detached P2-style reference, when present in
future authority, is diagnostic numeric evidence only and is never the gradient
authority.

Checkpoint recovery input is mandatory. The probe records path, SHA-256, size,
mode, source commit, selected epoch, and reconstructed provenance. It validates
the frozen source commit and trainer hash and never claims byte identity with the
lost checkpoints.

Parameters and persistent buffers are fingerprinted after load and through native,
candidate, direction, and order phases. The meaningful comparison starts after
checkpoint loading. Original model modes and RNG are restored. No optimizer or
scheduler is constructed and no checkpoint is written.

## Resources, decisions, and outputs

The probe records batch size, three candidates, native/replay forward counts,
peak CUDA allocated/reserved bytes, replay tensor count, and retained graph count.
It never changes batch size automatically and makes no throughput claim. CUDA OOM
is caught as a structured result.

The decision is derived in the required hierarchy: complete, valid/resource
constrained, stochastic repair, gradient repair, semantic repair, or blocked
contract. Scientific negative results do not by themselves become upstream
contract failures.

Exactly ten files are written atomically into a directory that must not already
exist. The process returns zero exactly when `blocking_reasons` is empty and two
otherwise.

## Static validation status

The four-file diff was reviewed statically only. Per execution policy, Python,
compilation, `--help`, probe/trainer/model/checkpoint execution, smoke tests,
training, evaluation, commits, and pushes were not performed.
