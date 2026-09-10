# D1 Partial Gradient Ownership: Design and Bounded Implementation Authority Candidate

## 1. Verdict, authority, and phase boundary

```text
VERDICT = PASS
PHASE = D1_CONTINUOUS_PARTIAL_GRADIENT_OWNERSHIP_DESIGN_AND_IMPLEMENTATION_AUTHORITY_AUTHORING
TRAINING_EVALUATION_KAGGLE_CHECKPOINT_MUTATION = NOT_AUTHORIZED
IMPLEMENTATION_IN_THIS_TASK = NOT_AUTHORIZED
ENDPOINT_REUSE = ELIGIBLE_IF_IMPLEMENTATION_EQUIVALENCE_PASSES
READY_FOR_INDEPENDENT_D1_DESIGN_IMPLEMENTATION_AUTHORITY_VERIFICATION = YES
```

This is the one compact D1 design + later-implementation authority.  It is
grounded in the checked source at commit
`5347c9e179e8365b591fe62884f8291b30e7f7ff` and in the frozen integrated
failure analysis, commit `5347c9e179e8365b591fe62884f8291b30e7f7ff`, blob
`806185add96086905eac8aa014ee9e8845907395`,
`reports/reason_router_p3w7_seed8192_factorial_integrated_failure_localization_analysis_report_candidate.md`.
Frozen scientific interpretation is commit
`0894a921bf7ed69151722e3ce2691eb49bf4f40f`, blob
`e4cf29f818566cad465d49703dba2e22978b1473`; frozen validated evidence is
commit `6dcef9520af2cb88691628a77b72f3fdd7042cd8`, blob
`17be9bead782f129de925148580b034d981079cd`.  The research vision and
hypothesis map are non-authority context only (main blobs
`13bcedfa229209c2d5657e7a81b4b52010552640` and
`46580cf00709654d452a81824cad7a3ae181d83c`).

The frozen results remain: A1-A0 is MIXED_OR_SEED_DEPENDENT; A2-A0 is
DESCRIPTIVELY_HARMFUL; A3-A2 is MIXED_OR_SEED_DEPENDENT with no reproducible
rescue; A3-A1 is DESCRIPTIVELY_HARMFUL; and the interaction is heterogeneous.
The matched rows establish A0->A2 277 broken/34 repaired, A1->A3 228 broken/15
repaired, 137 A1-correct REFUTE rows broken by A3 (100 REFUTE->SUPPORT and 37
REFUTE->NOT_ENTITLED), 24 C2 REFUTE stable IDs broken in all three seeds, a
descriptive C2 LEVEL_1_POLARITY_ASSOCIATED pattern, and a bounded C1
destination-level LEVEL_2_AUTHORIZATION_ASSOCIATED pattern.  No single small
harmful gradient edge is established.

D1 is therefore the minimum Generation-2 test of the failed *binary* G_G
intervention.  It changes only gradient-graph (G_G) behavior, not the
information graph (G_I) or the decision/authorization graph (G_D).  It adds no
owned recurrent state, no Generation-3 edge-specific ownership (D2), and no
multi-stream/semantic-state ContraMamba (D5).  Over-Isolation is a compatible
failure hypothesis, not a proven diagnosis.

## 2. Checked current implementation map

The authoritative model path is `src/contramamba/modeling_v6b_minimal.py`,
`ContraMambaV6BMinimal.forward` (ownership selection near lines 575-583;
consumer wiring near 586-758).  It accepts only `joint` and `explicit_local`.
`joint` passes owner tensors unchanged to every listed consumer.  In
`explicit_local`, owner dictionaries remain raw for owner-local losses while
downstream aliases are detached.  The frozen encoder is separately enforced by
the P2 resolver; it is not an ownership edge.

| Boundary | Downstream consumer | Controlled tensor occurrences | Current `explicit_local` detach tensors | Downstream consumer/loss whose upstream route is removed | Intentionally still available |
|---|---|---:|---|---|---|
| B1 | predicate | 4 | `claim_frame_state`, `evidence_frame_state`, `frame_pair_repr`, `frame_prob` | predicate head and its predicate BCE cannot update FrameGate | Frame BCE -> FrameGate; predicate BCE -> predicate head |
| B2 | sufficiency | 4 | `frame_pair_repr`, `predicate_pair_repr`, `frame_prob`, `predicate_coverage_prob` | sufficiency BCE cannot update FrameGate or predicate head through those inputs | frame/predicate local BCEs; sufficiency BCE -> sufficiency gate |
| B3 | polarity | 3 | `frame_pair_repr`, `predicate_pair_repr`, `sufficiency_repr` | polarity CE cannot update FrameGate, predicate head, or sufficiency gate | all preceding owner-local losses; polarity CE -> polarity head |
| B4 | decision_head / final CE | 5 | `frame_prob`, `predicate_coverage_prob`, `sufficiency_prob`, `positive_energy`, `negative_energy` | final 3-way CE cannot update any four local semantic owners | final CE -> decision composer/router parameters; local losses -> their owners |

There are exactly four ownership-controlled `explicit_local` conditional
boundary expression groups and 16 controlled tensor occurrences:
4 + 4 + 3 + 5 = 16.  B1 uses a detached `predicate_frame_inputs` dictionary;
B2, B3, and B4 use direct conditional `.detach()` arguments.  The output
witness at lines 1026-1032 records the four boundary states as
`frame_to_predicate_detached`, `frame_predicate_to_sufficiency_detached`,
`frame_predicate_sufficiency_to_polarity_detached`, and
`local_to_final_router_detached`.  `final_router_inputs_detached` is a
redundant observability/configuration boolean for the final-router input detach
state.  It is not a tensor edge, detach operation, ownership boundary, or
gradient-controlled expression group, and it is not counted in the four
boundaries or 16 tensor occurrences; it remains relevant only to reporting,
observability, and provenance/config semantics.

```text
OWNERSHIP_CONTROLLED_BOUNDARY_GROUPS = 4
OWNERSHIP_CONTROLLED_TENSOR_OCCURRENCES = 16
```

`scripts/train_controlled_v6b_minimal.py` supplies the P2 control plane:
`P2_ARM_CONTRACTS` (lines 200-205), `_p2_resolve_arm_contract` (318-509),
CLI `--gradient-ownership-mode` (11435-11440), model assignment (19840-19846),
`_vnext_forward_maybe_batched` forwarding (16262-16320),
`_p2_reason_router_losses` (620-750), checkpoint metadata/validation
(`_p2_checkpoint_metadata_from_args`, 16406-16492;
`_validate_model_checkpoint_metadata`, 16500-16537), row export
(`_add_reason_router_p2_prediction_exports`, 10535-10631), training report
(`reason_router_p2`, 24707-24716), and run provenance/runtime configuration
(24740-25081).  `tests/test_reason_router_p2_contract.py`, especially
`test_a0_a3_actual_production_autograd_ownership_matrix` (303-401) and
`test_a3_raw_owner_local_polarity_ce_gradients` (941-954), is the existing
ownership semantic contract.

`_p2_reason_router_losses` computes final CE from `output["logits"]`, plus
raw frame/predicate/sufficiency BCE and raw polarity CE; primary-reason CE is
weighted by `reason_loss_weight`.  D1 fixes that weight to zero.  Thus D1 must
not scale any owner-local contribution.  `frame_downstream_gradient_mode` is a
different legacy hook: `_install_framegate_gradient_ownership` (16168-16232)
detaches all FrameGate outputs for `frame_local_only`; P2 rejects that setting
in `_p2_resolve_arm_contract` (363-366).  D1 fixes it at `joint` and does not
alter that hook or its semantics.

## 3. D1 mathematical and mapped semantics

Define, for every tensor occurrence in the four tabled ownership edge groups,

```text
partial_grad(z, lambda) = stopgrad(z) + lambda * (z - stopgrad(z))
```

where `stopgrad` is `Tensor.detach()`.  Forward, it equals `z` for every
finite lambda.  Backward, its derivative with respect to `z` is lambda.
Replace *each and only each* current ownership-controlled conditional detach
occurrence with `partial_grad(z, lambda)` when mode is `partial`; use the raw
`z` when `joint` and preserve the existing `.detach()` branch unchanged when
`explicit_local`.

One shared scalar interpolates coherently across all four actual controlled
boundary groups: every current controlled site is the same binary choice
between raw owner tensor (joint) and its detached alias
(explicit_local), with no site-specific alternate transform or contrary
ownership direction.  At lambda=0 the partial aliases have the exact
`explicit_local` forward value and zero downstream-to-owner derivative; at
lambda=1 they have the exact joint forward value and derivative.  This is not
a license to refactor legacy branches into aliases; retaining their code paths
is required for backward compatibility and endpoint reuse.

For D1 only, `lambda = 0.5`, status
`PRE_SPECIFIED_SYMMETRIC_MIDPOINT_PROBE`.  It is the non-adaptive affine
midpoint between the only tested endpoints, not an optimized value, observed
curve selection, claimed optimum, or lambda sweep.  No other lambda is
authorized.  It is valid precisely because the mapped endpoint equivalence
above is required to pass before training.

## 4. Exact D1 experiment

Use `router=explicit_product`, `reason_loss_weight=0`, split seed `8192`,
training seeds `180,181,182`, the identical frozen A0/A2 encoder/backbone
freeze state, `frame_downstream_gradient_mode=joint`, reason order
FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED, diagnostic-only secondary
reasons, unchanged final 3-way semantics, and unchanged data/split/intervention
semantics.  Do not include conditional-first-blocker.

The coordinate is lambda=1 existing A0/joint, lambda=0 existing
A2/explicit_local, and lambda=0.5 the three new D1 runs.  The question is:
does halfway relaxation of hard isolation reduce its reproducible failure while
preserving useful final-task behavior?

Endpoint reuse is `ELIGIBLE_IF_IMPLEMENTATION_EQUIVALENCE_PASSES`, not now
approved evidence.  It becomes authorized only if all are demonstrated:
unchanged joint code-path semantics; unchanged explicit_local code-path
semantics; strictly additive/gated partial mode; independently tested partial
0=explicit_local and partial 1=joint parameter-gradient equivalence; unchanged
defaults; backward-compatible provenance; and no unrelated training/model
change.  Otherwise endpoint reuse is NOT_AUTHORIZED and a separate execution
decision must decide reruns.

Later comparisons are, per seed: P1 partial .5 versus explicit_local 0; P2
partial .5 versus joint/A0 1; P3 the descriptive ordered geometry 0 -> .5 ->
1.  Do not fit a curve or claim monotonicity from three points.  Primary metric
is macro-F1; safeguards are accuracy, NOT_ENTITLED/REFUTE/SUPPORT F1, and
prediction distribution.  The D1 diagnostic is frozen-A0-correct REFUTE rows
broken, preserving REFUTE->SUPPORT and REFUTE->NOT_ENTITLED where matched IDs
permit.

Hard-isolation receives descriptive support only if .5 moves predictably away
from 0: at least macro-F1, REFUTE F1, A0-correct REFUTE breaks,
REFUTE->SUPPORT, and accuracy are assessed per seed.  The strongest simple
pattern is improvement over 0 in macro-F1 and REFUTE failure across all three
seeds without compensating class collapse.  Failure to reproducibly improve
breakage and macro-F1 weakens the hypothesis; verified half gradients that
behave like 0 do not support simple interpolation; a new class collapse is not
success.  N=3 is descriptive; no significance test is authorized.

## 5. Later bounded implementation authority

Retain `joint` and `explicit_local`; add exactly `partial` and one scalar
`gradient_ownership_lambda`.  In partial mode it must be finite and in [0,1];
this D1 invocation requires exactly `0.5`.  Supplying lambda with a legacy
mode must fail closed.  The later patch may modify **only**:

1. `src/contramamba/modeling_v6b_minimal.py`
2. `scripts/train_controlled_v6b_minimal.py`
3. `tests/test_reason_router_p2_contract.py`

The model file implements the helper, mode validation, mapped aliases, and
configuration witness.  The trainer file adds bounded CLI/config resolution
(including a D1 explicit-product/partial contract), model forwarding, and mode
+ lambda in checkpoint metadata, training report, prediction exports, run
provenance, parsed/rendered command context, and all semantic/run identity
inputs already emitted there.  The test file proves the contract.  Existing
provenance machinery need not be changed: the trainer already serializes parsed
arguments and resolved runtime/P2 contract.  Any newly necessary file outside
this list is a scope mismatch requiring stop and report.

A partial artifact must unambiguously serialize both ownership mode and lambda;
lambda must affect every semantic run identity/provenance comparison that
distinguishes changed training semantics.  It must never serialize as joint or
explicit_local.  Existing defaults and legacy serialized values remain
unchanged.

Required future tests/gates, before any training:

1. Scalar autograd: forward identity and z-gradient multipliers 0/.5/1.
2. At every tabled site, partial 0 equals current explicit_local and partial 1
   equals current joint in outputs and affected parameter gradients.
3. At .5, downstream-to-upstream contribution is half within deterministic
   tolerance; owner-local contribution is unchanged.
4. Multi-loss: a tensor with owner-local plus downstream loss scales only the
   downstream contribution, never halves local ownership.
5. `frame_downstream_gradient_mode=joint` remains independent and unchanged.
6. Existing joint and explicit_local tests pass unchanged.
7. Config rejects lambda <0, >1, NaN, Inf, missing partial lambda, and any
   supplied lambda for a legacy mode.
8. Mode/lambda serialization and semantic run identity are distinguishable.

The registered pre-training gate is PASS only when all are PASS:
`FORWARD_IDENTITY`, `LAMBDA_0_EXPLICIT_LOCAL_GRADIENT_EQUIVALENCE`,
`LAMBDA_1_JOINT_GRADIENT_EQUIVALENCE`, `LAMBDA_05_GRADIENT_SCALING`,
`LOCAL_GRADIENT_PRESERVATION`, `LEGACY_JOINT_REGRESSION`,
`LEGACY_EXPLICIT_LOCAL_REGRESSION`, and `PROVENANCE_DISTINGUISHABILITY`.
Any failure blocks scientific execution.

## 6. Exclusions and release boundary

D1 does not establish an optimal lambda, a continuous curve, edge-specific or
adaptive ownership, causal responsibility of one G_G edge, G_I ownership,
multi-stream Mamba, semantic-state ownership, production readiness, or
generalization beyond this setting.

After independent verification, dedicated commit, push, and remote
commit/blob authentication, this spec may authorize only the bounded patch
above.  It does not authorize training, evaluation, Kaggle, .5 execution,
endpoint reruns, other lambdas/seeds, conditional-first-blocker D1 variants,
D2, or D5.  Execution requires: implementation -> independent
gradient-semantics verification -> registered/static gate -> implementation
commit/push -> separate execution release decision.
