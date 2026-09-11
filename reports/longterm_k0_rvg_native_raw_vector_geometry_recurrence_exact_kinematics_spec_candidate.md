# K0-RVG Native Raw Vector Geometry / Recurrence-Exact Kinematics Refinement Candidate

**Status:** mathematical / scientific-design refinement candidate only.

**Stage:** `K0-RVG — Native Raw Vector Geometry / Recurrence-Exact Kinematics`

**Purpose:** restore the original K0 scientific object as raw native recurrent-state motion before choosing a successor branch.

This document does not authorize training, evaluation, model forward, recurrent-state extraction, intervention, probe fitting, geometry fitting, branch-A execution, branch-B execution, or K4 execution.

It does not rewrite or invalidate historical K0/K1/K2/K3/K3C/K3T results.

It freezes a new prospective constraint for any future native-state trajectory study.

## 1. Branch decision is deferred

The latest K-series synthesis left a genuine fork:

- Branch A: further prospective confirmation of the already-observed D / displacement / P trajectory summaries;
- Branch B: return to the original confident-error precursor question with a genuinely new support design.

The current decision is:

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

Branch A is not abandoned:

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

Branch B is not yet active:

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

Before either branch is activated, the scientific object is constrained to raw native Mamba recurrent-state vector/tensor motion.

## 2. Raw native recurrent state is the primary scientific object

For frozen Mamba layer l and token/model step t, define:

`S_t^(l)`

as the exact native selective-SSM recurrent state with validated post-consumption semantics:

`post_consumption_s_t`

This is the internal recurrent state updated by the selective state-space recurrence.

It is not:

- final model hidden state;
- residual-stream hidden representation;
- output after `C_t` readout;
- skip/gate/output-projection representation;
- task-head logits;
- a learned probe representation;
- PCA embedding;
- a learned trajectory embedding;
- an outcome-selected subspace.

The primary object is the tensor state itself.

For a layer whose recurrent state has shape:

`d_inner × N`

the state space is written:

`S^(l) = R^(d_inner × N)`

with its frozen native tensor coordinates.

No flattening is required to define the scientific motion.

`vec(S_t)` may be used only as an algebraically equivalent serialization or linear-algebra notation.

It must not be treated as a learned representation.

## 3. Native coordinate geometry

The primary geometry is the frozen model's own native coordinate system at a single fixed layer.

For tensors X and Y of the same native state shape define the raw Frobenius inner product:

`<X,Y>_F = sum_(i,j) X_(i,j) Y_(i,j)`

and norm:

`||X||_F = sqrt(<X,X>_F)`

For nonzero X and Y define:

`cos_F(X,Y) = <X,Y>_F / (||X||_F ||Y||_F)`

This is the only primary geometry frozen by K0-RVG.

K0-RVG does **not** claim that raw Frobenius geometry is an intrinsic or representation-invariant physical metric of Mamba.

Its permitted claim scope is narrower:

**within the same frozen model, same layer, same native recurrent coordinate system, compare raw recurrent-state motion without learned or outcome-tuned geometry.**

## 4. Forbidden primary geometries

The following may not define the primary scientific space in the next native-vector study:

- whitening chosen or fit for effect separation;
- Mahalanobis geometry;
- learned metric;
- learned projection;
- PCA/ICA/NMF axis chosen from the confirmatory population;
- supervised subspace;
- probe-derived embedding;
- Jacobian axis selected after viewing outcomes;
- nonlinear detector embedding;
- outcome-selected channel subset;
- outcome-selected tensor mode subset;
- layer selected because it gives the strongest confirmatory result.

A future separately preregistered geometry-robustness study may investigate coordinate scaling or alternative metrics.

Such alternatives must not rescue a failed raw-native confirmatory result.

## 5. Cross-layer boundary

Every Mamba layer has its own native recurrent coordinate system.

Therefore K0-RVG forbids treating raw norms or raw angles from different layers as if they share one common physical unit.

Specifically, without a separately frozen normalization/mapping:

- do not subtract states from different layers;
- do not compute cross-layer velocity vectors;
- do not compare raw Frobenius speed magnitudes across layers as one metric scale;
- do not average raw vectors across layers;
- do not choose a best layer after outcome inspection.

Future work must freeze layer policy prospectively.

## 6. Primary raw vector/tensor kinematics

All vector/tensor objects below live directly in the native state tensor space.

### State displacement / velocity

`V_t^(l) = S_t^(l) - S_(t-1)^(l)`

This is the primary first-order motion object.

It is a tensor with the same shape as `S_t^(l)`.

### Delta-velocity / acceleration analogue

`DeltaV_t^(l) = V_t^(l) - V_(t-1)^(l)`

The symbol `A` is not used for acceleration because A is reserved for possible authorization notation in later decision-space work.

### Correction-control state difference

For aligned correction and control branches:

`DeltaCC_t^(l) = S_t^(l,corr) - S_t^(l,ctrl)`

This is a branch-separation tensor.

### Correction-control difference velocity

`DeltaVCC_t^(l) = DeltaCC_t^(l) - DeltaCC_(t-1)^(l)`

Equivalently:

`DeltaVCC_t^(l) = V_t^(l,corr) - V_t^(l,ctrl)`

This is the instantaneous vector/tensor response by which correction and control trajectories separate.

These four objects are scientific objects in their own right.

They must not be immediately replaced by one scalar summary.

## 7. Exact recurrence identity

For the frozen selective-SSM recurrence, write:

`S_t = G_t ⊙ S_(t-1) + W_t`

where:

`G_t = discrete_A_t`

and:

`W_t = deltaB_u_t`

under the validated slow-forward recurrence semantics.

Therefore the exact raw velocity is:

`V_t = S_t - S_(t-1)`

and algebraically:

`V_t = (G_t - 1) ⊙ S_(t-1) + W_t`

Define the exact natural-execution velocity components:

`V_t^(carry-change) = (G_t - 1) ⊙ S_(t-1)`

`V_t^(write) = W_t`

so that:

`V_t = V_t^(carry-change) + V_t^(write)`

This is an algebraic identity of the recurrence.

It introduces no learned axis and no causal intervention.

## 8. Relationship to retained contribution H

Historical K3/K3C work also used the full retained contribution:

`H_t = G_t ⊙ S_(t-1)`

Then:

`S_t = H_t + W_t`

and:

`V_t = H_t - S_(t-1) + W_t`

Therefore:

`V_t^(carry-change) = H_t - S_(t-1)`

K0-RVG distinguishes:

- **retained contribution:** `H_t = G_t ⊙ S_(t-1)`;
- **carry-change velocity component:** `H_t - S_(t-1)`;
- **write velocity component:** `W_t`.

These are not interchangeable.

The carry-change component measures how recurrence retention/decay changes the prior state during this token step.

The retained contribution H is the carried state contribution to the new state position.

## 9. Observational boundary of retain/write decomposition

K0-RVG uses the decomposition only as a **natural-execution observational identity**.

It does not authorize:

- W_EQ;
- G_EQ;
- H_EQ;
- WH_EQ;
- midpoint substitution;
- causal replay;
- counterfactual state editing;
- intervention-derived causal language.

Historical K3/K3C causal results remain historically frozen and are not reinterpreted as confirmatory evidence for K0-RVG.

Future natural-execution observation may record:

`S_(t-1), G_t, W_t, S_t`

and verify exact recurrence reconstruction.

That observation alone supports only algebraic/descriptive statements.

## 10. Direction is primary; magnitude is secondary

For nonzero raw tensor motions, direction is defined by Frobenius cosine.

Examples:

`cos_F(V_t, V_(t-1))`

`cos_F(V_t, V_(t_e-1))`

`cos_F(DeltaCC_t, DeltaCC_(t-1))`

`cos_F(DeltaVCC_t, V_(t_e-1))`

`cos_F(V_t^(write), V_t)`

`cos_F(V_t^(carry-change), V_t)`

`cos_F(V_t^(write), V_t^(carry-change))`

These retain information that is destroyed by scalar speed alone.

Near-zero vector handling must be frozen before any future confirmatory execution.

No observation may be removed after its scientific outcome is known.

## 11. Reversal, persistence, and rotation

K0-RVG permits prospective vector questions such as:

### Local persistence

Does:

`cos_F(V_t, V_(t-1))`

remain strongly positive?

### Reversal

Does motion after a critical evidence event satisfy:

`cos_F(V_post, V_pre) < 0`

under a frozen pre/post aggregation rule?

### Rotation

How does:

`cos_F(V_t, V_pre)`

change over event-relative time?

### Correction-response orientation

Does:

`DeltaVCC_t`

align with, oppose, or become orthogonal to the incoming trajectory?

### Separation-direction persistence

Does:

`DeltaCC_t`

retain a direction once correction/control trajectories diverge, or rotate over time?

These are vector-geometry questions.

No single one becomes confirmatory merely by being listed here.

## 12. Scalar summaries have secondary status

The following remain valid derived diagnostics:

- speed: `||V_t||_F`;
- delta-velocity magnitude: `||DeltaV_t||_F`;
- correction-control separation magnitude: `||DeltaCC_t||_F`;
- response magnitude: `||DeltaVCC_t||_F`;
- turning: `1 - cos_F(V_t,V_(t-1))`;
- path length;
- displacement norm;
- path efficiency;
- onset/latency/persistence summaries.

Existing D / displacement / P evidence is preserved.

But scalar summaries are not the primary scientific state object.

A future claim that only scalar magnitude differs, while prespecified vector-direction organization does not, must be reported as a scalar kinematic effect rather than a vector-geometry precursor.

## 13. Tensor structure must be preserved in artifacts

Even when calculations are implemented by flattening for efficient dot products, future scientific artifacts must preserve enough metadata to reconstruct the original tensor structure.

Required provenance includes:

- layer index;
- native state shape;
- axis meaning as implemented;
- dtype;
- state timing;
- exact source recurrence binding;
- canonical serialization order if tensor bytes are hashed;
- whether a statistic was computed tensor-wise or via an equivalent flattening.

Flattening must be demonstrably equivalent to Frobenius operations.

No scientific interpretation may be attached to an arbitrary flattened coordinate index without mapping it back to its native tensor axes.

## 14. Channel / state-mode inspection boundary

K0-RVG permits descriptive inspection of where raw motion energy occurs in the native tensor.

Examples include:

- per-`d_inner` channel Frobenius energy across the state-size axis;
- per-state-mode energy across channels;
- exact tensor-entry contribution to a frozen inner product.

However, channel or mode ranking from confirmatory outcomes may not be used to construct a new confirmatory subspace on the same data.

Any promoted channel/mode subset requires independent discovery and fresh confirmation.

## 15. Raw motion rank / singular-spectrum audit

A singular spectrum may be used as a **descriptive dimensionality audit** of raw motion.

For a frozen segment with tensor velocities `V_t`, vectorize only for linear algebra:

`M = [vec(V_t1)^T; ...; vec(V_tk)^T]`

and inspect singular values of M.

Permitted descriptive questions include:

- is motion approximately rank-1;
- how many independent directions are materially occupied;
- does rank expand after an evidence event;
- does correction/control divergence add a new direction.

The singular vectors are not primary learned axes.

No singular vector selected on confirmatory outcomes may become a same-data confirmatory projection.

No rank threshold may be tuned to maximize group separation.

## 16. Coordinate-robustness interpretation

K0-RVG intentionally does not try to solve representation invariance by learning a new metric.

Therefore the claim boundary is explicit.

A positive result may support:

**in this frozen Mamba model, at this frozen layer, in its native recurrent coordinates, a reproducible raw tensor-trajectory organization is observed.**

It may not by itself support:

- an intrinsic manifold metric;
- coordinate-free physical velocity;
- invariance under arbitrary state reparameterization;
- architecture-independent state geometry.

A later geometry-robustness study may test deterministic coordinate rescalings or alternative prespecified normalizations.

That later study is separate.

## 17. Event-relative organization

Preserve the original K0 time coordinates:

- absolute token index `t`;
- normalized progress `r=t/T`;
- semantic-event-relative time `tau=t-t_e`.

Where a critical evidence event is available, event-relative vector organization is scientifically primary.

A signal that first appears after the evidence event but before the answer is a pre-answer phenomenon.

It is not a pre-evidence predictor.

The event anchor must be deterministic and frozen before state trajectories are inspected.

## 18. Fresh-data and anti-tuning contract

For any future confirmatory native-vector study:

1. the model and checkpoint are frozen;
2. the layer policy is frozen;
3. native state capture semantics are frozen;
4. event anchor policy is frozen;
5. time window is frozen;
6. raw Frobenius geometry is frozen;
7. vector object family is frozen;
8. near-zero vector policy is frozen;
9. multiplicity procedure is frozen;
10. matching/support design is frozen;
11. no confirmatory outcome is used to choose a projection, subspace, metric, channel subset, singular vector, layer, or time window.

A failed confirmatory result cannot be rescued on the same population by changing any of the above.

## 19. Candidate falsification conditions

A future strong raw-vector claim is not supported if any applicable condition holds.

### RVG-F1 — no reproducible vector organization

Fresh confirmatory data show no preregistered difference in raw directional / vector-response organization.

### RVG-F2 — scalar-only effect

Only speed/path/displacement magnitudes reproduce while the preregistered directional/vector organization does not.

This may support a scalar kinematic observation.

It does not support the stronger raw-vector trajectory claim.

### RVG-F3 — terminal-only localization

The effect appears only at or after the terminal answer state when a precursor claim requires preterminal organization.

### RVG-F4 — confound explanation

The effect is explained by confidence, predicted class, gold class, length, lexical markers, intervention family, evidence position, or another frozen confound control.

### RVG-F5 — event nontracking

A claimed evidence-responsive vector phenomenon fails to move with prospectively manipulated semantic event timing where an event-order test is applicable.

### RVG-F6 — seed / population instability

The directional/vector organization does not reproduce beyond the population or seed that generated the hypothesis.

### RVG-F7 — outcome-selected subspace

The apparent vector phenomenon requires a subspace, singular direction, layer, channel, metric, or time window selected using the same confirmatory outcomes.

The confirmatory claim is invalid.

## 20. Relationship to A/B successor branches

K0-RVG is pre-branch.

It does not select Branch A.

It does not select Branch B.

Branch A may later ask whether known scalar summaries transport further.

Branch B may later ask whether confident-wrong and confident-correct trajectories differ.

Both branches, if activated, must respect the K0-RVG raw-native constraint unless a later authority explicitly supersedes it with a separate scientific rationale.

## 21. Decision-space boundary

Decision-relevant semantic projection is not part of K0-RVG.

The following remain downstream only:

- task-head/logit projection;
- class-margin coordinates;
- Authorization × Signed Polarity;
- learned linear readout;
- Jacobian-local projection;
- intervention-derived semantic axis.

K0-RVG asks first:

**what does the native recurrent state itself do in its own frozen coordinates?**

Only after reproducible raw vector organization is established may a later stage ask what that direction means epistemically.

## 22. Static instrumentation question

Before any new scientific execution, a read-only/static source audit should determine whether the validated frozen Mamba implementation can expose, with exact provenance:

- `S_(t-1)`;
- `G_t = discrete_A_t`;
- `W_t = deltaB_u_t`;
- post-update `S_t`;

at the same token step without modifying ordinary model outputs.

The audit must also verify the exact algebraic reconstruction:

`S_t == G_t ⊙ S_(t-1) + W_t`

within an explicitly frozen numerical equality/tolerance policy.

This audit is static/design work only unless separately authorized to execute synthetic instrumentation.

## 23. Next design stage

After K0-RVG is frozen, the next authorized activity is only:

`K0-RVG-S — Static Native Recurrence Observation Contract`

It may inspect frozen source code and existing validated instrumentation artifacts.

It must specify:

- exact source line/binding identities;
- capture timing of `S_(t-1), G_t, W_t, S_t`;
- tensor shapes and dtypes;
- noninterference requirements;
- serialization/hashing contract;
- recurrence-identity validation;
- synthetic-only validation requirements.

It may not run a new scientific population.

## 24. Authority state

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`RAW_NATIVE_SSM_STATE_PRIMARY_OBJECT = YES`

`LEARNED_OR_TUNED_GEOMETRY_PRIMARY = NO`

`RAW_FROBENIUS_NATIVE_GEOMETRY_PRIMARY = YES`

`RECURRENCE_EXACT_RETAIN_WRITE_OBSERVATION_MOTIVATED = YES`

`CAUSAL_RETAIN_WRITE_INTERVENTION_AUTHORIZED = NO`

`MODEL_FORWARD_AUTHORIZED = NO`

`RECURRENT_STATE_READ_AUTHORIZED = NO`

`K0_RVG_STATIC_SOURCE_AUDIT_AUTHORIZED_AFTER_FREEZE = YES`

`BRANCH_A_EXECUTION_AUTHORIZED = NO`

`BRANCH_B_EXECUTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

This document authorizes no scientific execution.
