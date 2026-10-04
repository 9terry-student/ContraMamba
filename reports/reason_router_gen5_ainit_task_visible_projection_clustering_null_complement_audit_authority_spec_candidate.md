# Gen5 A-init Task-Visible Projection Clustering / Null-Complement Audit Authority

SOURCE_FORWARD_JACOBIAN_EVIDENCE_FREEZE_COMMIT=a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e
SOURCE_FORWARD_JACOBIAN_EXECUTION_COMMIT=1c9d650c4cc06169c221c1525e93610ba495bdcb
SOURCE_TASK_SENSITIVITY_EVIDENCE_FREEZE_COMMIT=ed196d5003f279dbfe1dc9a631e84e22dc049ac7
SOURCE_FUNCTIONAL_EQUIVALENCE_EVIDENCE_COMMIT=5f079a66f7b0eb0caea30a8d5bc9a0fe757cc449

STATUS=READY_FOR_GEN5_AINIT_TASK_VISIBLE_PROJECTION_CLUSTERING_NULL_COMPLEMENT_AUDIT

TRAINING_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
PARAMETER_GRADIENT_UPDATE_ALLOWED=NO
ANALYSIS_AUTOGRAD_ALLOWED=NO
BACKWARD_METHOD_ALLOWED=NO
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
CUDA_EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_P0_DEV_ONLY

## Scientific question

The recovered true-forward-Jacobian evidence shows that same-training-RNG /
different-A-init layer-22 residuals are dominated in ambient energy by
directions outside the leading task-sensitive hidden subspace, while retaining
a small structured task-visible component.

This audit asks the sharper endpoint-geometry question:

> When the frozen 3x3 A-init x training-RNG layer-22 states are projected into
> the recovered task-sensitive hidden basis, how much absolute A-init
> separation remains, and how much A-init variation is relegated to the
> orthogonal complement?

This is a static trained-endpoint analysis. It does not measure an optimization
trajectory or training-time path.

## Frozen inputs

Use only:

- the frozen Phase3A P0 dev population, 840 rows, split seed 16384;
- the exact 3x3 A-init x training-RNG grid with seeds 6201, 6202, 6203;
- the exact frozen correction checkpoints already authenticated by the source
  Gen5 audits;
- the exact recovered task-sensitive basis from
  `reports/reason_router_gen5_forward_jacobian_recovery_runs/gen5-forward-jacobian-recovery-1c9d650-r4/recovered_task_sensitive_subspace.pt`;
- the frozen functional-fingerprint evidence used by the recovery stage.

No new scientific seed is authorized.

## Boundary

Analyze the output of `Phase2Layer22MixerWrapper.forward`.

For every frozen dev example and every grid cell, capture the layer-22 boundary
state only long enough to accumulate the authorized statistics below. Do not
persist full hidden states.

Use the frozen attention mask to include only valid token positions.

No gradient, Jacobian, backward pass, or optimizer operation is required or
authorized.

## Prospective task-sensitive dimensions

Use the already frozen leading-basis dimensions:

`k = {1,2,4,8,16,32,64,128,256}`

Do not choose k from the observed clustering result.

For recovered eigenvectors `V_k` and an aligned inter-cell residual
`d_t = h_t(left) - h_t(right)` at valid token position `t`, define:

`D_full = mean_t ||d_t||_2^2`

`D_k = mean_t ||V_k^T d_t||_2^2`

`D_perp_k = D_full - D_k`

`Q_k = D_k / D_full`

and the corresponding norm-compression factor:

`C_k = sqrt(Q_k)`

The recomputed `Q_k` values must authenticate against the recovered
forward-Jacobian artifact within a predeclared numerical tolerance.

## Pair classes

Report all 36 unordered grid-cell pairs and separately aggregate:

1. same training RNG / different A-init: exactly 9 pairs;
2. same A-init / different training RNG: exactly 9 pairs;
3. remaining different-A / different-RNG pairs as descriptive context only.

For each prospective k report:

- mean and range of `D_full`, `D_k`, and `D_perp_k`;
- mean and range of `Q_k` and `C_k`;
- ratio of same-RNG/different-A absolute distance to
  same-A/different-RNG absolute distance in full, projected, and complement
  spaces.

This ratio is descriptive geometry, not a statistical significance test.

## 3x3 factorial endpoint-variance decomposition

Exploit the full frozen 3x3 causal grid rather than relying only on pairwise
distances.

At each aligned valid-token observation, for the nine cell states `h[a,r]`,
compute the standard balanced two-factor decomposition:

- grand mean;
- A-init marginal main effect;
- training-RNG marginal main effect;
- A x RNG interaction residual.

Accumulate squared energy of these components over the frozen dev population
for:

- full 768-D hidden space;
- each recovered top-k task-sensitive space;
- each corresponding orthogonal complement.

Report absolute component energies and normalized fractions.

Primary descriptive contrasts:

- how the A-init main-effect energy changes from full space to top-k;
- how the A-init/RNG main-effect energy ratio changes from full space to top-k;
- whether A-init energy is preferentially retained in the null/complement
  space relative to the task-visible space;
- whether interaction energy materially changes the simple main-effect
  interpretation.

Do not infer an actual optimization path from these endpoint components.

## Functional authentication

For every frozen cell, authenticate final logits against the frozen functional
fingerprint under the same float32 tolerance used by the recovered
forward-Jacobian execution.

Stop if functional identity fails.

## Interpretation cases

### Strong task-visible collapse of A-init separation

Supported only if the absolute same-RNG/different-A distance and A-init
main-effect energy shrink strongly in low-k task-sensitive coordinates relative
to their full-space values, and the remaining separation is concentrated in
the orthogonal complement.

This would support the bounded statement that A-init selects substantially
different ambient representatives whose separation is mostly task-null or
low-gain at the tested layer-22 boundary.

### Persistent task-visible A-init separation

If A-init distance remains large relative to same-A/different-RNG controls even
in low-k task-sensitive coordinates, do not claim that A-init endpoints
collapse to one task-visible latent point.

The correct interpretation would be different task-visible representatives
that nevertheless remain nearly functionally equivalent at the final output.

### Mixed decomposition

If A-init separation is strongly reduced but not eliminated, report both the
dominant null/complement component and the residual structured task-visible
component without forcing a binary null/non-null interpretation.

## Required artifacts

Write only under:

`reports/reason_router_gen5_ainit_projection_clustering_runs/<run-name>/`

Required files:

- `ainit_projection_clustering_summary.json`
- `projection_clustering_metrics.pt`
- `run_provenance.json`

`projection_clustering_metrics.pt` may contain only compact per-example
distance/factorial-energy aggregates and fixed basis metadata required for
verification. It must not contain full hidden states, token-level hidden-state
dumps, gradients, or Jacobians.

## Runtime constraints

Use the same frozen Mamba snapshot, parent checkpoint, tokenizer/runtime, and
validated two-T4 Kaggle environment as the source evidence.

- model eval mode only;
- GPU 0 only within the validated 2xT4 environment;
- frozen-dev forward evaluation only;
- no autograd;
- no parameter gradients;
- no optimizer;
- no training;
- no checkpoint mutation;
- no confirmatory data;
- no output-conditioned k selection;
- preserve all parent/correction checkpoint identities.

## Scientific boundary

This audit can establish trained-endpoint projection geometry and factorial
variance allocation at the tested layer-22 boundary.

It cannot establish:

- an actual training trajectory;
- that A-init follows a distinct dynamical path;
- a universal null manifold;
- universal intrinsic task dimension;
- causal controllability of the projected components;
- behavior outside the frozen Phase3A P0 dev contract.

A causal visible-vs-null intervention or upstream precursor localization
requires a separate later authority after this audit is validated.
