# Gen5 Forward-Jacobian Gradient-Semantics Recovery Authority

SOURCE_TASK_SENSITIVITY_EVIDENCE_FREEZE_COMMIT=ed196d5003f279dbfe1dc9a631e84e22dc049ac7
SOURCE_TASK_SENSITIVITY_EXECUTION_COMMIT=b8b1a5e95c7932df2c0319e766d10b10f19b3081
SOURCE_INTERPOLATION_AUTHORITY_COMMIT=c356085720b855e3dd361ecaf0e829184e03d9ff
DIAGNOSTIC_INTERPOLATION_RUN=gen5-ainit-latent-interpolation-c356085-r1

STATUS=READY_FOR_TRUE_FORWARD_JACOBIAN_SEMANTIC_RECOVERY

TRAINING_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
PARAMETER_GRADIENT_UPDATE_ALLOWED=NO
ANALYSIS_AUTOGRAD_ALLOWED=YES_LAYER22_BOUNDARY_ONLY
BACKWARD_METHOD_ALLOWED=TORCH_AUTOGRAD_GRAD_ONLY
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
CUDA_EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_P0_DEV_ONLY
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

## Defect

The frozen historical evaluator uses:

`gradient_ownership_mode = edge_specific`

for the parent arm:

`G3-GROUP-D-HALF`

The frozen edge map has:

- `F_TO_D = 0.5`
- `P_TO_D = 0.5`
- `S_TO_D = 0.5`
- `Q_TO_D = 0.5`

while upstream ownership edges are `1.0`.

The model implements these ownership edges with a partial-gradient alias that
preserves the forward tensor value while scaling the backward derivative.

Therefore autograd through the historical `edge_specific` evaluation path does
not, by construction, necessarily equal the mathematical Jacobian of the
frozen forward map.

The prior task-sensitivity audit and the diagnostic interpolation execution
used the historical edge-specific backward path while interpreting it as the
forward Jacobian.

## Diagnostic witness

The diagnostic interpolation run authenticated all frozen forward endpoints,
but for same-training-RNG / different-A pairs reported:

- endpoint centered-margin distance mean:
  `0.0278227543645`
- sampled directional total variation mean:
  `0.0139188154129`
- quadrature residual mean:
  `0.013903939981`

A true path derivative must satisfy the fundamental theorem of calculus, and
the norm of the endpoint change cannot exceed directional total variation.

The approximately one-half scale is consistent with the frozen D-half
gradient-ownership contract.

The diagnostic interpolation run MUST NOT be collected or frozen as scientific
evidence.

## Recovery goal

Recover the true downstream forward Jacobian from the layer-22 boundary while
preserving the exact frozen forward function.

Use the same model parameters, inputs, masks, checkpoint identities, and
decision-mode forward computation.

For analysis-gradient calls only, evaluate the downstream head with:

`gradient_ownership_mode = joint`

and no edge-gradient lambda map.

Because ownership aliases are forward-value preserving, the recovery must
authenticate that `joint` and historical `edge_specific` produce identical
forward logits within a predeclared float32 tolerance.

If forward equality fails, stop.

## Required semantic authentication

On the frozen Phase3A P0 dev population and exact 3x3 cell grid:

1. authenticate historical edge-specific forward logits against the frozen
   functional-fingerprint evidence;
2. authenticate joint-mode forward logits against those same historical logits;
3. compute layer-22 boundary gradients under both modes;
4. report the elementwise / directional relationship between the two gradient
   fields;
5. do not assume a uniform factor of `0.5` unless it is numerically verified.

Required summary diagnostics include:

- maximum joint-vs-edge forward-logit absolute difference;
- cosine between joint and edge gradient fields;
- norm ratio `||g_edge|| / ||g_joint||`;
- maximum residual from the best scalar relation;
- per-margin and grouped summaries.

## Recovered task-sensitivity audit

Repeat the frozen A-init residual nullness / task-sensitive-subspace audit using
the `joint` analysis-gradient path.

Retain exactly the prior:

- 840 frozen dev rows;
- 3x3 A-init x training-RNG grid;
- layer-22 wrapper-output boundary;
- two centered-logit margin coordinates;
- same-A/different-RNG natural controls;
- eight frozen signed-permutation orientation controls;
- fixed hidden-subspace k values.

The recovered audit must report:

- true directional downstream gain;
- true local task-row-space residual-energy fraction;
- true 768x768 task-sensitive covariance and spectrum;
- residual projection into the leading task-sensitive hidden directions;
- actual/control contrasts.

The recovery must explicitly compare each invariant or changed metric with the
prior frozen task-sensitivity artifact.

## Interpretation boundary for prior frozen evidence

Until recovery completes:

- the prior absolute directional-gain values are NOT validated as true forward
  Jacobian gains;
- the prior row-space, normalized spectrum, subspace, and actual/control ratio
  conclusions may be retained only as hypotheses expected to be invariant
  under a uniform positive scalar gradient transformation;
- that invariance must be demonstrated, not assumed.

The provenance and forward functional authentication of the prior frozen run
remain valid.

## Interpolation recovery boundary

Do NOT rerun the full interpolation/path-integral audit in the same execution.

First recover the true task-sensitive geometry.

After that recovery is validated, decide whether a corrected interpolation
path-integral execution is still scientifically necessary.

The already observed forward-only interpolation values may be treated as
diagnostic evidence only until a clean recovery execution is completed.

## Required artifacts

Write only under:

`reports/reason_router_gen5_forward_jacobian_recovery_runs/<run-name>/`

Required files:

- `forward_jacobian_recovery_summary.json`
- `recovered_task_sensitive_subspace.pt`
- `run_provenance.json`

Do not persist full hidden states or full per-example gradients.

## Stop conditions

Stop if:

- source identity mismatches;
- parent/correction checkpoint identity mismatches;
- frozen dev row or encoding identity mismatches;
- endpoint forward authentication fails;
- joint-vs-edge forward values differ outside tolerance;
- analysis autograd touches parameters;
- `.backward()` is called;
- an optimizer is constructed;
- training or checkpoint mutation occurs;
- confirmatory data is accessed;
- an output collision exists.

## Result boundary

This recovery can establish the true local forward-Jacobian geometry at the
layer-22 boundary under the frozen Phase3A P0 dev contract and determine which
conclusions from the prior task-sensitivity audit survive the gradient-semantics
correction.

It does not establish a global latent manifold, latent controllability,
precursor dynamics, or behavior outside the frozen dev contract.
