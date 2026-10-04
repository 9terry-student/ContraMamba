# Gen5 A-init Internal Task-Visible Precursor Localization Audit Authority

SOURCE_CONFIRMATORY_EVIDENCE_FREEZE_COMMIT=1468938af9753fa9f4a511d4e7f740dea0110bba
SOURCE_DEV_CAUSAL_EVIDENCE_FREEZE_COMMIT=5694f962855bd2ab4f4035feb15cf1f4bfb3f784
SOURCE_FORWARD_JACOBIAN_RECOVERY_EVIDENCE_FREEZE_COMMIT=a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e
SOURCE_RESIDUAL_LOCALIZATION_EVIDENCE_FREEZE_COMMIT=3c0a3d8a67e9910f91de2354ba29a5c4b3b28942

STATUS=READY_FOR_INTERNAL_TASK_VISIBLE_PRECURSOR_LOCALIZATION

TRAINING_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
PARAMETER_GRADIENT_UPDATE_ALLOWED=NO
ANALYSIS_AUTOGRAD_ALLOWED=YES_DETACHED_INTERNAL_STAGE_LEAVES_ONLY
BACKWARD_METHOD_ALLOWED=TORCH_AUTOGRAD_GRAD_ONLY
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
CUDA_EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_P0_DEV_ONLY
TWO_GPU_EXECUTION_ALLOWED=YES_DETERMINISTIC_PAIR_SHARDING
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

## Scientific position

The A-init endpoint axis is now closed as a descriptive-and-causal endpoint
phenomenon.

The frozen one-shot confirmatory evidence established on a previously unseen
population that:

- A-init remains the dominant source of ambient layer-22 endpoint variation;
- most A-init residual energy lies outside the frozen compact task-sensitive
  subspace;
- the much smaller visible component carries essentially all measurable
  endpoint functional difference;
- the much larger complement is downstream effectively null / very low gain
  under the bounded endpoint intervention contract.

The earlier correction-residual localization already measured geometric
residual decay through:

`raw_write -> recurrent_state -> C_readout -> gated_scan -> layer22_out_proj -> final_logits`

and found the largest geometric reduction after the layer-22 correction output.

Therefore this stage MUST NOT repeat blockwise norm-decay localization.

The remaining mechanistic question is:

> At which internal correction transformation does the actual A-init residual
> first admit a stable task-visible-versus-complement causal decomposition?

This is an internal precursor question, not an upstream layer-0..21 precursor
question.

An upstream layer-0..21 A-init precursor search is prohibited because the
parent Mamba is frozen and the A-init-dependent trainable tensors first enter
inside the layer-22 correction. A first difference at layer 22 would be
structurally tautological.

## Frozen population and checkpoint grid

Use only the already consumed/frozen Phase3A P0 development population:

- rows: `840`
- split seed: `16384`
- arm: `G5-C0`
- pressure: `P0`
- valid-token target from frozen source evidence: `60094`

Use exactly the frozen 3x3 checkpoint grid:

- A-init seeds: `{6201,6202,6203}`
- training-RNG seeds: `{6201,6202,6203}`

Primary pair class:

- same training RNG;
- different A-init;
- exactly `9` unordered pairs.

Natural control pair class:

- same A-init;
- different training RNG;
- exactly `9` unordered pairs.

No new training seed, model seed, rank, checkpoint, data population, or
architecture is authorized.

The consumed `xg1_fact_9601..xg1_fact_9900` confirmatory population MUST NOT be
loaded.

## Exact internal stages

Use the exact frozen Phase2 correction semantics and localize only these ordered
stages:

1. `raw_write`
   - shape per token: `24576`
   - exact `B_theta(A_theta(x))` write for `G5-C0`
2. `recurrent_state`
   - shape per token: `24576`
   - exact correction recurrence state after the frozen discrete-A transition
3. `c_readout_pre_gate`
   - shape per token: `1536`
   - exact C-readout of the correction state before multiplication by the gate
4. `gated_scan`
   - shape per token: `1536`
   - exact C-readout after multiplication by the frozen native gate activation
5. `layer22_out_proj`
   - shape per token: `768`
   - exact correction contribution after the native mixer out-projection and
     before addition into the layer-22 mixer output

For `G5-C0`, `effective_write == raw_write`; do not create a duplicate
effective-write stage.

The native coefficient path producing discrete-A, C-readout coefficients, and
gate values is common across the 3x3 checkpoint grid for a fixed input because
the parent model and upstream computation are frozen.

## Exact stage replay requirement

For every dev stream chunk, compute the common frozen native coefficient path
once and replay the five correction stages for every frozen checkpoint.

The replay must authenticate against the already frozen correction-residual
localization semantics.

Required geometric cross-checks include the same-training-RNG / different-A
grouped normalized residuals at all five stages against the frozen source
artifact within a predeclared float32 replay tolerance.

The expected source means are:

- raw write: `0.459225933132`
- recurrent correction state: `0.235282854833`
- C readout before gate: `0.201189474462`
- gated correction scan: `0.139629099045`
- layer-22 out-projection contribution: `0.155009066845`

If the replay does not reproduce this chain, stop before precursor
interpretation.

## True-forward gradient semantics

Historical `edge_specific` backward semantics are NOT a mathematical forward
Jacobian and are prohibited for scientific sensitivity interpretation.

For analysis gradients only, use the recovered forward-equivalent `joint`
gradient-ownership semantics established by:

`a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e`

At every analysis boundary:

1. preserve the exact frozen forward value;
2. detach the selected internal stage tensor;
3. reintroduce it as an equal-valued leaf requiring gradient;
4. resume the exact frozen downstream computation from that stage;
5. evaluate final task logits with joint analysis-gradient semantics;
6. authenticate joint forward logits against historical edge-specific logits
   within a predeclared float32 tolerance.

No gradient may be requested with respect to model or correction parameters.

`.backward()` is prohibited.

Use only `torch.autograd.grad`.

## Task output coordinates

Use exactly the two nontrivial three-class margin coordinates:

`m_refute = logit_refute - logit_not_entitled`

`m_support = logit_support - logit_not_entitled`

For each source endpoint, dev chunk, and internal stage, obtain the exact
true-forward gradient rows:

`g_refute = grad(sum(m_refute), z_stage)`

`g_support = grad(sum(m_support), z_stage)`

Batch examples are independent; the summed-margin gradient therefore retains
the per-example gradient field without mixing examples.

## Actual pair residuals

For a frozen endpoint pair `(i,j)` at stage `s`:

`d_s = z_s(j) - z_s(i)`

Compute all scientific quantities at both endpoint references:

- source-local: Jacobian row space at endpoint `i`;
- target-local: Jacobian row space at endpoint `j` with residual `-d_s`.

Report both orientations and the symmetric grouped aggregate.

Do not select a favorable orientation.

## Exact local task-row-space projection

For each example at each stage, flatten the sequence-by-stage tensor only for
the purpose of the exact local two-row Jacobian calculation.

Let:

`J_s = [g_refute ; g_support]`

and let `P_s` be the Euclidean orthogonal projector onto `row(J_s)`.

Compute `P_s d_s` using only the 2x2 Gram system:

`P_s d_s = J_s^T (J_s J_s^T)^+ J_s d_s`

Use a deterministic float64 2x2 Gram/pseudoinverse calculation with a fixed
numerical tolerance recorded in provenance.

Do not materialize a full dense projector.

Define:

`d_visible_s = P_s d_s`

`d_complement_s = d_s - d_visible_s`

and the exact local row-space energy fraction:

`E_task_s = ||d_visible_s||^2 / ||d_s||^2`

Report numerator and denominator before division.

## Finite causal intervention at each stage

For every actual primary A-init pair and both endpoint orientations, perform
finite interventions at each internal stage.

At source endpoint `i`:

`z_visible = z_i + d_visible_s`

`z_complement = z_i + d_complement_s`

Resume the exact frozen downstream computation from stage `s` for:

1. real source stage `z_i`;
2. real target-consistent full residual `z_i + d_s`;
3. visible-only stage `z_visible`;
4. complement-only stage `z_complement`.

The full-residual replay must authenticate against the real target endpoint
logits within a predeclared float32 tolerance. If it does not, stop.

For centered three-class logits and the two-margin vector, accumulate:

- `E_full`
- `E_visible`
- `E_complement`
- `E_interaction`

where interaction is the residual:

`Delta_full - Delta_visible - Delta_complement`

Report:

- `R_visible = E_visible / E_full`
- `R_complement = E_complement / E_full`
- `R_interaction = E_interaction / E_full`

Also report prediction disagreement descriptively.

## Orientation controls

A two-output map from a high-dimensional internal state has a generically large
kernel. Therefore low `E_task_s` alone is not evidence of structured A-init
geometry.

For every actual residual, construct exactly eight deterministic
norm-preserving signed-permutation controls.

For each stage:

- preserve sequence position;
- flatten only the per-token feature dimensions;
- apply the same feature permutation and sign vector to all examples and token
  positions for one control index;
- preserve every token-vector norm exactly;
- derive permutation/sign seeds only from SHA256 of:
  `GEN5_INTERNAL_PRECURSOR_V1|<stage>|<control_index>`;
- use no training, evaluation, confirmatory, A-init, or RNG-seed namespace.

No final-logit outcome may influence control construction.

Using the already computed Jacobian rows, report for each control:

- directional gain `||J d_control|| / ||d_control||`;
- local task-row-space energy fraction;
- actual/control ratios.

No additional model forward is required for orientation controls.

## Out-projection authentication against frozen Jacobian recovery

At `layer22_out_proj`, the checkpoint-to-checkpoint correction-contribution
residual equals the layer-22 wrapper-output residual because the frozen native
mixer contribution is common across cells.

Therefore the `layer22_out_proj` local Jacobian audit must reproduce the frozen
true-forward layer-22 boundary nullness evidence within a predeclared numerical
tolerance.

Required authentication targets include the same-training-RNG/different-A
grouped values from the recovered forward-Jacobian evidence:

- directional gain: `0.00030510912159`
- local task-row-space squared-energy fraction: approximately
  `0.00415859942285`
- actual/signed-permutation directional-gain ratio: approximately `9.8220`
- actual/signed-permutation row-energy ratio: approximately `57.1938`

If this authentication fails, no internal precursor conclusion is allowed.

## Internal precursor criterion

The primary localization object is NOT raw residual magnitude.

For each ordered internal stage, define the stage as having a
`STABLE_FINITE_TASK_VISIBLE_DECOMPOSITION` only if, for the primary
same-training-RNG/different-A group, both centered logits and two-margin
readouts satisfy the symmetric grouped criteria:

- `0.60 <= R_visible <= 1.40`
- `R_complement <= 0.05`
- `R_interaction <= 0.05`

and the local row-space energy of the actual residual is enriched over the mean
of its eight signed-permutation controls by at least:

`E_task_actual / E_task_control_mean >= 5`

The **internal task-visible precursor stage** is the earliest stage in the fixed
order:

`raw_write -> recurrent_state -> c_readout_pre_gate -> gated_scan -> layer22_out_proj`

that satisfies all of the above criteria.

No stage may be skipped or reordered.

No threshold may be changed after execution.

If no stage satisfies the criterion, report:

`NO_STABLE_INTERNAL_TASK_VISIBLE_PRECURSOR_LOCALIZED`

and do not rescue the result by changing thresholds, pair subsets, controls, or
stage definitions.

## Interpretation cases

### RAW_WRITE_PRECURSOR

If `raw_write` is the earliest passing stage, the small task-visible component
is already present in the learned correction write itself. Later Mamba
recurrence/readout operations primarily filter or reshape the much larger
low-gain complement rather than creating task visibility de novo.

### RECURRENCE_PRECURSOR

If `recurrent_state` is earliest, the state recurrence is the first tested
operation at which a stable finite visible/complement causal decomposition
emerges.

### C_READOUT_PRECURSOR

If `c_readout_pre_gate` is earliest, the C readout is the first tested
operation that concentrates the functional component sufficiently.

### GATE_PRECURSOR

If `gated_scan` is earliest, gate multiplication is the first tested
local transformation yielding the stable functional decomposition.

### OUT_PROJ_PRECURSOR

If `layer22_out_proj` is earliest, the correction remains causally entangled
until the 768-dimensional mixer-output contribution.

### NO_LOCALIZED_PRECURSOR

If no stage passes, the endpoint decomposition is not attributable to a stable
single local correction boundary under this audit.

## Two-GPU execution contract

Use the validated Kaggle `2x Tesla T4` environment and use both GPUs for
scientific execution.

Shard complete pair-orientation work items deterministically before execution.

Each pair orientation and all five stages for that orientation must remain on a
single GPU.

Do not split one pair orientation across GPUs.

No cross-GPU scientific tensor reduction is permitted during an intervention.

Only pair/stage sufficient statistics may be merged on CPU after workers
complete.

The exact deterministic shard manifest must be written to provenance before
scientific metrics are interpreted.

Two-GPU use is runtime acceleration only and must not alter estimator semantics.

## Required outputs

Write only under:

`reports/reason_router_gen5_ainit_internal_precursor_localization_runs/<run-name>/`

Required files:

- `ainit_internal_precursor_localization_summary.json`
- `internal_stage_task_visible_metrics.pt`
- `run_provenance.json`

Do not persist:

- full hidden-state tensors;
- full correction-state tensors;
- full per-example gradient tensors;
- full Jacobian matrices.

Persist only pair/stage sufficient statistics, fixed control identities, gate
results, and provenance needed for exact interpretation.

## Required guards

All scientific outputs must record and require:

- `training_executed = false`
- `optimizer_constructed = false`
- `parameter_gradients_accumulated = false`
- `backward_method_called = false`
- `checkpoint_mutation = false`
- `confirmatory_9601_9900_loaded = false`
- `joint_analysis_gradient_semantics = true`
- `edge_joint_forward_authentication_pass = true`
- `outproj_recovery_authentication_pass = true`
- `residual_chain_authentication_pass = true`

Autograd execution on detached internal analysis leaves is expected and must be
recorded as such.

## Stop conditions

Stop before scientific interpretation if:

- any frozen evidence identity mismatches;
- parent or correction checkpoint identity mismatches;
- dev row order or encoding identity mismatches;
- the 3x3 grid is incomplete;
- the exact stage replay fails residual-chain authentication;
- joint and edge-specific forward logits differ outside tolerance;
- `layer22_out_proj` fails recovered-Jacobian authentication;
- a full-residual finite replay fails to reproduce the real target endpoint;
- a parameter gradient is requested or accumulated;
- `.backward()` is called;
- an optimizer is constructed;
- training or checkpoint mutation occurs;
- confirmatory 9601-9900 data are accessed;
- a pair orientation is split across GPUs;
- an output collision exists.

## Scientific boundary

This audit can localize the earliest tested internal layer-22 correction stage
at which the actual A-init residual admits a stable finite
task-visible/complement causal decomposition under the frozen Phase3A P0 dev
contract.

It may support a bounded mechanism statement about whether task visibility is
already encoded in the learned write or becomes concentrated by recurrence,
C-readout, gating, or out-projection.

It does NOT establish:

- an exact gauge symmetry;
- a universal Mamba state-space law;
- a global null manifold;
- arbitrary-perturbation invariance;
- an optimization-trajectory mechanism;
- a nontrivial precursor in frozen upstream layers 0..21;
- behavior outside the frozen task and model contract;
- Transformer inferiority or Mamba superiority.

A successful localization may motivate a later separately controlled
architecture comparison, but no architecture-comparative conclusion is
authorized here.
