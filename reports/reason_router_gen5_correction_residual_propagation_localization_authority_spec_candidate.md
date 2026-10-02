# Gen5 Correction Residual Propagation Localization Authority

SOURCE_QUOTIENT_EVIDENCE_FREEZE_COMMIT=0e6191fd54e23388abcce1abd9e01a453d2dc73c
SOURCE_QUOTIENT_EXECUTION_COMMIT=c731270221c4e0e131fb68173f40bf0ad8a2bfdd
SOURCE_FUNCTIONAL_EQUIVALENCE_EVIDENCE_COMMIT=5f079a66f7b0eb0caea30a8d5bc9a0fe757cc449
SOURCE_FUNCTIONAL_FINGERPRINT_EXECUTION_COMMIT=17a783c61e310d6298d4105233ea3f1b71ec8add

STATUS=READY_FOR_CORRECTION_RESIDUAL_PROPAGATION_LOCALIZATION

TRAINING_ALLOWED=NO
BACKWARD_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
CUDA_EVALUATION_ALLOWED=YES_ONE_FROZEN_PHASE3A_DEV_FORWARD_PLUS_OFFLINE_CORRECTION_REPLAY
MODEL_FORWARD_ALLOWED=YES_ONE_FROZEN_PHASE3A_DEV_FORWARD
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

## Scientific question

The frozen quotient audit established that same-training-RNG / different-A-init
pairs have a mean task-restricted raw correction-action normalized residual of:

`0.459227092986`

despite near-identical final correction-induced dev-logit fingerprints.

The present localization asks:

> At which frozen operation does the remaining different-A correction residual
> undergo the dominant collapse required to reconcile the layer-22 write
> difference with the near-identical final function?

## Exact population and source grid

Use only the frozen Phase3A dev set:

DEV_ROWS=840
SPLIT_SEED=16384
ARM=G5-C0
PRESSURE=P0

Use exactly the frozen 3x3 checkpoint grid over:

A_INIT_SEED in {6201,6202,6203}
TRAINING_RNG_SEED in {6201,6202,6203}

Authenticate all source checkpoint and provenance identities against the frozen
evidence commits.

No new seed, training population, confirmatory population, or checkpoint may be
introduced.

## One-forward state capture

Construct the exact frozen parent model and layer-22 wrapper with correction B
exactly zero.

Run exactly one eval-mode, no-grad forward over the frozen 840-row Phase3A P0
dev set.

Capture the exact `hidden_states` entering
`Phase2Layer22MixerWrapper.forward` together with the effective attention mask
for each of the four frozen stream chunks.

These layer-22 inputs are common to all nine correction cells because the
correction first intervenes inside layer 22.

No second full-model forward is authorized.

## Offline exact correction replay

After the one forward, replay only the already-frozen layer-22 correction branch
for each of the nine frozen A/B checkpoints using the captured common
layer-22 inputs.

The replay must reproduce the exact frozen streaming semantics:

1. `raw_write = B A x`
2. recurrent correction state:
   `state_t = discrete_A_t * state_(t-1) + raw_write_t`
3. pre-gate C readout:
   `read_t = sum_state(state_t * C_t)`
4. gated correction scan:
   `scan_t = read_t * act(gate_t)`
5. layer-22 correction contribution:
   `out_t = out_proj(scan_t)`

The frozen coefficient path producing `discrete_A_t`, `C_t`, and `gate_t` is
computed from the common captured layer-22 input using the exact native mixer
weights and frozen implementation semantics.

The final functional endpoint is not recomputed with additional model forwards.
Reuse the already frozen `cell_fingerprints.pt` from the validated functional
fingerprint execution.

## Replay semantic authentication

For at least one frozen cell and one captured stream chunk, compare the
localization replay's layer-22 correction contribution against the existing
frozen `_streaming_correction_impl` using the same input, attention mask, A,
and B tensors.

Require strict numerical agreement under a predeclared implementation-check
tolerance appropriate for float32 replay.

This comparison is an implementation-semantic guardrail only, not a scientific
endpoint.

## Stagewise sufficient statistics

Do not persist full hidden-state, correction-write, recurrent-state, or scan
tensors.

For each stage, accumulate only the exact 9x9 inner-product Gram matrix across
all valid dev-token positions and all stage dimensions.

Stages:

- `raw_write`
- `recurrent_state`
- `c_readout_pre_gate`
- `gated_scan`
- `layer22_out_proj`
- `final_delta_logits`
- `final_centered_delta_logits`

The first five stages use valid dev tokens only.

The final two stages use the frozen 840-example functional-fingerprint artifact.

## Primary pairwise metrics

For every unordered cell pair and every stage, derive from the accumulated Gram:

- cosine;
- normalized residual

`R_s(i,j) =
 ||v_s(i)-v_s(j)|| /
 sqrt(0.5 * (||v_s(i)||^2 + ||v_s(j)||^2))`

Primary factor groups:

1. same A-init, different training RNG;
2. same training RNG, different A-init.

Report all raw pairwise values plus grouped means and ranges.

## Residual survival localization

For every pair and each successive stage, define:

`S_step(s) = R_s / R_previous`

and relative to raw write:

`S_raw(s) = R_s / R_raw_write`

Report grouped means and ranges.

For the same-training-RNG / different-A-init group, define the descriptive
dominant-collapse stage as the transition with the smallest grouped mean
`S_step`.

This is a descriptive localization label only. No post-hoc significance
threshold or p-value is authorized.

## Interpretation cases

### Recurrence-dominant collapse

If the dominant residual reduction occurs from `raw_write` to
`recurrent_state`, the frozen Mamba state dynamics suppress the
initialization-specific write difference before readout.

### C-readout-dominant collapse

If the dominant reduction occurs from `recurrent_state` to
`c_readout_pre_gate`, distinct correction-state trajectories are strongly
identified by the same observable readout.

### Gate/out-projection collapse

If the dominant reduction occurs at `gated_scan` or `layer22_out_proj`, the
local mixer output map is the main local quotient.

### Downstream collapse

If substantial residual survives through `layer22_out_proj` but collapses only
at `final_delta_logits` or `final_centered_delta_logits`, the near-equivalence
is primarily produced by post-layer-22 propagation and task readout.

Mixed reductions across multiple stages must be reported as mixed rather than
forced into a single mechanistic label.

## Required artifacts

Write only under:

`reports/reason_router_gen5_correction_residual_localization_runs/<run-name>/`

Required files:

- `correction_residual_localization_summary.json`
- `stage_inner_product_grams.pt`
- `run_provenance.json`

No existing artifact may be overwritten.

## Runtime constraints

Use the same frozen Mamba snapshot, parent checkpoint, tokenizer/runtime, and
validated two-T4 Kaggle environment as the source runs.

Only GPU 0 is required.

The localization must:

- use `model.eval()`;
- use `torch.no_grad()`;
- perform exactly one full-model frozen-dev forward;
- perform only offline frozen correction replay afterward;
- construct no optimizer;
- call no backward;
- perform no training step;
- mutate no checkpoint;
- preserve parent parameter identity;
- load no confirmatory data.

## Stop conditions

Stop if:

- source evidence identity mismatches;
- the 3x3 checkpoint grid is incomplete;
- parent identity mismatches;
- frozen dev encoding or row order mismatches;
- layer-22 capture semantics differ from the frozen wrapper input;
- offline replay fails the frozen-backend semantic check;
- more than one full-model forward would be required;
- training, backward, or optimizer construction would occur;
- confirmatory data would be accessed;
- an output collision exists.

## Result boundary

This localization can identify where correction-specific residual differences
are suppressed under the frozen Phase3A P0 dev contract.

It does not establish exact global functional equivalence, a universal Mamba
filtering law, a formal gauge symmetry, or behavior outside the tested
population and configuration.
