# O0c Selective-SSM Native Recurrent-State Instrumentation Implementation Authority Candidate

## 1. Status, phase, and authority

**Phase:** STATIC IMPLEMENTATION-AUTHORITY RECOVERY AUTHORING ONLY.

**Candidate verdict:** PASS_READY_FOR_INDEPENDENT_VERIFICATION.

This recovery repairs only the independently identified token-index and artifact-schema defects. It is one bounded candidate implementation authority, not implementation, execution, or scientific evidence. SCIENTIFIC_CONCLUSION: NONE.

Authority precedence is: (1) current research-controller recovery instruction; (2) required HEAD cb529070cc68921fd917e99866bc29de9891751e and corrected-preflight validated evidence; (3) frozen native-state design authority commit 242ad9ed70fc995ebda560911a7d0dfd2f18f9b3, reports/longterm_o0c_selective_ssm_native_state_instrumentation_authority_spec_candidate.md; (4) accepted prerequisite verdict PASS_SAFE_TO_CONSUME_O0C_NATIVE_STATE_INSTRUMENTATION_AUTHORITY; then (5) AGENTS.md. A later conflict stops work.

This phase authorizes no observer/test implementation, staging, commit, push, model/tokenizer/dataset loading or revalidation, Transformers forward, generation, training, evaluation, Kaggle, package mutation, or preflight rerun.

## 2. Exact future implementation scope

After independent verification and later explicit implementation authorization, only these paths may be created or modified:

- scripts/observe_longterm_o0c_selective_ssm_native_state_dynamics.py
- tests/test_observe_longterm_o0c_selective_ssm_native_state_dynamics.py

No other file, installed package, data, model, report, or preflight code is in scope. A need for any other path is BLOCKED_SCOPE_OR_RECOVERY_AUTHORITY_REQUIRED.

## 3. Frozen runtime/source basis

Before any model/tokenizer load or scientific observation, fail closed unless Python is 3.12.13, NumPy 2.0.2, torch 2.10.0+cpu, Transformers 5.0.0, source resolution is PASS_RECONCILED_UNIQUE_TRANSFORMERS_SOURCE, and backend classification is BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN.

The required Mamba source is transformers.models.mamba.modeling_mamba, SHA256 4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83, bytes 39500. The required cache source is transformers.cache_utils, SHA256 6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc, bytes 60432. The capture location is MambaMixer.slow_forward at ordinary readout line 410.

These are source/root/byte/role/dispatch gates, not version-string gates. Reject shadowed roots, alternate source, byte mismatch, symbol ambiguity, line-role drift, unresolved code-object binding, CPU-dispatch ambiguity, cache/recurrent role reversal, or unavailable line event. No package install, patch, vendoring, copying, optimized-kernel fallback, or token-by-token replay is allowed.

The validated sequential source has fresh recurrence initialization at lines 375--378, state update at 409, readout at 410, final cache persistence at 417, and MambaMixer.forward dispatch at 424--436. Convolution cache is distinct.

## 4. Observation and non-interference contract

The only mechanism is a process-local, default-disabled Python frame-line observer bound to the validated MambaMixer.slow_forward code object and source roles. It uses Python 3.12.13 tracing only while an enabled observer forward runs. It must not replace, wrap, alter, or monkey-patch forward methods, hooks, cache methods, tensors, parameters, buffers, package files, inputs, or outputs.

At a line event for that code object at line 410, it must identify one registered eligible mixer; read only local ssm_state and the zero-based loop index; require an unambiguous valid index; copy exactly ssm_state.detach().clone(); and retain an observer-owned clone keyed by (forward_id, layer_identity, absolute_token_index). Line 410 occurs post-line-409 update and pre-readout consumption, so the clone is native recurrent s_t after x_t, not pre-update state or hidden/output state.

The clone is never written to frame locals, cache, module, input/output, parameter, buffer, or autograd path, and is copied again for artifact-owned CPU serialization. Restore the prior trace in finally and discard registrations. Disabled capture installs no trace and allocates no snapshot collection.

Tier 1 is only local ssm_state in this frame. Hidden states, last_hidden_state, convolution state/cache, final cache entry, scan workspace, and reconstructed substitute states are forbidden. Validate rank, nonempty dimensions, CPU float32 metadata, finiteness, and recurrent-local identity; ambiguity/nonfinite state blocks the member.

A mandatory local synthetic equivalence harness compares otherwise-identical capture-disabled/enabled forwards and requires torch.equal plus exact metadata equality for ordinary outputs, last_hidden_state, requested hidden states, shape, dtype, device, and output structure. It also requires unchanged parameters/requires_grad, no unauthorized buffer/cache mutation, deterministic keys, no observer feedback, and clone non-aliasing. Capture-disabled runs have tracing off. Exact equality cannot be weakened; failure is BLOCKED_NEW_PRE_EXECUTION_AUTHORITY_REQUIRED.

## 5. Frozen token, state, transition, and anchor contract

For serialized sequence:

~~~text
x_0, x_1, ..., x_T
~~~

T always means the **last valid absolute token index**, never token count. Everywhere:

~~~text
token_count = T + 1                         # number of serialized tokens
valid absolute token indices = {0, 1, ..., T}
s_-1 = fresh zero initial recurrent SSM state for that same member/layer forward
s_t = recurrent state immediately after consuming x_t
captured full-sequence states = s_0, s_1, ..., s_T
terminal_index = T = token_count - 1
anchor_terminal = s_T
transition_t = s_t - s_(t-1)
transition_0 = s_0 - s_-1
~~~

For every eligible Mamba layer and independently forwarded matched member, capture exactly one post-update snapshot at each t in [0, T] inclusive: exactly token_count = T + 1 captured states. Terminal is captured and last.

The mandatory completeness invariant, per eligible (pair_id, condition, layer_index), is:

~~~text
captured_indices == list(range(token_count))
~~~

Reject duplicate/missing token index, terminal_index not equal to token_count - 1, or last captured-state index not equal to terminal_index.

Anchors only, and in this exact order, are d-1, d, d+1, d+2, d+4, terminal T: anchor_pre_minus_1, anchor_divergence, anchor_post_plus_1, anchor_post_plus_2, anchor_post_plus_4, anchor_terminal. An unavailable anchor stays unavailable: no substitution. At d-1 require allclose(s_(d-1)^reference, s_(d-1)^member, rtol=0.0, atol=1e-6) for every comparison/layer; earlier transition history is under the same guard. Mismatch is failure, not signal.

## 6. Inherited controls and forward policy

Preserve model/tokenizer state-spaces/mamba-130m-hf at revision 5708daa364c50b880e7bd92eab456e0d34492ee9; CPU float32; eval, frozen parameters, inference-only; add_special_tokens=false; trust_remote_code=false; dataset data/longterm_o0b_matched_controls_v1.jsonl SHA256 75a675bee49cb26eb0935d364f0f5d090922dd01576dfc23294961b28394aec2; validation artifact reports/longterm_o0b_matched_controls_v1_validation.json SHA256 e8344ea3df54a3393aa8fa82dba19eb2baade9af9366687bb105f4ad348979ff; and serialization Claim: <claim>\nEvidence: <evidence>.

PAIR_ORDER is o0b_pair_001, o0b_pair_002, o0b_pair_003. CONDITION_ORDER is reference_sufficient, insufficient_matched, paraphrase_sufficient, surface_null_matched. There are exactly 12 independently initialized full-sequence forwards. Every layer/member starts fresh; no state, cache, or snapshot reuse. All eligible Mamba layers capture; no post-hoc layer, anchor, token, or member choice.

## 7. Frozen publication bundle and transaction

Schema version is longterm_o0c_selective_ssm_native_state_dynamics_v1. Serialization is canonical-json-v1/deterministic-npz-v1. A successful output is one collision-protected directory with exactly these seven files, in order:

1. manifest.json
2. state_rows.jsonl
3. full_recurrent_states.npz
4. paired_measurements.jsonl
5. summary.json
6. report.md
7. SHA256SUMS.txt

No eighth scientific artifact, alternate CSV/Parquet/HDF5 representation, pickle, overwrite, or merge is permitted. Build the entire bundle in a staging directory, validate it fully, and atomically rename/publish only if final target does not exist. Final-output and staging-directory collisions both fail closed.

Canonical JSON/JSONL is UTF-8, LF-only, final-LF, sorted object keys, no NaN/Infinity, and no volatile timestamp/hostname/username/UUID/branch-name scientific identity. JSONL is exactly one canonical JSON object per LF-terminated line.

### 7.1 full_recurrent_states.npz

This is complete Tier-1 trajectory storage, not anchors only. It has exactly one scientific array, vectors; extra members are rejected. vectors is finite, C-contiguous, two-dimensional little-endian float32 (<f4), shape [R,D]. Each row is exactly one (pair_id, condition, layer_index, absolute_token_index); D is the flattened native recurrent tensor element count. Each source is copied from ssm_state.detach().clone() before serialization. Original tensor shapes are in state_rows.jsonl and must reconstruct exactly.

~~~text
R = sum_over_all_members(token_count * eligible_layer_count)
~~~

Row order is exactly PAIR_ORDER, CONDITION_ORDER, layer_index ascending, then absolute_token_index ascending 0 through terminal_index inclusive. vector_index is the zero-based row number.

Follow the frozen O0b deterministic NPZ precedent: no pickle; deterministic NPY payload; deterministic ZIP member metadata/order; one array only; parser rejection of extra members, object/pickle arrays, wrong endian/dtype, nonfinite values, malformed arrays, and malformed/nondeterministic container structure where applicable. No alternative representation choice remains.

### 7.2 state_rows.jsonl

Exactly one canonical row per NPZ vector. Its exact field set, no extras:

~~~text
schema_version
pair_id
condition
token_count
terminal_index
absolute_token_index
layer_index
layer_role
state_source
state_timing
source_module
source_qualname
source_line
vector_index
tensor_shape
flattened_size
dtype
device
~~~

Fixed values: schema_version=longterm_o0c_selective_ssm_native_state_dynamics_v1; state_source=native_selective_ssm_recurrent_state; state_timing=post_consumption_s_t; source_module=transformers.models.mamba.modeling_mamba; source_qualname=MambaMixer.slow_forward; source_line=410; dtype=float32; device=cpu. vector_index maps one-to-one to vectors[vector_index], tensor_shape/flattened_size agree with it, token_count is number of serialized tokens, terminal_index=token_count-1, and 0 <= absolute_token_index <= terminal_index. No duplicate coordinates.

### 7.3 paired_measurements.jsonl

Only frozen Tier-1/Tier-2 measurements reconstructable from full_recurrent_states.npz appear. Tier-3 SSM-A/B/C/Delta diagnostics are excluded. Every canonical row has exactly these fields, no extras:

~~~text
schema_version
pair_id
comparison_id
reference_condition
member_condition
anchor_name
absolute_token_index
layer_index
reference_vector_index
member_vector_index
reference_previous_vector_index
member_previous_vector_index
normalized_l2_state_distance
reference_transition_l2
member_transition_l2
paired_transition_delta
transition_direction_cosine
pre_divergence_integrity_status
~~~

Comparison order: comparison-A (insufficient_matched vs reference_sufficient), comparison-B (paraphrase_sufficient vs reference_sufficient), comparison-C (surface_null_matched vs reference_sufficient). Anchor order is section 5.

Each value must reconstruct exactly from referenced rows. For t>0 previous indices identify s_(t-1). For t=0 both previous-index values are JSON null, and reconstruction uses that member/layer forward's exact zero s_-1; no other sentinel. Normalized-state zero norm, a zero transition norm where cosine is required, or any nonfinite required value fails closed. No substituted metric.

### 7.4 summary.json

Top-level exact fields:

~~~text
schema_version
experiment_name
measurement_order
comparison_order
anchor_order
rows
~~~

schema_version is the frozen schema version; experiment_name=longterm_o0c_selective_ssm_native_state_dynamics; measurement_order=[normalized_l2_state_distance, paired_transition_delta, transition_direction_cosine]; comparison_order=[comparison-A, comparison-B, comparison-C]; anchor_order is section 5 order.

Each row is keyed only by layer_index, layer_role, anchor_name, measurement_name and contains exactly:

~~~text
a_available_pair_ids
a_available_pair_count
a_mean
a_median
b_available_pair_ids
b_available_pair_count
b_mean
b_median
c_available_pair_ids
c_available_pair_count
c_mean
c_median
a_gt_b_comparable_pair_ids
a_gt_b_denominator
a_gt_b_count
a_gt_c_comparable_pair_ids
a_gt_c_denominator
a_gt_c_count
~~~

No significance test, learned threshold, best-layer/anchor field, heterogeneous aggregate, or unlisted summary field.

### 7.5 manifest.json

Its exact required top-level key contract is:

~~~text
schema_version
experiment_name
scientific_design_authority_commit
implementation_authority_commit
observer_implementation_commit
observer_script_path
observer_script_sha256
observer_script_bytes
dataset_path
dataset_sha256
validation_artifact_path
validation_artifact_sha256
model_id
model_revision
tokenizer_id
tokenizer_revision
model_trust_remote_code
tokenizer_trust_remote_code
add_special_tokens
device
dtype
expected_python_version
observed_python_version
expected_numpy_version
observed_numpy_version
expected_torch_version
observed_torch_version
expected_transformers_version
observed_transformers_version
transformers_distribution_root
transformers_import_root
source_resolution_classification
backend_classification
mamba_source_module
mamba_source_sha256
mamba_source_bytes
cache_source_module
cache_source_sha256
cache_source_bytes
capture_source_qualname
capture_source_line
capture_state_source
capture_state_timing
pair_order
condition_order
comparison_order
anchor_order
layer_descriptors
serialization_template
exact_command
run_name
required_artifacts
equivalence_gate_status
capture_completeness_status
provenance_status
execution_status
blocker
~~~

Frozen constants: schema_version=longterm_o0c_selective_ssm_native_state_dynamics_v1; experiment_name=longterm_o0c_selective_ssm_native_state_dynamics; scientific_design_authority_commit=242ad9ed70fc995ebda560911a7d0dfd2f18f9b3; observer_script_path=scripts/observe_longterm_o0c_selective_ssm_native_state_dynamics.py; model_id=tokenizer_id=state-spaces/mamba-130m-hf; model_revision=tokenizer_revision=5708daa364c50b880e7bd92eab456e0d34492ee9; model_trust_remote_code=false; tokenizer_trust_remote_code=false; add_special_tokens=false; device=cpu; dtype=float32; expected versions 3.12.13, 2.0.2, 2.10.0+cpu, 5.0.0; source_resolution_classification=PASS_RECONCILED_UNIQUE_TRANSFORMERS_SOURCE; backend_classification=BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN; Mamba/cache identity is section 3; capture_source_qualname=MambaMixer.slow_forward; capture_source_line=410; capture_state_source=native_selective_ssm_recurrent_state; capture_state_timing=post_consumption_s_t; serialization_template=canonical-json-v1/deterministic-npz-v1; required_artifacts is the seven-file section-7 list in order.

implementation_authority_commit equals the eventual commit freezing this authority and is never unknown/placeholder during scientific execution. observer_implementation_commit is a full 40-lowercase-hex commit; observer_script_sha256 is lowercase 64-hex; observer_script_bytes is positive integer. Observed versions are exact nonempty strings; roots are absolute nonempty normalized paths reconciled by runtime gate; exact_command/run_name are nonempty deterministic strings; layer_descriptors is the complete ordered descriptor array. Gate statuses are required nonempty closed-vocabulary status strings; blocker is nonempty when execution is unsuccessful and JSON null only when execution succeeds. No required provenance value can be unknown, n/a, empty, or null except that expressly permitted blocker null. Validator rejects key/schema/type/value drift.

### 7.6 report.md and SHA256SUMS.txt

report.md is deterministic rendering only of validated manifest, measurements, and summary. It introduces no extra measurement/conclusion and retains SCIENTIFIC_CONCLUSION: NONE.

SHA256SUMS.txt has exactly six lines covering every artifact except itself, lower-case SHA256, LF-only/final-LF, and syntax <64 lowercase hex><two spaces><filename>\n. Filenames are, in required-artifact order: manifest.json, state_rows.jsonl, full_recurrent_states.npz, paired_measurements.jsonl, summary.json, report.md. Reject any duplicate/missing/extra line or filename.

## 8. Required future local/synthetic tests

Only a later implementation-validation turn can implement tests; no network, Hugging Face download/loading, scientific dataset/model forward, generation, Kaggle, training, or evaluation. In addition to source binding, fresh-state, hidden/cache rejection, clone nonaliasing, provenance, and exact-equivalence tests, prove:

1. for synthetic x_0..x_T, token_count==T+1;
2. every layer/member captures exactly 0..token_count-1 inclusive;
3. terminal_index==token_count-1;
4. anchor_terminal resolves exactly terminal_index;
5. duplicate/missing coordinate fails;
6. row order is pair -> condition -> layer -> token;
7. vector_index round-trips exactly through state_rows.jsonl;
8. NPZ parser rejects extra arrays, pickle/object arrays, wrong endian/dtype, nonfinite values, malformed arrays, and malformed/nondeterministic container where applicable;
9. validator rejects missing/extra artifact, malformed canonical JSON/JSONL, manifest-key drift, schema drift, checksum mismatch, output collision, and staging collision;
10. each paired measurement reconstructs exactly from referenced vectors plus zero s_-1 when required; and
11. Tier-3 measurement/artifact fields are rejected rather than silently accepted.

Tests assert no scientific separation result. git diff --check and narrow tests are later requirements, not authorized now.

## 9. Exclusions and next action

Excluded: generic hooks; hidden/convolution Tier-1 capture; Tier-3 SSM-A/B/C/Delta diagnostics/artifacts; source/package changes; replay; new data; scientific execution; interpretation; causal/detector/calibration claims; promotion; environment recovery.

After later explicit implementation authorization, recheck authority/base/two-file scope, pass runtime/source and local tests, and obtain independent implementation verification of diff, trace restoration, semantics, transaction/schema validation, and evidence. Neither step authorizes execution; a separate explicit execution authority is required.

SCIENTIFIC_CONCLUSION: NONE

PASS_READY_FOR_INDEPENDENT_VERIFICATION

Exact next authorized action: independent static verification of this single recovered candidate against cb529070cc68921fd917e99866bc29de9891751e; do not implement or execute without a later explicit independently verified implementation authorization.
