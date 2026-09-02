# O0c - Selective-SSM Native Recurrent-State Instrumentation Feasibility And Matched-Control Design

## 1. Status / authority

Overall verdict:

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

This document is a single candidate authority/specification for a later native-precursor feasibility and observational-design study. It is authored in `STATIC O0c SCIENTIFIC / TECHNICAL DESIGN AUTHORITY AUTHORING ONLY` phase.

This document does not authorize implementation, tokenizer execution, model loading, model forward passes, generation, training, evaluation, Kaggle execution, package installation, package mutation, staging, commit, push, or modification of any existing artifact.

Authority order used:

1. Current controller instruction for this O0c task.
2. Frozen canonical O0b scientific interpretation commit: `f1dc559d546d20611d66b27684bbfa0f02afa696`.
3. Frozen O0b scientific runtime-version recovery execution authority: `5079b3cc738618d9afba25397b73d499432fcfc7`.
4. Frozen O0b repaired observer implementation: `44c5ba4f2204167f91c7f5564c6dbfcd82304035`.
5. Frozen O0b scientific design authority: `df461469cb087f7f5db1e41a2b08e65ea517ad8`.
6. `docs/CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md`.
7. `docs/CONTRAMAMBA_RESEARCH_VISION.md`.
8. Repository `AGENTS.md`.

Canonical authoring HEAD verified before authoring:

`f1dc559d546d20611d66b27684bbfa0f02afa696`

Read-only inspection performed:

- frozen O0b repaired observer at commit `44c5ba4f2204167f91c7f5564c6dbfcd82304035`, path `scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`;
- frozen O0b observer tests at commit `44c5ba4f2204167f91c7f5564c6dbfcd82304035`, path `tests/test_observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`;
- O0b scientific interpretation report candidate;
- O0b scientific execution/runtime recovery authorities;
- O0b scientific design, boundary-recovery, dataset/validator authority context;
- current Hypothesis Map and Vision;
- current local Hugging Face Transformers Mamba source interfaces only, without loading any pretrained model or tokenizer.

Local source caveat: the inspected local Transformers package is `5.12.1`, with `transformers/models/mamba/modeling_mamba.py` SHA256 `23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41` and byte size `32590`. Frozen O0b scientific execution used `transformers_version=5.0.0`. Therefore this local source inspection is feasibility evidence only. A future O0c implementation authority must independently bind the exact runtime source identity used for O0c before any scientific execution.

## 2. Scientific motivation from frozen O0b

Frozen O0b concluded:

`narrow sufficiency-sensitive precursor clue supported`

O0b measured Hugging Face `MambaModel` exposed hidden-state proxies under a strict matched-control design. The O0b interpretation is bounded: insufficient evidence retained a distinguishable hidden-state-proxy response relative to sufficient paraphrase and surface-null controls, after controlling claim bytes, serialization scaffold, token count, terminal position, and first-divergence-relative anchors.

O0b explicitly did not observe the selective-SSM recurrent state, convolutional cache state, selective-scan internal state matrices, or direct SSM-A/SSM-B/SSM-C/SSM-Delta dynamics. O0c exists because the O0b hidden-state-proxy clue is worth explaining, but hidden states are downstream exposed representations, not necessarily the lower-level recurrence/update state that gives Mamba its state-space character.

O0c is therefore an explanatory follow-up. It does not rewrite O0b, reinterpret O0b evidence, or strengthen O0b into a causal, predictive, calibrated, or deployable claim.

## 3. Exact O0c scientific question

O0c asks:

Do frozen native Mamba selective-SSM recurrent-state trajectories show a matched-control sufficiency-sensitive precursor clue corresponding to the bounded O0b hidden-state-proxy clue?

The permitted descriptive contrasts are:

- comparison-A = D(`insufficient_matched`, `reference_sufficient`);
- comparison-B = D(`paraphrase_sufficient`, `reference_sufficient`);
- comparison-C = D(`surface_null_matched`, `reference_sufficient`).

The expected scientific interpretation is limited to whether comparison-A is consistently larger than comparison-B and comparison-C across the frozen pairs, all eligible Mamba layers, and the frozen anchor schedule in native recurrent-state measurements.

### Notation disambiguation

O0c uses two unrelated naming systems that must never be interpreted as corresponding semantic variables.

| Names | Level | Meaning |
|---|---|---|
| comparison-A / comparison-B / comparison-C | Experiment-level matched-control distance contrasts | comparison-A = D(`insufficient_matched`, `reference_sufficient`); comparison-B = D(`paraphrase_sufficient`, `reference_sufficient`); comparison-C = D(`surface_null_matched`, `reference_sufficient`). |
| SSM-A / SSM-B / SSM-C / SSM-Delta | Implementation-level Mamba state-space quantities | Mamba selective-SSM transition/dynamics, input, readout, and discretization/step-size quantities. |

SSM-A/SSM-B/SSM-C/SSM-Delta are mechanistic model quantities, not Frame, Predicate, Sufficiency, Authorization, or experimental comparison labels.

This question is not:

- whether the model will hallucinate;
- whether a hidden or recurrent state defines hallucination probability;
- whether a detector can be calibrated;
- whether any layer/anchor can be selected after seeing results;
- whether observed state separation is causal;
- whether ContraMamba semantic ownership should be introduced.

## 4. Native-state ontology

O0c uses the following strict ontology.

### Exposed layer hidden states

Definition: the tensors returned through Hugging Face `output_hidden_states=True`, plus `last_hidden_state` when present. In the inspected local Mamba implementation, these are collected after each `MambaBlock` and after final norm.

O0c status: contextual baseline only. These were the O0b measurement target and must not be relabeled as native recurrent SSM state.

### Recurrent SSM state

Definition: the state variable updated by the selective SSM recurrence inside each Mamba mixer. In the inspected sequential implementation, the recurrence has the form:

```text
ssm_state_t = discrete_SSM-A_t * ssm_state_{t-1} + SSM-Delta_SSM-B_u_t
scan_output_t = SSM-C_t @ ssm_state_t
```

with per-layer state shape conceptually:

```text
batch x intermediate_size x ssm_state_size
```

O0c status: Tier 1 scientific target, if and only if it can be captured exactly without changing forward semantics.

### Convolutional state/cache

Definition: the short convolutional state used by the causal convolution path before the SSM recurrence. In the inspected cache implementation this is separate from `recurrent_states`, has its own update method, and stores convolution-window information rather than the SSM recurrence memory itself.

O0c status: not Tier 1. It may be recorded only as a technical cache-integrity diagnostic if a future authority proves it is needed to validate non-interference. It must not be interpreted as the recurrent SSM state.

### Selective-scan internal intermediates

Definition: temporary tensors used to compute scan outputs, such as `scan_outputs`, optional `all_h` in an associative scan path, or internal optimized-kernel temporaries. They may include the per-token scan trajectory or only outputs/final state depending on backend.

O0c status: admissible only when they are exact representations of the recurrent trajectory or required for reconstructing it. Implementation-specific temporaries must not become scientific variables merely because they exist.

### SSM-A/SSM-B/SSM-C/SSM-Delta-related quantities

Definition: quantities produced or consumed by the Mamba mixer to parameterize the selective SSM update:

- continuous or discretized SSM-A transition/dynamics factors;
- input-dependent SSM-B quantity;
- SSM-C readout quantity;
- SSM-Delta discretization/step-size values;
- `SSM-Delta_SSM-B_u` or equivalent input injection term.

O0c status: Tier 3 diagnostics only. They may be observed only under a future exact-capture contract and interpreted as update-parameter diagnostics, not as separately learned semantic variables. They must not be mapped onto ContraMamba Frame/Predicate/Sufficiency/Authorization owners.

### Implementation-specific temporary tensors

Definition: any local variable, transposed view, kernel workspace, dtype-cast buffer, or backend-specific helper tensor whose identity depends on implementation details rather than on the scientific recurrence definition.

O0c status: not authorized as scientific observation. It may be used only to prove exact reconstruction of an authorized Tier 1/2/3 quantity.

## 5. Backend/instrumentation feasibility

O0b execution reported fallback to the sequential Mamba implementation because optimized selective-state/scan kernels were unavailable. O0c must treat that fact carefully.

The inspected local Mamba implementation indicates:

- fast CUDA/kernel path is used only when all optional kernels are available, the device is CUDA-like, and tracing conditions permit;
- otherwise `MambaMixer.forward()` falls back to `slow_forward()`;
- `slow_forward()` explicitly computes `ssm_state` through a Python loop over sequence positions when associative scan is not used;
- on CPU/eval/non-tracing, the sequential loop appears to expose the desired recurrent state most directly;
- `cache_params` can store only final recurrent state and convolutional state, not the full per-token recurrent trajectory by default.

Feasibility conclusion:

O0c is technically feasible as a candidate design if a future implementation can instrument the sequential full-sequence recurrence to copy the per-token post-update `ssm_state_t` trajectory while returning exactly the same model outputs as the uninstrumented forward. O0c should prefer the sequential backend because it avoids optional optimized kernels and gives a direct recurrence loop surface. O0c must not install `mamba-ssm`, `causal-conv1d`, `mambapy`, kernels, or any package to obtain another backend.

Technical-provenance blocker for future implementation:

The exact upstream/runtime source identity for the O0c execution environment must be established before implementation authority. The local `transformers 5.12.1` source inspection cannot be assumed equivalent to the frozen O0b `transformers 5.0.0` runtime. A future O0c implementation authority must either:

- freeze a source hash for the exact installed `transformers.models.mamba.modeling_mamba` and `transformers.cache_utils` files used in the O0c runtime; or
- vendor/copy no code, but pin exact package version and source hash checks in the observer provenance gate before model loading.

This is a future technical-provenance gate, not a reason to guess.

## 6. Frozen input/control inheritance

O0c inherits the validated O0b matched-control dataset unless a future static authority proves a technical incompatibility before tokenizer or model execution.

Inherited constants:

- model/tokenizer: `state-spaces/mamba-130m-hf`;
- immutable HF model/tokenizer revision: `5708daa364c50b880e7bd92eab456e0d34492ee9`;
- CPU;
- float32;
- eval/frozen/inference only;
- `add_special_tokens=False`;
- `trust_remote_code=False`;
- O0b dataset path: `data/longterm_o0b_matched_controls_v1.jsonl`;
- O0b dataset SHA256: `75a675bee49cb26eb0935d364f0f5d090922dd01576dfc23294961b28394aec2`;
- O0b validation artifact path: `reports/longterm_o0b_matched_controls_v1_validation.json`;
- O0b validation artifact SHA256: `e8344ea3df54a3393aa8fa82dba19eb2baade9af9366687bb105f4ad348979ff`;
- pair IDs: `o0b_pair_001`, `o0b_pair_002`, `o0b_pair_003`;
- serialized form: `Claim: <claim>\nEvidence: <evidence>`;
- equal full serialized token counts within pair;
- identical claim bytes within pair;
- matched terminal positions;
- frozen first-divergence-relative anchors.

O0c must not invent a new semantic dataset, add more pairs, alter evidence wording, alter claims, or revalidate under a different tokenizer merely for convenience. Any future incompatibility with recurrent-state instrumentation must be reported as a blocker and handled by separate authority.

## 7. State/token indexing convention

O0c freezes this convention before execution:

For a serialized token sequence:

```text
x_0, x_1, ..., x_T
```

and a layer-specific recurrent state:

```text
s_{-1} = zero initial recurrent SSM state for that sequence/member/layer
s_t = recurrent SSM state after consuming token x_t
```

Therefore a state observation at absolute token index `t` means post-consumption state `s_t`, not pre-token state. The raw transition at token `t` is:

```text
transition_t = s_t - s_{t-1}
```

with `s_{-1}` defined as the zero initial state created for that same forward. O0c may observe `transition_0` from zero to the post-token-0 state, but the primary matched-control anchors are the inherited O0b anchor positions.

Anchor alignment:

- `anchor_pre_minus_1`: observe post-consumption state `s_{d-1}`, where `d` is first divergent token index. This is the last post-token recurrent state after an identical token prefix.
- `anchor_divergence`: observe `s_d`, the post-consumption state immediately after the first divergent token.
- `anchor_post_plus_1`: observe `s_{d+1}`.
- `anchor_post_plus_2`: observe `s_{d+2}`.
- `anchor_post_plus_4`: observe `s_{d+4}`.
- `anchor_terminal`: observe `s_T`, the post-consumption final recurrent state at the matched terminal token index.

If any anchor index is out of range under the frozen O0b validation artifact, it remains unavailable and must not be substituted.

The convention applies independently for each layer and each full-sequence forward. No recurrent state may be reused across matched members.

## 8. Minimal measurement set

O0c authorizes only measurements directly reconstructable from captured recurrent state arrays.

Tier 1 recurrent-state distance:

```text
D_state_l2(layer, anchor, member, reference)
= Euclidean distance between unit-normalized vec(s_t) tensors
```

The vector is the flattened recurrent SSM state for one layer at one post-consumption token index. If any vector has zero norm or non-finite values, the run fails closed.

Tier 2 transition magnitude:

```text
M_transition(layer, anchor, condition) = ||s_t - s_{t-1}||_2
paired_transition_delta = M_transition(member) - M_transition(reference)
```

This reports whether the divergent/control token causes a larger recurrent update, independent of absolute state displacement.

Tier 2 transition-direction cosine:

```text
cos_transition(layer, anchor, member, reference)
= cosine(s_t_member - s_{t-1}_member, s_t_reference - s_{t-1}_reference)
```

This is admissible only when both transition vectors have nonzero norm and when it answers a distinct direction question. It is not an independent copy of normalized state distance.

Tier 3 update-parameter diagnostics, if technically exact:

- `discrete_SSM-A_t` transition factor norms or paired differences;
- `SSM-B_t`, `SSM-C_t`, and SSM-Delta/time-step norms or paired differences;
- `SSM-Delta_SSM-B_u_t` injection norm.

These are diagnostics only. They must be stored separately and must not be folded into one aggregate.

Algebraic redundancy:

- cosine distance between the same unit-normalized recurrent state vectors is redundant with normalized L2 because `D_l2^2 = 2 * D_cos`;
- raw state norm and normalized state distance answer different questions, but raw norm alone is not the primary comparison-A/comparison-B/comparison-C separation measure;
- transition magnitude and transition-direction cosine are non-redundant with absolute state distance only when computed from `s_t - s_{t-1}` rather than from `s_t`;
- per-anchor metrics and terminal metrics are not interchangeable.

O0c must not create one heterogeneous aggregate scalar over state distance, transition magnitude, transition direction, layer, anchor, and optional SSM-A/SSM-B/SSM-C/SSM-Delta diagnostics.

## 9. Instrumentation non-interference contract

Any future O0c observer must prove exact forward non-interference before scientific execution.

Required equivalence:

- same input IDs;
- same attention mask behavior, if any;
- same CPU device;
- same float32 dtype;
- same model mode, `eval()`;
- frozen parameters with `requires_grad=False`;
- `torch.inference_mode()`;
- no stochastic behavior;
- no dropout/training path;
- no generated text;
- no output mutation;
- no cache behavior mutation;
- no parameter or buffer mutation except ordinary non-persistent internal temporaries already present in the uninstrumented path;
- no changed dtype/device through instrumentation;
- no extra model forward passes beyond the explicitly frozen design;
- no state reuse between reference/member/control sequences;
- no dependence on optional optimized kernels.

The future implementation must include a fail-closed equivalence gate comparing uninstrumented and instrumented outputs before any scientific observation is accepted. At minimum, the gate must compare `last_hidden_state` and all exposed hidden states returned by the ordinary model call under identical inputs. Tolerances must be frozen before execution. If bitwise equality is not technically guaranteed because copied instrumentation changes only observation side effects, a future authority must justify a strict numerical tolerance and prove that all differences are below it.

If instrumentation changes ordinary model outputs, O0c execution is invalid regardless of interesting recurrent-state measurements.

## 10. Pre-divergence invariant

For each pair, comparison, layer, and authorized native recurrent-state quantity:

If the reference and member token IDs are identical through absolute token index `t`, then their post-consumption recurrent SSM states must be identical through `s_t` within frozen tolerance.

At the inherited `anchor_pre_minus_1`, the future O0c observer must require:

```text
allclose(s_{d-1}^{reference}, s_{d-1}^{member}, rtol=0.0, atol=1e-6)
```

for every eligible Mamba layer and every comparison-A/comparison-B/comparison-C contrast. The default tolerance is inherited from O0b hidden-state pre-divergence checks. A future implementation authority may tighten it to exact equality if source inspection proves deterministic CPU float32 recurrence should be bit-identical with observation-only copying.

The same invariant applies to transition histories before divergence when stored:

```text
transition_i^{reference} == transition_i^{member}
for all i < d
```

within the same frozen tolerance.

A pre-divergence mismatch is an implementation/execution failure, not signal.

## 11. Layer/forward-count policy

Layer policy:

O0c must observe all eligible Mamba layers. A favorable layer subset must not be chosen after data. The default eligible set is every `MambaMixer`/Mamba block layer in the frozen model whose recurrent SSM state has the expected finite tensor shape.

A priori exclusion is allowed only if a future static implementation authority demonstrates a technical incompatibility before execution, for example a layer with no recurrent SSM state or a backend path that cannot expose exact state without semantic alteration. Any exclusion must be documented before scientific execution and must not depend on observed comparison-A/comparison-B/comparison-C results.

Forward-count policy:

The same 12 full-sequence forwards used by O0b are sufficient in principle:

```text
3 pair IDs x 4 conditions = 12 full-sequence forwards
```

O0c should capture all authorized internal states during each full-sequence forward. Token-by-token replay is not authorized when the full-sequence sequential recurrence can expose the exact same per-token states.

Token-by-token replay may be considered only under a future recovery/feasibility authority if full-sequence instrumentation is proven impossible. Such replay would require proving exact equivalence to full-sequence sequential recurrence, preserving cache reset between sequences, and documenting why the replay does not alter state semantics. It is not authorized by this candidate.

## 12. Artifact/provenance design

O0c should use deterministic binary arrays plus JSON/JSONL metadata.

Expected primary tensor size estimate:

For `state-spaces/mamba-130m-hf`, the common configuration is approximately 24 Mamba layers, `intermediate_size ~= 2 * hidden_size`, and `state_size ~= 16`. With hidden size around 768, the recurrent state per layer is approximately:

```text
1536 * 16 = 24576 float32 values
24576 * 4 bytes = 98304 bytes per state
```

O0b used 3 pairs, 3 comparisons to reference, 6 anchors, 2 vector roles, and 25 exposed hidden-state descriptors. For O0c recurrent state over 24 eligible Mamba layers:

```text
3 * 3 * 6 * 2 * 24 * 98304 bytes ~= 255 MB
```

This estimate covers anchor-only recurrent states, not full per-token trajectories. If a future implementation stores full trajectories for transition reconstruction, expected size may increase by roughly:

```text
total serialized token count across 12 forwards * eligible_layers * 98304 bytes
```

Given O0b's short matched controls, this may remain practical but must be computed exactly from the frozen validation artifact before execution. If full trajectories exceed a future storage budget, the future authority may instead store only the anchor states plus required preceding states `s_{t-1}` for transition metrics. It must still preserve reconstruction of every reported measurement.

Required artifact strategy:

- deterministic `.npz` or equivalent binary arrays with little-endian float32 and no pickle;
- separate arrays for recurrent states and transition vectors or enough adjacent state rows to reconstruct transitions;
- JSON/JSONL metadata for every vector row, including pair ID, condition, comparison ID, layer index, anchor name, absolute token index, state timing convention, vector role, tensor shape, dtype, source backend, and source line/function identity;
- separate JSONL for paired measurements;
- summary JSON containing only reconstructable descriptive aggregates;
- rendered Markdown report;
- `SHA256SUMS.txt` covering every artifact;
- manifest with repository HEAD, observer script SHA256/bytes, model/tokenizer ID and revision, package versions, source-file hashes for Mamba implementation and cache implementation, device, dtype, exact command, output directory, dataset SHA256, validation artifact SHA256, comparison order, anchor order, layer descriptors, tolerance values, execution status, and non-interference gate result.

The future implementation must not use timestamps, hostnames, usernames, random UUIDs, or mutable branch names as scientific identity fields unless separately marked as non-authoritative runtime logs. Commit SHAs and content hashes are the authority identities.

## 13. Falsification matrix

O0c must apply the following interpretation matrix without post-hoc layer or anchor selection.

| Outcome | Interpretation |
|---|---|
| Recurrent-state comparison-A consistently > comparison-B / comparison-C across multiple pairs, layers, and anchors | Supports a more native sufficiency-sensitive precursor clue. |
| O0b hidden-state clue exists but recurrent-state clue collapses | Suggests exposed representation sensitivity without corresponding persistent recurrent-state separation. |
| comparison-A, comparison-B, and comparison-C all similar | Indicates broad intervention sensitivity or weak specificity rather than sufficiency-sensitive separation. |
| comparison-B comparable to or exceeding comparison-A | Indicates wording/paraphrase sensitivity may explain the measured response. |
| comparison-C comparable to or exceeding comparison-A | Indicates surface/null token identity sensitivity may explain the measured response. |
| Only isolated layers, pairs, or anchors show comparison-A > comparison-B / comparison-C | Weak or unstable pattern; no favorable layer/anchor selection is licensed. |
| Pre-divergence recurrent-state mismatch | Invalid execution/instrumentation, not signal. |
| Instrumentation changes ordinary model outputs | Invalid instrumentation, not signal. |
| Exact recurrent state cannot be captured without changing semantics | O0c implementation blocked; do not substitute hidden states and call them native state. |
| Optional SSM-A/SSM-B/SSM-C/SSM-Delta diagnostics show separation but recurrent state does not | Diagnostic update-parameter sensitivity only; no recurrent-state precursor claim. |

No hard scientific PASS threshold, statistical significance claim, population estimate, generalization claim, hallucination score, or calibrated detector is authorized.

## 14. O0b/O0c/O1 boundary

O0b remains valid frozen evidence for a narrow sufficiency-sensitive hidden-state-proxy precursor clue. O0c does not rewrite O0b and does not alter its artifacts, reports, tests, or interpretation.

O0c is limited to feasibility and native-state observational design for frozen Mamba recurrent-state instrumentation under the O0b matched-control inputs. It may support or weaken the hypothesis that O0b's exposed hidden-state clue is visible closer to the actual selective-SSM recurrence.

O1, if later justified, is a separate future stage. O1 may ask expanded precursor-predictability or execution questions only under new authority. O0c does not authorize O1 implementation, O1 execution, expanded datasets, learned probes, threshold tuning, best-layer selection, generation, model modification, or semantic ownership.

## 15. Explicit non-authorizations

This candidate does not authorize:

- modifying any existing tracked file;
- modifying O0b reports, evidence artifacts, implementation, tests, data, or validation artifacts;
- modifying URP/reason-router files;
- modifying stage180 files;
- modifying protected temporary directories;
- modifying root patch files;
- modifying `docs/CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md`;
- modifying `docs/CONTRAMAMBA_RESEARCH_VISION.md`;
- modifying `AGENTS.md`;
- modifying cm tooling;
- importing/loading the pretrained model;
- downloading the model or tokenizer;
- executing tokenizer calls;
- executing model forwards;
- executing generation;
- training or fine-tuning;
- evaluation;
- installing packages or kernels;
- enabling optimized kernels;
- altering the pretrained Mamba;
- changing dataset identity, pair IDs, claims, evidence text, tokenization, anchors, seeds, labels, or split semantics;
- defining a hallucination probability;
- defining a calibrated detector;
- selecting a best layer or best anchor after seeing results;
- introducing ContraMamba semantic owners;
- implementing Frame/Predicate/Sufficiency/Authorization state channels;
- treating SSM-A/SSM-B/SSM-C/SSM-Delta tensors as semantic owners;
- claiming causality from observation alone;
- staging, committing, or pushing.

## 16. Preconditions for future implementation authority

A future implementation authority may be considered only after independent verification of this candidate and must satisfy all preconditions below before model/tokenizer execution:

1. Freeze the exact O0c implementation file scope.
2. Freeze the exact observer script path, commit, SHA256, byte count, and line-ending facts.
3. Freeze exact source identity for the runtime Mamba implementation and cache implementation.
4. Confirm the backend path to be instrumented is CPU sequential full-sequence recurrence, or explicitly document a source-provenance-compatible alternative.
5. Define the exact instrumentation mechanism and prove it is observation-only.
6. Define strict output-equivalence tests comparing instrumented and uninstrumented outputs.
7. Define exact recurrent-state tensor shape expectations per layer.
8. Define exact tolerated dtype/device behavior and fail closed on mismatch.
9. Define exact storage format, artifact names, schemas, checksums, and deterministic publication protocol.
10. Preserve O0b dataset/control constants and validation artifact identities.
11. Preserve the 12 full-sequence forward policy unless a separate feasibility authority proves this impossible.
12. Preserve state reset/no-reuse between all sequence members.
13. Preserve pre-divergence recurrent-state invariant.
14. Preserve comparison-A/comparison-B/comparison-C order and anchor order.
15. Preserve no-generation/no-training/no-evaluation/no-package-mutation boundaries.

The future implementation authority must include tests that fail closed for hidden-state/native-state conflation, cache-state/recurrent-state conflation, pre-divergence mismatch, changed output tensors, incomplete state rows, non-finite states, wrong source hash, wrong backend, wrong layer set, wrong anchor order, duplicate state reuse, and any metric not reconstructable from stored arrays.

## 17. Open technical blockers, if any

Open blocker for future implementation authority:

The exact Mamba source identity corresponding to the intended O0c runtime must be frozen before implementation. Local static inspection found a feasible sequential recurrence surface in Transformers `5.12.1`, but O0b's frozen scientific execution used Transformers `5.0.0`. O0c must not assume those source interfaces are identical.

Open feasibility question for future implementation authority:

Whether full-sequence sequential recurrence in the exact O0c runtime can expose every per-token `ssm_state_t` without altering outputs must be proven by source-bound implementation and equivalence tests. If exact capture is impossible, O0c must fail closed or require a new recovery/feasibility authority. It must not substitute exposed hidden states.

No blocker is identified at design-authoring level. The candidate is ready for independent verification as a static authority/specification.
