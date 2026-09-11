# K0-RVG-V Raw Recurrence Observer Validation / Scientific-Use Readiness Report Candidate

**Status:** independent implementation validation / readiness review candidate.

**Implementation commit under review:**

`fcfe161c12f4ed8ef37aff435554cc0660e477af`

**Implementation authority commit:**

`41dbeae22a3831300f0de2f1f44a4b49733d70a5`

**Implementation authority specification SHA256:**

`e1cd7eee0895ade2b9230a59f14a4e7f63f3b85c550d508961a16787be9df3b6`

This report reviews implementation correctness, synthetic execution success, provenance validity, and readiness boundaries.

It does not convert synthetic validation into scientific evidence.

It does not authorize scientific-population model execution or recurrent-state reading.

## 1. Verdict

Overall verdict:

`PASS_READY_FOR_PROSPECTIVE_RAW_VECTOR_PREREGISTRATION_DRAFT`

This means:

- the raw recurrence observer implementation is sufficiently validated to be referenced by a future prospective scientific observation preregistration;
- the observer may be treated as a validated measurement instrument under its frozen CPU slow-path runtime contract;
- no scientific execution is authorized by this report;
- no Branch A or Branch B successor is activated;
- K4 remains unauthorized.

## 2. Exact implementation scope

The implementation commit is the exact one-commit child of:

`41dbeae22a3831300f0de2f1f44a4b49733d70a5`

and adds exactly two files:

`scripts/longterm_k0_rvg_raw_recurrence_observer.py`

`tests/test_longterm_k0_rvg_raw_recurrence_observer.py`

No historical K2S/K2R/K3/K3C/K3T implementation file was modified.

The historical untracked K1 pair remained untouched.

## 3. Exact implementation identities

Observer:

`scripts/longterm_k0_rvg_raw_recurrence_observer.py`

SHA256:

`12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25`

Git blob at implementation commit:

`f2dbdfe52661eca384897578ab272e602e36deac`

Test:

`tests/test_longterm_k0_rvg_raw_recurrence_observer.py`

SHA256:

`54543c91b9c919ab6fa9cf8e0b5e37380681d776f81e70a06057ed69e7c95af4`

Git blob at implementation commit:

`4de6ae521311970680ed3fcd57218c9327ed6eca`

## 4. Frozen runtime and historical bridge

Validated runtime:

Model:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers:

`5.12.1`

Installed `modeling_mamba.py` SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Upstream Transformers Git blob:

`87987e3e6646d8d0f9f0048bdd8a155d99c845db`

Historical K2S bridge helper:

`scripts/longterm_k2s_pair_specific_event_dynamics.py`

SHA256:

`f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8`

Git blob:

`3a651fb508669bdcf72441a4869b863d6eee6c1f`

The new observer preserves the historical `post_consumption_s_t` timing by requiring byte-identical `S_post` snapshots against the K2S collector.

## 5. Frozen model/checkpoint provenance

Synthetic validation authenticated the seed180 handoff:

ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Common encoder canonical SHA256:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Common encoder raw-concat SHA256:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

Encoder tensor count:

`242`

Encoder numel:

`129135360`

Encoder raw bytes:

`516541440`

Encoder dtype set:

`torch.float32`

The observer additionally validated A0 task-model source lineage:

A0 model blob SHA:

`f0ddc0eda64937de6fcd27943e30a296082c01d5`

A0 heads tree SHA:

`68d26855aa511fcd41d6f395ae5f87177a162678`

## 6. Source binding

The observer independently binds the frozen Mamba slow path through both exact file identity and AST/source-structure analysis.

Frozen source-line semantics:

- `MambaMixer.slow_forward`: line 270;
- `discrete_A`: line 322;
- `deltaB_u`: line 324;
- sequential loop: line 349;
- recurrence update: line 350;
- post-update C readout: line 351;
- ordinary `MambaMixer.forward`: line 366.

The trace is bound to the exact verified `slow_forward.__code__` object.

The analyzer fails closed on missing/ambiguous source structure.

## 7. Captured scientific objects

For each requested `(layer, token)` coordinate the observer captures:

`S_prev = S_(t-1)`

`G = G_t`

`W = W_t`

`S_post = S_t`

without retaining aliases to live model tensors.

The frozen recurrence is:

`S_t = G_t ⊙ S_(t-1) + W_t`

Derived raw velocity quantities are:

`V_t = S_t - S_(t-1)`

`V_t^(carry-change) = (G_t - 1) ⊙ S_(t-1)`

`V_t^(write) = W_t`

The implementation does not learn or tune a geometry.

## 8. Focused test result

Post-commit focused test result:

`19 passed`

The focused suite includes:

- authority/scope constants;
- authority ancestry rather than exact-HEAD locking;
- real frozen Transformers source binding;
- structural analyzer positive fixture;
- malformed-source rejection;
- missing/ambiguous Mamba source rejection;
- ambiguous recurrence rejection;
- exact recurrence validation;
- recurrence mismatch fail-closed behavior;
- snapshot clone/device validation;
- nonfinite rejection;
- dtype/shape validation;
- mixer-exact expected-shape rejection;
- collector-reuse rejection;
- CLI scientific-path exclusion;
- absence of scientific-population loader calls;
- frozen K2S helper identity;
- complete tensor-role hashing.

## 9. Nonblocking test-hygiene note

The committed test file defines the Python function name:

`test_analyzer_rejects_ambiguous_recurrence`

twice.

The earlier definition is shadowed by the later definition at module import time.

This does not remove the required ambiguity coverage because the surviving test still constructs two sequential recurrence loops and requires:

`MAMBA_SEQUENTIAL_RECURRENCE_AMBIGUOUS`

and the real frozen source binding is tested separately.

Therefore:

`DUPLICATE_TEST_NAME = NONBLOCKING_HYGIENE_ONLY`

The executed count must be reported as exactly 19 tests.

No claim of an additional shadowed test is permitted.

A future maintenance-only cleanup may rename the shadowed fixture test, but this is not required before preregistration drafting and does not justify changing the frozen observer implementation.

## 10. Synthetic preflight population boundary

The synthetic preflight used fabricated non-study text only.

The manifest explicitly reported:

`scientific_population_accessed = false`

`scientific_recurrent_state_read = false`

No K2W/K2R/K3/K3C/K3T scientific population was loaded or enumerated by the observer.

The implementation exposes no scientific execution CLI flag.

## 11. Synthetic capture dimensions

Synthetic preflight captured:

layers:

`24`

tokens:

`33`

records:

`792`

because:

`24 × 33 = 792`

Captured native tensor shape:

`[1, 1536, 16]`

dtype:

`torch.float32`

snapshot device:

`cpu`

The implementation validates the shape against each actual mixer's:

`(1, intermediate_size, ssm_state_size)`

rather than merely accepting any rank-3 tensor.

## 12. Exact recurrence reconstruction

For every synthetic captured coordinate the observer reconstructed:

`S_reconstructed = G ⊙ S_prev + W`

and required:

`torch.equal(S_reconstructed, S_post)`

Result:

`PASS_EXACT`

This is the primary algebraic correctness gate.

There is no tolerance fallback for the source recurrence identity.

## 13. Velocity rearrangement validation

The secondary algebraic rearrangement compares:

`S_post - S_prev`

against:

`(G - 1) ⊙ S_prev + W`

with frozen float32 tolerance:

absolute tolerance:

`1e-6`

relative tolerance:

`1e-5`

Post-commit synthetic result:

maximum absolute residual:

`3.814697265625e-06`

maximum relative Frobenius residual:

`1.1752220934199782e-06`

maximum scaled tolerance residual:

`0.6157423257827759`

Result:

`PASS_TOLERANCE`

The scaled residual remains below 1.

The previously observed very large componentwise relative diagnostic from an earlier implementation draft was caused by division near zero and was removed before freeze.

It is not part of the frozen implementation evidence.

## 14. Noninterference

The same fabricated input was run with tracing disabled and enabled.

The output/logit tensors were required to be exactly equal.

Result:

`PASS_EXACT`

Therefore the observer did not measurably alter the frozen model output under the validated synthetic CPU slow-path run.

## 15. Fresh-forward identity

The same synthetic input was executed in a fresh forward with a fresh collector.

All four roles:

- `S_prev`;
- `G`;
- `W`;
- `S_post`;

were byte-identical at every captured coordinate.

Result:

`PASS_EXACT_ALL_ROLES`

## 16. Causal common-prefix identity

Four independent fabricated continuations sharing an exact prefix were executed.

For every prefix coordinate across all 24 layers, hashes of all four recurrence roles were identical across branches.

Result:

`PASS_EXACT_ALL_ROLES_ALL_LAYERS`

This validates causal prefix isolation for the observer.

It is not scientific evidence about any study population.

## 17. Historical K2S state-timing bridge

For the same fabricated input, the existing K2S collector captured its validated post-consumption state.

At every common coordinate:

`new_observer.S_post`

was byte-identical to:

`K2S_post_consumption_S_t`

Result:

`PASS_BYTE_IDENTICAL`

This establishes continuity between historical K-series state timing and the new raw-recurrence tuple observer.

## 18. Trace lifecycle

Validated:

- prior Python trace restoration;
- no snapshot aliasing;
- complete pre/post pair formation;
- duplicate/missing pair rejection in implementation contract;
- collector reuse rejection.

Results:

`TRACE_RESTORATION = PASS`

`SNAPSHOT_NONALIASING = PASS`

`PRE_POST_PAIR_COMPLETENESS = PASS_EXACT`

`COLLECTOR_REUSE_REJECTION = PASS`

## 19. Authority provenance after implementation commit

The frozen observer does not require runtime HEAD to equal its authority specification commit.

Instead it requires:

`41dbeae22a3831300f0de2f1f44a4b49733d70a5`

to be an ancestor of runtime HEAD.

The post-commit synthetic preflight was executed at:

`fcfe161c12f4ed8ef37aff435554cc0660e477af`

and reported:

`implementation_authority_is_ancestor = true`

This proves that the observer remains self-validating after its own implementation commit.

## 20. Independent repository review

Independent remote review confirms:

- implementation commit is exactly one commit ahead of the implementation authority;
- exactly two files were added;
- observer code contains authority ancestry validation;
- observer code contains mixer-exact shape validation;
- observer code contains exact recurrence reconstruction;
- observer code contains A0 model/head provenance checks;
- test code excludes scientific execution flags;
- test code checks that scientific population loader functions are not called.

No implementation-scope deviation was identified.

## 21. Code correctness conclusion

`CODE_CORRECTNESS = PASS_FOR_FROZEN_SYNTHETIC_CONTRACT`

The implementation satisfies the frozen K0-RVG-I measurement contract under the tested runtime.

This conclusion is limited to the bound CPU slow-path implementation.

It does not establish correctness for:

- CUDA fused kernels;
- associative scan;
- another Transformers version;
- another Mamba model family;
- another state layout.

## 22. Execution-success conclusion

`SYNTHETIC_EXECUTION_SUCCESS = YES`

This means the observer actually executed successfully on fabricated input after the implementation was committed.

It does not mean a scientific study executed successfully.

## 23. Artifact / provenance conclusion

`IMPLEMENTATION_PROVENANCE_VALID = YES`

The implementation commit, source/runtime identities, checkpoint identities, authority lineage, historical K2S bridge, and final dirty-state contract are all consistent with the frozen implementation specification.

The final local state after push contained only the historical K1 untracked pair.

## 24. Scientific conclusion boundary

`SCIENTIFIC_CONCLUSION_FROM_K0_RVG_V = NONE`

The observer validation establishes a measurement instrument.

It does not establish:

- a confident-error precursor;
- a vector-direction effect;
- a correction-response effect;
- a retained/write epistemic mechanism;
- a decision-space direction;
- Branch A;
- Branch B;
- K4.

## 25. Scientific-use readiness decision

The observer is sufficiently validated to be named as the frozen measurement instrument in a prospective preregistration.

Therefore:

`RAW_RECURRENCE_OBSERVER_VALIDATED = YES`

`READY_FOR_PREREGISTRATION_DRAFT = YES`

However:

`SCIENTIFIC_USE_EXECUTION_AUTHORIZED = NO`

A future scientific preregistration must separately freeze:

- the scientific population;
- study role and branch relationship;
- layer policy;
- token/event alignment;
- vector object family;
- directional statistics;
- scalar secondary summaries;
- multiplicity;
- zero-vector handling;
- falsification criteria;
- output/provenance schema.

Only a later execution-authority artifact may permit scientific recurrent-state reading.

## 26. Branch state

The pre-branch decision remains unchanged:

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

The observer validation itself does not favor either branch.

## 27. Next authorized design stage

After this K0-RVG-V report is frozen, the next allowed stage is:

`K0-RVG-P — Prospective Raw Native Vector Observation Preregistration`

K0-RVG-P is design-only.

It must define a fresh prospective raw-vector study without inspecting new scientific recurrent states.

It must preserve:

- raw native recurrent coordinates;
- no learned/tuned primary geometry;
- exact observer identity;
- no outcome-selected subspace;
- no post-hoc layer/window selection;
- Branch A and Branch B boundaries explicitly.

K0-RVG-P may determine whether a useful pre-branch observation study can be defined.

If it cannot, the report must stop and return to the A/B scientific fork rather than invent a study.

## 28. Authority state

`K0_RVG_IMPLEMENTATION_COMMIT = fcfe161c12f4ed8ef37aff435554cc0660e477af`

`RAW_RECURRENCE_OBSERVER_VALIDATED = YES`

`SYNTHETIC_EXECUTION_SUCCESS = YES`

`IMPLEMENTATION_PROVENANCE_VALID = YES`

`READY_FOR_PREREGISTRATION_DRAFT = YES`

`SCIENTIFIC_USE_EXECUTION_AUTHORIZED = NO`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`SCIENTIFIC_POPULATION_ACCESS_AUTHORIZED = NO`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`NEXT_STAGE = K0-RVG-P_PROSPECTIVE_RAW_NATIVE_VECTOR_OBSERVATION_PREREGISTRATION`

This report authorizes only drafting K0-RVG-P after this report is frozen.
