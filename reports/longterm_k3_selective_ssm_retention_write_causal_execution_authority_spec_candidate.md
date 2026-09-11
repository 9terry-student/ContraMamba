# K3 Selective-SSM Retention-vs-Write Scientific Execution Authority Specification Candidate

**Status:** SCIENTIFIC EXECUTION AUTHORITY.

**Scope:** exactly one K3 scientific intervention execution using the frozen K3 execution harness and the frozen K3 causal-decomposition preregistration.

This document does not modify the K3 scientific design. It only authorizes execution of the already-frozen design with the exact implementation identities below.

## 1. Machine-readable authority markers

`K3_EXECUTION_AUTHORITY_SCHEMA=k3-scientific-execution-authority-v1`

`K3_SCIENTIFIC_INTERVENTION_EXECUTION_AUTHORIZED=YES`

`K3_ONE_SCIENTIFIC_EXECUTION=YES`

`K3_EXECUTION_IMPLEMENTATION_COMMIT=519f188451bd4c6acb6b5b24b643621903e45fe8`

`K3_EXECUTION_RUNNER_SHA256=1dd9a9d13ff7e738f1080ad2ad36c8a6ab8279290ca4f84ea572acbe0b86f0f1`

`K3_EXECUTION_TEST_SHA256=20e2206009043d353db8ee2e0ed7448c2b94b442e768fb242dab7f91d1df1b45`

`K3_REPLAY_IMPLEMENTATION_COMMIT=a2f33796903466680091ccbede9f6728022e678f`

`K3_REPLAY_MODULE_SHA256=116e9daae8a4d62b19d8ab4e7a4d4171df8ad7ae99b08953fc1f2ba49563a60e`

`K3_PREREG_COMMIT=20032bb53d77416bb7eb25411eb4f77b008648b4`

## 2. Governing scientific authority

The scientific design is frozen by:

`reports/longterm_k3_selective_ssm_retention_write_causal_decomposition_prereg_candidate.md`

at commit:

`20032bb53d77416bb7eb25411eb4f77b008648b4`

with SHA256:

`7561f4188921b645eba3d7108bcd007b7c223389b5e1abc512cc8d61af976c84`

That preregistration remains the sole authority for:

- the layer-23 intervention target;
- W=8;
- intervention onset at first correction/control token divergence d;
- arithmetic-midpoint W equalization;
- arithmetic-midpoint G equalization;
- GW equalization as an integrity control only;
- the four K2R state metrics R, D, DISP, P;
- the aligned attenuation quantities;
- the eight confirmatory tests;
- exact sign tests;
- Holm correction with m=8 and alpha=0.05;
- the full-support, contradiction, and not-established verdict logic;
- anti-rescue rules;
- the K4 boundary.

This execution authority does not amend any of those definitions.

## 3. Frozen implementation chain

The structural replay implementation is frozen at:

`a2f33796903466680091ccbede9f6728022e678f`

The scientific execution harness is frozen at:

`519f188451bd4c6acb6b5b24b643621903e45fe8`

The execution-authority commit containing this file must be:

1. the immediate child of `519f188451bd4c6acb6b5b24b643621903e45fe8`;
2. a one-file commit containing only:
   `reports/longterm_k3_selective_ssm_retention_write_causal_execution_authority_spec_candidate.md`;
3. the exact runtime HEAD for the authorized scientific execution.

No code modification is authorized between the frozen harness commit and this execution-authority commit.

No commit may be added after this authority commit before the authorized scientific run.

## 4. Pre-execution implementation evidence

The frozen structural replay implementation passed its focused suite and synthetic replay preflight before this authority was issued.

The validated synthetic replay evidence includes:

- component capture noninterference = `PASS_EXACT`;
- natural structural replay = `PASS_EXACT`;
- sham replay = `PASS_EXACT`;
- GW equalization paired-state collapse = `PASS_EXACT`;
- synthetic known-term replay = `PASS_EXACT`;
- scientific population intervention executed = `false`;
- scientific outcome values read during implementation preflight = `false`;
- capture layer = 23;
- component dtype = float32;
- device = CPU;
- recurrence source SHA256 =
  `23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`;
- Transformers version = 5.12.1.

The combined replay + execution-harness focused suite passed 49 tests before the harness was frozen.

These are implementation/equivalence results, not K3 scientific evidence.

## 5. Frozen scientific population

Use exactly the archived K2R claim-disjoint population:

`reports/longterm_k2r_claim_disjoint_replication_52bd363_v1/candidate_pool.jsonl`

SHA256:

`00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4`

Count:

`300`

Reciprocal inference blocks:

`150`

Use the exact frozen ordering and reciprocal mapping.

No item filtering, replacement, downweighting, resampling, or re-pairing is authorized.

## 6. Frozen baseline scientific evidence

Before accepting any K3 intervention statistic, the execution harness must reproduce the archived K2R baseline block artifact:

`reports/longterm_k2r_claim_disjoint_replication_52bd363_v1/block_metrics.jsonl`

SHA256:

`e4982e8a57e15863227d17f080e7a7a699fb14100387fc36c7c47ab68962c8de`

The K3 natural baseline must reproduce all:

`150 blocks x 4 metrics = 600`

archived K2R block values under the harness's frozen exact-comparison contract.

Failure of archived-baseline reproduction invalidates the K3 scientific execution.

No intervention result may be interpreted after such a failure.

## 7. Model and runtime identity

Use the authenticated seed180 A0 encoder realization already frozen in the K3 preregistration.

Expected handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Expected checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Expected encoder canonical digest:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Expected raw-concatenation digest:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

HF model:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers:

`5.12.1`

Mamba recurrence-source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Scientific device:

`CPU`

GPU / fast-kernel execution is not authorized.

## 8. Authorized intervention conditions

The authorized scientific conditions are exactly:

- `BASE`
- `W_EQ`
- `G_EQ`

The integrity-only control is:

- `GW_EQ`

The intervention span is exactly:

`d ... p+8`

where d is the first correction/control token divergence within each matched or swapped branch pair.

W equalization uses the exact arithmetic midpoint of paired natural write terms.

G equalization uses the exact arithmetic midpoint of paired natural retention coefficients.

No zeroing, alternate clamp, alternate midpoint, log-space averaging, renormalization, intervention-onset shift, or alternate layer/window is authorized.

## 9. Required scientific integrity gates

The authorized execution must fail closed unless all of the following hold:

- authority provenance and one-file commit topology pass;
- repo dirty-state contract passes;
- frozen runner/test blobs are unchanged;
- frozen replay module/test blobs are unchanged;
- K3 prereg bytes are unchanged;
- K2R candidate, block, and manifest bytes are unchanged;
- authenticated handoff and encoder fingerprints pass;
- CPU sequential recurrence source identity passes;
- synthetic replay preflight re-passes;
- natural replay is exact on every scientific branch;
- sham replay is exact on every scientific branch;
- GW_EQ pair collapse is exact for every matched and swapped pair;
- archived K2R baseline reproduction passes for all 600 block-metric values;
- all required output artifacts are written only after successful computation.

Expected scientific replay integrity counts:

- natural branch exact = 1200;
- sham branch exact = 1200;
- GW pair exact = 600.

## 10. Confirmatory scientific family

The only confirmatory test order is:

1. `R_ATT`
2. `R_SEL`
3. `D_ATT`
4. `D_SEL`
5. `DISP_ATT`
6. `DISP_SEL`
7. `P_ATT`
8. `P_SEL`

Holm family size:

`8`

Alpha:

`0.05`

Full causal-specialization support requires all eight tests to meet the frozen preregistered directional criterion.

No partial result may be promoted to the full K3 success label.

## 11. Frozen scientific verdicts

The only permitted overall K3 verdicts are:

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CAUSALLY_SUPPORTED`

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CONTRADICTED`

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_NOT_ESTABLISHED`

The implementation must derive the verdict mechanically from the frozen eight-test rule.

## 12. One-run execution scope

This authority authorizes exactly one K3 scientific execution on the exact frozen 300-item population.

The authorized run may:

- capture frozen layer-23 G, W, and native recurrent states;
- perform BASE, W_EQ, and G_EQ structural replay;
- perform GW_EQ integrity replay;
- calculate the frozen recipient/block quantities;
- calculate the eight frozen confirmatory tests;
- emit the frozen scientific artifacts.

This authority does not authorize:

- a rerun because a scientific result is unfavorable;
- alternate intervention variants;
- alternate layer or window;
- best-time selection;
- endpoint removal;
- threshold sweeps;
- post-hoc changes to Holm family;
- training or fine-tuning;
- learned probes;
- K4 execution.

A technically failed run caused by an integrity/provenance/software fault must be treated as a failure-recovery event under the repository recovery rules; it must not be silently replaced by an unrecorded second scientific attempt.

## 13. Output and provenance

Scientific output must be written outside the repository to a fresh, nonexistent directory accepted by the frozen harness.

The harness must emit and cryptographically bind its required scientific artifacts, including:

- item metrics;
- block metrics;
- primary statistics;
- integrity report;
- manifest;
- scientific report;
- SHA256SUMS.

The manifest must bind this execution-authority commit and authority-file SHA, exact runner/test identities, replay implementation identity, prereg identity, runtime identities, population identities, recurrence identity, intervention definition, integrity gates, and primary statistics.

No result should be interpreted before post-execution artifact/provenance validation passes.

## 14. Claim boundary after execution

Even a full K3 positive result would support only the narrow causal claim frozen in the K3 preregistration:

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CAUSALLY_SUPPORTED`

It would not by itself establish:

- task-decision causality;
- authorization/entitlement causality;
- native-state necessity or sufficiency outside the replay intervention;
- confident-error prediction;
- detector utility;
- natural-corpus generalization;
- external-distribution generalization.

K4 remains separately preregistered future work.

## 15. Final authority state

`K3_EXECUTION_AUTHORITY_READY=YES`

`K3_EXECUTION_CODE_FROZEN=YES`

`K3_REPLAY_EQUIVALENCE_GATE=PASS`

`K3_SCIENTIFIC_RESULT_AVAILABLE=NO`

`K4_EXECUTION_AUTHORIZED=NO`

Once this exact file is committed as the immediate one-file child of the frozen execution-harness commit, one K3 scientific execution is authorized.
