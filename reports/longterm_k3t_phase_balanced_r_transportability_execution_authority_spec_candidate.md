# K3T Phase-Balanced R Transportability Scientific Execution Authority Candidate

**Status:** one-run scientific execution authority candidate.

**Stage:** `K3T — Phase-Balanced R Transportability`.

This document authorizes exactly one bounded K3T scientific recurrent-state execution after this document is committed as the immediate one-file child of the frozen K3T implementation commit.

It does not amend the K3T preregistration.

It does not authorize a K3T rerun.

It does not authorize K3C rerun.

It does not authorize K4.

## 1. Exact authority markers

`K3T_EXECUTION_AUTHORITY_SCHEMA=k3t-scientific-execution-authority-v1`

`K3T_SCIENTIFIC_RECURRENT_STATE_EXECUTION_AUTHORIZED=YES`

`K3T_ONE_SCIENTIFIC_EXECUTION=YES`

`K3T_EXECUTION_IMPLEMENTATION_COMMIT=1a2ed06ad22ec7c293acd827d11646c92390cf0d`

`K3T_EXECUTION_RUNNER_SHA256=acb9390ac8fafc64edcac07562e84c4d9de99df844b41ad8e82652640e7527a3`

`K3T_EXECUTION_TEST_SHA256=144870ce18ae6e57d12a179a88bab853b7a1d1d88ae658d818786ee0a7f85d88`

`K3T_PREREG_COMMIT=58cbcd316c7714ddc8c041c2a2ec4376e79a4bd0`

`K3T_PREREG_SHA256=57b5bd2375fbacb7ef5e22260f3f1ecf71dba839f383f7726c03544e33a45cc5`

The scientific runner recognizes exactly these eight K3T execution-authority marker keys.

## 2. Frozen authority topology

Authority path:

`reports/longterm_k3t_phase_balanced_r_transportability_execution_authority_spec_candidate.md`

The authority commit must:

1. be the immediate child of:

   `1a2ed06ad22ec7c293acd827d11646c92390cf0d`

2. add exactly the authority path above;
3. modify no other tracked file;
4. remain the exact runtime HEAD during the scientific execution.

No commit may be inserted between the implementation commit and this authority commit.

No commit may be added after this authority commit before the one scientific execution.

The runtime branch must remain:

`longterm-k-series-native-state-kinematics`

The only allowed working-tree dirt at scientific execution time is the historical K1 untracked pair:

- `scripts/longterm_k1_native_state_kinematics.py`
- `tests/test_longterm_k1_native_state_kinematics.py`

Any other dirty state blocks execution.

## 3. Frozen preregistration

Preregistration path:

`reports/longterm_k3t_phase_balanced_r_transportability_prereg_candidate.md`

Preregistration commit:

`58cbcd316c7714ddc8c041c2a2ec4376e79a4bd0`

Preregistration SHA256:

`57b5bd2375fbacb7ef5e22260f3f1ecf71dba839f383f7726c03544e33a45cc5`

K3T remains:

`SAME_GENERATOR_CLAIM_DISJOINT_PHASE_BALANCED_PROSPECTIVE_R_REPLICATION`

K3T is observational.

K3T is not an external-distribution replication.

K3T is not a causal-mechanism experiment.

## 4. Frozen implementation

Implementation commit:

`1a2ed06ad22ec7c293acd827d11646c92390cf0d`

Runner:

`scripts/longterm_k3t_phase_balanced_r_transportability.py`

Runner SHA256:

`acb9390ac8fafc64edcac07562e84c4d9de99df844b41ad8e82652640e7527a3`

Runner Git blob at implementation commit:

`40bd739f2cf6c6c2fa8d66a8b4acc284cccb66a5`

Test:

`tests/test_longterm_k3t_phase_balanced_r_transportability.py`

Test SHA256:

`144870ce18ae6e57d12a179a88bab853b7a1d1d88ae658d818786ee0a7f85d88`

Test Git blob at implementation commit:

`815c471d50098706ec4e7048de29c2aa87a0c475`

The implementation commit is exactly one commit after the preregistration commit and adds only the runner and test files.

## 5. Completed non-scientific validation

The following implementation validation completed before this authority:

- combined frozen K2S/K3T focused suite: `53 passed`;
- recovered K3T focused suite: `33 passed`;
- direct CLI help: PASS;
- state-blind K3T population reconstruction: PASS;
- frozen generated-source SHA reconstruction: PASS;
- frozen candidate-pool SHA reconstruction: PASS;
- frozen reciprocal-mapping SHA reconstruction: PASS;
- exact 168-phase x2 balance: PASS;
- exact per-phase correction-source one-to-one balance: PASS;
- claim-disjointness against K2W: PASS;
- claim-disjointness against K2R/K3: PASS;
- claim-disjointness against K3C: PASS;
- tokenizer-only W=8 feasibility: PASS;
- seed180 handoff authentication: PASS;
- encoder fingerprint validation: PASS;
- synthetic native recurrent-state instrumentation: PASS;
- synthetic trace-logit noninterference: PASS_EXACT;
- synthetic fresh-state isolation: PASS_EXACT;
- synthetic causal common-prefix state identity across all layers: PASS_EXACT;
- CPU sequential recurrence validation: PASS;
- scientific mode without authority: fail-closed PASS;
- committed implementation without authority: fail-closed PASS.

The non-scientific validation read no K3T scientific-population recurrent state.

## 6. Frozen K3T population

Generator:

`scripts/build_controlled_v5.py`

Generator SHA256:

`4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c`

Generator Git blob:

`baee23a9f71333125f4a8735c2c92d20cab7eb4f`

Global template slice:

`[900:1236]`

Exact pair-ID range:

`generated_fact_901`

through:

`generated_fact_1236`

Item count:

`336`

Reciprocal block count:

`168`

Nonnumeric lexical phase period:

`168`

Required phase count:

exactly 2 items per phase.

Required correction-source balance:

`{"none":168,"polarity_flip":168}`

Required stronger phase/source balance:

every phase has exactly one `none` item and one `polarity_flip` item.

Generated source rows:

`4368`

Generated-source canonical SHA256:

`fb699bcc99e00b8c437fd49c204345933d61c69215af853746bc4639545c7400`

Candidate-pool canonical SHA256:

`d95d245e358ff497ea50b95e4f54d1192be64d09fec2c06538fc15f75b09ef70`

Reciprocal-mapping canonical SHA256:

`1c458054540ac0d38857cb3f55595e0b286428225ef11ed22059981dfd1e28ad`

Any mismatch blocks scientific recurrent-state read.

## 7. Frozen prior-population disjointness

K3T must remain zero-overlap on pair ID, exact claim SHA, and exact claim text against:

K2W candidate pool:

`abf693d3267cc4e3dd27a8127d2948b36fdf8ba24e135a643215f0f31a26d808`

K2R/K3 candidate pool:

`00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4`

K3C candidate pool:

`9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e`

Any overlap blocks execution.

## 8. Frozen tokenizer feasibility

Model/tokenizer family:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers:

`5.12.1`

Tokenizer class:

`GPTNeoXTokenizer`

Tokenizer mode:

fast tokenizer required.

Frozen feasibility:

- total items = 336;
- reciprocal blocks = 168;
- matched valid = 336;
- swapped valid = 336;
- matched d-p = `{"2":168,"3":168}`;
- swapped d-p = `{"2":168,"3":168}`;
- matched correction availability = `[23,28]`;
- matched control availability = `[20,27]`;
- swapped correction availability = `[23,28]`;
- swapped control availability = `[20,27]`;
- prefix marginal preserved = true;
- correction marginal preserved = true;
- control marginal preserved = true.

Window:

`W=8`

Any mismatch blocks scientific execution.

## 9. Frozen checkpoint and encoder

Handoff ZIP:

`C:\Users\Home1\Downloads\p3w7-seed8192-a0-seed180-replacement-r1-retry1_55debe94f0d1.zip`

Expected ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Expected selected checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Encoder canonical digest:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Encoder raw-concatenation digest:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

Encoder tensor count:

`242`

Encoder total numel:

`129135360`

Encoder raw bytes:

`516541440`

Encoder dtype:

`torch.float32`

No training or fine-tuning is authorized.

## 10. Frozen native-state runtime

Scientific device:

`cpu`

Scientific recurrence:

`MambaMixer.slow_forward`

Installed recurrence source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

State timing:

`post_consumption_s_t`

Primary scientific layer:

`23`

Fast-kernel or GPU scientific execution is not authorized.

The sequential-fallback warning from Transformers is expected.

## 11. Frozen scientific measurement

Primary endpoint:

`R`

For each branch, layer-23 native recurrent-state speed over W=8 is:

`speed_(p+k) = ||S_(p+k) - S_(p+k-1)||_F`

for:

`k=1,...,8`

Branch R:

the mean of the eight raw Frobenius speeds.

For recipient i:

`Delta_R_matched(i) = R(M_corr(i)) - R(M_ctrl(i))`

`Delta_R_swapped(i) = R(S_corr(i)) - R(S_ctrl(i))`

Pair-specificity:

`X_R(i) = |Delta_R_matched(i)| - |Delta_R_swapped(i)|`

For reciprocal block b containing items a and c:

`B_R(b) = (X_R(a) + X_R(c)) / 2`

The block is the inferential unit.

There are exactly 168 confirmatory block values.

## 12. Frozen primary inference

K3T has exactly one confirmatory endpoint.

Endpoint:

`R`

Expected direction:

positive.

Required:

`n_valid = 168`

Undefined/nonfinite R is an integrity failure.

Zeros are allowed and excluded from n_eff.

Promotion floor:

`n_eff >= 30`

Primary test:

two-sided exact sign test.

Family size:

`m=1`

Alpha:

`0.05`

Rank-biserial sign effect:

`(positive - negative) / n_eff`

Verdict rules are exactly:

If corrected/raw p <= 0.05 and effect > 0:

`R_PHASE_BALANCED_TRANSPORTABILITY_SIGNAL_REPLICATED`

If corrected/raw p <= 0.05 and effect < 0:

`R_PHASE_BALANCED_TRANSPORTABILITY_SIGNAL_CONTRADICTED`

Otherwise:

`R_PHASE_BALANCED_TRANSPORTABILITY_NOT_ESTABLISHED`

No D, displacement, P, or path-length diagnostic may modify this verdict.

## 13. Interpretation boundary

A positive K3T result does not erase K3C.

A positive K3T result does not by itself establish unrestricted:

`R_CROSS_CLAIM_TRANSPORTABILITY_ESTABLISHED`

A null result leaves:

`R_CROSS_CLAIM_TRANSPORTABILITY_NOT_ESTABLISHED`

in force.

An opposite significant result strengthens evidence against stable positive R transportability in the tested controlled-generator regime.

No K3T result establishes a causal W/G/H mechanism.

## 14. Reference diagnostics

The scientific runner may serialize:

- D mean turning;
- displacement;
- P efficiency;
- path length.

These are:

`REFERENCE_DIAGNOSTICS_ONLY`

They have no K3T confirmatory p-values.

They cannot alter the R verdict.

They cannot authorize a three-of-four rule.

## 15. Scientific output contract

The scientific execution must write to a new directory outside the repository.

The authority freezes the output naming rule:

`C:\Users\Home1\Desktop\ContraMamba-K3T-Runs\k3t-r-transportability-<AUTHORITY_COMMIT_FIRST12>`

The directory must not exist before execution.

Required artifact family:

- `generated_source.jsonl`
- `candidate_pool.jsonl`
- `reciprocal_mapping.json`
- `item_metrics.jsonl`
- `block_metrics.jsonl`
- `primary_stats.json`
- `integrity.json`
- `report.md`
- `manifest.json`
- `SHA256SUMS.txt`

The scientific artifacts must independently bind:

- authority commit and authority SHA;
- preregistration commit and SHA;
- implementation runner/test SHAs;
- generator identity;
- K2S helper identity;
- population identities;
- prior-overlap checks;
- HF/runtime identities;
- handoff and encoder identities;
- synthetic instrumentation preflight;
- measurement contract;
- primary statistics;
- artifact SHA256 values.

## 16. One-run execution semantics

This authority permits exactly one scientific execution.

Before K3T scientific-population recurrent-state read begins, a provenance/hash/runtime/output-path blocker may stop execution without creating scientific evidence.

Once any K3T scientific-population recurrent-state execution begins, the authority is treated as consumed unless a separately frozen recovery authority proves otherwise.

An unfavorable result never authorizes rerun.

A null result never authorizes rerun.

A software failure after scientific state read begins never silently authorizes retry.

Do not launch the scientific command as a probe.

Do not reuse an output directory.

Do not make another commit before the scientific run.

## 17. Frozen scientific command template

At the authority commit, the exact scientific command form is:

```powershell
python scripts/longterm_k3t_phase_balanced_r_transportability.py `
  --scientific `
  --seed180-handoff "C:\Users\Home1\Downloads\p3w7-seed8192-a0-seed180-replacement-r1-retry1_55debe94f0d1.zip" `
  --hf-revision "5708daa364c50b880e7bd92eab456e0d34492ee9" `
  --output-dir "C:\Users\Home1\Desktop\ContraMamba-K3T-Runs\k3t-r-transportability-<AUTHORITY_COMMIT_FIRST12>"
```

`<AUTHORITY_COMMIT_FIRST12>` is replaced only after this authority document is committed.

No other scientific CLI mode is authorized.

## 18. Anti-rescue restrictions

The one scientific execution must not:

- change population slice `[900:1236]`;
- replace any claim;
- search another phase-balanced slice;
- change 168-phase definition;
- change stable-ID sort;
- change reciprocal pairing;
- change W=8;
- change layer 23;
- change raw Frobenius speed;
- change R pair-specificity;
- drop zero blocks;
- use item-level pseudo-replication;
- change the two-sided exact sign test;
- add post-outcome phase/source/p/d filters;
- promote reference diagnostics;
- reinterpret K3C as successful;
- run K4.

## 19. Post-run handling

A successful runner exit establishes execution success only.

It does not by itself establish artifact/provenance validity.

It does not by itself establish the scientific conclusion.

After the single execution:

1. preserve the output directory unchanged;
2. independently validate every artifact SHA and manifest binding;
3. reconstruct all 168 B_R values and primary statistics independently;
4. classify the scientific verdict only after artifact/provenance validation passes;
5. freeze a closure/archive report before any successor decision.

## 20. Authority state

`K3T_SCIENTIFIC_EXECUTION_AUTHORITY = ONE_RUN`

`K3T_EXECUTION_AUTHORITY_CONSUMED = NO`

`K3T_RERUN_AUTHORIZED = NO`

`K3C_RERUN_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

This document becomes active scientific execution authority only when committed with exact bytes as the immediate one-file child of implementation commit:

`1a2ed06ad22ec7c293acd827d11646c92390cf0d`
