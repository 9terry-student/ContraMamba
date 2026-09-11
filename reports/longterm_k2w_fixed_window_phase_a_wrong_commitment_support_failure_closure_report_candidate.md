# K2W Phase-A Wrong-Commitment Support Failure Closure Report Candidate

**Status:** RESULT / INTERPRETATION REPORT ONLY.
**Authority status:** NOT EXECUTION AUTHORITY.
**Preregistration status:** NOT A NEW K2W PREREGISTRATION.

## Exact execution identity

This report archives and interprets the completed K2W Phase-A execution at runtime Git HEAD `c7c7a0c218bb64d083f0da86e2929990d87bc4ed`.

The frozen K2W scientific/design preregistration authority is `cc5386e730c333209eb070b14025c5368038e247`.

The executed Phase-A script SHA256 is:

`84e01f13fa8a0f2f52cc0a0be730c1d610615b9562c7b77074b0b1835418a826`

The frozen source Git blob is commit `8eb7386e0344117d026c0e6ab172018bb98a698e`, path:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Physical SHA256:

`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

Semantic SHA256:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

The pinned Hugging Face model/tokenizer identity was `state-spaces/mamba-130m-hf` at exact revision `5708daa364c50b880e7bd92eab456e0d34492ee9`.

## Imported execution artifacts

The completed Phase-A artifacts were copied byte-for-byte from:

`C:\Users\Home1\Desktop\ContraMamba-K2W-Runs\k2w-phase-a-c7c7a0c218bb`

into:

`reports/longterm_k2w_fixed_window_phase_a_c7c7a0c218bb/`

| Artifact | SHA256 |
| --- | --- |
| `candidate_pool.jsonl` | `abf693d3267cc4e3dd27a8127d2948b36fdf8ba24e135a643215f0f31a26d808` |
| `screening.jsonl` | `e15b4b750ece295223a13047c33a4dd339c87526ccf3d24d0aece61e4f6e3205` |
| `eligible_ids.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `final_confirmatory_ids.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `phase_a_manifest.json` | `5447c1739c6a0825fcf3eba4bf5797b76d308e70f5c8ba3808b9f913b6d01c64` |

The two empty ID-list files intentionally have the SHA256 identity of an empty byte sequence.

## Construction and provenance result

Phase A completed normally.

All prespecified construction gates passed:

- N_total_attempts = 300
- N_semantically_valid = 300
- N_exact_prefix_valid = 300
- N_event_divergence_valid = 300
- N_window8_available = 300
- N_construction_valid = 300
- N_duplicate_excluded = 0

The frozen event geometry was reproduced exactly.

`tau_minus_p`:

- 1 -> 149
- 2 -> 151

Correction post-tau availability:

- 19 -> 4
- 20 -> 5
- 21 -> 7
- 22 -> 118
- 23 -> 61
- 24 -> 70
- 25 -> 34
- 26 -> 1

Control post-tau availability:

- 16 -> 2
- 17 -> 3
- 18 -> 7
- 19 -> 38
- 20 -> 53
- 21 -> 116
- 22 -> 47
- 23 -> 12
- 24 -> 22

All three authenticated A0 checkpoints strict-loaded successfully.

Each recorded the same authenticated common encoder:

- canonical digest: `48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`
- secondary raw-concatenation digest: `968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`
- tensor count: 242
- numel: 129135360
- raw bytes: 516541440
- dtype: torch.float32

Therefore K2W did not fail at source construction, tokenization, exact-prefix identity, event divergence, fixed-window availability, checkpoint reconstruction, or common-encoder provenance.

## Frozen wrong-commitment screening result

All 300 construction-valid candidates reached the frozen three-observer Phase-A screening.

Eligibility required exactly:

- gold(prefix) = NOT_ENTITLED
- seed180 = SUPPORT
- seed181 = SUPPORT
- seed182 = SUPPORT

There was no confidence or margin threshold.

The completed result was:

- N_ELIGIBLE = 0
- N_FINAL = 0
- phase_a_verdict = `INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE`

The categorical read-only diagnostic produced:

| Seed | REFUTE | NOT_ENTITLED | SUPPORT |
| --- | ---: | ---: | ---: |
| seed180 | 50 | 244 | 6 |
| seed181 | 5 | 295 | 0 |
| seed182 | 0 | 300 | 0 |

SUPPORT-vote counts across the three frozen observers were:

- 0 SUPPORT votes = 294
- 1 SUPPORT vote = 6
- 2 SUPPORT votes = 0
- 3 SUPPORT votes = 0

Exact three-seed prediction triples were:

- NOT_ENTITLED | NOT_ENTITLED | NOT_ENTITLED = 244
- REFUTE | NOT_ENTITLED | NOT_ENTITLED = 49
- SUPPORT | REFUTE | NOT_ENTITLED = 4
- SUPPORT | NOT_ENTITLED | NOT_ENTITLED = 2
- REFUTE | REFUTE | NOT_ENTITLED = 1

No candidate had two or three SUPPORT votes.

In particular, seed181 produced zero SUPPORT predictions and seed182 produced NOT_ENTITLED for all 300 prefixes.

These categorical counts are descriptive interpretation of the already-completed frozen screening. They do not authorize a new threshold, a new voting rule, or post-hoc candidate selection.

## Scientific interpretation

Unlike K2, K2W had a fully valid construction population and all 300 candidates reached the scientific entrance screen.

Therefore the frozen Phase-A verdict is correctly interpreted as a genuine support failure for the preregistered unanimous wrong-commitment entrance condition.

It is not a tokenizer/construction failure and it is not an infrastructure failure.

However, native recurrent state was never captured and the K2W R/D/P fixed-window endpoints were never evaluated.

Accordingly:

`K2W_PHASE_A_SCIENTIFIC_CLOSURE = INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE`

`K2W_HYPOTHESIS_SUPPORTED = NO`

`K2W_HYPOTHESIS_FALSIFIED = NO`

`NATIVE_STATE_OBSERVED = NO`

`PHASE_B_AUTHORIZED = NO`

This result does **not** establish:

- H_NULL;
- absence of a native Mamba precursor;
- absence of event-aligned corrective state dynamics;
- absence of recurrent-state kinematic structure;
- contradiction with O0c;
- failure of the overall K-series hypothesis.

It establishes only that the frozen K2W experiment has no admissible population satisfying its prospectively frozen `3-of-3 SUPPORT` wrong-commitment entrance condition on the frozen 300-item source.

## No K2W rescue

K2W must not be rescued by changing the completed entrance contract.

In particular, this closure does not authorize:

- changing 3-of-3 SUPPORT to 2-of-3 or 1-of-3;
- selecting the six seed180 SUPPORT cases;
- adding confidence or margin thresholds;
- sweeping thresholds;
- changing the prefix construction;
- selecting another observer realization;
- replacing the frozen source;
- choosing candidates from prediction probabilities;
- entering Phase B despite `N_ELIGIBLE = 0`.

Any future design that does not require unanimous A0 false commitment must be a separately preregistered successor experiment, not a K2W amendment.

## Stage status

K2W is **CLOSED at Phase A** as scientifically inconclusive due to wrong-commitment support failure.

Phase B was not entered.

No K2W native-state observation or causal recurrent-state intervention was performed.
