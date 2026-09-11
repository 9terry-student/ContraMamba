# K2 Phase-A Construction-Support Failure Closure Report Candidate

**Status:** RESULT / INTERPRETATION REPORT ONLY.
**Authority status:** NOT EXECUTION AUTHORITY.
**Preregistration status:** NOT A NEW K2 PREREGISTRATION.

## Exact execution identity

This report archives and interprets the completed K2 Phase-A run whose runtime Git HEAD was `36bef92717bfea10cd37326ab8a28688657b32f5`.  Its corrected frozen K2 preregistration authority is `f2543949a23749f9a1119f88b06105d217a6172a`.  The recorded Phase-A script SHA256 is `aae516b2a921a20b43a5184a2d9af05e8e9a05f7963b0b14f5d53a1bd32b5c65`.

The manifest binds the source to the physical frozen Git blob at commit `8eb7386e0344117d026c0e6ab172018bb98a698e`, path `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`: physical SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` and semantic SHA256 `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`.

The pinned Hugging Face model/tokenizer revision is `state-spaces/mamba-130m-hf` at `5708daa364c50b880e7bd92eab456e0d34492ee9`; the completed run resolved that exact revision with a fast `GPTNeoXTokenizer`, `add_special_tokens=false`, and `trust_remote_code=false`.

## Imported artifact provenance

The following files were copied byte-for-byte from `C:\Users\Home1\Desktop\ContraMamba-K2-Runs\k2-phase-a-36bef92717bf` into `reports/longterm_k2_exact_prefix_phase_a_36bef92717bf/` and independently SHA256-verified against their source files:

| Artifact | SHA256 |
| --- | --- |
| `candidate_pool.jsonl` | `18d0f7cedfe3ade8a8ed4d8649ee3556f94284387c191d8eb4d648865e6219d1` |
| `screening.jsonl` | `ee1c818ed295413578506222e54134261ef06e86e31ebf5137c40f98d7d12173` |
| `eligible_ids.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `final_confirmatory_ids.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `phase_a_manifest.json` | `d0d6480b983191530e257007550d6bff6ce0f3d4d2fda1f40085da5830ed5bde` |

`eligible_ids.jsonl` and `final_confirmatory_ids.jsonl` are intentionally empty: each has SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, the valid SHA256 identity of an empty byte sequence.

The manifest’s candidate, screening, eligible, and final SHA256 fields match these imported bytes.  Its script identity, frozen-source physical and semantic identities, pinned HF tokenizer-file identities, three handoff ZIP and checkpoint identities, and common-encoder provenance are internally consistent.  The authenticated common encoder is recorded identically for seed180, seed181, and seed182: normative canonical digest `48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`, secondary raw-concatenation digest `968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`, 242 float32 tensors, 129135360 total elements, and 516541440 raw bytes.

## Technical execution conclusion

Phase A completed normally.  The recorded source, pinned HF snapshot, handoffs, selected checkpoints, and common-encoder checks passed.  No native-state capture occurred.  The Phase-A manifest records 300 total attempts, zero construction-valid candidates, zero duplicate exclusions, zero eligible items, and zero final confirmatory items.

## Construction result and read-only tokenizer diagnostic

The imported candidate pool has exactly 300 rows: 300 have `construction_status == "invalid"`, zero are valid, and zero are excluded.  All 300 have the single construction failure label `INVALID_CONTINUATION_TOKEN_CONTRACT`.

Using only the exact tokenizer snapshot pinned in the manifest, read-only diagnostics reproduced:

| Diagnostic | Count |
| --- | ---: |
| `CORR_LENGTH_OUT_OF_RANGE` | 296 |
| `CTRL_LENGTH_OUT_OF_RANGE` | 272 |
| `UNEQUAL_CONTINUATION_LENGTH` | 276 |

No other token-contract subreason occurred: every tokenized branch retained the required literal prefix and no candidate had an equal first continuation token.  Continuation-length difference counts were `0 -> 24`, `1 -> 95`, `2 -> 30`, `3 -> 146`, and `4 -> 5`.  These observations are descriptive reproduction of the completed diagnostic and do not alter the frozen K2 contract.

## Interpretation correction and scientific closure

The manifest machine verdict, `INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE`, **MUST NOT** be interpreted as evidence that the three frozen A0 observers failed the wrong-commitment criterion.  There were zero construction-valid candidates, so the scientific wrong-commitment eligibility question had no admissible population.

```
K2_PHASE_A_SCIENTIFIC_CLOSURE = INCONCLUSIVE_DUE_TO_CONSTRUCTION_SUPPORT_FAILURE
K2_HYPOTHESIS_SUPPORTED = NO
K2_HYPOTHESIS_FALSIFIED = NO
NATIVE_STATE_OBSERVED = NO
PHASE_B_AUTHORIZED = NO
```

This result does **not** establish H_NULL, absence of a native Mamba precursor, absence of corrective state dynamics, contradiction with O0c, or failure of the overall K-series hypothesis.  It establishes only that the frozen K2 prospective construction is not feasible on the frozen 300-item source under its prespecified tokenizer continuation contract.

No continuation-length relaxation, truncation, synonym rewrite, padding, regenerated controls, threshold tuning, or post-hoc candidate substitution is authorized within K2.

## Stage status

K2 is **CLOSED** at Phase A as scientifically inconclusive due to construction support failure.  Phase B was not entered.
