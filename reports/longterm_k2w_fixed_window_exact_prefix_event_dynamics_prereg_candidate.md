# K2W Fixed-Window Exact-Prefix Event Dynamics Preregistration Candidate

**Status:** successor scientific/design preregistration candidate; no implementation or execution authority until independently reviewed and committed.

**Frozen prospective design verdict:** `PASS_READY_TO_FREEZE_K2W_PREREG`.

## 1. Status, authority, and purpose

Authority order is: the current controller instruction; K0 `7bd1cf824cd53c7f6cf6215346b42cabf351b70a`; K1 closure/K2 theory `e8227945401d83b952e35b056343e3e1cb19e18f`; corrected K2 preregistration `f2543949a23749f9a1119f88b06105d217a6172a`; K2 implementation/execution `36bef92717bfea10cd37326ab8a28688657b32f5`; K2 closure `386ef0763a0dd8470c22617581af196a324f222f`; and O0c native-state semantics `ff2fb076f6e66a34a632515bb8502d8b1c90ad7f`.

K2 is closed: `INCONCLUSIVE_DUE_TO_CONSTRUCTION_SUPPORT_FAILURE`.  K2W is a successor experiment.  It does not amend, reopen, overwrite, rescue, or reinterpret K2.  In particular, K2 was not statistically negative, and K2's `N_eligible=0` was not observer failure: K2 had no construction-valid population to screen.

K2 required both total post-prefix continuation lengths in `[8,24]` and exact equality of full continuation-token counts.  In its 300 attempted recipes, construction-valid was zero and all 300 were `INVALID_CONTINUATION_TOKEN_CONTRACT`; diagnostic counts were correction out of range 296, control out of range 272, and unequal continuation length 276.  Length differences were `0:24, 1:95, 2:30, 3:146, 4:5`.  The literal prefix and first-divergence conditions were not the failure.

K2W instead prospectively defines its scientific estimand on a fixed post-divergence event-relative segment.  It never uses the unused suffix.  This document, if independently reviewed and committed, is the complete scientific/design authority for bounded K2W implementation, Phase A screening, and conditional Phase B native-state execution; no separate authority/specification document is required.

## 2. Frozen source, roles, and semantic contract

The sole source is the frozen Git blob at commit `8eb7386e0344117d026c0e6ab172018bb98a698e`:

    reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl

Its physical SHA256 is `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`; its semantic SHA256 is `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`.  Source bytes are read from that blob; a CRLF working-tree materialization is not a substitute for those frozen bytes.  No source regeneration, synonym search, candidate substitution, Unicode normalization, source-evidence truncation, or generated filler is permitted.

For each source pair, derive one recipe exactly as follows: `T` is the `evidence_truncation` row; `E` is the `entity_swap` row; `Q` is the `polarity_flip` row when it is `REFUTE`, otherwise the `none` row when it is `REFUTE`.  Require exactly one of each selected role; byte-identical claims across `T`, `E`, and `Q`; `T = NOT_ENTITLED`, primary failure `sufficiency`, sufficiency label 0; `E = NOT_ENTITLED`, primary failure `frame`, polarity `NONE`, and evidence distinct from `T`; and `Q = REFUTE` with polarity `REFUTE`.

Thus the semantic comparison is fixed: `C` is decisive same-claim REFUTE evidence, while `N` is non-corrective same-pair control evidence leaving `NOT_ENTITLED`.  The comparison does not assert semantic matching of their actual evidence tokens.

## 3. Sole neutral construction and token/event contract

For every semantically valid recipe, form these exact UTF-8 strings, without normalization or special tokens:

    P_i = "Claim: " + T.claim + "\nEvidence: " + T.evidence + "\nAdditional evidence:\n"
    C_i = Q.evidence
    N_i = E.evidence
    X_i^corr = P_i + C_i
    X_i^ctrl = P_i + N_i

This single branch-neutral common frame is identical on both branches.  The K2 branch-specific prose templates `This is false:` and `A separate event:` are not used in K2W construction.

Tokenize `P_i`, `X_i^corr`, and `X_i^ctrl` separately and unpadded using `state-spaces/mamba-130m-hf`, revision `5708daa364c50b880e7bd92eab456e0d34492ee9`, `trust_remote_code=false`, and `add_special_tokens=false`.  Before Phase A, bind resolved tokenizer/config files, paths, byte counts, and SHA256s to this revision; any mismatch is `INSTRUMENTATION_OR_PROVENANCE_FAILURE`.

Let `p_i = len(tokens(P_i)) - 1`.  Exact construction requires `tokens(P_i)` to be a prefix of both complete branch token arrays.  Define `d_i` as the smallest token index strictly greater than `p_i` for which correction and control token IDs differ, and require it to exist.  Define `tau_i = d_i - 1`; it is the last token with exactly identical branch history.  Require `tau_i >= 1` so the final common-history velocity `V_tau` is defined; otherwise fail `INVALID_PRESTATE`.  `tau_i` may exceed `p_i` when evidence strings share initial tokens.  Require IDs, unpadded attention masks, and zero-based positions to be exactly identical through `tau_i`; no native state may determine `p_i`, `d_i`, `tau_i`, construction validity, or selection.  Failures are `INVALID_EXACT_PREFIX` or `INVALID_EVENT_DIVERGENCE` as applicable.

## 4. Fixed primary estimand and why suffix matching is absent

The frozen primary event window is `W = {1,2,3,4,5,6,7,8}`, so primary state positions are `tau_i+1` through `tau_i+8`.  Construction requires only that both branches contain at least eight tokens after `tau_i`; otherwise label `INVALID_WINDOW_AVAILABILITY`.

K2W does **not** require equal total continuation length, total continuation length at most 24, equal terminal positions, branch padding, or truncation to equalize branches.  These are absent, not relaxed K2 constraints: K2W has a different fixed-window estimand and excludes every token after `tau_i+8` from the primary estimand.

At comparison `k=1,...,8`, each branch has consumed exactly `k` tokens after the same last-common state, with identical recurrent parameters and starting state, and is compared at the same event-relative token count.  In a causal Mamba forward, later branch suffix length cannot causally affect `S_(tau_i+k)` for `k <= 8`; unequal unused future suffix length is therefore not itself a primary confound.  This does not claim semantic matching of correction and control tokens.  K2W identifies only the effect of this predefined corrective-evidence sequence relative to this predefined non-corrective sequence.

## 5. State-blind Phase A and final N

Freeze the full construction-valid candidate pool before head screening, retain all failures, and resolve duplicate base-claim hashes deterministically before screening (retain the lexicographically smallest `(pair_id, stable_item_id)`).  `stable_item_id` is a prospectively frozen SHA256 ordering identity over canonical recipe fields; it is also the ordering key if selection is needed.  No model outcome, native state, head prediction, confidence, margin, recovery, or inertia may alter source construction, duplicate handling, or pool order.

Phase A evaluates `P_i` only.  Eligibility is exactly:

    gold(P_i) = NOT_ENTITLED
    seed180 A0 observer = SUPPORT
    seed181 A0 observer = SUPPORT
    seed182 A0 observer = SUPPORT

There is no confidence or margin threshold.  Confidence/margin are diagnostic only.  No native-state selection is permitted.  Bind candidate-pool, screening-artifact, eligible-ID-list, and final-ID-list SHA256s before any Phase B work.

If `N_ELIGIBLE < 30`, issue `INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE` and stop before native-state capture.  If `30 <= N_ELIGIBLE <= 64`, use all eligible items.  If `N_ELIGIBLE > 64`, select 64 by ascending SHA256 of the prospectively canonical stable-item representation, with ascending stable ID as tie-break.  No threshold relaxation follows screening.

## 6. Conditional Phase B native state and integrity

Phase B is conditional on valid Phase A and `N_ELIGIBLE >= 30`.  Use one canonical seed180 realization and the authenticated common A0 encoder.  For seed180/181/182, require the normative common-encoder canonical fingerprint `48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`, plus exactly 242 float32 tensors, 129135360 numel, and 516541440 raw bytes.  This is the corrected K2/A0 encoder provenance; the three heads remain eligibility/secondary decision observers, not native-state replications.

Capture only the direct selective-SSM recurrent state, post-consumption `s_t`, under the validated O0c semantics: frame-local SSM state cloned after recurrent update and before readout.  Hidden-state proxies, learned probes, cache final states, and reconstructed state are forbidden.  Require all 24 layers (0--23), exact common prestates through `tau_i`, complete valid-token rows, finite float32 state, fresh-state isolation, trace noninterference, and complete source/runtime/checkpoint/tokenizer provenance.  Any failure is `INVALID_PRESTATE` or `INSTRUMENTATION_OR_PROVENANCE_FAILURE` and stops rather than fabricating an endpoint.

## 7. Bounded primary kinematics

The sole primary layer is layer 23.  Set `epsilon = 1e-12`.  For branch `b` and `k=1,...,8`, let:

    V_(tau+k)^b = S_(tau+k)^b - S_(tau+k-1)^b

The primary family contains exactly `R`, `D`, and `P`:

    mean_speed_b = (1/8) sum_k ||V_(tau+k)^b||_F
    R_i = mean_speed_corr - mean_speed_ctrl

    turn_(tau+k)^b = 1 - cosine(V_(tau+k)^b, V_(tau+k-1)^b)
    D_i = mean_valid_turn_corr - mean_valid_turn_ctrl

For `k=1`, the adjacent prior velocity is final common-history `V_tau`.  A turn is undefined when either adjacent velocity norm is at most epsilon.  If either branch has zero valid turns, `D_i = UNDEFINED`.

    L_b = sum_k ||V_(tau+k)^b||_F
    eta_b = ||S_(tau+8)^b - S_tau||_F / (L_b + epsilon)
    P_i = eta_corr - eta_ctrl

No metric joins the primary family.  Layers 0--22 and full trajectories are descriptive only.

## 8. Inference, decisions, and promotion

The scientific unit is the base item and all endpoints are paired branch differences.  For each of `R`, `D`, and `P`, use a two-sided exact binomial sign test; exclude exact zeros from `n_eff`; disclose undefined values; set `RAW_P=1` if `n_eff=0`; and report the rank-biserial sign effect.  Report positive, negative, zero, undefined, valid, and effective counts for each endpoint.

The fixed Holm family has `m=3`, alpha `.05`, and tie order `R < D < P`; report raw p, Holm-adjusted p, and rejection for every member.  Promotion requires valid construction, instrumentation, and provenance; `N_ELIGIBLE >= 30`; and at least one Holm rejection among `R/D/P`.  Statistical significance is not mechanism.

After the complete correction only, the frozen heads may provide secondary descriptive outcomes: `RECOVERY` when all are `REFUTE`, `INERTIA` when all are `SUPPORT`, and `DISAGREEMENT` otherwise.  They cannot rescue or modify the primary family.

## 9. Failure labels and interpretation boundary

Use at least these noninterchangeable labels: `INCONCLUSIVE_DUE_TO_CONSTRUCTION_SUPPORT_FAILURE`, `INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE`, `INVALID_EXACT_PREFIX`, `INVALID_EVENT_DIVERGENCE`, `INVALID_WINDOW_AVAILABILITY`, `INVALID_PRESTATE`, `INSTRUMENTATION_OR_PROVENANCE_FAILURE`, and `PRIMARY_NULL_AFTER_VALID_EXECUTION`.  Only the last is a K2W scientific null-like outcome; infrastructure or construction-support failure is never `H_NULL`.

The strongest allowed positive language is `FIXED_WINDOW_POST_EVENT_NATIVE_STATE_RESPONSE_OBSERVED`, or `DESIGNED_CORRECTIVE_CONTINUATION_RELATIVE_TO_CONTROL_FIXED_WINDOW_DYNAMICS_OBSERVED`.  K2W does not support claims of a generic corrective-evidence mechanism, necessity, sufficiency, causal recurrent-state mechanism, epistemic-state ontology, or deployable error detector.  K3 remains reserved for direct recurrent-state intervention after a valid native-state phenomenon exists.

## 10. One state-blind tokenizer feasibility audit

Before this candidate was written, one read-only audit evaluated only the frozen source blob, deterministic source labels, and the pinned tokenizer/config.  It loaded no checkpoint or head and performed no model forward, confidence calculation, native-state capture, training, evaluation, or Kaggle work.  It tested this exact neutral construction and `W=8` once; it searched neither templates nor window sizes.  Thus `W=8` was frozen before any K2W scientific execution.

| Audit count | Value |
| --- | ---: |
| N_total source recipes | 300 |
| N_semantically_valid | 300 |
| N_exact_P_token_prefix_valid | 300 |
| N_event_divergence_valid | 300 |
| N_window8_available | 300 |
| N_K2W_construction_valid | 300 |

For the 300 construction-valid recipes, `tau_i-p_i` was `1:149, 2:151`.  Available post-`tau` correction tokens were `19:4, 20:5, 21:7, 22:118, 23:61, 24:70, 25:34, 26:1` (min/median/mean/max `19/23/22.9267/26`).  Control availability was `16:2, 17:3, 18:7, 19:38, 20:53, 21:116, 22:47, 23:12, 24:22` (min/median/mean/max `16/21/20.8833/24`).  Therefore this candidate meets the construction floor; it is not `K2W_DESIGN_FEASIBILITY_BLOCKED`.

## 11. Execution boundary and provenance manifest

Phase B may begin only after the Phase-A hashes and floor are bound.  A future manifest must bind this committed authority identity, implementation identity, K2 closure commit `386ef0763a0dd8470c22617581af196a324f222f`, source blob commit `8eb7386e0344117d026c0e6ab172018bb98a698e`, source/tokenizer/config/checkpoint/head/O0c hashes, exact inputs and token arrays, `p/d/tau`, integrity results, endpoint inputs, and output SHA256s.  Any mismatch fails closed.

Creation of this candidate authorizes no implementation, checkpoint loading, model forward, A0 prediction, eligibility screening, native-state capture, training, evaluation, Kaggle, staging, commit, or push.
