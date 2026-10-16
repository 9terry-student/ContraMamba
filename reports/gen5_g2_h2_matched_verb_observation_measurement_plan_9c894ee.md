# Gen5 G2-H2 — frozen matched-verb observational measurement plan

**Phase:** prospective *design/measurement freeze only*. No implementation, training, evaluation, checkpoint mutation, Kaggle execution, or Phase3B promotion is authorized by this document.

**Repository:** `9terry-student/ContraMamba`, branch `gen5-causal-role-state-ownership`, frozen input commit `9c894ee22b786ad572c92d39cd0e47df3366eb80`. Historical H1 import/report commit `d84909876278e1808e634fb6f207eeec221b24c7`. Cohort construction source checkout `40a279a28ac088b8728f828038127343f598fc9e` is **provenance**, not the HEAD to use for future G2-H2 execution.

**Exact input:** `reports/gen5_g2_h2_static_matched_verb_candidates/g2h2-static-matched-verbs-40a279a-r1/`:
- `matched_verb_candidate.jsonl` SHA256 `fecdf9e1e3940a3c5e9d893df4f4a810130359c00f1ece5c1c0b1599fc6127c9`
- `static_candidate_manifest.json` SHA256 `d186781cc55749990c168b8f9f0a98186b16e8ca581d203250275b096979fb71`
- Frozen tokenizer revision `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`, tokenizer.json SHA256 `b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`, `tokenizers==0.22.2`.

## 1. Scientific question and bounds

**Question:** When the *same serialized input* differs by exactly one evidence token—NEG `denies` (token 31647) versus POS `confirms` (23849)—does the pre-existing frozen G2 layer-22 correction show an arm-dependent change in the final REFUTE–SUPPORT distinction and in its *output* at fixed evidence token positions? POS `asserts` (17086) is an independently prespecified *lexical positive* sensitivity/control.

Population is the **same frozen 120-pair Phase3A dev set**, with no independent holdout. This is an *exploratory matched-input contrast*, not new confirmatory generalization. Proposed labels NEG=REFUTE, POS=SUPPORT are **not frozen independent annotations** and must not be treated as truth for CE/accuracy claims. The intervention here is a controlled **input token substitution**, NOT a causal intervention on the layer-22 state or correction write. The observer's `wrapped_mixer_output − native_mixer_output` is not the raw SSM recurrent write.

Conditions per pair: `NEG_DENIES`, `POS_CONFIRMS`, `POS_ASSERTS` (120 × 3 = 360 rows), heldout donor seeds `6201,6202,6203`, frozen arms `ZERO,FULL`. No coefficients, tuning, training, or new model architecture. Keep the frozen G2 model/tokenizer, 3 seed-specific corrections, output head, masks and class index map `REFUTE=0, NOT_ENTITLED=1, SUPPORT=2` unchanged. The H1 seven-cell `C0/C6` summary and pair-join functions are **not** valid on these H2 three-condition rows; construct a separate read-only three-condition join and use new unique H2 row IDs.

## 2. Hard input, provenance and numerical validity gates

Before any future GPU work, a **CPU-only** gate must check all 120 original pair IDs/order and three conditions, artifact SHA256, frozen source C6 text/IDs, frozen tokenizer identity, untruncated `claim[:63]+EOS(0)+evidence[:64]` serialization to length ≤128, proposed-label isolation and absence of pair/condition duplicates. For each pair, require identical `claim_token_ids`, serialized length, attention/claim/evidence masks, one shared `verb_serialized_token_index`, **exactly one** differing token ID across all three condition pairs and identical IDs at all other positions. Recompute these checks from archived candidate IDs; original XG1 source remains immutable. Verify no row-ID collisions with H1.

For a **separately authorized future observation**, load the exact frozen parent/checkpoint/donor/kernel binaries and audit their SHA256, no-grad eval mode, native mixer restoration, masks, gradient/optimizer/checkpoint immutability and zero CUDA work until the CPU gates pass. Match H1's model forward and `_historical_forward_from_hidden` logit path. Do not insert candidate semantic labels into model internals or inherit H1's seven-cell label validation. For identical input, unhooked and observed hidden states/logits must be bitwise identical; each `ZERO` correction tensor must be exactly zero; all output tensors/logits must be finite. On original `NEG_DENIES`, the 3-seed × ZERO/FULL logits must reproduce the frozen H1/G2-P0 C6 references within the already frozen `1e-5` logit tolerance. No historical reference exists for the new POS cases; **do not claim POS reference reproduction**. Report both per-cell errors and maximum error. A failed gate blocks *interpretation* even if the program exits successfully.

Frozen H1 output is at `reports/gen5_g2_h1_full_readonly_observation_runs/g2h1-full-observation-direct-40a279a-r1/` (unregistered handoff; do not relabel it `cm import`-registered). Any future H2 execution must use a new, separate run root pinned to its own actual execution commit; neither `40a279a` nor H1's output path may be reused.

## 3. Prespecified final-logit estimands (primary scientific readout)

Let `m(p,s,a,c) = z_REFUTE − z_SUPPORT` for pair `p`, heldout seed `s`, arm `a ∈ {ZERO,FULL}`, and condition `c ∈ {N,C,A}` for `denies,confirms,asserts`.

- **Primary** `D_NC(p,s,a) = m(N) − m(C)`; **arm interaction** `Δ_NC(p,s) = D_NC(FULL) − D_NC(ZERO)`.
- **Sensitivity** `D_NA(p,s,a) = m(N) − m(A)`, `Δ_NA = D_NA(FULL) − D_NA(ZERO)`.
- **Lexical positive control** `D_CA(p,s,a) = m(C) − m(A)`, `Δ_CA = D_CA(FULL) − D_CA(ZERO)`.

Identity `D_NA = D_NC + D_CA` and `Δ_NA = Δ_NC + Δ_CA` must hold at pair/seed/arm level within floating-point tolerance. These are **logit score differences, not classification accuracy**. The previously observed G2-P0 `C6−C0` inversion is motivation but is **not** the new matched input's result.

Report unpooled per-seed: mean, median, sign counts out of 120, quantiles and per-pair rows for `D` in both arms and `Δ` for all three comparisons. Treat **120 distinct factual pairs**, not 360 condition rows or 3 seeds × 120 as independent samples. Do not select a favorable seed or statistical threshold after seeing results. Never use new proposed labels to produce a supervised performance score.

**Directional H2 hypothesis:** `mean_p Δ_NC(p,s) < 0` (FULL compresses REFUTE–SUPPORT NEG-vs-POS separation relative to ZERO). Prespecified sensitivity: `mean_p Δ_NA < 0` for each seed. A *selectivity-compatible descriptive pattern* additionally requires `|mean_p Δ_NC| > |mean_p Δ_CA|` **for each of all 3 seeds** and the same comparison for `Δ_NA`. Report within-pair directional counts without a post-hoc cutoff. These are *exploratory descriptive decision rules only*, not statistical proof, promotion criteria, or a claim of mediation.

## 4. Prespecified layer-22 output measurements

Let `q(p,s,a,c,t) = h_wrapped22(p,s,a,c,t) − h_native22(p,s,a,c,t)` (vector of output channels), computed on detached copies after a read-only forward. For each matched pair let `t*` be the **shared serialized verb token index**. Predeclare regions **relative to t***:

- **PRE**: all valid serialized positions strictly before `t*` (a common-token negative control).
- **VERB**: `t*` only.
- **POST8**: valid positions `t*+1 ... t*+8` (verified to exist in all 360 candidates; the frozen cohort has at least 12 post-verb tokens per candidate).
- **EVIDENCE**: all evidence positions, identical masks and token counts for N,C,A.

Primary output readout, computed **only for FULL** and per pair/seed, is the paired *vector difference RMS across channels*: `E_NC(R) = sqrt(mean_{t∈R,channel} [q_N(t) − q_C(t)]²)`. Also `E_NA(R)` and positive-lexical `E_CA(R)`. Report `VERB` and `POST8` independently, with `PRE` and `EVIDENCE` as controls/secondary; don't pool different windows to optimize an effect. These are distances, not directional suppression or true causal writes. Report sign/magnitude of signed `mean`, `mean_abs` and `RMS` differences of q by condition in each prespecified region as secondary descriptive statistics; q magnitude alone must not be called semantic suppression. ZERO q must be exactly zero, so ZERO q distances must also be exactly zero.

`PRE` q_N, q_C and q_A should match under a causal left-to-right backbone on identical prefixes. Verify maximal absolute deviations and classify any disagreement above numerical `1e-5` as an **unexpected-prefix/numerical gate failure to diagnose**, not a scientific effect. Report the same PRE control on native outputs. Comparisons of layer-22 output distances with `Δ_NC`/`Δ_NA` may be recorded by pair (descriptive Pearson/Spearman), but **no correlation counts as mediation**.

## 5. Predeclared dispositions and stop rules

**BLOCKED / invalid execution evidence** if any input SHA/identity/single-position/mask check fails, model/donor/kernel/runtime identity drifts, NEG reference drift exceeds 1e-5, hook changes model outputs, ZERO q ≠ 0, source artifacts are overwritten, unapproved GPU/train/backward operations occur, or artifact provenance is incomplete. Preserve diagnostics; never auto-retry/overwrite.

**NOT SUPPORTED (direction)** if any seed fails `mean Δ_NC < 0`, or the `denies−asserts` sensitivity fails `mean Δ_NA < 0` in any seed. **NOT DISTINCT FROM LEXICAL CONTROL** if the prespecified arm-effect dominance over `confirms−asserts` fails in any seed; report the actual contrasts, not just the verdict. **OBSERVATIONAL SELECTIVITY-COMPATIBLE ONLY** when all direction and lexical rules hold and the **120-pair mean** output distances satisfy `mean E_NC(R) > mean E_CA(R)` and `mean E_NA(R) > mean E_CA(R)` **for each seed and both prespecified regions R ∈ {VERB,POST8}**. Otherwise classify observed layer-22 signal as nonselective, mixed or undetermined with full numeric evidence. A zero or inconsistent PRE control requires numerical diagnosis before those comparisons are interpreted.

Even full agreement permits only an **exploratory matched-token association**, never layer-22 necessity/sufficiency, causal mediator identity or raw recurrent-write attribution. `PHASE3B=BLOCKED_UNCHANGED` in every case. Testing mediation would require a distinct, precisely specified causal intervention and equivalence gates. **No GPU execution or implementation is authorized by this measurement plan itself.**

## 6. Next bounded validation (not part of this document's execution)

The next authorized work should be a small **CPU-only read-only H2 three-condition schema/metric selftest** against the frozen 360 candidate records, including synthetic logits and synthetic output tensors for identity, sign, zero and prefix checks, without model loading. Keep H1 code/outputs immutable. After this CPU work independently passes and is committed, decide separately whether a real-model GPU observation is justified and authorized.
