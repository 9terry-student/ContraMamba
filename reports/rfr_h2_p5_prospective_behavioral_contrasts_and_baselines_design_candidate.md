# ContraMamba RFR-H2-P5 — Prospective Behavioral Contrasts, Baselines, and Decision Rules (Candidate)

- Date: 2026-10-24 (Asia/Seoul)
- Branch: `gen5-causal-role-state-ownership`
- Inspected GitHub HEAD: `ae5f1a751b44eb29545799ce1c7c70cf277bb37c`
- Phase: **PROSPECTIVE SCIENTIFIC DESIGN / REPORT ONLY**.
- Status: `H2_P5_BEHAVIORAL_PROTOCOL_SPECIFIED / MODEL_EVALUATION=NOT_AUTHORIZED / SCIENTIFIC_RESULT=NONE`.
- Authority boundary: This document freezes **prospective evaluation questions and readout/contrast definitions**, not execution permission. It does not authorize tokenizer/model inference, prompt/data materialization, independent-root generation, learning, calibration, evaluation, checkpoint changes, cache-state manipulation, native-state intervention, GPU/Kaggle, H3 promotion, or Phase3B activation. No frozen P0–P4 implementation is changed.

## 1. Why P5 is necessary; source and evidence ledger

- D0 `reports/rfr_d0_dependency_aware_revision_scientific_design_candidate.md` fixes authenticated newer A-update versus sham/unauthenticated contradiction, prerequisite C, derived `B := A AND C`, independent U, logical PRE/POST/DELAYED, and primary D1 oracle class order `REFUTE=0 / NOT_ENTITLED=1 / SUPPORT=2`.
- P0 `reports/rfr_h2_p0_temporal_serialization_readout_identifiability_design_candidate.md` specifies single causal-stream input with no post-update evidence replay, matched queries after identical pre-query prefixes, and a strict **H2 functional behavior versus H3 causal mediation** separation.
- P1 `scripts/rfr_h2_p1_temporal_serialization_contract_cpu.py` is an exact provisional surface and oracle contract, not a sampled corpus; it retains the literal `Answer: ` text at the query endpoint. P3 provides the readout-only seam by moving its trailing space to the answer continuation, without modifying P1's surface or answer text.
- P2 `reports/rfr_h2_p2_tokenizer_lm_readout_static_feasibility_decision_candidate.md` excludes the historical Gen3/Gen4 claim/evidence router as a decoder for RFR A/B/U.
- P3 `scripts/rfr_h2_p3_tokenizer_continuation_conformance_cpu.py` and its operator-reported conformance fix **one compatible, distinct token per class**, 216 hypothetical prompt contexts, 648 candidate checks, and 864 anchors; no RFR model behavior was scored.
- P4 `reports/rfr_h2_p4_frozen_causal_lm_runtime_feasibility_report_candidate.md` records operator-reported pinned-weight SHA256 verification, tied-LM-head CPU load with no missing/unexpected keys, and one **shape-only** forward on synthetic repeated token ID 42 of length 151; no RFR prompt forward or class score.
- D3-M8 `reports/rfr_d3_m8_semantic_micro_world_diversity_decision_candidate.md`: `CURRENT_CONTRACT_TASK_BEARING_DIVERSITY_NOT_DEMONSTRATED / INDEPENDENT_ROOTS_NOT_ESTABLISHED`. The existing 8 Boolean strata and related descendants are **not** 8 IID real-world problem families, and M1–M8 do not authorize confirmatory extrapolation.

**Research question for a separately authorized future diagnostic:** Can the pinned frozen Mamba causal LM use an authenticated temporal revision, dependent rule, and unrelated retained fact in one causal stream, under the specific fixed English serialization, rather than merely follow the latest sentence, a validity marker, a constant answer token, or a query-only shortcut?

## 2. Locked input, inference, and class decision definitions — prospective only

- Model/tokenizer: `state-spaces/mamba-130m-hf`, both at revision `5708daa364c50b880e7bd92eab456e0d34492ee9`. Model weights SHA256 `1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea`. P4 runtime identity Windows, Python 3.13.2, torch 2.12.1, transformers 4.45.0, installed Mamba source SHA256 `0fb3d343b2016aa4abe0c493ae143dc2193e1148bdd8792e9486ad04f5e30d71`. P3 snapshot hashes are the input-tokenizer identity; Gen4 alternative revision must not be substituted.
- Frozen P1 string components: `RULE_TEXT`, exactly three t0 records, one optional t1 A record, optional single fixed neutral delay, then exactly one `Q(A)`, `Q(B)` or `Q(U)`. Physical PRE excludes t1. P3 readout context is the same visible prompt **without the trailing space after `Answer:`**. No t0 or t1 evidence is appended after the designated t1 except the prescribed neutral delay and terminal query. No prompt variants or source-alias swaps inside the primary protocol.
- P3 continuation tokens (single token in all checked contexts): `REFUTE` = ` False` = **16708**, `NOT_ENTITLED` = ` Unknown` = **32471**, `SUPPORT` = ` True` = **11793**. Preserve full-case class ID order `(0,1,2)` and P1/D1 mapping. Do not attach EOS, append explanatory sentences, use a trained probe, or substitute legacy router logits.
- Future model call, if separately authorized: `model.eval()`, CPU float32, `torch.inference_mode()`, `use_cache=False`, `labels=None`, `output_hidden_states=False`, `return_dict=True`, no generation; compute *only* final query-context next-token scores `z=(logits[0,-1,16708], logits[0,-1,32471], logits[0,-1,11793])` for expected output shape `(1,sequence_length,50280)` in the declared label order. Do not reuse sequence-wide pooled logits, train or tune a calibrator, or choose among multiple templates after seeing labels. The full prompt (not appended answer) is the model input. **P5 does not execute this call.**
- For future diagnostic, prediction `y_hat = argmax(z)` only when **all three scores finite** and the maximum is unique; otherwise `INVALID_SCORE_OR_TIE`, counted as an error for every affected comparison. Exact ties are never arbitrarily broken in favor of a class. `m_y = z_y - max_{k != y}(z_k)` is a signed gold-relative margin; correct iff `m_y > 0` and valid. Within-three-token `softmax(z)` may be reported for descriptive ranking **only**: it is not calibrated probability over full vocabulary nor a three-way entitlement certificate.
- No class thresholds, per-class offsets, prior correction, length normalization, temperature tuning, hidden-state pooling, prompt selection, or test-label feedback. Because all three continuations are single tokens, token-length mismatch has been removed within P3, but vocabulary frequency, lexical prior and training distribution biases remain uncontrolled. Report unconditional token preference diagnostics without claiming them to be ground-truth corrections.
- P3 maximum **completed** token sequence length in its hypothetical grid was 151. P4's synthetic length-151 smoke supports a single shape, not a general context guarantee. Any new surface, alternative delay, unseen tokenizer suffix or shifted token anchor requires a fresh separate preflight.

## 3. Exact paired truth-table invariants and analysis unit

Let one *logical stratum* be `(a0,c0,u0) in {F,T}^3`. This finite logical grid has eight members and is **exhaustively enumerated but not sampled from an independent task-family population**. The model-free oracle is frozen D1; not inferred from model predictions.

| Readout/arm | A | B | U |
|---|---|---|---|
| PRE (all three nominal arms, physically identical) | `a0` | `a0 AND c0` | `u0` |
| POST/DELAYED `VALID_FLIP` | `NOT a0` | `(NOT a0) AND c0` | `u0` |
| POST/DELAYED `VALID_SHAM` | `a0` | `a0 AND c0` | `u0` |
| POST/DELAYED `INVALID_FLIP` | `a0` | `a0 AND c0` | `u0` |

- **A revision:** across both flip directions, A changes on `VALID_FLIP` at POST and DELAYED, never on `VALID_SHAM` or `INVALID_FLIP`.
- **Dependency B:** when `c0=T`, valid flip changes B with A while sham/unauthenticated flip do not; when `c0=F`, B stays REFUTE across all arms even though valid A flips.
- **Retain U:** U is identical across all three arms and all readouts, for both values of `u0`.
- **PRE leakage:** because PRE physically lacks t1, count PRE as **three distinct queries per stratum**, not nine independent arm cases; all nominal-arm predictions should be bit-identical when the full prompt is identical. A discrepant PRE arm prediction is an implementation/provenance failure, not a scientific discovery.
- **POST/DELAYED:** for each stratum, 3 arms × 3 queries × 2 readouts = 18 distinct prompt contexts. With three unique PRE queries this gives **21 unique contexts per stratum**, or 168 unique combinations over the eight strata; P3's 216 nominal arm×readout×query contexts include 48 duplicate PRE prompt presentations. These are merely design counts, **not** executed model cases or independent observations.
- **Delay:** compare POST versus DELAYED under the same `(stratum,arm,query)`; the only permitted difference before query is the frozen neutral interval. This measures short synthetic distractor robustness only, not memory across distinct requests or indefinite retention.
- The elementary paired reporting unit is the **entire logical stratum with all related arms/readouts/queries**. For a later independently sampled cohort, the sampling unit would instead be a provenance-linked **case family**, not each row, prompt, token, prediction, or paraphrase.

## 4. Prospective metrics and interpretations

For each stratum define Boolean strict correctness for each of 21 unique contexts, counting nonfinite scores and ties as failures. All results below are **definitions**, not observations:

- `PRE_joint`: all three A/B/U PRE answers correct, evaluated once per stratum.
- `POST_joint`: all nine answers (three arms × A/B/U) correct at POST; `DELAYED_joint`: the analogous nine answers at DELAYED.
- `A_revision_joint`: all relevant POST/DELAYED A predictions satisfy flip, sham and invalid controls **in both flip directions across matched strata**. Do not infer true forgetting from change of A output.
- `B_dependency_joint(C=T)`: correct A-dependent B flip and non-flip controls when C is true; `B_blocked_joint(C=F)`: B always REFUTE despite legitimate A changes.
- `U_retention_joint`: all U answers correct across PRE/POST/DELAYED and arms, for both truth values.
- `INVALID_resistance_joint`: INVALID_FLIP preserves the t0 A/B truth under unauthenticated contradiction, at POST and DELAYED.
- `CASE_joint`: conjunction of `PRE_joint`, `POST_joint`, `DELAYED_joint`; report number/8 and each stratum's result, with **no statistical CI, IID significance, or generalization claim**. The highest narrow claim available on all-eight success is **"this pinned model/readout obeyed the specified fixed-law Boolean grid in the tested runtime"**, not general RFR memory.
- Report full `(query, c0, a0, u0, arm, readout)` cell results, confusion counts for `REFUTE/SUPPORT`, all three raw logits, gold-relative margins, conditional candidate-only softmax and exact error vectors. Never use only pooled row accuracy, marginal class counts or a few cherry-picked successful examples.
- A strict *logical-grid smoke acceptance* is `CASE_joint=8/8` plus every specified conditional contrast and audit gate PASS; a partial pattern may be described as partial diagnostic behavior but not as complete H2 grid success. This **does not set a population-level efficacy threshold**: confirmatory acceptance criteria, family sampling plan, confidence intervals, and independent-root capacity remain `UNSET` pending a separately designed study.

## 5. Baselines and alternative explanations — predeclare before looking at logits

**Level A: deterministic, no-new-prompt baseline predictions on the exact same logical strata.** They are future analytic comparisons, not trained models; their eventual oracle-match counts must be computed from fixed rules, not adjusted to fit LM errors:

1. `CONSTANT_REFUTE`, `CONSTANT_NOT_ENTITLED`, `CONSTANT_SUPPORT`: output that token for every query/readout; class-prior/majority collapse controls.
2. `LATEST_ASSERTION`: at PRE use the authenticated t0 baseline; at POST/DELAYED use the visually latest t1 value for A even if source is unauthenticated, use t0 C/U, derive B by the stated conjunction. Tests the propensity to accept unauthenticated contradiction.
3. `T0_ONLY`: never update A from t1; retain t0 C/U and derive B. Detects lack of valid revision.
4. `A_PROPAGATION_WITHOUT_C`: reconstruct A using the admitted-update rule but answer B as though `C=T` regardless of the actual C, while retaining U. Detects *all-dependents-follow-A* behavior and must fail on at least some `C=F` cases.
5. `QUERY_ONLY_CONSTANT_PER_TARGET`: for each query target A/B/U choose a constant predeclared answer independently of arm/readout; a diagnostic family, not a selected optimal post-hoc baseline. For complete transparency report its entire finite 3^3 table if used; **do not select the maximum after observing the LM**.
6. `AUTH_STATUS_ONLY`: permitted information is just the displayed t1 authentication flag/source alias (or absence at PRE) plus query name, **not** the actual a0/c0/u0 values. Any fixed mapping for this baseline must be committed before LM outputs. It cannot generally solve both signs of A, conditional B and U.
7. `CANDIDATE_TOKEN_PRIOR`: report which among the three fixed answer logits dominates for predeclared answer-neutral controls **only if the control prompts pass a separate tokenizer contract**. This is a descriptive lexical-prior check, not a probability calibration or an excuse to shift the P3 logits. Until such controls are specified, `PRIOR_CONTROL_NOT_DESIGNED`.

**Level B: new counterfactual prompt manipulations are NOT included in the current P3-conformant primary surface.** They require separate prospective text/tokenizer/oracle and independence checks: crossover `REV_A`/`UNVERIFIED_A` alias with explicit authority flag held fixed; rule removal/rewriting; withholding C; changing evidence presentation order independently of logical time; neutral distractor length variants; source/lexical paraphrase; and class-token neutral prompts. These are **proposed future falsifiers**, not free new variants or P5-authorized executions.

**Shortcut limitation:** In the current P1 surface, source aliases `REV_A` and `UNVERIFIED_A` are perfectly associated with authenticated status. Consequently even a hypothetical positive Level-A result cannot independently identify reliance on the Boolean authority field versus the alias. Report `SOURCE_ALIAS_CONFOUNDED` until Level-B controlled crossover exists. The rule is explicitly provided, so successful B outputs cannot by themselves prove a learned abstract dependency operator beyond this specific template.

## 6. Stop conditions and claim ladder

- `BLOCKED_VERSION_OR_BYTES`: model/checkpoint, config, tokenizer revision/files, Mamba runtime source, P1/P3 query seam, class IDs or declared case surface differ from P3/P4. Never silently migrate to the Gen4 128-token tokenizer or a different execution backend.
- `BLOCKED_INPUT_OR_READOUT`: invalid token anchors; truncation; future-arm leakage at PRE; noncausal score location; answer included in input; pooled sequence states; use of trained router; cache assumptions; candidate ID drift, duplicate/tied/nonfinite class scores. Preserve full score vectors and failed cases.
- `BLOCKED_ORACLE_OR_PROVENANCE`: event/source timestamp or D1 class meaning changes, any unapproved new roots/splits, case-family duplicates counted as independent, or output evidence missing source/commit/checkpoint hashes. If local console output is not archived, disclose that limitation.
- `NO_H2_GRID_SUPPORT` if A does not distinguish valid versus invalid/sham updates, if B fails either C gate, if U changes, or if POST-only success collapses at the fixed DELAYED condition. Interpret failures by conditional cells; do not rescue them with tuned thresholds, prompts or verbalizers.
- `BLOCKED_SHORTCUT_RESISTANCE_CLAIM`: if source/alias confounding is not addressed, do not claim authority-cue robustness even when the primary grid is correct. Baseline comparison can reject specific deterministic explanations, not all possible shortcuts.
- `BLOCKED_THREE_WAY_EPISTEMIC_CLAIM`: the eight strict-Boolean worlds contain no gold `NOT_ENTITLED`, even though the three-token scorer contains it. The missing-premise/unknown family must be separately designed **before** any such competence claim.
- `BLOCKED_CONFIRMATORY_GENERALIZATION`: D3-M8 did not establish independent task-bearing micro-worlds or IID roots; 8/8 is at most exact-grid diagnostic success. No scientific performance promotion, p-value, confidence interval or dataset-generalization claim.
- `BLOCKED_H3_CAUSAL_CLAIM`: neither output correctness nor final logits nor observational hidden trajectories establish raw recurrent-state necessity, sufficiency, Retain/Forget/Revise operators, or causal mediation. Later H3 needs separately authorized matched perturbation, removal/restoration, B rescue and U preservation, source/runtime-specific raw-state capture.

**Decision ladder:** `P0-P4 technical preparation` → `P5 prospective fixed-law behavioral protocol` → only upon separate execution authority, `exploratory fixed-grid behavior` → only after separate scientific redesign + independent cases/controls, `potential generalization evidence` → separately, `H3 raw-state causal intervention`. None of these later gates is granted by a P5 design commit.

## 7. P5 disposition and next bounded task

**P5 report-only completion condition:** review this single static design document against frozen D0/P0–P4 and D3-M8, ensure no scope drift, and freeze **one Markdown file**. This does not add an evaluator or data artifact and does not change existing frozen files.

**Recommended next bounded research action after P5 freeze:** H2-P6 **static evaluation-contract review**, limited to evaluating whether all required baseline/score/provenance fields can be specified without changing D1 or P3 and to identifying what extra scientific and execution authorization would be needed before any single CPU exploratory behavior run. No run, inference, tokenizer re-execution, training, sampling, or shortcut-control surface generation is approved by P5.

**Final:** `H2_P5=PROSPECTIVE_DESIGN_CANDIDATE / RFR_BEHAVIOR=NOT_EVALUATED / H2_GRID_ACCEPTANCE=NOT_EVALUATED / INDEPENDENT_CASES=NOT_ESTABLISHED / H3_MEDIATION=NOT_ESTABLISHED / MODEL_EVALUATION_AUTHORITY=NO / PHASE3B=BLOCKED_UNCHANGED / GPU=OFF`.
