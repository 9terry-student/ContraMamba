# ContraMamba RFR-H2: Frozen Readout Scientific Closure

- Date: 2026-10-25 (Asia/Seoul)
- Branch: `gen5-causal-role-state-ownership`
- Status: **CLOSED — INELIGIBLE_STOP_RFR_PATH** for this specific model/serialization/fixed-readout family
- Scope: Scientific result archival only. Not new experimental authority, not a mechanism or architecture rejection.
- Source: operator-reported execution logs and validated artifact summaries. H2-168 predictions and summary were separately inspected in chat; basic-16 raw artifact files remain on the operator's PC and are to be archived verbatim. The assistant did not independently rerun either model forward.

## 1. Fixed H2 Boolean-grid diagnostic (negative)

- Execution commit: `6148207713a59de49325bc7d56541ff7e31dc6d6`
- Run: `rfr_h2_p6_grid168_mamba130m_cpu_6148207_20261025`
- Frozen model/tokenizer: `state-spaces/mamba-130m-hf` @ `5708daa364c50b880e7bd92eab456e0d34492ee9`.
- Execution: CPU float32; no training, GPU or Kaggle; Mamba sequential fallback.
- 8 related Boolean strata x 21 unique contexts = 168; not independent task-family samples.
- Gold: `REFUTE=98`, `SUPPORT=70`, `NOT_ENTITLED=0`.
- Observed: `SUPPORT=168/168`; correct `70/168`; `CASE_joint=0/8`. PRE `10/24`, POST `30/72`, DELAYED `30/72`.
- Frozen baseline oracle matches: `ALWAYS_SUPPORT=70/168`, `ALWAYS_REFUTE=98/168`, `T0_ONLY=144/168`, `LATEST_ASSERTION=144/168`, `IGNORE_C=154/168`, `AUTH_FLAG_ONLY=82/168`.
- Conditional cell accuracy cannot be interpreted as selective updating because every actual prediction was SUPPORT; most clearly `B_BLOCKED_C_FALSE=0/24`.
- Original artifacts: `manifest.json`, `predictions.jsonl`, `baseline_predictions.jsonl`, `summary.json`, `report.md`, `SHA256SUMS.txt`.

## 2. Prospectively frozen 16-item basic-readout qualification (negative)

- Execution commit: `7413f1525ee81a3a79b3d88b3347de10496ad1dc`
- Run: `rfr_h2_basic16_mamba130m_cpu_7413f15_20261025`
- Template: `Facts:\nP is FALSE.\nQ is TRUE.\nQuestion: Is P true?\nAnswer:` (same scaffold for 16 combinations); no rule, revision, authentication, conjunction or delay.
- 4 `(P,Q)` Boolean values x 2 record orders x 2 queried variables = 16 distinct prompts; gold `REFUTE=8`, `SUPPORT=8`.
- Same model/revision and frozen three candidate next-token scores: `REFUTE=' False' (16708)`, `NOT_ENTITLED=' Unknown' (32471)`, `SUPPORT=' True' (11793)`.
- Prospective strict acceptance: **16/16**; no thresholds, calibration, prompt selection or retest.
- Observed: `SUPPORT=16/16`, `correct=8/16`, `strict_pass=false`. For opposite truth-value `(P,Q)` pairs: query-target-dependent prediction switch occurred in **0/4** matched pairs; both-correct **0/4**.
- Reported execution and output validator: `ARTIFACT_SHA256=PASS`, `COMMIT_IDENTITY=PASS`, `PREDICTIONS=16`, `ARTIFACT_VALIDATION=PASS` and CLEAN git worktree. These are operator-reported checks, not a new independent re-execution by the assistant.
- Original artifacts: `manifest.json`, `predictions.jsonl`, `summary.json`, `report.md`, `SHA256SUMS.txt`.

## 3. Scientific decision and claim limits

1. **Accept the two negative observations.** The frozen pretrained Mamba-130M, two tested input templates and the fixed three-token argmax failed both the complex H2 task and the elementary P/Q readout qualification. The RFR-specific failure is not localized to revision: already the PRE and basic P/Q tasks fail.
2. **Honor the predeclared stopping rule.** Set `INELIGIBLE_STOP_RFR_PATH` for this frozen model/serialization/readout pathway; do not re-run, tune prompts, adjust class offsets, fit probes, train, or promote to H3 interventions to rescue this result.
3. **Do not overgeneralize.** This does **not** show that Mamba in general lacks factual/semantic competence, that internal relevant information is absent, or that a learned/native RFR mechanism is impossible. No independent-root generalization, unknown-gold competence, alias-invariant authentication interpretation or recurrent-state causal mediation is established.
4. **Raw-logit diagnostics are exploratory only.** Input-dependent margins exist, but similar directionality on irrelevant U and blocked B rules out a selective-RFR interpretation based on those contrasts alone; no post-hoc threshold rescue.
5. **Next research state.** Close this pathway and preserve artifacts without changing P0–P6. Any future test with a different model, training, prompt scheme or readout would be a **new scientific hypothesis**, not a continuation that overturns these frozen negative results.

### Gate ledger

| Layer | Verdict |
|---|---|
| Code/contract validation | PASS (85 user-reported unit/regression tests before basic-16 run) |
| Fixed-grid execution | COMPLETED (operator reported) |
| Fixed-grid artifact validation | PASS (operator reported) |
| Basic-16 execution | COMPLETED (operator reported) |
| Basic-16 artifact validation | PASS (operator reported) |
| H2 fixed-readout scientific success | **NO SUPPORT** |
| P/Q basic-readout eligibility | **FAIL — 8/16 vs required 16/16** |
| H3 native-state causal mediation | NOT ESTABLISHED |
| Phase3B | BLOCKED_UNCHANGED |
| GPU / training | OFF / NOT PERFORMED |

**Final decision:** `RFR_H2_FIXED_READOUT_PATH=CLOSED_INELIGIBLE / NEGATIVE_RESULTS=PRESERVED / ADDITIONAL_EVALUATION=NO / PHASE3B=BLOCKED_UNCHANGED`.
