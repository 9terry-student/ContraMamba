# ContraMamba RFR-H2-P6 — Unified Pre-Experiment Protocol and Execution Contract (Candidate)

- Date: 2026-10-24 (Asia/Seoul)
- Branch: `gen5-causal-role-state-ownership`
- Verified base commit: `9195fe8f2f76b39b81de0f34e6ea4a19065456b7` (P5 report freeze; operator states clean worktree)
- Phase: **single consolidated pre-experiment design**, not execution or model-evaluation authority.
- Status: `P6_UNIFIED_DESIGN_READY_FOR_FREEZE / EXPLORATORY_H2_DIAGNOSTIC_SPECIFIED / EXECUTION=NOT_AUTHORIZED / PHASE3B=BLOCKED_UNCHANGED`.
- Scope boundary: No modifications to D0/D1/D3/M1–M8/P0–P5, Gen4/Gen5 core, model/checkpoint/tokenizer, datasets or splits. No Codex, training, inference, new roots, tokenizer rerun, native-state interventions, Kaggle or GPU during this design. A successful code gate will not authorize evaluation.

## 1. Research question, claim ceiling and decision

**One question:** Under the *already frozen* English rule/evidence/query serialization, does the pinned pretrained Mamba causal LM produce correct authentication-sensitive A revision, C-gated B consequence, unrelated U retention and prescribed short neutral-delay behavior in one causal token stream?

**Decision:** Bundle every remaining pre-experiment **design** requirement here. Do not create serial P7/P8 design-only phases by default. The next task following this report's user-controlled freeze is a **single bounded CPU-only evaluator implementation + adversarial tests**, with model execution explicitly disabled by default; only a subsequent, separately authorized exploratory run may execute the 168 unique RFR prompts. If implementation exposes a substantive scientific ambiguity, amend this contract rather than silently tuning it.

Claim ceilings, independent of implementation quality:
1. Success on the fixed grid establishes at most *observed behavior on the pinned fixed-rule Boolean grid, model, tokenizer, verbalizers, template, runtime and prescribed delay*.
2. It does **not** establish independently sampled task-family generalization, a three-way epistemic competency (`NOT_ENTITLED` has no gold primary examples), alias-invariant authentication semantics, or H3 selective recurrent-state causal necessity/sufficiency.
3. D3-M8's `INDEPENDENT_ROOTS_NOT_ESTABLISHED` and `CURRENT_CONTRACT_TASK_BEARING_DIVERSITY_NOT_DEMONSTRATED` remain binding. No confidence intervals, p-values, 8 independent roots or ICML-ready causal claims from this experiment.

## 2. Frozen model, runtime and readout

- Model and tokenizer: `state-spaces/mamba-130m-hf` @ `5708daa364c50b880e7bd92eab456e0d34492ee9`; never mix historical Gen4 tokenizer revision `40e5d2bd...`.
- `model.safetensors`: 516,567,560 bytes, SHA256 `1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea`; `config.json`: SHA256 `784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`.
- P3 tokenizer SHA256: `tokenizer.json=3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8`; `tokenizer_config.json=fcd5669efe1150240c13ee4bd863316de4f2abd14cb1806a8cdbcbea6577bc99`; `special_tokens_map.json=10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c`.
- Operator-tested Windows runtime: Python 3.13.2, PyTorch 2.12.1, Transformers 4.45.0, tokenizers 0.20.3, safetensors 0.8.0; installed `modeling_mamba.py` 37,982 bytes, SHA256 `0fb3d343b2016aa4abe0c493ae143dc2193e1148bdd8792e9486ad04f5e30d71`. Any drift is a hard stop, not an invitation to update versions.
- Prior P4 **operator console smoke**: no missing/unexpected checkpoint keys, actual LM-head/input-embedding weight tying, one CPU `input_ids=(1,151)` on repeated ID 42 with finite logits `(1,151,50280)`, `use_cache=False`. This is technical feasibility only; no archived execution record and no RFR prompt/model evaluation.
- Future evaluation calls (only after separate authorization): CPU float32, `model.eval()`, `torch.inference_mode()`, `use_cache=False`, `labels=None`, `output_hidden_states=False`, `return_dict=True`; no beam search/generation, answer appended to input, GPU, trainable head/probe, external class mapping, cache state reuse or pooling over tokens.
- Freeze exact P1 clauses `RULE_TEXT`, `_t0_text`, `_t1_text`, `DELAY_TEXT`, `_query_text` and their D1 semantics. PRE physically contains no t1. Query A/B/U branches only after shared prefix; no evidence replay after t1.
- P3 **readout seam only**: visible P1 `Answer: ` remains unchanged in completed text; to score next-token class, remove its terminal space from the *model input*, so the prompt ends in `Answer:`. Exactly one candidate continuation per class includes the space: `REFUTE=' False' → 16708`, `NOT_ENTITLED=' Unknown' → 32471`, `SUPPORT=' True' → 11793`. No EOS. Do not edit frozen P1 source.
- P3 established 216 nominal contexts, 864 anchors, 648 candidate checks, fixed single-token IDs and complete sequence length 113–151; re-authenticate all relevant exact files, P3 anchors and key identities **before** any future evaluation. Prompt-only length maximum 150. No silent truncation. P4 shape smoke is not a blanket maximum-context guarantee.

## 3. Finite diagnostic population and exact causal contrasts

One logical stratum is `(a0,c0,u0) ∈ {F,T}³`, giving **8 exhaustively enumerated related Boolean strata, not independent sampled cases**. For each stratum:

- Three unique PRE prompts: `A/B/U` at t0, with future arm hidden and **identical bytes and token IDs across all nominal arms**.
- Eighteen unique post-t1 contexts: `{VALID_FLIP, VALID_SHAM, INVALID_FLIP} × {POST, DELAYED} × {A,B,U}`.
- **21 unique prompts per stratum; 168 total.** Do not run/recount the extra 48 nominal-arm duplicate PRE presentations. Nominal 216 is used solely for factor-contract crosschecks.
- At t1, valid newer authenticated flip revises A; authenticated sham and unauthenticated flip preserve t0 A. `C` and `U` remain t0-authenticated. `B = A AND C` under D1 strong Kleene; strict Boolean primary has only `REFUTE/SUPPORT` gold.
- All full-case assertions: A changes iff authenticated flip; B follows changed A iff `C=T` and remains `REFUTE` when `C=F`; U retains its original value at PRE/POST/DELAYED; INVALID_FLIP does not overwrite A/B; DELAYED equals POST gold under frozen neutral interval. Both directions of A and both U truths must appear.
- Paired unit is the **entire logical stratum**, not row or token. Preserve linked identifiers for its 21 contexts but do not imply independence. No seed, RNG, new root, split, paraphrase or sampling claim.

## 4. Predetermined score and decision rules

- For each unique prompt only, compute `z = (logits[0,-1,16708], logits[0,-1,32471], logits[0,-1,11793])` with class order `(REFUTE=0, NOT_ENTITLED=1, SUPPORT=2)`.
- Reject wrong tensor shape, unexpected vocab, nonfinite class logits, non-causal position, shifted label IDs, changed tokenization or tied maximum. A valid prediction requires **one unique maximum**; all invalid cases count as incorrect and remain in the output audit, not silently skipped.
- Record all three raw logits, winning class, gold class and `margin = z_gold − max(z_other classes)`; optionally also stable `softmax(z)` **within these three candidates only**, explicitly uncalibrated and not a complete-vocabulary probability.
- No temperatures, thresholds, per-class prior subtraction, class balancing, preferred template/verbalizer selection, best-layer search, gold-assisted calibration, model adaptation or post-hoc rescue. Flat exact ties are errors.
- Report full confusion by `(query, C, a0, u0, arm, readout)`; gold class coverage; answer-token prediction frequencies; PRE/POST/DELAYED; normalized candidate-only scores and signed margin distributions as descriptive statistics, never independent-error estimates.
- Per-stratum `PRE_joint` requires all three unique PRE predictions correct; `POST_joint` and `DELAYED_joint` each require all nine corresponding predictions correct. `CASE_joint = PRE_joint ∧ POST_joint ∧ DELAYED_joint`. Aggregate `CASE_joint` as count / **8** with every individual stratum visible. **8/8 is necessary for full fixed-grid smoke success**, never sufficient for generalized RFR or H3 causal claims.
- Also show stricter conditional contrasts: A update (both directions), B gate `C=T`, B blocked `C=F`, U retention (both values), INVALID trust resistance, POST→DELAYED retention. Every failure is a recorded negative pattern; no moving the goalposts to averaged accuracy.

## 5. Baselines: integrate comparisons in the same report, no extra model calls

Every baseline is **deterministic** on the same 168 logical prompt identities and compared to the frozen oracle. Compute analytically in the CPU report function, independently of LM logits; no fit/search/selection:

| Baseline ID | Exact decision rule | Diagnostic |
|---|---|---|
| `ALWAYS_REFUTE` | REFUTE for every context | Constant/class collapse |
| `ALWAYS_NOT_ENTITLED` | NOT_ENTITLED for every context | Unsupported unknown-token prior |
| `ALWAYS_SUPPORT` | SUPPORT for every context | Constant/class collapse |
| `T0_ONLY` | Use t0 A/C/U at PRE, POST and DELAYED; derive B=A AND C | Failure to revise |
| `LATEST_ASSERTION` | PRE: t0 values; POST/DELAYED: take t1's A value regardless of authentication, keep C/U; derive B | Trust-blind latest-message following |
| `IGNORE_C` | Admit newest authenticated A as D1, but predict B from A alone (`C` treated as TRUE); retain t0 U | Missing prerequisite/dependency shortcut |
| `QUERY_ONLY` | A→REFUTE, B→REFUTE, U→SUPPORT for *all* contexts; fixed before looking at logits | Target-only fixed answer |
| `AUTH_FLAG_ONLY` | PRE→REFUTE; POST/DELAYED authenticated t1→SUPPORT, unauthenticated t1→REFUTE for *any* query; no values read | Trust-marker-only shortcut |

- `SOURCE_ALIAS_CONFOUNDED`: The frozen text uses `REV_A` exactly when authenticated and `UNVERIFIED_A` exactly when unauthenticated. No source-invariance/authority-cue robustness claim can be inferred from primary grid success. An alias/authority crossover requires new surface + tokenizer contracts in a **later independently decided** study, not a free P6 extension.
- `TOKEN_PRIOR_CONTROL_NOT_DESIGNED`: Candidate-token frequency bias and answer-neutral prompt baseline cannot be corrected/claimed absent a *separately predeclared token-valid neutral control*. Do not invent one from observed logits; simply report relative candidate frequency and limitations.
- These deterministic baselines are not a learned reference architecture and have different information restrictions. State their information budgets and count their oracle matches, never call them causal interventions or train/dev/test baselines.

## 6. One implementation task, one execution boundary, one complete evidence bundle

**Implement once after P6 is frozen**; the implementation stage may add only a dedicated evaluator and its tests (e.g. `scripts/rfr_h2_p6_exploratory_grid_cpu.py`, `tests/test_rfr_h2_p6_exploratory_grid_cpu.py`; exact names are proposal until scope authority). The evaluator must default to **no model execution**, exposing a pure model-free contract/baseline/audit path and a separately gated, explicitly enabled inference path. Never silently run at import or when tests are invoked.

Implementation tests (using fake logits/stubs, no real model): exact 168 unique contexts; 48 duplicate PRE nominal rows deduplicated; 8-stratum D1 labels; all paired truth contrasts; fixed 3-class score ordering; invalid/tied/nonfinite results counted as errors; baseline expected outputs; stable ordering and record/manifest hashes; fail closed on changed alias/scaffold, tokenizer/checkpoint/source, class IDs, prompt byte and token anchors, input truncation, hidden gold/arm leakage; no generated roots or data writes in no-run mode. Run narrow tests once and user-run `cm ship` once before manual commit.

**A later exploratory run requires a separate explicit execution decision**, a specific newly frozen commit/clean HEAD, verified local checkpoint/tokenizer/runtime hashes and budget, exact CPU-only command, and new run identity bound to that commit; no reusing an earlier commit's results. No training, GPU/Kaggle or H3 instrumentation. If run cost or model compatibility warrants change, stop and report before changing the contract.

Upon approved execution, produce **one** non-overwriting, manifest- and checksum-bound artifact directory containing at minimum:
- `manifest.json`: exact HEAD, branch, dirty status, model/tokenizer/revision + all file SHA256, installed runtime + Mamba source SHA256, P1/P3/P5/P6 source hash, frozen readout IDs and configuration, UTC timestamp, run name and scope `EXPLORATORY_NONCONFIRMATORY`.
- `predictions.jsonl`: all **168** unique `(world, arm or PRE, readout, query)` records, serialized prompt hash and token count, model-gold-free input identity, frozen D1 gold, full three candidate logits, finite/tie status, prediction, signed gold margin and limited softmax diagnostics; avoid copying private paths or uploading raw checkpoint bytes.
- `baseline_predictions.jsonl`: deterministic baseline predictions and gold comparisons for the same 168 records; fully frozen rules and their information budgets.
- `summary.json` and `report.md`: full strata/pair/joint confusion, all 8 stratum results, baseline tables, failures, missing NE coverage, alias/prior confounds, scientific claim ceiling, no statistical inference.
- `SHA256SUMS.txt` over all other artifacts; fail closed on missing/extra artifacts, partial runs, duplicate IDs, invalid hashes or mismatch in HEAD, tokenizer, checkpoint, source or sample count.

Do not auto-stage results; after scientific and provenance validity review separately decide which small reports to commit. The user's local runtime controls all execution and git staging/commit/push.

## 7. Execution readiness and stop/go criteria

**Passed from existing reported evidence:** H2-P1 42 tests; H2-P3 15 tests + pinned tokenizer 216/648/864 static checks; H2-P4 authenticated weights, tied head, one finite `(1,151,50280)` sequential-CPU smoke; P5 frozen prospective scientific design; P6 a unified specification only. These **do not** constitute an evaluator implementation or RFR experiment.

**Still required before a first exploratory run:** (i) dedicated evaluator+adversarial tests verified from exact new commit; (ii) complete prompt and oracle coverage, unique-context dedup and baseline checks; (iii) authenticated preflight of checkpoint/config/tokenizer/Mamba source and current local environment; (iv) fail-closed readout/output artifact/provenance contract; (v) separate explicit CPU exploratory-run authorization. Until then `EXECUTION=BLOCKED`.

**Hard scientific claim blockers even after 8/8:** zero primary NOT_ENTITLED gold, source aliases perfectly confounded with authentication, no independently sampled task-bearing roots, no controlled token-prior baseline, no H3 intervention. A non-PASS can be reported as a negative/partial *fixed-grid observation*, without prompt tuning or expanding the study after viewing outputs.

**No further preparatory phase slicing:** Do not create P7/P8 paperwork merely to reconfirm previously frozen semantics, rehash the same files in chat, or add decorative provenance. Escalate only upon a new empirical failure or genuine scientific design fork.

**Final:** `H2_P6=UNIFIED_PREEXPERIMENT_CONTRACT_CANDIDATE / NEXT=ONE_BOUNDED_IMPLEMENTATION_AND_TEST_TASK / EVALUATION=NOT_AUTHORIZED / INDEPENDENT_COHORT=NOT_ESTABLISHED / H3_MEDIATION=NOT_ESTABLISHED / PHASE3B=BLOCKED_UNCHANGED / GPU=OFF`.
