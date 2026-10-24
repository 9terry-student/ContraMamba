# ContraMamba RFR-H2-P0 — Temporal Serialization and Readout Identifiability Design (Candidate)

- Date: 2026-10-24 (Asia/Seoul)
- Branch: `gen5-causal-role-state-ownership`
- Inspected baseline HEAD: `c31eab558c5b66acc671241df1ed67603161a926`
- Status: `RFR_H2_P0_STATIC_DESIGN_ONLY / NO_MODEL_OR_DATA_EXECUTION / NO_CLAIM_PROMOTION`
- Purpose: separate temporal evidence-conditioned *behavior* from native-state *mediation*, without altering the frozen RFR oracle, Gen4 observer or Gen5 architecture.
- Authority boundary: This candidate documents a prospective falsifiable design. It authorizes **no** root generation, seed, split, text/row materialization, tokenizer loading, model execution/inference/evaluation/training, checkpoint mutation, native-state intervention, Kaggle/GPU, or Phase3B promotion.

## 1. Frozen sources and the unresolved scientific question

- `docs/CONTRAMAMBA_RETAIN_FORGET_REVISE_COMPUTATIONAL_HYPOTHESIS_2026_10_16.md` defines Retain, Forget and Revise as separate computational hypotheses, demands functional use and causal controls, and states that correct output alone does not identify a raw-state mechanism.
- `reports/rfr_d0_dependency_aware_revision_scientific_design_candidate.md` defines `A`, prerequisite `C`, `B := A AND C`, unrelated `U`, timestamped provenance-qualified evidence and paired `VALID_FLIP`/`VALID_SHAM`/`INVALID_FLIP` contrasts. D0 first asks whether labels and tasks can be defined, not whether native memory was observed.
- `scripts/rfr_d1_deterministic_oracle_cpu.py`, frozen D2/D3 and M1–M3 provide deterministic oracle/contract checks and in-RAM synthetic probes, not model behavior. M4–M8 reject claims of independently sampled or task-bearing-diverse roots; the M1–M3 family frame must not be promoted to a confirmatory cohort.
- `scripts/rfr_d3_m2_shortcut_surface_cpu.py` visibly repeats all initial evidence alongside t1 in POST/DELAYED prompts. Consequently its prompt-level task permits rereading the presented evidence, and by itself cannot distinguish functional recomputation from persistent state-mediated revision.
- `scripts/reason_router_gen4_native_mamba_state_measurement.py` and `scripts/reason_router_gen4_native_mamba_state_extraction.py` demonstrate *a bounded mechanism for capturing token-indexed raw post-consumption `ssm_state` within one `model.mamba(input_ids=...)` forward*. Their frozen source hash, runtime, layer-11, shape, 128-token and historic checkpoint/data conditions are **not** an RFR-compatible gate or RFR execution permission.
- `src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py` uses a historically trained claim/evidence router and readout; matching class labels do **not** establish an RFR A/B/U decoder. `src/contramamba/gen5_phase2_state_update_ownership.py` explicitly rejects `cache_params` and `cache_position` in its current reference wrapper.

**Decision:** The next scientific target is a *prospective single-causal-stream temporal functional-use design* for H2, not a claim of identified state mediation. H3 raw-state perturbation/removal/restoration remains a separate later question.

## 2. One causal sequence; no repeated records after the update

Define one root world with a frozen, explicitly presented rule and initial authenticated records, then separately present a t1 event, neutral delay and a final query. Conceptual order (not generated text):

```text
[rule B := A AND C; synthetic authority/recency rules]
[t0: authenticated A=a0; authenticated C=c0; authenticated U=u0; time=10]
[t1: exactly one A event; stated_value in {a0, not a0}; time=20; source_authenticated={true,false}]
[optional neutral delay tokens: cannot assert A/C/U, authenticate or override sources]
[query, *only here*: Q(A) or Q(B) or Q(U)]
[answer continuation scored at the causal LM endpoint]
```

- `PRE`: truncate the stream after t0 and ask the query; do not add t1 or leak its planned arm in any header. Its three nominal arms therefore share exactly the **same input bytes** for a fixed root, query and permitted surface realization.
- `POST`: include t1 and ask immediately afterwards, without neutral delay.
- `DELAYED`: include the **same** t1 bytes as POST, add a separately specified strictly neutral interval, then ask. Delayed success means bounded retention under the specified interval, **not** persistent memory across calls or parameter learning.
- The source authentication and logical timestamps must be available **in task text**, not privileged to the oracle or hidden metadata. Source alias, validity signal, wording, ordering and token lengths must be checked as potential shortcuts, not silently assumed balanced.
- PRE/POST/DELAYED are separate causal-prefix observations; they are not a single run that reveals three answers and revisits them. A/B/U queries branch only **after** the common pre-query prefix. Query-target labels, root/arm/split IDs, answer/gold and provenance manifests must never appear in this prefix.
- The PRE stream and the prefix common to all arms must be byte-identical for a fixed root/surface realization. For any arm/readout, A/B/U pre-query prefixes must be **token-identical**; query text can differ only after the shared anchor. The same root's post-t1 states may differ between arms, which is the intended manipulation.
- `event_time=10/20` is synthetic logical time. Token index is model-computation time; neither is wall-clock time. Keep separate anchors for rule end, t0 end, t1 boundary, delay end and pre-query boundary.
- The freeze of an exact natural-language serialization, answer format, tokenizer token IDs, delay length, maximal sequence length and anchors requires a later **separate** static/CPU-only contract gate; none is picked by this document. Never silently truncate the dependency rule, certification, changed record or query.

## 3. Preserve D1 truth constraints and matched causal contrasts

For strict-Boolean initial `(a0,c0,u0)` and each arm, oracle expectations are:

| Arm | PRE A | POST/DELAYED A | POST/DELAYED B | POST/DELAYED U |
|---|---|---|---|---|
| `VALID_FLIP` | `a0` | `not a0` | `(not a0) AND c0` | `u0` |
| `VALID_SHAM` | `a0` | `a0` | `a0 AND c0` | `u0` |
| `INVALID_FLIP` | `a0` | `a0` | `a0 AND c0` | `u0` |

- At `C=T`, VALID_FLIP must change B in the appropriate direction; at `C=F`, **B stays REFUTE** across arms even when A flips. For both values of `U`, its readout must be preserved at all time points.
- `INVALID_FLIP` tests unauthorized contradiction, whereas `VALID_SHAM` controls mere authenticated event presence. The three-arm primary estimand must remain intact; optional `MISSING_PREMISE` is a separately predefined later family.
- **Epistemic limit:** the eight strict-Boolean primary worlds contain no unknown A/C/U baseline, so the correct primary labels cover `SUPPORT`/`REFUTE` only. The primary design does **not** test correct `NOT_ENTITLED` handling, even if a future readout outputs three labels. Never infer three-way epistemic competence from two-label support.
- Preserve D1's `REFUTE=0`, `NOT_ENTITLED=1`, `SUPPORT=2` oracle convention for all reporting; do not infer mapping from historic Gen3/Gen4 tensor-column comments or labels.
- Row accuracies are correlated descendants. No independent root-family count, train/dev/test isolation, statistical power, model-generalization claim or family-wise confidence interval is established by this prospective design.

## 4. Readout choice: causal continuation, not the legacy claim/evidence router

**Prospective primary readout:** a *frozen, revision-pinned, causal language-model head* on a compatible native Mamba backbone, evaluated at the final query boundary with a predeclared three-answer continuation/verbalizer scheme and teacher-forced conditional log-likelihood. This choice introduces **no learned RFR readout** and avoids the historical claim/evidence Frame/Predicate/Sufficiency router.

- Feasibility is **UNVERIFIED**: existence, exact model and LM-head weights, tokenizer revision, vocabulary, special-token policy, causal forward mode, token budgets, answer continuation tokenization, and whether all class candidates can be scored comparably have not been audited. Do not claim that the Gen4 classifier checkpoint automatically supplies this head, or that two models/checkpoints are comparable.
- Before any inference, freeze a single response scaffold, one continuation per `REFUTE`/`NOT_ENTITLED`/`SUPPORT`, handling of preceding whitespace/newline and EOS, and **a single candidate-score aggregation rule**. Compute scores only from the same pre-query prefix. Multiple-token verbalizers and differing continuation lengths can introduce class-prior/length bias; they require a prospective normalization/control decision or block, never opportunistic calibration after seeing answers.
- If no trustworthy, class-comparable training-free readout exists, record `BLOCKED_UNVALIDATED_FUNCTIONAL_READOUT`. Do **not** fit a probe, train a new classifier, use oracle/gold-based prompt selection, or substitute the historic router within this stage.
- All eventual model behavior must be compared with constant-class, latest assertion, trust-marker-only, query-only, `C`-omission/rule-agnostic and source/template baselines, including conditional `(query, C, arm, readout, a0, u0)` and matched-family contrasts. Report full class score vectors and paired joint case success if a later evaluation is authorized.
- Even if a separately authorized frozen LM fails the behavioral contrasts, that outcome would be limited to the tested model, continuation readout and task formulation; it cannot alone prove that no useful native-state content exists. Conversely a successful continuation is functional evidence, not causal-state identification.

## 5. Identifiability ladder and independent negative explanations

| Claim | Minimum prospective observation | Not justified by that observation |
|---|---|---|
| `H2-FUNCTIONAL-USE` | Correct A update only on authenticated flip; B conditional on C; U preserved; PRE/POST/DELAYED matched contrasts with no replayed evidence after t1 | A particular recurrent site performs the revision |
| `H2-DELAYED-USE` | The same conditional behavior after a neutral, controlled delay | Persistence across independent requests; parameter learning |
| `H2-SHORTCUT-RESISTANCE` | Conditional performance rejects constant-class, latest-assertion, trust-marker, rule-agnostic and query-only explanations | Absence of all lexical/template shortcuts or independent-root generalization |
| `H3-STATE-ASSOCIATION` | Later, prefix-only raw post-update state capture at frozen event anchors and a predeclared layer | Necessary/sufficient state mediation; decodable signal is not enough |
| `H3-CAUSAL-MEDIATION` | Later, site-/time-/norm-matched intervention, necessity/replacement/restoration, B rescue and U preservation in held-out controls | Automatically proves distinct Retain/Forget/Revise operators or a new architecture |

A native Mamba causal forward uses preceding tokens through model computation rather than explicit whole-prompt rereading at the output endpoint, provided the head is truly causal and uses only the terminal token. Even so, a successful terminal readout does not identify **which** recurrent layer/site stores a revised belief or whether information was selectively forgotten. If future implementation pools all intermediate token states for classification, the alleged causal-memory bottleneck is broken and H2 state-memory interpretation must be downgraded.

The existing Gen4 observer is source- and runtime-pinned to a particular native `MambaMixer.slow_forward`, primary layer 11 and exact post-consumption state shape. A fresh RFR runtime binding and anchor/shape audit would be needed later; do not repurpose historic hashes, assume another backend's equivalence, reuse Q1/Q3 layers for convenience, or use output proxies as raw write.

## 6. Explicit fail-closed decisions and bounded next action

- `BLOCKED_REPEATED_EVIDENCE`: if t0 or t1 records recur verbatim or as an answer-bearing summary after t1, the test cannot claim no-replay temporal use.
- `BLOCKED_QUERY_PREFIX_LEAKAGE`: if the A/B/U branch affects any token before the designated query anchor or if hidden family/arm/oracle fields become model visible.
- `BLOCKED_SERIALIZATION_OR_TOKENIZER`: if readout candidates, authority, rule, timestamps, anchors or token budgets cannot be frozen and validated under a pinned tokenizer.
- `BLOCKED_UNVALIDATED_FUNCTIONAL_READOUT`: if no causal terminal readout and comparable three-class scoring can be established without post-hoc fitting.
- `BLOCKED_SURFACE_OR_INDEPENDENCE`: if M1–M3 related rows are used to imply confirmatory population independence, or semantics of D1 authentication are silently changed.
- `BLOCKED_H3_PROMOTION`: if correct answers, post-update state differences or a decoded signal are reported as causal mediation without removal/restoration and matched functional controls.

**P0 result:** `H2_P0_PROSPECTIVE_DESIGN_COMPLETE / SERIALIZED_PROMPTS=NONE / TOKENIZER_AUDIT=NOT_EVALUATED / READOUT_FEASIBILITY=NOT_EVALUATED / MODEL_EVALUATION=NOT_EVALUATED / NATIVE_STATE_CAUSAL_MEDIATION=NOT_ESTABLISHED / INDEPENDENT_COHORT=NOT_ESTABLISHED / PHASE3B=BLOCKED_UNCHANGED`.

**Next bounded scientific action (not authorized by this design alone):** a CPU-only, deterministic **H2-P1 static serialization/continuation-contract feasibility check**, with no seed, new sampled root, materialized dataset, tokenizer/model load, trained readout, cache continuation, or evaluation. Use deliberately hand-authored, clearly hypothetical example descriptions (not newly counted sampled roots) and independently checked invariants for causal ordering, prefix equality, D1 labels and query isolation. If selecting actual token IDs or a model revision is required, stop and perform a separate pre-execution tokenizer/readout feasibility review. No Gen4/Gen5 frozen files are to be changed by P0.
