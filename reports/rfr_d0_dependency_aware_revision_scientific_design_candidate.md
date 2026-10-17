# ContraMamba RFR-D0 — Dependency-Aware Revision Structural Cohort Design

**Status:** `SCIENTIFIC_DESIGN_CANDIDATE / DESIGN_ONLY / NO_DATA_GENERATION / NO_SCIENTIFIC_EXECUTION`
**Date:** 2026-10-17 (Asia/Seoul)
**Repository:** `9terry-student/ContraMamba`
**Design baseline HEAD:** `4c98858fa2b9d3a90f2329dafd70655036c249e1`
**Branch:** `gen5-causal-role-state-ownership`
**Proposed file:** `reports/rfr_d0_dependency_aware_revision_scientific_design_candidate.md`

## 0. Authority and evidence boundary

This document is a **new scientific-design candidate**, not an execution authority and not evidence for Retain–Forget–Revise (RFR). Its scientific objective is inherited from:

- `docs/CONTRAMAMBA_LONG_TERM_RESEARCH_PROGRAM_2026_10_16.md` (long-term intent, not execution authority);
- `docs/CONTRAMAMBA_RETAIN_FORGET_REVISE_COMPUTATIONAL_HYPOTHESIS_2026_10_16.md` (RFR hypothesis, not execution authority);
- the frozen Gen5 Phase3 XG1 source/generator artifacts (comparison/negative historical context, not data to relabel).

The operator's **read-only schema audit**, performed at baseline HEAD, reported:

```text
FACT_ROWS=600
SEVEN_CELL_ROWS=4200
DEV_PAIRS=120
EXPLICIT_DEPENDENCY_FIELD_CANDIDATES=[]
RFR_XG1_SINGLE_SOURCE_PAIR_STRUCTURE_PASS
SOURCE_PAIRS=600;ROWS=4200;CELLS_PER_PAIR=7
CROSS_PAIR_REFERENCES_BY_SCHEMA=NONE
DEPENDENCY_AWARE_REVISION=NOT_DEFINED
MODEL=OFF;GPU=OFF;FILES_WRITTEN=0
```

Repository inspection confirmed that `scripts/prepare_reason_router_gen5_phase3_static.py` derives C6 `REFUTE` from explicit denial, whereas the original XG1 generator provides six within-source contrasts. These are not a time-dependent **dependency-aware belief revision** population. The absence of a dependency *field* is a schema finding; this design does not claim to prove the impossibility of every implicit natural-language inference in the old rows.

**Immutable historical objects:** XG1 rows, labels, dev/train pair split, active encoding, old generated facts, Phase2/3A results, G2 H2/B artifacts, checkpoint, tokenizer, native transport and v10 artifacts. No reuse of the existing 120 dev pairs as a newly independent confirmatory cohort.

## 1. One new falsifiable question

> Given an explicit dependency rule and time-stamped, provenance-qualified evidence, can a model update proposition **A**, recompute its dependent conclusion **B**, and preserve unrelated proposition **U**, including after distractors — without replacing revision by a blanket REFUTE or NOT_ENTITLED readout?

RFR-D0 is first a **data/label feasibility design**. Even a later input–output success would show only bounded **functional use of revised evidence**. It would not by itself establish a native Mamba state write, persistent memory, independent Retain/Forget/Revise operator, or causal mediation.

## 2. Minimal logical world and epistemic oracle

For each *new* independent synthetic case, define four proposition identifiers without recycling frozen XG1 source pair IDs:

- `A`: revisable base proposition;
- `C`: separately verified prerequisite;
- `B`: derived proposition with frozen rule `B := A AND C`;
- `U`: verified proposition unrelated to A, B, and C.

At `t0`, authoritative records explicitly certify `A=a0`, `C=c0`, `U=u0`, with `a0,c0,u0 ∈ {0,1}`. **Negative** evidence is an explicit certified false assertion, not merely the lack of positive evidence. The model is supplied the dependency rule; this is not a test of spontaneously discovering a hidden graph.

At `t1`, the evidence event has *separate*, fully specified attributes: target variable, stated value, authenticated source status, and event time. The oracle admits an update only if the source is authenticated and its event time is newer than the stored valid record. A later-presented but older record does **not** override a newer authenticated correction. Unauthenticated claims have no power to override an established authenticated record; the latter remains available. This is a **stipulated synthetic authority protocol**, not a claim about real-world evidence trust.

With three-valued epistemic state `T`, `F`, `?`, set:

```text
B = F  if A=F or C=F
B = T  if A=T and C=T
B = ?  otherwise
```

For an affirmative query `Q(X): "Is X true under the current verified record?"`:

```text
X=T -> SUPPORT (2)
X=F -> REFUTE (0)
X=? -> NOT_ENTITLED (1)
```

These are labels **relative to the stipulated source-validity protocol and explicit derivation**, not objective truth claims about real-world facts. The `NOT_ENTITLED` state is not a synonym for *forgetting*; it denotes absence of a justified answer under the protocol.

## 3. Fixed D0 manipulation family

The primary comparison is a within-case, paired **three-arm** update design. Every case has one fixed `a0,c0,u0` and the same rule/identifiers. Variants differ by one admissible event at `t1`:

| Arm | `t1` message value | Source status | Oracle A after `t1` | Diagnostic purpose |
|---|---|---|---|---|
| `VALID_FLIP` | `NOT a0` | authenticated, newer | `NOT a0` | genuine revision and possible dependent change |
| `VALID_SHAM` | `a0` | authenticated, newer | `a0` | matched update without revision |
| `INVALID_FLIP` | `NOT a0` | unauthenticated | `a0` | contradiction wording alone must not trigger revision |

All three arms retain `C=c0` and `U=u0`. An optional independent `MISSING_PREMISE` eligibility-control family may withhold initial C authority and test `B=?` when A=T, while preserving `B=F` whenever A=F. This **control is not part of the three-arm primary estimand**, and must not be added after seeing any outcomes.

Factors required in a later cohort specification, before any generator code:

- `a0 ∈ {0,1}`: both revision directions `T→F`, `F→T`;
- `c0 ∈ {0,1}`: only `c0=1` makes B change under an A flip; `c0=0` is the *blocked-dependency* control;
- `u0 ∈ {0,1}`: preserve both true and false unrelated facts;
- `query ∈ {A,B,U}`: do not only measure A;
- `readout ∈ {PRE, POST, DELAYED}`: direct and later functional use.

The **case ID**, not the row, condition, query, prompt paraphrase, or delay realization, is the independent sampling unit. All arms, readouts and paraphrases for the same case must share a single split. Number of cases, independent generator families, final train/dev/test sizes and prospective power/precision thresholds remain **UNSET** in D0: this document does not silently choose sample sizes or authorize tests.

## 4. Logical falsification checks before model execution

For each case, derive the labels mechanically from the oracle. Mandatory pairwise assertions:

1. `VALID_FLIP` changes `A` relative to both `VALID_SHAM` and `INVALID_FLIP`, for either original truth value.
2. When `C=T`, the same valid flip changes `B` in the corresponding direction; this is the **dependent consequence**. The other two arms retain the old B.
3. When `C=F`, `B=F` across all three arms even if A changes. Failure here identifies a simple *all-dependents-flip* shortcut.
4. In all primary arms and time conditions `U=u0` is constant. Its correctness is an indispensable collateral-damage control.
5. `PRE` uses only `t0`; `POST` uses `t0+t1`; `DELAYED` uses `t0+t1+neutral distractors`. Distractors cannot add an admissible event about A, C or U.
6. For fixed semantic world and authenticated timestamps, changing the **presentation order** of already-timestamped records must not change the oracle. This is an order/recency shortcut control, distinct from changing the **event timestamps** themselves.
7. Opposite truth directions and opposite query answers must occur in each retained split, or feasibility fails and the cohort is not promoted.
8. `REFUTE` and `NOT_ENTITLED` must never be collapsed or relabeled to inflate accuracy.

These assertions define **generator/label invariants**, not achieved scientific findings. Any non-determinism, circular derivation, overwritten original fact, inconsistent evidence ranking, or label conflict is a hard blocker.

## 5. Future row-level contract (design only)

A later separately authorized generator may materialize an **independent**, versioned artifact family, but the proposed semantic schema must carry at minimum:

```text
case_id, case_family_id, variant_id, update_arm,
initial_A, initial_C, initial_U,
derivation_rule_id,
t0_authoritative_record_ids, t1_event_id,
t1_source_authenticated, t1_event_time,
presentation_order_id, readout_time, distractor_id,
query_target, target_status, final_label,
oracle_version, generator_version, split_group_id
```

For each event, also preserve source provenance, exact input text, timestamp, and immutable assertion content; for every B label, preserve the *input A/C state* used by the oracle. `split_group_id` must link all realizations from the same abstract case, including reverse-label permutations. Materialize detailed per-record SHA256 and a corpus manifest when (and only when) data generation receives separate authority. Do **not** reuse `xg1_fact_...` identifiers.

The token count, event positions, serialized bytes, and actual tokenizer identity are not yet frozen. A subsequent tokenizer feasibility gate must reject truncation that removes the rule, evidence authority, correction, or query. Lexical format, source-authentication marker, update wording, query position, and neutral distractor length must be counterbalanced or explicitly flagged as confounds; **do not invent a token-match PASS before checking an authenticated tokenizer**.

## 6. Future experimental estimands — not executed in D0

If a later cohort and evaluation authority pass, report **joint case-wise functional success**, not just mean accuracy:

- `Revise`: A correct at POST/DELAYED on VALID_FLIP, in both directions;
- `Dependency use`: B correct at POST/DELAYED for `C=T`, including comparison against both SHAM and INVALID_FLIP;
- `Blocked dependency`: B unchanged when `C=F`;
- `Retain`: U correct and stable across PRE/POST/DELAYED;
- `Evidence entitlement`: INVALID_FLIP must not trigger an unauthorized A/B update; unknown controls must map to NOT_ENTITLED where the oracle requires;
- `Joint`: all applicable A/B/U answers correct **for the same case and variant**.

Report per-condition confusion matrices, REFUTE/SUPPORT/NOT_ENTITLED by target, flip-direction, dependency gate, trust status and delay. Include always-REFUTE, always-SUPPORT, always-NOT_ENTITLED, last-mention, source-status-only, and rule-agnostic response baselines. Keep paired scores within case; no treating repeated queries as independent observations. No p-values, effect sizes, sample sizes, or promotion criteria are selected from downstream outputs.

A future native-state claim would additionally require **prefix-only** time-aligned raw recurrent-state measurement (not wrapper output), frozen site and time, sham/norm/lexical controls, and independent intervention with removal/restoration and downstream functional-use checks. Correct B output alone is not state mediation; correlation of state trajectories alone is not causal identification.

## 7. Specific failure hypotheses and stopping rules

- **Simple output-prior collapse:** high average score but minority class and/or dependent B fails → *functional revision not supported*.
- **Lexical contradiction shortcut:** INVALID_FLIP behaves like VALID_FLIP → *authority-sensitive revision not supported*.
- **Immediate-only correction:** POST succeeds but DELAYED fails → *retention across distractors not supported*.
- **Rule-insensitive readout:** B flips with A when `C=F` → *dependency-aware revision not supported*.
- **Collateral overwrite:** U changes following A update → *selective retention not supported*.
- **Unknown-state collapse:** missing/insufficient evidence is confidently treated as T/F → *evidence entitlement not supported*.
- **Unbalanced design or shortcut leakage:** class, template, trust marker, split-case family or token position trivially determines the answer → stop cohort preparation; fix prospectively before observing model outcomes, never through post-hoc test tuning.
- **Insufficient independent case capacity:** do not launch confirmatory model/state experiments; revise the future sampling frame separately.

The existing G2 and v10 limitations, including three-way NE class collapse, unresolved fine-anchor formal equivalence and native-state causal mediation, remain unchanged and cannot be promoted by this D0 design.

## 8. Exact current boundary and next allowable transition

**Current phase:** `SCIENTIFIC_DESIGN_CANDIDATE_AUTHORING_ONLY`.

**Allowed now:** read historical code/reports, create/review **this single design markdown**, validate its exact bytes and one-file Git scope, stage it for user-controlled review via `cm ship`. An optional user-controlled commit/push may freeze the candidate as a design, not as execution authority.

**Not authorized now:** generating any new case row; modifying old datasets/splits/labels or scripts; tokenizer or model execution; training/evaluation/Kaggle/GPU; native-state measurement; new statistical claim; H3 intervention; RFR mechanism promotion; Phase3B activation.

**Next permitted scientific step, after D0 design freeze:** create a **separately scoped, CPU-only RFR-D1 generator/label-contract specification and deterministic oracle self-tests**; independently verify case-level split isolation, all three arm truth tables, and negative controls. D1 implementation, data materialization and any model execution require their own explicit stage scope and validation; D0 PASS alone grants none of them.

**Final disposition:** `RFR_D0_DESIGN_CANDIDATE_READY_FOR_ONE_FILE_REVIEW; SCIENCE_RESULT=NONE; TRAINING=NO; EVALUATION=NO; GPU=OFF; PHASE3B=BLOCKED_UNCHANGED`.
