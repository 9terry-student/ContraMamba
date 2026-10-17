# ContraMamba RFR-D1 — Deterministic Oracle and Generator/Label Contract

**Status:** `D1_SPECIFICATION_CANDIDATE / CPU_ORACLE_TEST_SCOPE_ONLY / NO_DATA_MATERIALIZATION`
**Date:** 2026-10-17 (Asia/Seoul)
**Frozen D0 parent:** `835bf75a49d6bd98ae524b4e5a576601561f080d`
**Frozen D0 file:** `reports/rfr_d0_dependency_aware_revision_scientific_design_candidate.md`
**Branch:** `gen5-causal-role-state-ownership`
**Proposed destination:** `reports/rfr_d1_deterministic_oracle_and_generator_contract.md`

## 1. Phase and authority boundary

This file is a narrowly bounded D1 *contract specification*, subordinate to the frozen D0 design and existing repository `AGENTS.md`. It is not an RFR science result, generated dataset, implementation/experiment authority, model-evaluation protocol, or checkpoint execution gate. The historical XG1 source, 600 facts, 4,200 seven-cell rows, split, labels, active encoding and all G2/native/v10 artifacts remain immutable.

The immediate deliverable is only this Markdown specification. A later, independently scoped implementation may add a **pure CPU deterministic logical oracle and synthetic unit tests**, but it must be authorized against this frozen contract before files are changed. Corpus materialization, tokenizer/model invocation, real inference, training, Kaggle and CUDA are outside D1's current scope. This document does not choose the future case count, independent generator family, sample size or scientific promotion threshold.

## 2. Formal world and interpretation

Each abstract *case* has distinct identifiers for propositions `A`, `C`, `B`, `U`, and a fixed derivation rule `B := A AND C`. The variable `A` is the only revisable base proposition in the primary family; `C` is a separately verified premise; `U` is independently verified and unrelated. All three base values at `t0`, `a0,c0,u0`, are Boolean. Negative evidence means *authenticated assertion of false*, not missing information.

Truth domain: `{T,F,?}`; use strong-Kleene conjunction for the derived proposition:

| A | C | B := A AND C |
|---|---|---|
| T | T | T |
| T | F | F |
| T | ? | ? |
| F | T | F |
| F | F | F |
| F | ? | F |
| ? | T | ? |
| ? | F | F |
| ? | ? | ? |

Under the stipulated, *synthetic* record-validity protocol, an affirmative query about a verified proposition maps `T → SUPPORT (2)`, `F → REFUTE (0)`, `? → NOT_ENTITLED (1)`. Never conflate `F` with `?` or use `NOT_ENTITLED` as an operational definition of Forget. These labels are relative to this synthetic evidence protocol, not externally verified world truth.

### 2.1 Event acceptance

Each event contains `event_id`, `target`, `asserted_value`, `source_authenticated`, `event_time`, and immutable provenance. Model event time is a logical timestamp; it is **not** the textual presentation position.

The primary fixture uses authenticated `t0` source records for A, C and U at logical time `10`. `t1` is at logical time `20`. An admissible update to a base proposition requires **authenticated source and strictly newer event time** than its most recent admitted record. Otherwise the previous admitted state persists. A conflicting authenticated assertion at equal timestamp is rejected as ambiguous rather than accepted by presentation order; duplicate identical assertions are idempotent. Missing initial authority produces `?` rather than implicitly `F`. Derivation rule and C/U baseline records cannot be overwritten by a primary A-arm event.

The evaluator is deterministic and order-invariant for uniquely timestamped, valid event sets. It may reorder presentation without changing which record has temporal precedence. For equal-timestamp conflicting authenticated assertions, the test must explicitly expect a hard invalid-input rejection, not guess a winner. Duplicate `event_id` with incompatible payload is also a hard failure.

## 3. Fixed factorial oracle fixtures — no dataset generation

A deterministic self-test enumerates exactly the eight baseline assignments `2 × 2 × 2`: `(a0,c0,u0) ∈ {F,T}³`. This is an **in-memory logic fixture**, not eight training examples, not a random sample, and not a prospective cohort or statistical inference.

For each baseline assignment, enumerate three paired arms:

| Arm | t1 asserted A | Authenticated | t1 newer | Admitted A (POST) |
|---|---|---|---|---|
| `VALID_FLIP` | `NOT a0` | yes | yes | `NOT a0` |
| `VALID_SHAM` | `a0` | yes | yes | `a0` |
| `INVALID_FLIP` | `NOT a0` | no | yes | `a0` |

`C=c0` and `U=u0` are unchanged in every primary arm. Each arm is queried at `PRE`, `POST`, `DELAYED` for each target `A`, `B`, `U`:

- `PRE`: observe authenticated `t0` records only.
- `POST`: observe `t0` plus the arm-specific `t1` event.
- `DELAYED`: same admitted state as `POST`, followed by neutral distractors **without any admissible event concerning A, C or U**.

Thus the fixed primary fixture contains `8 cases × 3 arms × 3 readouts × 3 queries = 216` logical answer checks. `PRE` values are identical across arms for the same abstract case; repeated checks do not create independent samples. The 24 case-arm combinations are the internal state-level assertions before the query/readout expansion. `DELAYED` must equal `POST` at oracle level; future *model* failure across delays would be an experimental observation, not an oracle change.

### 3.1 Non-negotiable paired invariants

1. `VALID_FLIP` flips A at POST and DELAYED in both directions; SHAM and INVALID retain original A.
2. For `c0=T`, B flips with A **only** in VALID_FLIP; this is the dependent-consequence test.
3. For `c0=F`, B always remains F, even for VALID_FLIP. This blocks blanket flipping of derived conclusions.
4. U retains `u0` for every arm and readout; verify both U=T and U=F.
5. SHAM vs INVALID_FLIP have equal oracle answers yet differ in authentication and contradiction wording; neither must override t0.
6. The three final label integers and names obey exactly `REFUTE=0`, `NOT_ENTITLED=1`, `SUPPORT=2`; never silently map F→unknown.
7. Permuting presentation order without changing event timestamps leaves all accepted states and answers unchanged. A *newer* valid record, not the last presented sentence, takes precedence.
8. All model-facing primary query answers are attributable to the fixed rule and authenticated records; do not inject gold endpoints into future input prompts.

### 3.2 Auxiliary deterministic negative tests (separate from the 216 primary checks)

- **Untrusted replacement:** unauthenticated newer assertion of `NOT a0` must not override t0.
- **Stale authenticated replacement:** authenticated assertion of `NOT a0` at time `5` must not override authenticated t0 at time `10`, even if presented last.
- **Conflicting equal-time authority:** two authenticated contradictory assertions for one variable at identical time fail closed rather than adopting textual order.
- **Duplicate event ID conflict:** identical ID with inconsistent content fails closed; exact duplicates are idempotent.
- **Missing premise C:** explicitly remove C's original authenticated record. With A=T, `B=?` and `NOT_ENTITLED`; with A=F, `B=F` and `REFUTE`. This is eligibility-only and **not** a fourth primary arm.
- **Missing A and C:** when `A=?`, `C=F`, B remains F; when both unknown, B is unknown.
- **Unauthorized write to C/U:** primary arm must reject an event targeting non-A base propositions.
- **Distractor contamination:** an apparent fact-changing distractor must be rejected as non-neutral rather than silently incorporated into DELAYED.
- **Malformed input:** invalid Boolean, unknown target, missing source identity or event time, repeated case ID with incompatible case world, cyclical rule or unknown `oracle_version` must fail closed.

## 4. Future generator and provenance contract

The following is a *schema specification*, not an instruction to materialize rows now. Every eventual record must retain:

`case_id`, `case_family_id`, `split_group_id`, `variant_id`, `update_arm`, `initial_A`, `initial_C`, `initial_U`, `derivation_rule_id`, `t0_authoritative_record_ids`, `t1_event_id`, `t1_source_authenticated`, `t1_event_time`, `presentation_order_id`, `readout_time`, `distractor_id`, `query_target`, `target_status`, `final_label`, `oracle_version`, `generator_version`.

Additionally preserve the complete event list with original assertions, authenticated-source identifiers, logical timestamps and immutable event IDs; retain oracle-intermediate A/C states when labeling B; include case-level target state and a canonical hash of each event payload, serialized input bytes and output row once a future separately authorized implementation exists. A null event ID at PRE must not imply that POST events were secretly visible.

**Do not reuse** `xg1_fact_...` IDs, legacy XG1 dev rows, or historical provenance hashes as if they identify new cases. No tokenizer identity or truncation success is asserted here. Before later model execution, an authenticated tokenizer feasibility check must prove that rule, provenance, correction and query survive serialization within the actual context budget.

### 4.1 Split-isolation contract

`case_id` is the sampling unit and `split_group_id` binds **all** arms, queries, readouts, presentation permutations, paraphrases, delays and truth-direction reversals in the same abstract case family. In a future generator, one `split_group_id` may appear in exactly one of train/dev/test. Enforce mutual disjointness of `case_id` and `case_family_id` across splits when identities are assigned at family level. Identical lexical templates and source-family cues across splits must receive explicit leakage scrutiny; distinct row IDs alone never prove independence.

The eight Boolean logical fixtures **are not** train/dev/test and must never be presented as independent case groups. Case count, template inventory, distractor lengths, split fractions, random seed and prospective precision/power criterion remain `UNSET` until a separate dataset-design authority. A later split feasibility gate must verify both flip directions, C true/false, U true/false and positive/negative answer labels in every retained split before scientific testing. In a primary family where all A/C/U are initially known, `NOT_ENTITLED` need not occur; evaluate it through the separately defined missing-premise control, not by altering core labels to balance classes.

## 5. Static test acceptance / implementation scope for a *later* D1 code phase

The minimal later allowed code delta, **only after separately authorizing and freezing the implementation scope**, should be two new files, with no modification to historical scripts:

- `scripts/rfr_d1_deterministic_oracle_cpu.py`: Python standard-library-only, pure functions implementing the fixed truth and event acceptance rule. No filesystem writes, model, tokenizer, torch, network, Git mutation or GPU.
- `tests/test_rfr_d1_deterministic_oracle_cpu.py`: deterministic exhaustive contract tests covering the 216 primary checks and all auxiliary negative controls, plus refusal to write materialized datasets.

Positive tests must independently calculate expected outputs from the frozen truth table and arm semantics rather than calling the oracle to generate its own expected values. Otherwise a shared implementation bug could manufacture a false PASS. Tests must check stable schema, zero input mutation, exact order semantics, and rejection paths. A later, separately authorized code review will fix concrete module signatures and exact implementation file hashes before code freeze.

Implementation correctness (unit tests), scientific execution, artifact/provenance validity and RFR mechanism conclusions are **four different assertions**. A deterministic oracle PASS shows only the proposed dataset labels are internally coherent, not that a language model can revise beliefs or that native state mediates this capacity.

## 6. Decision and stop conditions

- **D1 spec accepted:** this document is frozen as a single reviewed artifact, with no new generated examples.
- **D1 implementation eligible for separate scope:** the exact 216 primary check specification and negative-control semantics are complete and internally non-contradictory.
- **D1 implementation blocked:** contradictory oracle mappings, undefined equal-time tie policy, PRE future leakage, missing split-group isolation, incorrect truth label mapping, or undocumented post-hoc changes to arm semantics.
- **Dataset creation blocked:** case-family generator, actual source attestations, serialization/tokenization feasibility, group split plan, independent case capacity and future evaluation protocol are not yet separately frozen.
- **Science not established:** output success is not state mediation, and D1 contains no model output at all.

The subsequent step, if this design is committed, is a separate exact two-file CPU implementation/testing task under this frozen contract, **not** automatic generation of a dataset or authorization of a scientific run.

**Final status:** `RFR_D1_ORACLE_SPEC_CANDIDATE / SCIENCE_RESULT=NONE / CODE_EXECUTION=NO / DATA_GENERATION=NO / MODEL=OFF / GPU=OFF / PHASE3B=BLOCKED_UNCHANGED`.
