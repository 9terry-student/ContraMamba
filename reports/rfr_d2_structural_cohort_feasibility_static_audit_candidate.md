# ContraMamba RFR-D2 — Structural Cohort Feasibility / Design-Matrix Audit

**Status:** `STATIC_DESIGN_MATRIX_PASS / REAL_COHORT_FEASIBILITY_NOT_ESTABLISHED / NO_DATA_GENERATION`

**Branch:** `gen5-causal-role-state-ownership`

**Frozen baseline HEAD:** `45a510c7a85947cbd9637df8163182fb9159c3a5`

**Frozen sources:**
- `reports/rfr_d0_dependency_aware_revision_scientific_design_candidate.md`
- `reports/rfr_d1_deterministic_oracle_and_generator_contract.md`
- `reports/rfr_d1_deterministic_oracle_cpu_implementation_scope.md`
- `scripts/rfr_d1_deterministic_oracle_cpu.py`
- `tests/test_rfr_d1_deterministic_oracle_cpu.py`

**Evidence distinction:** The operator reported `19/19` D1 CPU unit tests PASS, `cm ship` PASS, and clean two-file commit/push at the frozen HEAD. The D2 counts below were derived in a separate Python-standard-library Boolean enumeration, independently of the D1 implementation's expected-answer function. This static enumeration is *not* a test of a model, independent samples, or a generated cohort.

## 1. Primary design matrix — exact deterministic counts

Eight baseline truth assignments `(a0,c0,u0) ∈ {F,T}³` crossed with:
- three arms: `VALID_FLIP`, `VALID_SHAM`, `INVALID_FLIP`;
- three readouts: `PRE`, `POST`, `DELAYED`;
- three queried propositions: `A`, `B=A AND C`, `U`.

Thus `8 × 3 × 3 × 3 = 216` **logical checks**, not 216 statistically independent cases.

| Query | REFUTE | SUPPORT | NOT_ENTITLED | Interpretation |
|---|---:|---:|---:|---|
| A | 36 | 36 | 0 | Balanced |
| B | 54 | 18 | 0 | 75% REFUTE from conjunction, even without model error |
| U | 36 | 36 | 0 | Balanced |
| All | **126** | **90** | **0** | Primary family is intrinsically two-label |

Every one of the nine `arm × readout` blocks has exactly 24 checks: **14 REFUTE / 10 SUPPORT**. Consequently *pooled label rates by arm or readout are identical*, even though the case-paired conditional answer transitions differ. A method that reports only arm-level label histograms cannot detect successful revision.

From PRE to POST, among the eight baseline worlds:

| Arm | A changes | B changes | U changes |
|---|---:|---:|---:|
| VALID_FLIP | **8/8** | **4/8** | **0/8** |
| VALID_SHAM | 0/8 | 0/8 | 0/8 |
| INVALID_FLIP | 0/8 | 0/8 | 0/8 |

B changes only for `C=T`; with `C=F`, B must remain false even if A flips. The same POST state must persist through DELAYED at the logical-oracle level.

## 2. Interpretation and leakage gates

1. **Primary endpoints cannot be a generic three-way benchmark.** The primary Boolean-known fixture has zero NOT_ENTITLED cases by definition. Do not inject unknown labels merely to balance classes. The frozen *separate* missing-premise/eligibility controls supply the unknown/NOT_ENTITLED behavior.
2. **B-specific class imbalance is structural.** Always-REFUTE already scores 75% on B, so B accuracy alone cannot establish dependent revision. Require case-wise paired B transitions gated by C, plus matched VALID_SHAM and INVALID_FLIP controls.
3. **Global label-marginal equality masks functional differences.** Under the balanced logical factorial, the three update arms have the same marginal label counts. Use arm-sensitive, same-case joint A/B/U criteria; do not treat 216 re-queries as independent replicates.
4. **U preservation must be joint, not averaged.** Report whether the *same case* correctly updates A, conditionally updates B, and leaves U unchanged at both POST and DELAYED.
5. **Authority and logical time are distinguishable.** A lexical contradiction, the most recently *presented* sentence, and a newer **authenticated** event are not synonymous. Preserve `INVALID_FLIP`, timestamp permutations and stale authenticated records as controls.
6. **Rule and lexical shortcut controls are prospective.** Counterbalance identifiers, proposition wording, polarity expressions, authentication markers, source IDs, timestamps, query order and distractor surfaces *within each case family*. A single template with a unique answer cue is insufficient, even if the oracle is correct.
7. **Future model claims require a fresh independently sampled population.** The eight worlds here are truth-table configurations, not independent source families. Historical XG1 dev pairs are not eligible to be relabeled as independent RFR confirmatory cases. `case_id`, `case_family_id`, `split_group_id` must be kept together across arm, readout, query, delays and wording permutations.
8. **Truncation feasibility is unknown.** There is no authenticated tokenizer or bounded serialized-text test yet. No context-size, lexical equivalence or token-balance PASS may be inferred from this audit.

## 3. Exact feasibility disposition

- `D1_CPU_IMPLEMENTATION_TESTS=PASS_OPERATOR_REPORTED_19_OF_19`
- `D1_TWO_FILE_FREEZE=CONFIRMED_BY_OPERATOR_PUSH_LOG`
- `D2_INDEPENDENT_BOOLEAN_FACTORIAL_ENUMERATION=PASS`
- `PRIMARY_LOGICAL_DESIGN_IDENTIFIABLE=YES_FOR_PAIRED_CONDITIONAL_TARGETS`
- `PRIMARY_NOT_ENTITLED_COUNT=0_BY_DESIGN`
- `B_ANSWER_IMBALANCE=STRUCTURAL_75_PERCENT_REFUTE`
- `GENERATOR_FAMILY_AND_LEXICAL_CONTROLS=NOT_FROZEN`
- `INDEPENDENT_CASE_CAPACITY=NOT_ESTABLISHED`
- `ACTUAL_SPLIT_FEASIBILITY=NOT_ESTABLISHED`
- `TOKENIZER_AND_SERIALIZATION_FEASIBILITY=NOT_EVALUATED`
- `RFR_D2_REAL_COHORT_FEASIBILITY=NOT_ESTABLISHED`
- `RFR_FUNCTIONAL_SUCCESS=NOT_EVALUATED`
- `NATIVE_STATE_MEDIATION=NOT_ESTABLISHED`

**Next bounded phase:** Freeze a *separate*, source-independent structural generator/lexical-counterbalance and split-family design. The proposal must keep `B=A AND C`, the D1 oracle and three primary arms immutable; explicitly define how independent case families will be constructed, how all family variants share a split, and which label/template/recency shortcuts are rejected before outcomes. Actual data materialization requires separate implementation/execution scope. Do not choose case counts, power, model seeds, context budget, architecture, or success thresholds using model outcomes.

No dataset row was generated or saved, no tokenizer or model was loaded, no GPU/CUDA, Kaggle, training/evaluation or scientific state measurements were run. This is an analysis candidate, not a generator authorization or a new positive scientific result. `PHASE3B=BLOCKED_UNCHANGED`.
