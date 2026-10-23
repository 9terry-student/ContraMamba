# ContraMamba RFR-D3 — Minimal Independent-Family Renderer and Materialization Contract

- **Date:** 2026-10-17 (Asia/Seoul)
- **Status:** `D3_CONTRACT_CANDIDATE / STATIC_ONLY / NO_MATERIALIZATION`
- **Baseline:** `gen5-causal-role-state-ownership` at `ae021d271572b911de6c89b9b9263fcece835e49`
- **Controlling frozen sources:** D0 scientific design; D1 oracle/generator contract and `scripts/rfr_d1_deterministic_oracle_cpu.py`; D2 independent-cohort design, validator scope, `scripts/rfr_d2_structural_feasibility_cpu.py`, and designated tests.
- **Decision:** Establish a single prospective D3 contract. No new code, data rows, split assignments, tokenizer use, model execution, or evaluation is authorized by this file.

## 1. Unit of independence and admissible claim

A **root case-family** is one independently drawn *synthetic micro-world specification*, before arm/readout/query/paraphrase/presentation-order descendants exist. It contains an independently instantiated entity configuration, A/C/U record identities, source-attestation identities, textual construction choices, and a provenance-bearing generator seed/substream. Every derivative stays in that one `case_family_id` and `split_group_id`. A reversed truth direction, renamed entities, changed arm, changed wording, or moved query is **not** a new independent unit. Root-family count, prompt count, and number of distinct logical problems must be reported separately.

The primary family always has the same conjunction `B := A AND C` and only eight baseline Boolean worlds. Independent procedural draws may create distinguishable synthetic *surface micro-worlds*, but cannot create new independent logical laws. Fingerprint disjointness and random seed separation are necessary audit conditions, **not proof** of semantic/statistical independence. If the realized diversity is only renaming of the same eight worlds, any later empirical claim must be limited to surface generalization; broader confirmatory independence remains `NOT_ESTABLISHED`.

Do not choose `K_s`, total family count, split proportions, seed, predictive success threshold, power, tokenizer budget, or model hyperparameters in this contract.

## 2. Two nonoverlapping surface grammar families

Define two *candidate* renderer namespaces, not concrete example prompts:

| Domain | Independent surface vocabulary and grammar | Logical roles (not extra factual implications) |
|---|---|---|
| `REGISTRY_RECORD` | Synthetic ledger, record, filing, verification vocabulary; document/event grammar; invented non-real identifiers | A = registered property status; C = separate prerequisite status; B = stipulated joint eligibility under `A AND C`; U = unrelated register status |
| `INSTRUMENT_LOG` | Synthetic apparatus, channel, calibration, reading vocabulary; instrument/event grammar; invented non-real identifiers | A = instrument property status; C = separate prerequisite status; B = stipulated joint operating condition under `A AND C`; U = unrelated apparatus status |

Domain lexicon pools and full sentence-template families are disjoint except for unavoidable logical/timestamp notation. No renderer can infer truth from an entity name, status-word choice, source name, or fixed query location. Both domains must support both signed polarities, all eight baseline truth strata, every primary arm and query, both flip directions, multiple equivalent rule phrasings, presentation-order permutations, and neutral delayed distractors. Every rule phrasing must explicitly convey conjunction and no implicit real-world dependencies. `B` is derived from A and C; it is **never** supplied as a certified base fact.

A renderer receives typed records; it must not invent/alter oracle facts. Grammar/paraphrase alternatives must preserve explicit *verified false* versus *absent evidence* and show authentication and logical event time unambiguously. External commonsense truth does not define targets.

## 3. Minimal prospective typed interface (not yet implemented)

Future implementation, only under separate implementation scope, shall provide these conceptual operations:

- `validate_root_spec(root)`: consume a declared root world and distinct identity/provenance namespaces; check strict Boolean A/C/U, rule/version, and source/record consistency. It must not create independent roots from alternate names or truth reversals.
- `expand_primary_plan(root)`: declare the D2-required 27 `(arm, readout, query)` cells per declared order/paraphrase realization. Reference frozen D2 metadata schema and `audit_variant_plan`; never count 27 descendants as 27 independent roots.
- `render_primary_variant(root, variant, renderer_spec)`: deterministic, side-effect-free construction of typed visible event and query components. **No call or materialization is authorized yet.** A future actual generator must maintain one-to-one traceability to its root and each event, and separately store model-visible text and sealed metadata.
- `audit_materialized_cohort(rows, registry, plan)`: a future *independent* audit over actual authorized text/records, separate from D2 plan validation. No materialized PASS may be inferred from D2 returns.

**Input namespaces:** root identities `case_family_id`, `split_group_id`, `semantic_world_id`, `entity_tuple_fingerprint`, `semantic_world_fingerprint`, `source_attestation_fingerprint`, `lexicon_namespace_id`, `renderer_family_id`, `template_family_id`, `surface_domain`; A/C/U truth; `B_AND_A_C`; `RFR_D1_ORACLE_V1`; event protocol; explicit root ancestry. A future implementation may add strictly validated fields but may not reinterpret frozen D1/D2 fields.

**Typed event:** `event_id`, `target`, strict-Boolean `asserted_value`, strict-Boolean `source_authenticated`, integer `event_time` (not Boolean), nonempty `source_id` and `provenance_id`; preserve exact D1 rejection semantics. Base t0 A/C/U records: authenticated at logical time 10. Primary t1 target: only A at time 20. `VALID_FLIP` = opposite A, authenticated; `VALID_SHAM` = unchanged A, authenticated; `INVALID_FLIP` = opposite A, unauthenticated.

**Outputs once separately authorized:** a `visible_input` consisting only of task rule, selected visible records, their authentication evidence and timestamps, an affirmative query, and optional neutral delay material; plus a separate `sealed_metadata` containing root/variant/split identities, oracle truth states, gold labels, arm, generator/renderer identity, provenance and hashes. Never concatenate sealed metadata or answers into visible input, filenames presented to the model, or task instructions.

`PRE` serialization must physically exclude the entire t1 record/metadata; it is not enough to ignore t1 in scoring. `POST` includes t0 and t1. `DELAYED` preserves POST's admissible state and appends only neutral distractors. PRE content may intentionally be identical across arm descendants of one root: mark such within-family duplicates as *expected equivalences*, rather than independent rows; cross-split duplicate prompts remain a blocker.

## 4. Prospective matched-contrast and balance obligations

Before materialization, a declared plan must cover the following *within root or matched-stratum* contrasts; before model execution, verify them against actual text and oracle results:

1. **Authentication:** `VALID_FLIP` versus `INVALID_FLIP` shares asserted opposite A, target, and logical t1 time; only legitimate authentication evidence/provenance differs. Verification evidence is visible and task-relevant, but source names and formatting must not directly encode a gold endpoint.
2. **Correction direction:** `VALID_FLIP` versus `VALID_SHAM` shares a valid update-site/time protocol but differs in asserted polarity. Both F→T and T→F occur for A across root strata.
3. **Dependency blocking:** for the same A trajectory compare `C=T` and `C=F`: B changes only under authenticated A flip when C is true; B remains F when C is false. U is independent with both polarities.
4. **Logical time:** cross event *presentation order* with the truth strata and arms. A later-presented untrusted or stale authenticated statement does not overrule a newer admissible record. D1 stale/equal-time conflict tests remain auxiliary controls, not a fourth primary arm.
5. **Negation / query / delay:** cross unambiguous positive and explicit-negative grammar, A/B/U query order and positions, neutral distractor templates and DELAYED placement. No query-position or negation token may serve as a deterministic gold-label code by design.
6. **Shortcut coverage:** predeclare label-majority, t1-content-only, authentication-only, last-mentioned, query-only, rule-agnostic, and source/template-only attacks. For each suspected cue require concrete matched counterexamples or label-conditional audit; do not demand impossible B=50/50 marginals or suppress legitimate authenticity cues.

`NOT_ENTITLED` remains absent from the primary Boolean-known cohort. A separate missing-premise eligibility control follows frozen D1 strong-Kleene semantics and must not contaminate primary endpoints.

## 5. Identity, split and duplicate checking (materialized-phase gate, not accomplished)

Root-family ID → one split group → at most one frozen split; every 27-cell declaration, arm, wording, order, reverse polarity and delay descendant inherits that split. Validate using D2 registry/variant audits and D1 `verify_split_isolation`, then independently recheck realized rows. Canonical event-structure, entity-tuple, source-attestation, renderer-family, paraphrase-family, exact serialized prompt bytes and normalized prompt-content hashes must be indexed across splits. Root/semantic/fingerprint reuse and cross-split exact content duplicates are blockers; near-duplicates require recorded review, not a post-hoc convenient similarity threshold. Canonical hashing must be versioned and deterministic, with canonicalized typed event structure distinguished from raw UTF-8 prompt bytes.

The design has **two surface domains only**. This does *not* suffice to claim three mutually disjoint renderer sets for `train/dev/test`. In particular, the frozen D2 `holdout_contract=RENDERER` checks disjointness across **every assigned split**, not merely train versus test. Do not assert a global three-way renderer holdout with two renderers. No actual splits are assigned here; a later authorized design must either use an honest within-renderer grouped split (`holdout_contract=NONE`, shared renderer reported), or independently justify a different held-out-renderer protocol before claiming it. Do not quietly weaken D2 holdout checks.

## 6. Gates and explicit status boundary

| Gate | Minimal eventual evidence | Current status |
|---|---|---|
| D2 plan | D2 3 audits on declared registries/variant inventory | Existing validator frozen; no D3 registry executed |
| D3-G1 semantics | Actual typed events reproduce frozen D1 A/B/U oracle per arm/readout/query; PRE no future content; POST=DELAYED oracle | `NOT_EVALUATED` |
| D3-G2 family | Real root ancestry, all descendants in one split, exact/near-duplicate audit, capacity by 8 strata × domain × split | `NOT_EVALUATED` |
| D3-G3 surface | Rendered conditional cue tables, matched contrasts, no metadata/answer leakage | `NOT_EVALUATED` |
| D3-G4 shortcuts | Seven predefined shortcut attacks on actual authorized inputs with conditional errors (not just pooled accuracy) | `NOT_EVALUATED` |
| D3-G5 serialization | Separate gate with **pinned actual tokenizer revision**, exact serialization/chat template, context size, token indices/survival of rule, auth evidence, logical timestamp, t1 assertion, query; truncation fail-closed | `NOT_EVALUATED` |
| D3-G6 provenance | Exact generator/renderer/grammar versions, root seed/substream, plan identity, raw text and event hashes, manifest, family coverage | `NOT_EVALUATED` |

Only after separate explicit data-materialization authority and successful independent realized-row audits may a result be called `MATERIALIZED_COHORT_FEASIBILITY_PASS`. D2 `PASS_PLAN_CHECK`, a unit-test PASS, nonzero family count, or unique fingerprints are never a substitute. Tokenizer/serialization feasibility is a **later separate gate**, and no model execution is authorized even after a materialized audit PASS without its own phase authorization.

## 7. Next bounded implementation, not authorized by this file

If this contract is frozen and a subsequent implementation task is authorized, constrain the first code delta to **two new files only**:

- `scripts/rfr_d3_cohort_contract_cpu.py`: Python-stdlib-only, pure *spec/grammar-definition and static contract-checking* functions; no actual prompt creation, corpus materialization, file I/O, random root generation, tokenizer or network calls.
- `tests/test_rfr_d3_cohort_contract_cpu.py`: independently constructed metadata/grammar fixtures and adversarial tests. Cover two-domain grammar disjointness; D1 event-schema and polarity invariants; 27-cell plan compatibility; PRE exclusion rules; matched-contrast declarations; prohibited visible metadata; negative/unknown separation; split inheritance; false holdout claim; immutable inputs and no side effects.

Validation for this *later* implementation: `python -m py_compile scripts/rfr_d3_cohort_contract_cpu.py tests/test_rfr_d3_cohort_contract_cpu.py`, `python -m unittest discover -s tests -p "test_rfr_d3_cohort_contract_cpu.py" -v`, `python -m unittest discover -s tests -p "test_rfr_d1_deterministic_oracle_cpu.py" -v`, `python -m unittest discover -s tests -p "test_rfr_d2_structural_feasibility_cpu.py" -v`, `git diff --check`, then `cm ship`. No CLI unless separately scoped, and no gate is presumed registered. Tests must independently calculate negative expectations rather than asking production code for expected answers. Actual generator, renderer execution, cohort file writes, splitting, tokenizer or model execution require a separate, later authorization.

**Current conclusion:** `D3_MINIMAL_STATIC_CONTRACT_DRAFTED / COHORT_CAPACITY_NOT_ESTABLISHED / MATERIALIZED_DATA=NONE / TOKENIZER_NOT_EVALUATED / RFR_FUNCTIONAL_REVISION_NOT_EVALUATED / NATIVE_STATE_CAUSAL_MEDIATION_NOT_ESTABLISHED / PHASE3B=BLOCKED_UNCHANGED`.
