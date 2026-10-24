# ContraMamba RFR-D3-M8 — Semantic Micro-World Diversity Feasibility Decision

- Date: 2026-10-24 (Asia/Seoul)
- Baseline: `gen5-causal-role-state-ownership` at `c6364bf4953f0d8f7a19bad87997a428ad21800e`
- Status: `M8_REPORT_ONLY / CURRENT_CONTRACT_TASK_BEARING_DIVERSITY_NOT_DEMONSTRATED`
- Scope: static inspection of frozen D1, D3, M2, M6 and M7 only. No code modification, root drawing, data materialization, new experiment, RNG, split, tokenizer, or model call.

## 1. Exact scientific question

Can the **current** frozen primary event and M6 graph contracts support meaningfully different task-bearing case structures, rather than graph fingerprints made different by names, non-oracle decoration or attestation-hop topology, while preserving the D1 rule `B := A AND C`?

**Decision:** No such variation is demonstrated by the present representation. This is a bounded negative feasibility assessment of the present protocol, **not** an impossibility theorem about independently sampled cases under a fixed rule. Keep the M7 claim ceiling: `FIXED_LAW_SURFACE_GENERALIZATION_ONLY / INDEPENDENT_ROOTS_NOT_ESTABLISHED`.

## 2. Exact implementation constraints

1. D1's `evaluate_records` computes the newest authenticated Boolean for A/C/U by **logical time** and derives B using strong-Kleene conjunction. `source_id` and `provenance_id` identify typed records, but attestation-graph paths have no D1 evaluator semantics. The source-authenticated Boolean is already part of each event.
2. Frozen D3 `_validated_events` requires **exactly three** t0 authenticated base records (A, C, U) for PRE and **exactly four** records for POST/DELAYED, adding only the permitted t1 A event. Additional conflicting-source records, multiple entities, or conditional authentication-graph inference are not part of this primary protocol.
3. M6 graph requires one `ENTITY_MAIN`, exactly one `RECORD_A/C/U` each and one matching `ATTEST_A/C/U` each. Optional nodes are at most three `ATTEST_HOP` and neutral `AUX_NEUTRAL`; allowed task-bearing edge kinds are fixed `HAS_RECORD`, `HAS_ATTESTATION`, and `CHAIN_LINK` (plus neutral-context edges). M6 `_graph_signature` drops auxiliary decoration but distinguishes some attestation-hop shapes.
4. The actual M2 `render` displays D1 event fields and grammar; it does **not** consume or serialize the M6 `instance_graph`. M6 is a hypothetical metadata-only contract, not an integrated generator or graph-grounded task.

For fixed `(a0,c0,u0, arm, readout, query)`, let `P` denote the D1 typed-event projection: `(target, asserted_value, source_authenticated, event_time)` for each record, with event identities held valid and its permitted readout. Then the oracle label is `Y = F_D1(P,query,readout)`. A change to an M6 graph that does not change `P` **cannot** change `Y` under the frozen D1 semantics. Non-isomorphic graph signatures alone are insufficient evidence that distinct *task-bearing* micro-worlds have been instantiated.

## 3. Constructive example and falsifier (on paper; not sampled data)

Consider the same Boolean stratum `A=T, C=T, U=F` under `VALID_FLIP`. At t0, the three authenticated base values yield `B=T`; at t1, an authenticated A=F at time 20 yields `B=F`. U remains F. Hold the typed events fixed in both examples.

- **G0:** one entity, three base records and their direct record-attestation links; no attestation hops.
- **G1:** the same core roles and D1 events, with one additional `ATTEST_HOP` connected to `ATTEST_A` via `CHAIN_LINK`.

G0 and G1 can have different M6 canonical graph signatures. But the D1 state transition and answers to A/B/U at PRE/POST/DELAYED remain identical. G1's additional chain has **no defined role in deciding** the Boolean `source_authenticated`, and the frozen M2 text renderer would not display it. Thus counting G0 and G1 as demonstrably different task-bearing instances would be unsupported.

**Prospective falsifier:** Mask names/IDs; remove neutral-only decorations and attestation-hop variation that has no authenticated-status or admissible-record interpretation. If the resulting role/evidence projection is identical, or if the purported distinction never enters either the frozen oracle semantics or actual rendered input, classify the difference as `TASK_IRRELEVANT_SURFACE_OR_METADATA_VARIATION`, not as verified independent task-bearing diversity. A counterexample would need an explicit, prospectively specified *semantics-preserving yet task-relevant* observable variation (and a corresponding independent construction/audit) that changes the evidence-processing challenge rather than only the metadata fingerprint. None is present in the inspected M2/D3/M6 contracts.

## 4. Decision boundary and next research direction

- `M8_CURRENT_CONTRACT_DIVERSITY=NOT_DEMONSTRATED`; do not claim a confirmed independently sampled surface cohort, IID observations, a three-way renderer holdout, generalized logical laws, or native-state mediation.
- **Close the present M1–M6 independent-root expansion loop.** Do not add M9 metadata validators, arbitrary attestation-hop families, renamed roots, or generated corpus rows merely to obtain distinct graph hashes.
- The already-verified M1–M3 in-memory rows remain **diagnostic fixed-law surface probes**, not a train/dev/test-ready cohort. Their rule-agnostic pooled heuristic and conditional B/C failure warrant interpreting any eventual modeling result by `query × C × arm × readout` and matched contrasts, not pooled accuracy.
- If the scientific objective later requires genuinely distinct evidence-processing structures, that is a **new scientific task-design question**. Examples that might become meaningful only under separately specified semantics include multiple conflicting sources or deriving admissibility from verification chains. These **would change the present D3 event contract and/or D1 authentication rule**, so they are *not authorized or recommended as an implicit extension* of the frozen primary experiment.
- No further root-independent feasibility claim is justified without (a) an explicit genuinely task-relevant variability axis, (b) a prospective sampling protocol and actual audit of generated outputs, and (c) separate execution authority. Neither (a)–(c) is satisfied here.

**Final:** `RFR-D3-M8=BOUNDED_NEGATIVE_FEASIBILITY / INDEPENDENT_SURFACE_COHORT=NOT_ESTABLISHED / MATERIALIZED_COHORT=NONE / MODEL_EVALUATION=NOT_EVALUATED / NATIVE_STATE_CAUSAL_MEDIATION=NOT_ESTABLISHED / PHASE3B=BLOCKED_UNCHANGED`.
