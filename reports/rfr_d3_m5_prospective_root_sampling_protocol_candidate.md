# ContraMamba RFR-D3-M5 — Prospective Independent Root-Sampling Protocol (Candidate)

- Date: 2026-10-23 (Asia/Seoul)
- Baseline: `gen5-causal-role-state-ownership` at `92098502d99f8e46ae67e4ab2bc5c0ecce53485e`
- Status: `M5_SAMPLING_PROTOCOL_CANDIDATE / STATIC_ONLY / PLAN_ONLY`
- Purpose: decide what *future* sampling and independent inspection would need to establish before any root-family could be counted as a candidate independent synthetic surface micro-world.
- Authority: frozen D0/D1 semantics; D2 cohort design and `scripts/rfr_d2_structural_feasibility_cpu.py`; D3 renderer/materialization contract, D3 static validator, M1/M2 pilots, M3 overlap audit and M4 root-frame feasibility validator. This document adds neither implementation nor execution authority.
- No sampling, RNG invocation, seed selection, row generation, split assignment, tokenizer/model use, file-based corpus creation, or scientific claim promotion occurred in M5.

## 1. Why M4 cannot be turned into a sampling PASS by adding IDs

M4 establishes that M2/M3's 16 provisional roots are two deterministic presentations of the same eight `(A,C,U)` Boolean worlds, not verified independent draws. M3 found 4,608 cross-family equal-role-structure *row pairs* and 1,728 controlled one-U-literal neighbor pairs. These counts describe comparisons among related rows, not new independent cases. The frozen primary law remains `B := A AND C`; its eight worlds are logical *strata*, not eight independent logical laws. The currently supported inference is bounded surface robustness, never broad law generalization or native-state causality.

**Decision:** Keep M1–M4 unchanged. M5 defines prospective conditions only. Do not create 48 renamed roots and mistake a completed coverage grid for independence.

## 2. Three separate units; no denominator substitution

| Unit | Definition | Admissible count |
|---|---|---|
| `logical_world` | One of the eight strict Boolean `(a0,c0,u0)` strata under `B_AND_A_C` | At most 8 distinct Boolean configurations under this task |
| `root_case_family` | One separately instantiated and auditable synthetic entity/record/attestation/renderer *micro-world draw* before arm/readout/query descendants | Candidate surface sampling unit only after draw provenance and independent audit |
| `variant_row` | Descendant of a root for arm × readout × query × permitted order/paraphrase/source rotation | Correlated repeated measurement, **never** a new root |

A `root_case_family` is *not* made independent by changing names, reversing truth, choosing another domain, reordering statements, changing source alias, or creating another random-looking identifier. Independently seeded surface variables do not prove semantic independence. If generated realizations are solely isomorphic substitutions of the same eight worlds, the final scientific claim must remain explicitly **surface generalization conditional on one fixed law**.

## 3. Prospective sampling frame and order of operations

1. Before realization, freeze the generator and renderer code versions, the grammar inventory, the event/serialization schema, permitted micro-world construction operators, fingerprint canonicalization rules, intended sampled-family counts by `(domain, A, C, U, eventual split)`, and a root-manifest specification. No model responses may enter selection.
2. Specify an *independent root-draw unit*: its entity-role configuration, three typed base-record identities, authenticated-source/attestation chain, and grammar/template construction choices must have separately designated draw substreams. Conditional truth stratum and chosen surface domain are **design factors**, not outcomes of an identifier. The root draw must not be copied from another root or derived from a reversal/paraphrase of it.
3. Keep the master seed and precise derivation/counter-substream algorithm prospective until independently frozen for an authorized execution; do not pick them in M5. Each root must later record a reproducible root-draw identifier, version, substream derivation and independent source of each draw component. A nonce/unique hash without verified generation history is not enough.
4. Permit both `REGISTRY_RECORD` and `INSTRUMENT_LOG`, both A correction directions and all eight Boolean strata. Construct actual *task-bearing* entity, record and attestation configurations in each domain; mere decoration, arbitrary synonyms, or an ID change do not count as new semantic micro-world structures. Any auxiliary graph/context fields must be provably neutral for the frozen A/C/U/B oracle.
5. A proposed split policy can describe `train/dev/test` family-group isolation, but M5 does not assign any root or descendant to a split. With only two renderer domains, `holdout_contract=NONE` is the honest proposed three-split setting; do not claim a three-way `RENDERER` holdout. Prospective split construction and family inheritance must be frozen before realizing/evaluating cases.
6. For each realized root in a *later separately authorized stage*, expand the frozen 27 arm/readout/query cells per declared surface realization and inherit one `case_family_id`/`split_group_id` throughout. Verify t0 at 10, t1 A at 20, strict D1 authentication, PRE physical future exclusion, POST/DELAYED logical equality, paired arm contrasts, and U preservation.
7. Do not silently filter awkward roots, edit grammar until surface heuristics improve, or redefine a threshold after seeing generated examples or model outcomes. Record every attempted root (including rejected attempts and reasons) so selection can be audited.

The full 8-stratum × 2-domain × 3-split grid has **48 coverage slots if one root is placed in each cell**. This is not an authorized size, prescribed split allocation, statistical power estimate, proof of meaningful independence, or guarantee that all cells are feasible. No `K_s`, sample total, seed, split ratio, similarity cutoff, tokenizer limit or performance threshold is chosen here.

## 4. Minimum root manifest (later artifact; none created now)

Before any sampling execution, fix a versioned typed manifest whose *per-root* sealed fields include:

- `case_family_id`, `split_group_id`, `semantic_world_id`, `surface_domain`, `renderer_family_id`, `lexicon_namespace_id`, `template_family_id`, strict-Boolean `root_initial_A/C/U`, `derivation_rule_id=B_AND_A_C`, `oracle_version=RFR_D1_ORACLE_V1`;
- frozen `generator_version`, `renderer_version`, `grammar_version`, `canonicalization_version`, `root_draw_protocol_id`, `root_draw_substream_id`, exact recorded source/counter identities of the four draw components, and frozen seed derivation specification;
- typed entity/role graph with explicitly declared non-oracle auxiliary facts; A/C/U base-record and source-attestation graph; verified lineage/ancestry links; fingerprints for canonical instance structure, entity tuples, attestation chain, renderer/template family, and event sequence;
- an attempt ledger (draw attempt ID, accepted/rejected flag, deterministic reason, no silent retries), immutable plan identity, and eventual split linkage; exact raw UTF-8 bytes and normalized-content hashes only once actual text is separately authorized.

Never expose root IDs, split/group labels, truth stratum, seed, gold labels, provenance manifest, or oracle internals in model-visible prompts. The task-relevant *source authentication evidence* remains visible; source names/aliases must not deterministically encode authorization or answers across the population.

## 5. Independent audit: compare different notions of identity

Use separate canonicalizations, all prospectively versioned, and never conflate them:

| Index | What it preserves | Interpretation |
|---|---|---|
| `CORE_LOGIC_SIGNATURE` | Frozen law, Boolean stratum, arm, readout, query, admissible timestamp/authority roles | Equal across families **by design**; not a cross-split blocker by itself |
| `INSTANCE_GRAPH_SIGNATURE` | Task-bearing entity/record/attestation relations, ignoring cosmetic identifiers but retaining authentic semantic structure | Cross-root reuse or an isomorphic copied root is a blocker to a claim of independently instantiated cases; log and investigate |
| `ANCESTRY_FINGERPRINT` | Origin root and all descendants including truth reversals, wording/order/alias variations | Root-group reuse or cross-split descendant leakage is a blocker |
| `SURFACE_FAMILY_SIGNATURE` | Template/lexicon/source construction pattern with identifiers masked | Detects shortcut-prone systematic duplication; not automatically independent evidence |
| `RAW_PROMPT_SHA256` | Actual rendered UTF-8 bytes | Exact cross-split duplicates block; expected identical PRE descendants **within one root** are recorded, not recounted |
| `NORMALIZED_CONTENT_HASH` / reviewed neighbors | Casefold/whitespace-normalized text plus explicitly recorded masking/normalization choices | Candidate cross-split near-duplicates require independent review; never silently invent an after-the-fact threshold |

Logical signatures recurring across train/dev/test are necessary for a controlled Boolean benchmark, whereas copied *instance graphs* across families compromise the proposed independent unit. A graph fingerprint collision can show repetition; distinct hashes alone never establish independence. Full inspection also checks that identity fields, template choice, query position, source alias and authentication evidence do not encode a deterministic unintended answer shortcut after conditioning on the task-relevant factors.

## 6. Prospective pass/fail checks and falsifiers

- **Frame/lineage:** all draw components and source substreams are traceable to a root; neither a renamed twin nor a reversed-truth descendant is counted as an independent root. A missing source/attempt ledger means `BLOCKED_MISSING_DRAW_PROVENANCE`.
- **Semantic construction:** if all realized cases reduce to the same instance graph after identifier masking, return `BLOCKED_ISOMORPHIC_SURFACE_ONLY`, even if all raw prompts and hashes differ. If there are genuinely different auxiliary graphs that do not modify the oracle, separately establish *surface-instance* diversity; still make no new-law generalization claim.
- **Family/split leakage:** each lineage belongs to one split; the same instance graph, entity tuple, attestation chain, descendant, or exact input is not reused across splits. Reject a purported three-way renderer holdout with only two renderer namespaces. Split checks remain `NOT_EVALUATED` until actual assignments are authorized and present.
- **Truth/causal contrasts:** full truth-stratum coverage; both correction directions; paired VALID_FLIP/SHAM/INVALID matched events; C=T update versus C=F block, U retention, PRE/POST/DELAYED invariants. Reject altered D1 semantics rather than adjusting labels.
- **Surface shortcuts:** independently evaluate the seven predeclared attacks (majority, t1 assertion, authentication, last-mentioned, query-only, rule-agnostic, source/template-only) on *authorized realized text*. Report conditional matrices by query × C × arm × readout, plus ordered paired contrasts; do not promote accuracy alone, including the M2/M3 rule-agnostic pooled 91.67%.
- **Selection integrity:** no retries without attempt provenance, outcome-based trimming, silent seed replacement, retrospective threshold tuning or result-driven template selection. Missing rejected-draw accounting is a blocker.
- **Tokenizer gate:** remains a later **separate** revision-pinned serialization/truncation audit, not implied by text-level success. Likewise no model training/evaluation or recurrent-state intervention is authorized.

An auditor other than the prospective generator implementation must later check the actual realized rows and their draw provenance. The generating code cannot self-certify population independence with one `PASS` flag. Planned metadata may receive only `PASS_PLAN_ONLY`; realized evidence requires separate authority and actual audit.

## 7. Current decision / next bounded action

- `M4_CURRENT_FRAME_INDEPENDENCE_BLOCKED` remains unchanged for M1/M2/M3 roots.
- `M5_PROTOCOL=PROPOSED_STATIC_ONLY`; `M5_ROOT_DRAWS=NONE`; `M5_SPLITS=NONE`; `M5_MANIFEST=NOT_CREATED`; `INDEPENDENT_SURFACE_COHORT=NOT_ESTABLISHED`.
- Do **not** build or run a new case generator yet. If subsequently needed, first implement a narrowly scoped **metadata-only M5 manifest/ancestry/instance-graph contract validator** with independently constructed adversarial fixtures, reusing frozen D1–D3/M4 checks. No sampled rows, files, seeds, tokenizer or model execution in that step.
- Scientific outcome remains `MATERIALIZED_COHORT_FEASIBILITY=NOT_EVALUATED / LOGICAL_GENERALIZATION=BLOCKED_BY_FROZEN_PRIMARY_RULE / NATIVE_STATE_CAUSAL_MEDIATION=NOT_ESTABLISHED / PHASE3B=BLOCKED_UNCHANGED`.
