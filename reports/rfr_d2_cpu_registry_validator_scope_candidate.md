# RFR-D2 — CPU-Only Family Registry / Counterbalance Plan Validator Scope

- **Date:** 2026-10-17 (Asia/Seoul)
- **Status:** `D2_VALIDATOR_IMPLEMENTATION_SCOPE_CANDIDATE / NO_DATA_MATERIALIZATION`
- **Branch / source HEAD:** `gen5-causal-role-state-ownership` / `2dc84cbb6ad04a14bcfae8df928a03de1603e131`
- **Controlling documents:** `reports/rfr_d2_independent_cohort_counterbalance_design_candidate.md`, D2 static audit, frozen D0/D1 contracts and D1 CPU oracle.
- **Purpose:** Verify the internal consistency of a *declared* family-registry and counterbalance plan, before any actual case rows, text, split assignments, or model runs exist. This stage cannot establish realized independent capacity or leakage-free prompts.

## 1. Exact implementation boundary (only after this scope is frozen)

Add exactly two files; do not modify existing files:

1. `scripts/rfr_d2_structural_feasibility_cpu.py` — pure Python-standard-library functions; in-memory inputs and returned audit dictionaries only.
2. `tests/test_rfr_d2_structural_feasibility_cpu.py` — independent, adversarial, synthetic *metadata-only* unit tests; no corpus examples or files written.

Permitted functions: `audit_family_registry(families)`, `audit_variant_plan(variants, families)`, and `audit_counterbalance_plan(plan, families)`; private helpers remain inside the single script. Input and output schemas must be fixed in the module docstrings and tests, with strict types and fail-closed errors. No CLI, file I/O, output artifacts, new dependencies, model/tokenizer, Git/network, subprocess, CUDA or Kaggle. Import of the existing frozen D1 CPU oracle is allowed only for its constant truth/label semantics, not to compute the tests' expected answers.

## 2. Family-registry metadata and identity rules

Each registry entry denotes a *proposed independent root family*, not an example or a claim of semantic independence. Required fields: `case_family_id`, `split_group_id`, `surface_domain`, `renderer_family_id`, `lexicon_namespace_id`, `root_initial_A`, `root_initial_C`, `root_initial_U` (strict Boolean), `semantic_world_fingerprint`, `entity_tuple_fingerprint`, `source_attestation_fingerprint`, `template_family_id`, `derivation_rule_id='B_AND_A_C'`, `oracle_version='RFR_D1_ORACLE_V1'`. Fingerprints are explicit opaque nonempty identities supplied by a future design; **do not synthesize them by hashing invented text**. `split` is optional/unassigned in the design phase; if present, it is one of `train/dev/test`. An unassigned family cannot count toward a realized split feasibility PASS.

Reject duplicate/inconsistent family or group IDs, reused root-world/entity/attestation fingerprints masquerading as different independent families, cross-split reuse of a namespace or template family where the plan claims holdout isolation, and any contradictory metadata. Shared `renderer_family_id` is allowed for an ordinary within-renderer split but must be reported, not misrepresented as held-out-renderer independence. Require exact grouping: one family -> one group -> at most one split; all derived variants of one root family inherit that same group and split. A reverse-polarity variant is a *dependent descendant* of its root and never counts as an extra independent family. Distinct IDs or hashes alone do not prove semantic or statistical independence; return `CONTENT_INDEPENDENCE=UNVERIFIED`.

## 3. Plan-only factorial and lexical gates

`audit_variant_plan` accepts only variant *declarations* (`case_family_id`, `variant_id`, `update_arm`, `readout_time`, `query_target`, `presentation_order_id`, `paraphrase_family_id`) referring to an existing root family. No prompt, evidence text, target label or oracle target state is allowed. Validate enums, unique variant identity, group inheritance, missing/duplicate arm-readout-query cells, and complete primary 3 x 3 x 3 coverage for each declared root-family plan; any additional planned order/paraphrase realization must be grouped with its root. This reports planning coverage, **not actual 27 generated examples**.

`audit_counterbalance_plan` accepts only a declared, strictly typed contingency inventory keyed by split, root truth stratum, domain/renderer, authentication-marker family, source-name family, polarity wording family, timestamp/presentation-order pattern, query and readout. Validate that every intended source/wording/order cue is crossed with both attainable correct answers under the frozen D1 truth logic; explicitly mark impossible balance for B (75% REFUTE) and absence of primary NOT_ENTITLED. Reject deterministic accidental label cues where matched contrasting cases are required; authentication truth itself is an intended legitimate cue and must **not** be hidden. Do not require impossible perfectly uniform cue-label correlations. Missing inventory -> `NOT_EVALUATED`, never PASS. No arbitrary near-duplicate threshold: exact fingerprint collisions block; lexical near-duplicate review remains pending until real text exists.

## 4. Acceptance and non-claims

- **G1 LOGIC:** D1 oracle tests remain separately authoritative; optionally cross-check plan algebra using an independently written Boolean table. Do not count 216 queries as 216 independent cases.
- **G2 FAMILY:** internal registry/group consistency, declared fingerprint collisions and conditional split assignment PASS/FAIL; realized split isolation remains `NOT_EVALUATED` without a frozen assignment.
- **G3 SURFACE:** planned exposure/counterbalance consistency only. Actual lexical leak audit `NOT_EVALUATED`.
- **G4 ATTACKS:** declare intended shortcut families (majority, t1-only, authentication-only, last mention, query-only, rule-agnostic, source/template-only). Their actual performance is `NOT_EVALUATED` until separately authorized examples exist.
- **G5 SERIALIZATION:** `NOT_EVALUATED` without a pinned tokenizer and real serialized text.
- **G6 CAPACITY:** count registry root families per initial truth stratum and renderer, without claiming their substantive independence, target sample size or statistical power. Never count paraphrases/arms as independent families.
- **G7 PROVENANCE:** input validation only; hashes, manifests and materialization provenance `NOT_EVALUATED`.

All returned reports must visibly distinguish `PASS_PLAN_CHECK`, `FAIL`, and `NOT_EVALUATED`. A missing plan may not be silently assumed balanced. Tests independently construct valid and deliberately inconsistent **metadata fixtures**, and cover missing fields, wrong types, family ID collisions, split crossing, dependent reversals, inappropriate renderer holdout claims, incomplete 27-cell planning, and false full-feasibility promotions. Static no-side-effect inspection and input-immutability tests are mandatory.

## 5. Scope and stopping conditions

No data materialization, generated prompt text, actual train/dev/test assignment, tokenizer/model execution, training/evaluation, CUDA, Kaggle, state intervention, new scientific effect claim, or Phase3B transition. No Codex. After the one-file scope review and operator commit/push, implement only the two new files, run narrow CPU unit tests and `cm ship`, then stop for manual two-file commit review. A plan-validator PASS does **not** authorize dataset generation.

**Disposition:** `RFR_D2_REGISTRY_VALIDATOR_SCOPE_READY_FOR_ONE_FILE_REVIEW / DATA=NONE / SCIENCE=NONE / GPU=OFF / PHASE3B=BLOCKED_UNCHANGED`.
