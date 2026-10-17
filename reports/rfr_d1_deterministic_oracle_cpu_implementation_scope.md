# RFR-D1 — Exact Two-File CPU Oracle Implementation Scope

**Status:** `IMPLEMENTATION_SCOPE_CANDIDATE / NOT_YET_FROZEN / NO_IMPLEMENTATION_EXECUTED`
**Date:** 2026-10-17 (Asia/Seoul)
**Branch:** `gen5-causal-role-state-ownership`
**Source HEAD:** `8afa06fc7b7d2d01f6cdb3cc22749314ae14e599`
**Parent contract:** `reports/rfr_d1_deterministic_oracle_and_generator_contract.md` at the source HEAD
**Parent design:** `reports/rfr_d0_dependency_aware_revision_scientific_design_candidate.md`

## 1. Phase and authorization boundary

This is an implementation **scope freeze candidate**, not a new scientific design, scientific evidence, data-generation authority, or permission to execute a model. The source HEAD contains the committed D0/D1 contracts. The implementation phase may begin **only after this exact scope is committed and the clean new HEAD is confirmed**. This scope intentionally restates only executable boundaries that the D1 contract left to implementation review; the parent D0/D1 contracts control semantics if there is any discrepancy.

**Exactly two new files may be added in the later implementation phase:**

1. `scripts/rfr_d1_deterministic_oracle_cpu.py`
2. `tests/test_rfr_d1_deterministic_oracle_cpu.py`

Do not edit existing repository files, including D0/D1 contracts, XG1 data/splits/labels/encoding, historical generators, model code, checkpoints, and G2/native/v10 artifacts. Do not add a requirements file or third implementation file. No Codex. No dataset materialization, tokenizer, model loading/inference, training, evaluation, Kaggle, CUDA, native-state measurements or Phase3B transition. No local commit/push until exact two-file scope, tests and `cm ship` are reviewed by the user.

## 2. Pure functions / API boundary

Implement immutable-input, Python-standard-library-only functions. API names below are fixed for the implementation; internal private helpers may be added **inside the one script only**.

- `strong_kleene_and(a, c) -> str`: accepts only `'T'`, `'F'`, `'?'`; implements the nine-row frozen truth table.
- `truth_to_label(value) -> tuple[str, int]`: exactly `'F' -> ('REFUTE', 0)`, `'?' -> ('NOT_ENTITLED', 1)`, `'T' -> ('SUPPORT', 2)`.
- `evaluate_records(events, query_target, readout, distractors=()) -> dict`: pure deterministic evaluation; events are maps with `event_id`, `target`, `asserted_value` (strict Boolean), `source_authenticated` (strict Boolean), `event_time` (integer, excluding bool), and nonempty immutable source/provenance identities. `query_target` is `A`, `B`, or `U`; `readout` is `PRE`, `POST`, or `DELAYED`. Valid base targets are `A`, `C`, `U`, with derived `B := A AND C` never writable. `PRE` sees time-10 t0 authority only; `POST` sees t0 and t1; `DELAYED` uses the same admitted records plus verified neutral distractors. Return a stable mapping with at least `A`, `C`, `B`, `U` three-valued statuses, `query_target`, `target_status`, `final_label`, and `final_label_id`.
- `primary_fixture(a0, c0, u0, arm, readout, query_target) -> dict`: evaluate **only in memory** the exact D1 Boolean primary fixture. Valid arms: `VALID_FLIP`, `VALID_SHAM`, `INVALID_FLIP`; strict Boolean baselines; untrusted flip never overrides t0. The function returns the same result structure as `evaluate_records`; it neither yields nor serializes a dataset row.
- `verify_split_isolation(assignments) -> bool`: read-only structural validation of in-memory maps containing `case_id`, `case_family_id`, `split_group_id`, and `split` from `train`, `dev`, `test`. Same abstract family/group and case ID must never cross splits; no splitting by rows, queries or arms. Contradictory reuse of case identity or family/group associations fails closed.

The implementation may expose `ORACLE_VERSION = 'RFR_D1_ORACLE_V1'`; version mismatches must be rejected when present, and unknown rule IDs (other than `B_AND_A_C`) are rejected. No implicit coercion of malformed types or absent identities. No dynamic plugins or file access. Functions must not mutate their input events, distractors or assignments.

## 3. Record consistency and temporal precedence

Validate the full event set independently of presentation order. Authenticate only records with a nonempty `source_id` and `provenance_id`. Missing t0 A/C/U record yields `'?'`, never `'F'`. A record may replace an admitted record only if authenticated and **strictly later** by logical timestamp. A conflicting pair of authenticated assertions at the same target/time rejects the entire evaluation regardless of their presentation order; exact duplicate events are idempotent. Same `event_id` with changed payload rejects even when the altered event would otherwise be ignored. Unauthenticated later assertions and authenticated stale assertions cannot defeat a newer authenticated state. Reject illegal event targets, malformed values, unsupported timestamp types, unknown oracle version or derivation rule.

`primary_fixture` is restricted to an `A` update; C and U cannot be written by the primary arm. Generic records support only baseline C/U authority and must reject later authenticated primary-style changes to C/U. A `DELAYED` distractor is explicitly neutral metadata, not a fourth admissible event: reject embedded event-like material or claims of an A/C/U update. No parsing natural-language truth from token strings is authorized.

## 4. Independent tests and mandatory acceptance

Tests must independently derive expected results from Boolean `a0`, `c0`, `u0` and arm identity, **without calling production oracle functions to compute expected values**. Minimum coverage:

- Exactly `8 × 3 × 3 × 3 = 216` primary fixture assertions, including both A flip directions; B flips iff C is true; U is unchanged; `DELAYED == POST`; all expected labels verified as name and integer.
- The nine exhaustive strong-Kleene conjunction cases, including missing A/C evidence, plus explicit `F` versus `?` label distinction.
- Event-order permutations (same timestamp semantics), authenticated stale event versus later trusted event, an unauthenticated newer flip, same-time trusted conflicts, incompatible reused `event_id`, and identical event duplicates.
- Unauthorized C/U/B updates; malformed identifiers, missing source/provenance, strict Boolean and timestamp types, unknown oracle/rule/version, invalid arm/readout/query; contaminating distractors rejected.
- In-memory split-isolation positive/negative cases, including conflicting case/family/group reuse; input immutability and deterministic repeated calls.
- AST/module checks or equivalent monkeypatch-free assertions that the target module uses Python standard library only; no filesystem write API, dataset artifact creation, model/tokenizer, torch, CUDA, network or Git mutation. Tests run on in-memory fixtures only.

Use the narrowly scoped local check:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B -m unittest discover -s tests -p "test_rfr_d1_deterministic_oracle_cpu.py" -v
if ($LASTEXITCODE -ne 0) { throw 'RFR_D1_ORACLE_TEST_FAILED' }
git diff --check
cm ship
```

A PASS means CPU oracle *implementation contract correctness only*; it does **not** validate a real corpus, tokenizer capacity, scientific effect, native recurrent revision, generalization, or causal mediation. Any ambiguity in event provenance/ties or any failed negative control is a blocker, not a reason to soften tests or add a new intervention arm.

## 5. Stop conditions and handoff

- First freeze this **one** scope document with an exact-file `cm ship` review and a user-executed commit/push. The branch must remain `gen5-causal-role-state-ownership` and clean.
- Then implement only the two whitelisted files, run the stated CPU checks, confirm exactly two files staged and nothing else changed, and request the next manual commit/push review.
- Do not use Kaggle, run model code, generate or save cases, assign actual train/dev/test split, or choose sample sizes/metrics here.

**Disposition:** `D1_TWO_FILE_CPU_IMPLEMENTATION_SCOPE_READY_FOR_FREEZE / SCIENCE_RESULT=NONE / GPU=OFF / PHASE3B=BLOCKED_UNCHANGED`.
