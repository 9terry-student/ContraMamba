# Gen5 v10 — Bounded Paired Runner: Minimal Implementation Scope v1

**Status:** `DESIGN_AND_IMPLEMENTATION_SCOPE_CANDIDATE / NOT_AN_EXECUTION_GATE`\
**Date:** 2026-10-11 (Asia/Seoul)\
**Phase:** CPU/static implementation planning only; no CUDA, Kaggle, model execution, population evaluation, or training.

## 1. Frozen source of truth

- Branch/worktree: `gen5-causal-role-state-ownership`, `C:\g5-state`.
- Last authenticated remote HEAD: `2b1aa3e3737c899337bd51d7a1fe6cb308b15511`, containing the frozen bounded E0–E3 record verifier.
- Versioned policy from commit `14608d234f6c0e1e8d7cd446b3e384cf77fc86da`: `reports/native_recurrent_generator_anchor_pair_localization_v10_numeric_contract_and_candidate_freeze_spec.md`, blob `661c627f09034c34f9830b18b7e9938c72c05925`.
- Original frozen script `scripts/audit_native_recurrent_generator_anchor_pair_localization.py`, blob `f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635`; frozen test blob `26c90ada9a74a2ac2716c880e58d15df5bd01704`.
- v10 hybrid blob `56cbcf3b2b4c47eeb807a14778a93ec8d60ffbf0`; shared GEMM blob `9003fb9bd23b2a6680ef210a51845058622badd2`.
- Bounded CPU verifier script at `2b1aa3e`: blob `7f547adc42a6e113dd179cb0f4988336ee476b17`. Do not modify it in this phase.

## 2. Architectural decision

Build a **separately versioned bounded paired-engine adapter**, not a production full-840 runner. The sole purpose is to arrange that the **same authenticated source tensor bundle** is fed independently through (a) the original frozen fine reducer and (b) v10 Hybrid, retaining all original outer coarse/fine guard semantics and emitting engine-separated, shape-checked records for later verification.

The original frozen `run_full()` uses `_spawn_worker()` to launch `__file__` with `--run-worker`, so simply importing the v10 function in a parent process does **not** integrate v10 into child workers. No transient global symbol replacement/monkeypatching is accepted as the execution architecture. The bounded runner must have its own explicit entrypoint and process boundary; any future fork of worker logic must be a separately identified script with an independently reviewed minimal semantic diff from the frozen worker.

A frozen audit `run_worker` or `run_full` **must never** be invoked expecting a swapped reducer. Do not edit the original audit script, original tests, original execution gate, or the existing candidate/verifier blobs.

## 3. Only the next CPU/static implementation scope

Create exactly these new files under a future implementation step, and nothing else:

1. `scripts/gen5_v10_bounded_pair_adapter_candidate.py` — pure injected-module adapter, receives already-constructed `component`, `log_a`, `survival`, `attention_mask`, `role_masks`, `fine_masks`, `shape` and a frozen `audit` module; never loads data, model, tokenizer, checkpoint, or CUDA by itself.
2. `tests/test_gen5_v10_bounded_pair_adapter_candidate.py` — CPU-only positive/negative tests and frozen-module integration tests.
3. `reports/native_recurrent_generator_anchor_pair_localization_v10_bounded_pair_adapter_implementation_scope.md` — this scope document when installed in the repo.

No new worker, exporter, CLI `--run`, JSON output schema, or GPU entrypoint is authorized in this initial patch. Adapter tests must enforce CPU-only and must not create or mutate any user data or Git files.

### Adapter contract

- Verify K4, B32 or B8, S128, W24576 on production-like geometry; explicitly flagged **CPU synthetic** geometry may use reduced width. Never accept CUDA with the synthetic flag.
- Compute coarse statistics with the genuine frozen `coarse._component_region_pair_stats` once, without substituting a proxy.
- Produce frozen fine-gap using the actual frozen `_fine_pair_gap_interference_fft` and v10 fine-gap using the pinned `hybrid_fine_pair_gap`; for B8, v10 delegates to the entire frozen reducer unchanged.
- For both fine-gap outputs independently, execute an explicitly factored copy of frozen `_component_fine_pair_stats`'s *post-reducer* parent-reconstruction and record emission logic, with a test verifying unchanged original replay calls and original tolerance/normalization semantics. No replay weakening or normalization drift. Coarse stats must be identical inputs for both engines, not independently fabricated.
- Return clearly separated frozen and v10 records plus exact dispatch counts. Never relabel engine diagnostics to make full-record comparison PASS. Preserve all record fields including `fine_parent_reconstruction_abs_max` and `fine_parent_reconstruction_rel_max`.
- Avoid stale CUDA streams, nondeterministic TF32 settings or implicit dtype conversion; no CUDA experiment in this scope.
- The adapter shall never claim E0/E1/E2/E3 formally PASS: data/tensor provenance and upstream production replay cannot be established by CPU injected fixtures.

## 4. CPU validation before another freeze

Run exactly the relevant new tests and a static import/plan check, plus `git -C C:\g5-state diff --check`; verify the new adapter returns frozen/v10 rows on real frozen synthetic CPU fixtures, has B32 fast/GEMM mixed and B8 routes, independently fails on parent-tamper, omitted fine cell, NaN, wrong geometry, or unexpected output shape. Ensure source strings cannot select a production entrypoint. Inspect `git -C C:\g5-state status --short` and stage only explicitly reviewed new files. No commit/push until a separate manual instruction after successful scope/CPU verification.

## 5. What is deliberately not yet authorized

- No full-width GPU paired benchmark or performance claim from the adapter tests.
- No real checkpoint/data/model forward, Kaggle session, 840-row worker/merge, scientific evaluation, training, or optimizer.
- No change to E0–E3 verdict semantics: legacy full-record FAIL remains FAIL for the observed fallback records; E0 and complete E1 are not authenticated merely by paired numerical records. Existing bounded verifier's `NOT_EVALUATED` fields remain unchanged.
- No worker/result/provenance schema reuse for v10. When later authorized, new versioned runner/export artifacts must capture tensor-shape/dtype/content hashes, source/worker/batch identity, head/blobs, runtime settings, dispatch counts, and actual frozen guard outcomes. A **separately scoped** v2 verifier will then authenticate this new schema. Until that implementation and a new bounded execution gate exist, `SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`.

## 6. Explicit stop conditions

Stop on branch or blob drift, dirty/unreviewed worktree, unapproved file delta, replacing genuine coarse/frozen replay with stand-ins outside tests, changed tolerances, missing field identity, modified candidate script/verifier, unexplained E2 numerical mismatch, altered E3 diagnostic, or any implicit CUDA/GPU action. Preserve all existing work.

**Disposition:** `NEXT_ACTION=CPU_STATIC_BOUNDED_PAIR_ADAPTER_IMPLEMENTATION_PREP`; `CODEX=NO`; `GPU=OFF`; `TRAINING=NO`; `COMMIT_PUSH=NO`; `FULL_840_SCIENCE=BLOCKED`.
