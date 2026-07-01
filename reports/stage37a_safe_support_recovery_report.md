# Stage37-A: Conservative Safe SUPPORT Recovery (shadow-only)

## Background: Stage36-A safety baseline

Stage36-A's four conservative support-safety blockers were effective on the
Stage35-A adversarial probe: overclaim, exception, and location-scope
SUPPORT errors were driven to zero, and REFUTE-to-SUPPORT leakage stayed at
zero.

| Metric | Stage35-A baseline | Stage36-A result |
|---|---:|---:|
| Shadow macro-F1 | 0.5832 | 0.6390 |
| `adv_overclaim_to_support` | 58 | 0 |
| `adv_exception_to_support_error` | 8 | 0 |
| `adv_location_scope_to_support_error` | 25 | 0 |
| `adv_refute_to_support` | 0 | 0 |
| `adv_support_shadow_support` | 91 / 250 | 91 / 250 |
| `adv_subset_support_recovered` | 60 / 175 | 60 / 175 |
| `adv_numeric_support_recovered` | 0 / 25 | 0 / 25 |
| `stage35_scope_safety` | unsafe | safe |
| `stage36_support_blocker_fired_count` | -- | 58 |
| `stage36_exception_blocker_fired_count` | -- | 8 |
| `stage36_not_all_blocker_fired_count` | -- | 25 |
| `stage36_location_scope_blocker_fired_count` | -- | 25 |

Blocking every unsafe proposed SUPPORT also blocked the rows that were
*correctly* SUPPORT but happened to match the same surface patterns, so
support recovery stayed flat at 91/250 (36.4%) and subset recovery at
60/175 (34.3%). The three remaining bad-row patterns identified after
Stage36-A are all rows where the gold label is SUPPORT and the post-Stage36
shadow label is stuck at NOT_ENTITLED/REFUTE:

1. `no X except Y -> Y SUPPORT` (included-subset double-negative)
2. `all X and all Z -> subset of X SUPPORT` (coordination universal)
3. `all N X -> subset among X SUPPORT` (numeric universal)

## What Stage37-A adds

Stage37-A adds three conservative, deterministic safe-SUPPORT-recovery rules
that run **strictly after** Stage36's post-blocker shadow label is known.
Stage37 never overrides a fired Stage36 blocker, and only ever recovers a
label that is currently NOT_ENTITLED (or REFUTE, only when explicitly
opted in) to SUPPORT.

- `--stage37-recover-no-except-included-subset`: recovers SUPPORT for
  `No X except Y ...` evidence when the claim's subject word-set is an
  *exact* match to the included subset `Y` and the claim/evidence predicates
  overlap >= 0.5. Exact-match (not overlap-ratio) is required deliberately:
  a ratio-based check would conflate `night-shift workers` with
  `day-shift workers` because both share the tokens `shift`/`workers`.
- `--stage37-recover-coordination-universal-subset`: recovers SUPPORT for
  `All X and all Z were/had P` evidence when the claim's subject contains
  the head noun of `X` or `Z` (e.g. `night-shift workers` is a subset of
  `workers`) and predicates overlap >= 0.5. Does not match `All X and some Z`
  (only literal `and all` coordination is matched).
- `--stage37-recover-numeric-universal-subset`: recovers SUPPORT for
  `All N X were/had/received P` evidence when the claim uses `among the X`
  phrasing or otherwise names a subset whose head is `X`, and predicates
  overlap >= 0.5. Never matches evidence without a leading `all`, so a
  reverse-numeric overclaim (`3 X were P` -> `All 12 X were P`) cannot fire.

Before running any of the three rules, Stage37 always checks four reusable
hazard detectors that wrap the existing Stage36 blocker logic verbatim (no
duplicated/diverging logic):

- `has_excluded_subset_hazard` (Stage36's `all X except Y` blocker)
- `has_not_all_existential_hazard` (Stage36's `not all/every X` blocker)
- `has_location_scope_mismatch` (Stage36's location-scope blocker)
- `has_temporal_scope_mismatch` (Stage36's temporal-scope blocker)

If any hazard check trips, or a Stage36 support blocker already fired on
that row, Stage37 never fires -- it cannot override Stage36.

All flags are gated behind `--stage37-use-safe-support-recovery` (master
switch), `--stage37-safe-support-export` (include `stage37_*` diagnostic
fields), and `--stage37-safe-support-shadow-mode` (allow a fired rule to
actually replace the exported shadow label). All default to **off**. When
off, behavior is identical to Stage36-A. `--stage37-allow-recover-from-refute`
(default off) additionally allows recovery from a post-Stage36 REFUTE label,
not just NOT_ENTITLED.

## No model or final-output changes

Stage37-A never modifies:

- model architecture, the Mamba backbone, or any training/loss/dataloader
  code path,
- `final_logits`, `final_probs`, or `pred_label`/`pred_final_label`,
- checkpoint selection,
- Stage31/Stage34/Stage35 data,
- Stage36 blocker behavior (Stage37 only reads Stage36's output; it cannot
  override a fired blocker),
- Stage33-F rule extraction itself (only a call site was added after the
  Stage36 call site, immediately before `exported.append(item)`).

It only ever changes the exported `stage32_shadow_label` /
`stage33_conditional_shadow_label` diagnostic fields, and only when
`--stage37-safe-support-shadow-mode` is explicitly enabled.

## Before/after support recovery (expected, pending Kaggle validation)

| Metric | Post-Stage36 (baseline) | Expected after Stage37-A |
|---|---:|---:|
| `adv_support_shadow_support` | 91 / 250 (36.4%) | higher |
| `adv_subset_support_recovered` | 60 / 175 (34.3%) | higher |
| `adv_numeric_support_recovered` | 0 / 25 | higher (if numeric flag enabled) |
| `adv_overclaim_to_support` | 0 | must remain 0 |
| `adv_exception_to_support_error` | 0 | must remain 0 |
| `adv_location_scope_to_support_error` | 0 | must remain 0 |
| `adv_refute_to_support` | 0 | must remain 0 (or not meaningfully increase) |

The evaluator (`scripts/evaluate_stage35_adversarial_coverage.py`) now
resolves the shadow label with priority `stage37_final_shadow_label` >
`stage36_final_shadow_label` > `stage32_shadow_label` >
`stage33_conditional_shadow_label`, so all normal Stage35/36 counters
automatically reflect the post-Stage37 label whenever Stage37 fields are
present. It also emits paired before/after counters
(`stage37_original_*` / `stage37_post_*`) and a `stage37_decision` label
(`STAGE37A_SAFE_SUPPORT_RECOVERY_EFFECTIVE` /
`STAGE37A_RECOVERY_TOO_UNSAFE` / `STAGE37A_RECOVERY_TOO_WEAK` /
`STAGE37A_DIAGNOSTIC_ONLY`) whenever `stage37_*` fields are present. Runs
without Stage37 fields keep the Stage35/Stage36 labels unchanged (verified
against a synthetic legacy predictions file with only `stage36_*` fields).

## Remaining risks

- Passive/active-voice SUPPORT recovery is not addressed by any Stage37-A
  rule and remains unresolved.
- Coordination parsing (`--stage37-recover-coordination-universal-subset`)
  is heuristic head-noun overlap, not real parsing; it is deliberately loose
  (per spec, `night-shift workers` matches `workers`) and could overclaim on
  contrived multi-word whole phrases with a shared head noun but a different
  qualifier (e.g. `north workers` vs. claim `night workers`). This trade-off
  mirrors the explicit examples in the design spec but should be monitored
  via `stage37_blocked_by_*_count` and `stage37_post_overclaim_to_support`
  on every run.
- Numeric subset recovery (`--stage37-recover-numeric-universal-subset`) is
  synthetic in the sense that it depends on a specific `among the X` claim
  phrasing or exact head-noun subset phrasing; real-world claim phrasing
  that expresses the same fact differently will not be recovered.
- Stage37 remains shadow-only and diagnostic-only: it never touches
  `final_logits`, `final_probs`, `pred_label`, or checkpoint selection, and
  must not be used for training, calibration, threshold selection, loss, or
  Kaggle selection.
- All Stage35-A/Stage36-A baseline numbers in this report are carried over
  from the task's reported prior runs; actual post-Stage37-A numbers require
  a fresh Kaggle validation run per task instructions (no training or
  evaluation was executed locally to produce this report).
