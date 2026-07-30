# ContraMamba Reason-Preserving Authorization Router P3 Experimental Execution Specification

## 1. Executive decision

```text
overall_p3_decision = P3_BLOCKED_BY_UNRESOLVED_EXECUTION_CONTRACT
a0_phase_decision = P3_A0_PHASE_READY_FOR_EXECUTION
a1_a3_phase_decision = P3_A1_A3_BLOCKED
```

This report is execution-spec only. It performs no implementation, training, evaluation, model loading, tests, artifact validation, or Kaggle execution. A0 is Phase-1 runnable because it has no positive reason-loss weight. A1/A3 are blocked until experimental positive reason-loss weight authority is established. A2 is also blocked because Phase 3 must be released as a matched factorial A1-A3 batch. Full mechanism comparison remains blocked.

Commit identity is split into three placeholders:

- `P2_IMPLEMENTATION_TESTED_COMMIT_SHA`: commit where the 26 P2 contract tests passed.
- `P3_EXECUTION_SPEC_COMMIT_SHA`: commit containing this report and manifest.
- `P3_EXECUTION_CHECKOUT_COMMIT_SHA`: Kaggle checkout commit; planned to equal the spec commit.

## 2. Authority and inspected sources

| Priority | Source | Role |
|---:|---|---|
| 1 | `reason_router_p2.patch`, `reason_router_p2_final_review.patch`, `reason_router_p2_final_review_v7.patch` | In-repository P2/final-review contract. |
| 2 | `tests/test_reason_router_p2_contract.py` | Contract reference; user reports `26 passed`, `P2_TEST_RETURN_CODE=0`. |
| 3 | `scripts/train_controlled_v6b_minimal.py` | Parser defaults, P2 fail-fast resolver, report/checkpoint metadata, prediction resolver, A0 join helpers. |
| 4 | `src/contramamba/heads/entitlement_decision.py` | Router numerical contract and reason class order. |
| 5 | `src/contramamba/modeling_v6b_minimal.py` | Mamba wrapper, frozen encoder path, explicit-local ownership path. |
| 6 | Stage195 specs/manifest builder | Canonical CUDA/Mamba envelope, seeds, sidecar, split seed, epochs, metric, run-root style. |
| 7 | `scripts/train_controlled_v5.py`, `scripts/build_controlled_v5.py` | Optimizer defaults and production JSONL load/split functions. |

No standalone file named exactly "Reason-Preserving Authorization Router P1 final report" was found, so the P2 patches/tests plus current implementation are the highest available reason-router authority.

## 3. Resolved canonical execution configuration

| Parameter | Resolved value | Evidence / field | Why fixed |
|---|---|---|---|
| Trainer | `scripts/train_controlled_v6b_minimal.py` | `build_parser`, `_p2_resolve_arm_contract` | Current P2-capable trainer. |
| Architecture | `v6b_minimal` | P2 resolver | P2 fail-fast requirement. |
| Backbone/model | `mamba`, `state-spaces/mamba-130m-hf` | Stage195 envelope, model default | Canonical Mamba-130m path. |
| Device | `cuda` | Stage195 envelope | Canonical CUDA/Mamba execution. |
| Encoder | frozen | `--freeze-encoder true` | P2 rejects unfrozen encoder. |
| Frame downstream mode | `joint` | `--frame-downstream-gradient-mode joint` | P2 rejects legacy local-only mode. |
| Clean-main dataset | `data/controlled_v5_v3_without_time_swap.jsonl` | `_STAGE187_AUTHORITATIVE_DATA` | Authoritative no-time-swap data. |
| Dataset SHA-256 | `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640` | `_STAGE187_DATASET_SHA256` | Data identity gate. |
| Integrity sidecar | `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl` | `_STAGE187_AUTHORITATIVE_SIDECAR` | P2 metadata source. |
| Sidecar semantic SHA-256 | `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc` | `_STAGE187_SIDECAR_SEMANTIC_SHA256` | Sidecar identity gate. |
| Seeds | `180`, `181`, `182` | Stage195 tail-three seeds | Canonical tail seeds. |
| Split seed/policy | `174`, `fixed_explicit_split_seed` | `--split-seed 174` | Same train/dev universe. |
| Dev ratio | `0.2` | parser/Stage195 | Fixed dev universe. |
| Epochs | `20` | Stage195 | Canonical controlled length. |
| Max length | `128` | parser/Stage195 | Encoding comparability. |
| Optimizer/LR | AdamW via `v5.build_optimizer`, `--lr 0.001` | v5 optimizer path | Existing training path. |
| Weight decay | actual `1e-4`; no CLI weight-decay field | `AdamW(... weight_decay=1e-4)` | Optimizer identity. |
| Class weighting/sampler | `none`, `balanced_sampler=false` | parser defaults | Avoid weighting/sampler drift. |
| Selection metric | `final_macro_f1` | `--select-metric` | Checkpoint comparability. |
| Selected checkpoint | `<run_dir>/selected_checkpoint.pt` | `--save-selected-checkpoint` | Artifact identity. |
| Run directory | `/kaggle/working/ContraMamba/reports/reason_router_p2_p3_runs/seed{seed}/{arm}` | manifest convention | Seed/arm isolation. |

Arm contract:

| Arm | Router | Gradient ownership | Primary reason CE |
|---|---|---|---|
| A0 | `explicit_product` | `joint` | none, weight `0.0` |
| A1 | `conditional_first_blocker` | `joint` | yes, weight `UNRESOLVED_EXECUTION_PARAMETER` |
| A2 | `explicit_product` | `explicit_local` | none, weight `0.0` |
| A3 | `conditional_first_blocker` | `explicit_local` | yes, weight `UNRESOLVED_EXECUTION_PARAMETER` |

Reason order remains `FRAME`, `PREDICATE`, `SUFFICIENCY`, `AUTHORIZED`; secondary reasons are diagnostic-only; A3 final 3-way CE is router-only; A3 local polarity CE uses the raw `PolarityEnergyHead` owner path; A1 local losses retain raw upstream ownership; EMA is observer/baseline-only; q-only decomposition is not causal; E0 algebraic-equivalence is preserved.

Prediction JSONL derivation authority: `_prediction_export_jsonl_path(output_json: Path)` in `scripts/train_controlled_v6b_minimal.py` returns `output_json.with_name(f"{output_json.stem}_predictions.jsonl")`. The stem comes from `--output-json`; the trainer writes via `write_jsonl(_prediction_export_output_jsonl, _prediction_export_rows_for_jsonl)`.

Authoritative dev universe authority: `v5.load_jsonl` ultimately uses `scripts.build_controlled_v5.load_jsonl`; production split is `v5.split_by_pair_id(records, dev_ratio=args.dev_ratio, seed=resolved_split_seed)`; implementation validates records, shuffles sorted pair IDs with `random.Random(seed)`, and preserves row order. Current trainer split identity helper is `_p2_row_identity_hash(records)` but P3 audit extends the immutable hash to include normalized gold label. Label normalization is `_s28e_normalize_label`; prediction identity uses `_p2_row_identity`, `_p2_reference_pair_id`, `_p2_a0_reference_key`; A0 join uses `_p2_load_a0_reference_predictions` and `_p2_validate_a0_reference_for_universe`.

## 4. Unresolved parameters

| Parameter | Status | Impact |
|---|---|---|
| `reason_loss_weight_A1_A3` | `UNRESOLVED_EXECUTION_PARAMETER` | Blocks A1/A3 and the matched Phase 3 batch, including A2 release. |
| Per-run `ownership_violation_count` | `P3_BLOCKED_BY_MISSING_EXECUTION_OBSERVABILITY` | Not currently emitted per run; do not infer zero from tests/config. |
| Per-run checkpoint metadata contract summary | `P3_BLOCKED_BY_MISSING_EXECUTION_OBSERVABILITY` | Checkpoint load is outside report-only P3 scope. |
| Stable true-SUPPORT harm tolerance | `UNRESOLVED_EXECUTION_PARAMETER` | Blocks final mechanism interpretation. |
| Polarity-error non-increase threshold beyond direction | `UNRESOLVED_EXECUTION_PARAMETER` | Report seed direction only. |

The value `1.0` in unit fixtures, exact-resume metadata tests, or example loss construction is not experiment hyperparameter authority. It validates positive-weight handling and metadata preservation only.

## 5. P2 fail-fast compatibility audit

| Item | P3 command handling |
|---|---|
| `architecture=v6b_minimal` | Explicitly set. |
| Mamba backbone | Explicit `--backbone mamba --model-name state-spaces/mamba-130m-hf`. |
| Encoder frozen | Explicit `--freeze-encoder true`. |
| `frame_downstream_gradient_mode=joint` | Explicitly set. |
| P2 comparator flags | Omit `--use-temporal-comparator` and `--use-predicate-comparator`; P2 rejects flag presence. |
| Teacher observer | Parser default off; omit. |
| external/OOD/bridge paths | Do not pass paths. |
| Ranking loss | Explicit `--ranking-weight 0.0`. |
| Intervention/pairwise loss | Omit; default safe. |
| Compatible-positive margin | Explicit `--compatible-positive-margin-weight 0.0` and `--compatible-positive-margin-logit 0.0`. |
| Boundary/frame/predicate/preservation losses | Defaults off/zero; omit. |
| Stage174/175/177 objectives | Defaults off/zero/no path; omit. |
| Pair-contrastive frame objective | Default off/no path; omit. |
| Temporal diagnostic/residual/adapter/channel objectives | Defaults off/zero/no path; omit. |
| v7 objectives | v6b architecture and defaults off; omit. |
| time_swap data | Use no-time-swap dataset and SHA audit. |
| P2 sidecar | Explicit path and expected semantic SHA. |
| A0 reference | Never pass for A0; required for future same-seed A1-A3 only. |

Conclusion: A0 commands are compatible with P2 fail-fast gates after explicit objective neutralization. A1-A3 remain blocked.

P2 resolver parser-default audit for `_p2_resolve_arm_contract()` collections:

| Resolver collection | Option(s) | Parser default | Classification | P3 A0 handling |
|---|---|---|---|---|
| direct CLI flag check | `--use-temporal-comparator`, `--use-predicate-comparator` | parser default `True`, but P2 rejects CLI flag presence and resolves both false internally | `FLAG_MUST_BE_OMITTED` | Omit both flags; preflight asserts absence. |
| `incompatible_options` | `temporal_adapter_final_penalty_scale`, `temporal_channel_gated_penalty_scale` | `0.0` | `SAFE_DEFAULT` | Omit. |
| `incompatible_options` | `use_temporal_adapter_final_penalty`, `use_temporal_channel_gated_penalty`, `vnext_enable_segmented_dual_pass`, `use_temporal_diagnostic_loss`, `use_temporal_channel_loss`, `use_temporal_adapter_loss` | `False` | `SAFE_DEFAULT` | Omit. |
| `incompatible_options` | `architecture` | explicit `v6b_minimal` | `SAFE_DEFAULT` with explicit architecture | Set `--architecture v6b_minimal`. |
| forbidden paths | OOD, external, Stage43, Stage57/66/75/80A bridge paths | `None` | `PATH_MUST_BE_OMITTED` | Do not pass paths. |
| forbidden bridge modes | `stage57_bridge_train_mode`, `stage66_bridge_train_mode`, `stage75_bridge_train_mode`, `stage80a_bridge_train_mode` | `none` | `SAFE_DEFAULT` | Omit. |
| forbidden external flags | `enable_external_eval`, `enable_stage43_external_eval`, `stage43_external_enable_shadow_export` | `False` | `SAFE_DEFAULT` | Omit. |
| `objective_options` | `ranking_weight` | `2.0` from v5 parser | `EXPLICIT_ZERO_REQUIRED` | Pass `--ranking-weight 0.0`. |
| `objective_options` | `compatible_positive_margin_weight`, `compatible_positive_margin_logit` | `0.0`, `0.0` | `SAFE_DEFAULT`; explicit neutralization required by P3 manifest | Pass both as `0.0`. |
| `objective_options` | `stage174c_clean_pairwise_mode`, `stage174c_clean_pairwise_weight` | `off`, `0.0` | `SAFE_DEFAULT`; explicit neutralization required by P3 manifest | Pass `off` and `0.0`. |
| `objective_options` | `stage174c_clean_polarity_preservation_weight` | `1.0` | `EXPLICIT_ZERO_REQUIRED` | Pass `--stage174c-clean-polarity-preservation-weight 0.0`. |
| `objective_options` | `stage175b_support_anchor_mode`, `stage175b_support_anchor_weight` | `off`, `0.0` | `SAFE_DEFAULT`; explicit neutralization required by P3 manifest | Pass `off` and `0.0`. |
| `objective_options` | `stage177c_frame_pairwise_mode`, `stage177c_frame_pairwise_weight` | `off`, `0.0` | `SAFE_DEFAULT`; explicit neutralization required by P3 manifest | Pass `off` and `0.0`. |
| `objective_options` | intervention/loss-sweep/boundary/frame-violation/predicate-isolation/preservation/pair-contrastive/temporal/v7 use flags | `False` | `SAFE_DEFAULT` | Omit flags. |
| `objective_options` | corresponding objective weights except `ranking_weight` and Stage174-C polarity preservation | `0.0` | `SAFE_DEFAULT` | Omit unless included in explicit neutralization above. |
| `objective_options` | `pair_contrastive_frame_data` | `None` | `PATH_MUST_BE_OMITTED` | Do not pass path. |

Known non-safe defaults found by static parser inspection: `ranking_weight = 2.0` and `stage174c_clean_polarity_preservation_weight = 1.0`. Both are explicitly neutralized in A0 commands.

## 6. A0-first execution design

A0 Phase 1 is runnable for seeds `180`, `181`, and `182`. Each run uses `P3_EXECUTION_CHECKOUT_COMMIT_SHA`, records `P2_IMPLEMENTATION_TESTED_COMMIT_SHA`, and requires the four P2 files to be unchanged between those commits.

A0 command template:

```text
python scripts/train_controlled_v6b_minimal.py --data data/controlled_v5_v3_without_time_swap.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --epochs 20 --max-length 128 --dev-ratio 0.2 --seed <SEED> --split-seed 174 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --ranking-weight 0.0 --class-weighting none --stage174c-clean-pairwise-mode off --stage174c-clean-pairwise-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --stage175b-support-anchor-mode off --stage175b-support-anchor-weight 0.0 --stage177c-frame-pairwise-mode off --stage177c-frame-pairwise-weight 0.0 --compatible-positive-margin-logit 0.0 --lr 0.001 --reason-router-arm A0 --controlled-integrity-sidecar-path reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc --compatible-positive-margin-weight 0.0 --save-selected-checkpoint --selected-checkpoint-filename selected_checkpoint.pt --output-json /kaggle/working/ContraMamba/reports/reason_router_p2_p3_runs/seed<SEED>/A0/training_report.json --output-predictions-json /kaggle/working/ContraMamba/reports/reason_router_p2_p3_runs/seed<SEED>/A0/clean_dev_predictions.json
```

The command intentionally omits `--reason-loss-weight`.

## 7. Immutable A0 reference contract

Seed-local references:

```text
seed180 A1/A2/A3 -> seed180 A0 reference
seed181 A1/A2/A3 -> seed181 A0 reference
seed182 A1/A2/A3 -> seed182 A0 reference
```

The primary immutable reference is the selected-checkpoint prediction JSONL derived from `--output-json`, not `clean_dev_predictions.json`. The authoritative dev universe comes from production load/split, not a hard-coded row count. `720` may be a sanity check only.

## 8. A1-A3 matched execution contract

A1-A3 are not runnable in this specification.

```text
P3_A1_A3_BLOCKED: UNRESOLVED_REASON_LOSS_WEIGHT
```

After a later authority resolves `reason_loss_weight_A1_A3`, A1/A3 templates must include `--reason-loss-weight <RESOLVED_POSITIVE_REASON_LOSS_WEIGHT>` and same-seed `--reason-router-a0-reference-predictions`. A2 keeps `reason_loss_weight = 0.0`, but remains blocked because the matched Phase 3 factorial batch is not released until A1/A3 have an established positive reason-loss weight.

The future Phase 3 dependency gate must re-read persisted A0 audit JSON from disk and verify: same-seed `status == PASS`, audit execution commit equals current execution commit, dataset SHA equals current dataset SHA, sidecar semantic SHA equals current sidecar SHA, split seed matches, dev identity hash matches, and A0 prediction SHA is recorded.

## 9. Run matrix

| Phase | run_id | arm | seed | runnable | blocked_by | dependency |
|---|---|---|---:|---|---|---|
| 1 | `p3_seed180_A0` | A0 | 180 | true | none | none |
| 1 | `p3_seed181_A0` | A0 | 181 | true | none | none |
| 1 | `p3_seed182_A0` | A0 | 182 | true | none | none |
| 3 | `p3_seed180_A1` | A1 | 180 | false | `UNRESOLVED_REASON_LOSS_WEIGHT` | `p3_seed180_A0_REFERENCE_AUDIT where status == PASS` |
| 3 | `p3_seed180_A2` | A2 | 180 | false | `P3_A1_A3_MATCHED_EXECUTION_CONTRACT_NOT_RELEASED` | `p3_seed180_A0_REFERENCE_AUDIT where status == PASS` |
| 3 | `p3_seed180_A3` | A3 | 180 | false | `UNRESOLVED_REASON_LOSS_WEIGHT` | `p3_seed180_A0_REFERENCE_AUDIT where status == PASS` |
| 3 | `p3_seed181_A1` | A1 | 181 | false | `UNRESOLVED_REASON_LOSS_WEIGHT` | `p3_seed181_A0_REFERENCE_AUDIT where status == PASS` |
| 3 | `p3_seed181_A2` | A2 | 181 | false | `P3_A1_A3_MATCHED_EXECUTION_CONTRACT_NOT_RELEASED` | `p3_seed181_A0_REFERENCE_AUDIT where status == PASS` |
| 3 | `p3_seed181_A3` | A3 | 181 | false | `UNRESOLVED_REASON_LOSS_WEIGHT` | `p3_seed181_A0_REFERENCE_AUDIT where status == PASS` |
| 3 | `p3_seed182_A1` | A1 | 182 | false | `UNRESOLVED_REASON_LOSS_WEIGHT` | `p3_seed182_A0_REFERENCE_AUDIT where status == PASS` |
| 3 | `p3_seed182_A2` | A2 | 182 | false | `P3_A1_A3_MATCHED_EXECUTION_CONTRACT_NOT_RELEASED` | `p3_seed182_A0_REFERENCE_AUDIT where status == PASS` |
| 3 | `p3_seed182_A3` | A3 | 182 | false | `UNRESOLVED_REASON_LOSS_WEIGHT` | `p3_seed182_A0_REFERENCE_AUDIT where status == PASS` |

Execution order:

```text
Phase 1: A0 seed-by-seed execution
Phase 2: each A0 prediction/reference audit
Phase 3: currently blocked; after unblock, A0-audit-PASS seeds only may run A1-A3
Phase 4: arm-level aggregation
Phase 5: causal comparison gate
```

## 10. Output artifact schema

Required per A0 run: `training_report.json`, primary `training_report_predictions.jsonl`, auxiliary `clean_dev_predictions.json`, `selected_checkpoint.pt`, and seed-local `A0_REFERENCE_AUDIT.json`.

A0 audit JSON fields are standardized as:

```text
audit_id
run_id
seed
status
errors
execution_commit
p2_implementation_tested_commit
output_dir
reference_prediction_path
prediction_sha256
selected_checkpoint_path
selected_checkpoint_sha256
report_path
report_sha256
selected_epoch
selected_epoch_source
row_count
unique_row_id_count
unique_row_pair_count
authoritative_dev_row_count
authoritative_dev_row_identity_hash
prediction_joined_dev_row_identity_hash
gold_counts
prediction_counts
a0_false_entitlement_count
a0_stable_true_support_count
data_path
dataset_sha256_expected
dataset_sha256_observed
sidecar_path
sidecar_semantic_sha256_expected
sidecar_semantic_sha256_observed
split_seed
split_policy
dev_ratio
```

Use `selected_checkpoint_sha256`; do not use `checkpoint_sha256`. Null `selected_epoch`, `prediction_sha256`, `selected_checkpoint_sha256`, `report_sha256`, `authoritative_dev_row_identity_hash`, or `prediction_joined_dev_row_identity_hash` is audit failure.

## 11. A0 audit gate

Failure status is `P3_A0_REFERENCE_AUDIT_FAILED`; A1-A3 must not run for that seed after future release.

Identity gates: current `HEAD` equals `P3_EXECUTION_CHECKOUT_COMMIT_SHA`; `git status --short` is empty; the four P2 files are unchanged between tested and execution commits; data path/SHA and sidecar path/semantic SHA match expected and observed; split seed is `174`; policy is `fixed_explicit_split_seed`; dev ratio is `0.2`.

Prediction universe exact join gates: production dev split row count equals prediction row count; no missing source or prediction row ID; no source or prediction duplicate row ID; source and prediction row-ID sets are equal; `(row_id, pair_id)` sets are equal; normalized gold labels match; no unknown external class; prediction is one of `REFUTE`, `NOT_ENTITLED`, `SUPPORT`; no extra or missing prediction row.

Dev identity hash serialization:

```text
row_id<TAB>pair_id<TAB>normalized_gold_label<NEWLINE>
```

The authoritative hash uses production dev split order. The prediction hash uses the same source order after exact join. Record `authoritative_dev_row_identity_hash` and `prediction_joined_dev_row_identity_hash`; mismatch fails.

Artifact gates: prediction JSONL SHA-256, selected checkpoint SHA-256, and report SHA-256 must be recorded; output path must be seed-local; no cross-seed reference reuse. Selected epoch must be found at `runs.single.best_epoch` first, then top-level `best_epoch`, and `selected_epoch_source` must record the field used.

## 12. Evaluation and causal comparison

Required post-unblock metrics include standard accuracy/macro-F1/per-label recall/prediction counts, A0-fixed recovery and SUPPORT harm metrics, reason diagnostics, and contract diagnostics.

Diagnostics availability:

| Diagnostic | P3 status |
|---|---|
| accuracy, macro-F1, per-label recall, prediction counts | `AVAILABLE_AND_COLLECTED` from report/prediction artifacts after a run exists. |
| A0 false-entitlement recovery and stable true-SUPPORT harm | `DERIVABLE_FROM_EXISTING_ARTIFACTS` by same-seed exact join. |
| polarity error | `DERIVABLE_FROM_EXISTING_ARTIFACTS` from normalized gold/pred labels. |
| reason confusion matrix | `DERIVABLE_FROM_EXISTING_ARTIFACTS` if row-level reason fields are emitted; not assumed as an aggregate. |
| reason eligible/ignored counts | `DERIVABLE_FROM_EXISTING_ARTIFACTS` from row/report reason-supervision fields when present. |
| local loss applicable/ignored counts | `DERIVABLE_FROM_EXISTING_ARTIFACTS` from P2 loss summaries when present. |
| q/reason/internal/collapsed normalization errors | `DERIVABLE_FROM_EXISTING_ARTIFACTS` from emitted mass/probability fields. |
| nonfinite/negative count | `DERIVABLE_FROM_EXISTING_ARTIFACTS` by scanning emitted numeric fields. |
| A0 reference join status | `AVAILABLE_AND_COLLECTED` in A0 audit JSON; future A1-A3 must read production join metadata. |
| ownership violation count | `NOT_CURRENTLY_EMITTED_PER_RUN`; do not assume zero from tests. |
| checkpoint metadata contract | `NOT_CURRENTLY_EMITTED_PER_RUN_SUMMARY`; checkpoint loading is outside report-only P3 scope. |

Post-unblock comparisons: A1-A0, A2-A0, A3-A1, A3-A2, A3-A0. For each metric report seed deltas, mean delta, positive/negative/zero direction counts, and all-seed direction consistency. Do not report means alone.

Decision gate: `P3_INVALID_EXECUTION` for reference join, ownership, numerical contract, artifact identity, checkpoint metadata, wrong-seed A0 reference, or forbidden objective failure. `P3_MECHANISM_SUPPORTED` requires later unblocked evidence; no numeric harm/polarity threshold is invented. Otherwise use `P3_MECHANISM_NOT_SUPPORTED` only after valid execution.

## 13. Kaggle subprocess cells

These cells are templates for Kaggle. All shell commands use Python `subprocess`; no notebook `!command` or `%cd` is used. Replace placeholders explicitly; do not auto-guess commit hashes.

```python
# Cell 1: exact commit checkout, clean HEAD, and P2 implementation tree check
import json, subprocess, pathlib, hashlib, collections
REPO = pathlib.Path("/kaggle/working/ContraMamba")
P2_IMPLEMENTATION_TESTED_COMMIT = "<P2_IMPLEMENTATION_TESTED_COMMIT_SHA>"
P3_EXECUTION_SPEC_COMMIT = "<P3_EXECUTION_SPEC_COMMIT_SHA>"
P3_EXECUTION_CHECKOUT_COMMIT = "<P3_EXECUTION_CHECKOUT_COMMIT_SHA>"
P2_FILES = [
    "scripts/train_controlled_v6b_minimal.py",
    "src/contramamba/heads/entitlement_decision.py",
    "src/contramamba/modeling_v6b_minimal.py",
    "tests/test_reason_router_p2_contract.py",
]
assert P3_EXECUTION_SPEC_COMMIT == P3_EXECUTION_CHECKOUT_COMMIT, {
    "p3_execution_spec_commit": P3_EXECUTION_SPEC_COMMIT,
    "p3_execution_checkout_commit": P3_EXECUTION_CHECKOUT_COMMIT,
}
def run(cmd, cwd=REPO, check=True):
    print("$", " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr: print(result.stderr)
    if check and result.returncode != 0: raise RuntimeError(cmd)
    return result
run(["git", "fetch", "origin"])
run(["git", "checkout", P3_EXECUTION_CHECKOUT_COMMIT])
assert run(["git", "rev-parse", "HEAD"]).stdout.strip() == P3_EXECUTION_CHECKOUT_COMMIT
assert run(["git", "status", "--short"]).stdout.strip() == ""
run(["git", "diff", "--exit-code", f"{P2_IMPLEMENTATION_TESTED_COMMIT}..{P3_EXECUTION_CHECKOUT_COMMIT}", "--", *P2_FILES])
print(
    json.dumps(
        {
            "p2_implementation_tested_commit": P2_IMPLEMENTATION_TESTED_COMMIT,
            "p3_execution_spec_commit": P3_EXECUTION_SPEC_COMMIT,
            "p3_execution_checkout_commit": P3_EXECUTION_CHECKOUT_COMMIT,
            "status": "P3_COMMIT_IDENTITY_CHECK_PASS",
        },
        indent=2,
    )
)
```

```python
# Cell 2: environment/device confirmation
run(["python", "-c", "import torch, transformers; print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count()); print('transformers', transformers.__version__)"])
```

```python
# Cell 3: A0 manifest preview and constants
RUN_ROOT = REPO / "reports" / "reason_router_p2_p3_runs"
DATA = "data/controlled_v5_v3_without_time_swap.jsonl"
DATA_SHA_EXPECTED = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
SIDECAR = "reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl"
SIDECAR_SHA_EXPECTED = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
SEEDS = [180, 181, 182]
SPLIT_SEED = 174
DEV_RATIO = 0.2
SPLIT_POLICY = "fixed_explicit_split_seed"
VALID_EXTERNAL_CLASSES = {"REFUTE", "NOT_ENTITLED", "SUPPORT"}
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()
print(json.dumps({"a0_runs": [f"p3_seed{s}_A0" for s in SEEDS], "run_root": str(RUN_ROOT)}, indent=2))
```

```python
# Cell 4: seed-by-seed A0 execution
# The JSONL path follows production _prediction_export_jsonl_path(output_json).
def run_paths(seed, arm):
    out = RUN_ROOT / f"seed{seed}" / arm
    output_json = out / "training_report.json"
    prediction_jsonl = output_json.with_name(f"{output_json.stem}_predictions.jsonl")
    assert prediction_jsonl == out / "training_report_predictions.jsonl"
    return {"dir": out, "report": output_json, "prediction_jsonl": prediction_jsonl, "prediction_json": out / "clean_dev_predictions.json", "checkpoint": out / "selected_checkpoint.pt", "audit": out / "A0_REFERENCE_AUDIT.json"}
def a0_command(seed):
    p = run_paths(seed, "A0")
    return ["python", "scripts/train_controlled_v6b_minimal.py", "--data", DATA, "--architecture", "v6b_minimal", "--backbone", "mamba", "--model-name", "state-spaces/mamba-130m-hf", "--freeze-encoder", "true", "--frame-downstream-gradient-mode", "joint", "--epochs", "20", "--max-length", "128", "--dev-ratio", str(DEV_RATIO), "--seed", str(seed), "--split-seed", str(SPLIT_SEED), "--device", "cuda", "--flag-source", "controlled_heuristic", "--select-metric", "final_macro_f1", "--ranking-weight", "0.0", "--class-weighting", "none", "--stage174c-clean-pairwise-mode", "off", "--stage174c-clean-pairwise-weight", "0.0", "--stage174c-clean-polarity-preservation-weight", "0.0", "--stage175b-support-anchor-mode", "off", "--stage175b-support-anchor-weight", "0.0", "--stage177c-frame-pairwise-mode", "off", "--stage177c-frame-pairwise-weight", "0.0", "--compatible-positive-margin-logit", "0.0", "--lr", "0.001", "--reason-router-arm", "A0", "--controlled-integrity-sidecar-path", SIDECAR, "--expected-integrity-sidecar-semantic-sha256", SIDECAR_SHA_EXPECTED, "--compatible-positive-margin-weight", "0.0", "--save-selected-checkpoint", "--selected-checkpoint-filename", "selected_checkpoint.pt", "--output-json", str(p["report"]), "--output-predictions-json", str(p["prediction_json"])]
def cli_value(argv, flag):
    positions = [i for i, value in enumerate(argv) if value == flag]
    if len(positions) != 1:
        raise RuntimeError(f"Expected exactly one {flag}, found {len(positions)}")
    index = positions[0]
    if index + 1 >= len(argv):
        raise RuntimeError(f"Missing value after {flag}")
    return argv[index + 1]

EXPECTED_P2_NEUTRALIZATION = {
    "--stage174c-clean-pairwise-mode": "off",
    "--stage174c-clean-pairwise-weight": "0.0",
    "--stage174c-clean-polarity-preservation-weight": "0.0",
    "--stage175b-support-anchor-mode": "off",
    "--stage175b-support-anchor-weight": "0.0",
    "--stage177c-frame-pairwise-mode": "off",
    "--stage177c-frame-pairwise-weight": "0.0",
    "--compatible-positive-margin-logit": "0.0",
    "--compatible-positive-margin-weight": "0.0",
    "--ranking-weight": "0.0",
    "--class-weighting": "none",
}

a0_runs = [{"run_id": f"p3_seed{seed}_A0", "command_argv": a0_command(seed)} for seed in SEEDS]
for run_spec in a0_runs:
    argv = run_spec["command_argv"]
    for flag, expected in EXPECTED_P2_NEUTRALIZATION.items():
        observed = cli_value(argv, flag)
        assert observed == expected, {"run_id": run_spec["run_id"], "flag": flag, "expected": expected, "observed": observed}
    assert "--use-temporal-comparator" not in argv
    assert "--use-predicate-comparator" not in argv
print("P3_A0_P2_NEUTRALIZATION_PREFLIGHT_PASS")
for run_spec in a0_runs: run(run_spec["command_argv"])
```

```python
# Cell 5: A0 output existence check
for seed in SEEDS:
    p = run_paths(seed, "A0")
    missing = [k for k in ("report", "prediction_jsonl", "prediction_json", "checkpoint") if not p[k].exists()]
    if missing: raise RuntimeError(f"P3_A0_OUTPUT_MISSING seed={seed} missing={missing}")
print("P3_A0_OUTPUT_EXISTENCE_CHECK_PASS")
```

```python
# Cell 6: A0 SHA-256 and exact-universe audit
from scripts import train_controlled_v5 as v5
from scripts import train_controlled_v6b_minimal as trainer

def load_jsonl(path):
    with open(path, encoding="utf-8") as f: return [json.loads(line) for line in f if line.strip()]
def semantic_sidecar_sha(path): return trainer._stage187_semantic_sidecar_sha256(load_jsonl(path))
def pred_gold(row): return trainer._s28e_normalize_label(row.get("gold_label") or row.get("gold_final_label") or row.get("final_label"))
def pred_label(row): return trainer._s28e_normalize_label(row.get("pred_label") or row.get("prediction") or row.get("pred_final_label"))
def source_tuple(row): return (trainer._p2_row_identity(row), trainer._p2_reference_pair_id(row), trainer._s28e_normalize_label(row.get("final_label")))
def identity_hash(rows):
    h = hashlib.sha256()
    for row_id, pair_id, gold in rows: h.update(f"{row_id}\t{pair_id}\t{gold}\n".encode("utf-8"))
    return h.hexdigest()
def selected_epoch(report):
    candidates = [("runs.single.best_epoch", (((report.get("runs") or {}).get("single") or {}).get("best_epoch"))), ("best_epoch", report.get("best_epoch"))]
    for src, val in candidates:
        if isinstance(val, int): return val, src
    return None, None

def audit_a0(seed):
    p = run_paths(seed, "A0"); errors = []
    data_observed = sha256_file(REPO / DATA); sidecar_observed = semantic_sidecar_sha(REPO / SIDECAR)
    if data_observed != DATA_SHA_EXPECTED: errors.append("DATASET_SHA_MISMATCH")
    if sidecar_observed != SIDECAR_SHA_EXPECTED: errors.append("SIDECAR_SEMANTIC_SHA_MISMATCH")
    records = v5.load_jsonl(REPO / DATA)
    _, dev_records = v5.split_by_pair_id(records, dev_ratio=DEV_RATIO, seed=SPLIT_SEED)
    source_rows = [source_tuple(r) for r in dev_records]
    pred_rows = load_jsonl(p["prediction_jsonl"])
    pred_by_pair = {}; pred_ids = []
    for r in pred_rows:
        key = (trainer._p2_row_identity(r, r), trainer._p2_reference_pair_id(r, r)); pred_ids.append(key[0])
        if key in pred_by_pair: errors.append(f"PREDICTION_DUPLICATE_ROW_PAIR:{key}")
        pred_by_pair[key] = r
    source_ids = [x[0] for x in source_rows]; source_pairs = [(x[0], x[1]) for x in source_rows]
    if len(dev_records) != len(pred_rows): errors.append("ROW_COUNT_MISMATCH")
    if "" in source_ids: errors.append("SOURCE_ROW_ID_MISSING")
    if "" in pred_ids: errors.append("PREDICTION_ROW_ID_MISSING")
    if len(set(source_ids)) != len(source_ids): errors.append("SOURCE_DUPLICATE_ROW_ID")
    if len(set(pred_ids)) != len(pred_ids): errors.append("PREDICTION_DUPLICATE_ROW_ID")
    if set(source_ids) != set(pred_ids): errors.append("ROW_ID_SET_MISMATCH")
    if set(source_pairs) != set(pred_by_pair): errors.append("ROW_PAIR_SET_MISMATCH")
    joined = []; gold_values = []; pred_values = []
    for r in dev_records:
        key = (trainer._p2_row_identity(r), trainer._p2_reference_pair_id(r)); pr = pred_by_pair.get(key)
        if pr is None: errors.append(f"MISSING_PREDICTION_ROW:{key}"); continue
        gold = trainer._s28e_normalize_label(r.get("final_label")); pgold = pred_gold(pr); ppred = pred_label(pr)
        if pgold != gold: errors.append(f"GOLD_LABEL_MISMATCH:{key}")
        if pgold not in VALID_EXTERNAL_CLASSES: errors.append(f"UNKNOWN_EXTERNAL_GOLD:{key}")
        if ppred not in VALID_EXTERNAL_CLASSES: errors.append(f"UNKNOWN_EXTERNAL_PREDICTION:{key}")
        joined.append((key[0], key[1], pgold)); gold_values.append(gold); pred_values.append(ppred)
    report = json.load(open(p["report"], encoding="utf-8")); epoch, epoch_source = selected_epoch(report)
    authoritative_hash = identity_hash(source_rows); prediction_hash = identity_hash(joined)
    if authoritative_hash != prediction_hash: errors.append("DEV_ROW_IDENTITY_HASH_MISMATCH")
    if epoch is None: errors.append("SELECTED_EPOCH_MISSING")
    audit = {"audit_id": f"p3_seed{seed}_A0_REFERENCE_AUDIT", "run_id": f"p3_seed{seed}_A0", "seed": seed, "status": "PASS" if not errors else "P3_A0_REFERENCE_AUDIT_FAILED", "errors": errors, "execution_commit": P3_EXECUTION_CHECKOUT_COMMIT, "p2_implementation_tested_commit": P2_IMPLEMENTATION_TESTED_COMMIT, "output_dir": str(p["dir"]), "reference_prediction_path": str(p["prediction_jsonl"]), "prediction_sha256": sha256_file(p["prediction_jsonl"]), "selected_checkpoint_path": str(p["checkpoint"]), "selected_checkpoint_sha256": sha256_file(p["checkpoint"]), "report_path": str(p["report"]), "report_sha256": sha256_file(p["report"]), "selected_epoch": epoch, "selected_epoch_source": epoch_source, "row_count": len(pred_rows), "unique_row_id_count": len(set(pred_ids)), "unique_row_pair_count": len(set(pred_by_pair)), "authoritative_dev_row_count": len(dev_records), "authoritative_dev_row_identity_hash": authoritative_hash, "prediction_joined_dev_row_identity_hash": prediction_hash, "gold_counts": dict(collections.Counter(gold_values)), "prediction_counts": dict(collections.Counter(pred_values)), "a0_false_entitlement_count": sum(g == "NOT_ENTITLED" and pr in {"REFUTE", "SUPPORT"} for g, pr in zip(gold_values, pred_values)), "a0_stable_true_support_count": sum(g == "SUPPORT" and pr == "SUPPORT" for g, pr in zip(gold_values, pred_values)), "data_path": DATA, "dataset_sha256_expected": DATA_SHA_EXPECTED, "dataset_sha256_observed": data_observed, "sidecar_path": SIDECAR, "sidecar_semantic_sha256_expected": SIDECAR_SHA_EXPECTED, "sidecar_semantic_sha256_observed": sidecar_observed, "split_seed": SPLIT_SEED, "split_policy": SPLIT_POLICY, "dev_ratio": DEV_RATIO}
    for field in ["selected_epoch", "selected_epoch_source", "prediction_sha256", "selected_checkpoint_sha256", "report_sha256", "authoritative_dev_row_identity_hash", "prediction_joined_dev_row_identity_hash"]:
        if audit.get(field) is None: audit["errors"].append(f"NULL_REQUIRED_FIELD:{field}"); audit["status"] = "P3_A0_REFERENCE_AUDIT_FAILED"
    p["audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if audit["status"] != "PASS": raise RuntimeError(json.dumps(audit, indent=2))
    return audit

a0_audits = [audit_a0(seed) for seed in SEEDS]
```

```python
# Cell 7: A0 audit markdown summary
print("| seed | status | rows | dev_hash | pred_sha256 | false_entitlement | stable_true_SUPPORT |")
print("|---:|---|---:|---|---|---:|---:|")
for a in a0_audits: print(f"| {a['seed']} | {a['status']} | {a['row_count']} | `{a['authoritative_dev_row_identity_hash']}` | `{a['prediction_sha256']}` | {a['a0_false_entitlement_count']} | {a['a0_stable_true_support_count']} |")
```

```python
# Cell 8: Phase 3 dependency gate, currently blocked
def verify_a0_audit_from_disk(seed):
    expected_paths = run_paths(seed, "A0")
    manifest_declared_audit_path = RUN_ROOT / f"seed{seed}" / "A0" / "A0_REFERENCE_AUDIT.json"
    errors = []
    if expected_paths["audit"] != manifest_declared_audit_path:
        errors.append("A0_AUDIT_PATH_MISMATCH")
    audit = json.loads(expected_paths["audit"].read_text(encoding="utf-8"))
    expected_run_id = f"p3_seed{seed}_A0"
    expected_output_dir = str(expected_paths["dir"])
    expected_reference_path = str(expected_paths["prediction_jsonl"])
    if audit.get("status") != "PASS": errors.append("A0_AUDIT_NOT_PASS")
    if audit.get("execution_commit") != P3_EXECUTION_CHECKOUT_COMMIT: errors.append("A0_AUDIT_EXECUTION_COMMIT_MISMATCH")
    if audit.get("dataset_sha256_observed") != sha256_file(REPO / DATA): errors.append("A0_AUDIT_DATASET_SHA_MISMATCH")
    if audit.get("sidecar_semantic_sha256_observed") != semantic_sidecar_sha(REPO / SIDECAR): errors.append("A0_AUDIT_SIDECAR_SHA_MISMATCH")
    if audit.get("split_seed") != SPLIT_SEED: errors.append("A0_AUDIT_SPLIT_SEED_MISMATCH")
    if audit.get("authoritative_dev_row_identity_hash") != audit.get("prediction_joined_dev_row_identity_hash"): errors.append("A0_AUDIT_DEV_HASH_MISMATCH")
    if not audit.get("prediction_sha256"): errors.append("A0_AUDIT_PREDICTION_SHA_MISSING")
    if audit.get("seed") != seed: errors.append("A0_AUDIT_SEED_MISMATCH")
    if audit.get("run_id") != expected_run_id: errors.append("A0_AUDIT_RUN_ID_MISMATCH")
    if audit.get("output_dir") != expected_output_dir: errors.append("A0_AUDIT_OUTPUT_DIR_MISMATCH")
    if audit.get("reference_prediction_path") != expected_reference_path: errors.append("A0_REFERENCE_PATH_MISMATCH")
    if not expected_paths["prediction_jsonl"].exists():
        errors.append("A0_REFERENCE_FILE_MISSING")
    elif audit.get("prediction_sha256") != sha256_file(expected_paths["prediction_jsonl"]):
        errors.append("A0_REFERENCE_SHA_MISMATCH")
    if not expected_paths["checkpoint"].exists():
        errors.append("A0_SELECTED_CHECKPOINT_MISSING")
    elif audit.get("selected_checkpoint_sha256") != sha256_file(expected_paths["checkpoint"]):
        errors.append("A0_SELECTED_CHECKPOINT_SHA_MISMATCH")
    if not expected_paths["report"].exists():
        errors.append("A0_REPORT_MISSING")
    elif audit.get("report_sha256") != sha256_file(expected_paths["report"]):
        errors.append("A0_REPORT_SHA_MISMATCH")
    if errors: raise RuntimeError(f"P3_A0_REFERENCE_DEPENDENCY_GATE_FAILED seed={seed}: {errors}")
    return audit
for seed in SEEDS: verify_a0_audit_from_disk(seed)
print("P3_A1_A3_BLOCKED: UNRESOLVED_REASON_LOSS_WEIGHT")
```

```python
# Cell 9: post-unblock arm/seed collection template; do not run until Phase 3 is released
print("POST_UNBLOCK_ANALYSIS_TEMPLATE_ONLY: A1_A3 blocked until reason_loss_weight_A1_A3 is resolved")
```

```python
# Cell 10-12: post-unblock aggregate and causal comparison contract
def rate(num, den): return None if den in (0, None) else num / den
def add_rates(df):
    df["recovery_rate"] = df.apply(lambda r: rate(r["recovered_false_entitlement"], r["a0_false_entitlement_population"]), axis=1)
    df["total_support_harm_rate"] = df.apply(lambda r: rate(r["support_to_ne_harm"] + r["support_to_refute_harm"], r["a0_stable_true_support_population"]), axis=1)
    return df
def comparison_table(df, metric):
    rows = []
    for left, right in [("A1", "A0"), ("A2", "A0"), ("A3", "A1"), ("A3", "A2"), ("A3", "A0")]:
        deltas = []
        for seed in sorted(df["seed"].unique()):
            delta = df[(df.seed == seed) & (df.arm == left)][metric].iloc[0] - df[(df.seed == seed) & (df.arm == right)][metric].iloc[0]
            deltas.append(delta); rows.append({"comparison": f"{left}-{right}", "seed": seed, "metric": metric, "delta": delta})
        signs = [1 if d > 0 else -1 if d < 0 else 0 for d in deltas]
        rows.append({"comparison": f"{left}-{right}", "seed": "MEAN", "metric": metric, "delta": sum(deltas) / len(deltas), "positive_count": signs.count(1), "negative_count": signs.count(-1), "zero_count": signs.count(0), "all_seed_direction_consistent": len(set(signs)) == 1})
    return rows
print("POST_UNBLOCK_ANALYSIS_TEMPLATE_ONLY: requires A1/A2/A3 artifacts")
```

## 14. Risks and invalidation conditions

Invalidation conditions include forbidden objective activation, running A1/A3 with placeholder or fixture-derived reason-loss weight,
Seed180 first attempt failure record:

```text
seed180 first attempt:
return_code = 2
failure_stage = argparse P2 fail-fast
failure_option = stage174c_clean_polarity_preservation_weight
observed_value = 1.0
training_started = false
artifacts_authorized_as_reference = false
```

Any directory or partial file from this failed prelaunch attempt is not an immutable A0 reference. running A2 before matched Phase 3 release, wrong-seed A0 reference, trusting in-memory A0 audit state instead of audit JSON, SHA mismatch, fixed-row-count PASS, missing selected epoch, or assuming per-run ownership/checkpoint diagnostics from tests.

## 15. Final P3 decision

```text
overall_p3_decision = P3_BLOCKED_BY_UNRESOLVED_EXECUTION_CONTRACT
a0_phase_decision = P3_A0_PHASE_READY_FOR_EXECUTION
a1_a3_phase_decision = P3_A1_A3_BLOCKED
```

Additional full-causal block reason:

```text
P3_BLOCKED_BY_MISSING_EXECUTION_OBSERVABILITY
```

This specification authorizes A0 Phase 1 only. It does not authorize A1, A2, A3, aggregate causal comparison, `P3_MECHANISM_SUPPORTED`, or P3 PASS.
