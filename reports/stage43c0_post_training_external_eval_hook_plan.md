# Stage43-C0 Post-Training External Evaluation Hook Plan

Preparation only. No training, local training, mini training, smoke test,
`py_compile`, help check, Kaggle command, evaluation, or external evaluation
was run to produce this plan.

## 1. Decision

**Decision:** `STAGE43C0_POST_TRAINING_EXTERNAL_EVAL_HOOK_PLAN_PREPARED`

## 2. Background

Stage43-B1 acquired two valid external fact-verification files:

- VitaminC validation sample1000:
  - `data/stage43b1_vitaminc_validation_sample1000.jsonl`
  - decision: `STAGE43B1_FACTVER_ACQUISITION_READY`
  - accepted rows: 1000
  - rejected rows: 0
  - label counts: SUPPORT 500, REFUTE 355, NOT_ENTITLED 145
  - interpretation: in-family fact-verification external validation

- Climate-FEVER test sample1000:
  - `data/stage43b1_climate_fever_test_sample1000.jsonl`
  - decision: `STAGE43B1_FACTVER_ACQUISITION_READY`
  - accepted rows: 903
  - rejected rows: 97
  - label counts: SUPPORT 438, REFUTE 164, NOT_ENTITLED 301
  - rejected reason: `disputed_label_excluded` 97
  - interpretation: cross-domain fact-verification external validation

Stage43-B2 standalone evaluation is scaffold-only and does not yet implement
real checkpoint inference. Prior checkpoint files are unavailable after the
runtime reset. Stage43-C0 therefore adds a post-training hook inside the
existing training runner so external evaluation can run after normal
best-state restoration.

## 3. Hook Timing

The hook runs only after:

- normal training is complete
- normal clean-dev checkpoint selection is complete
- the selected/best model state has been restored
- normal internal/dev reporting remains unchanged

Exact timing: `post_training_after_best_state_restore`.

## 4. CLI Flags

New optional flags:

- `--stage43-external-factver-jsonl`, repeatable
- `--stage43-external-output-dir`, default `reports`
- `--stage43-external-run-prefix`, default `stage43c0`
- `--stage43-external-max-rows`
- `--stage43-external-batch-size`
- `--enable-stage43-external-eval`

The hook does not run by default. Supplying JSONL paths without
`--enable-stage43-external-eval` records a skip note and does not evaluate.

## 5. Outputs

For each input JSONL:

- `{output_dir}/{run_prefix}_{dataset_stem}_external_factver_report.json`
- `{output_dir}/{run_prefix}_{dataset_stem}_external_factver_report.md`
- `{output_dir}/{run_prefix}_{dataset_stem}_external_factver_predictions.jsonl`

For multiple inputs:

- `{output_dir}/{run_prefix}_external_factver_aggregate_report.json`
- `{output_dir}/{run_prefix}_external_factver_aggregate_report.md`

## 6. Leakage Policy

External files are never used for training, calibration, checkpoint selection,
threshold selection, loss design, model selection, or composer behavior
changes.

Any failure on Climate-FEVER should be interpreted as a cross-domain
limitation, not as a training signal.

## 7. Guardrails

Stage43-C0 must not:

- use Stage43-B1 data in training
- use Stage43-B1 data in checkpoint selection
- use Stage43-B1 data in threshold selection
- use Stage43-B1 data for calibration
- modify model/composer behavior based on external results
- silently run external eval by default
- report heuristic-only results as PASS
- alter Stage39-C composer behavior
- treat Stage34/35 synthetic probes as naturalistic external validation

## 8. Recommendation

Use Stage43-C0 only as an explicit post-training eval-only hook after normal
best-state restoration. Report VitaminC as in-family external
fact-verification validation and Climate-FEVER as cross-domain
fact-verification validation.
