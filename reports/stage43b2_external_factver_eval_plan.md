# Stage43-B2 External Fact-Verification Evaluation Plan

Preparation only. No training, tuning, smoke test, `py_compile`, help check,
Kaggle command, external evaluation, or model evaluation was run to produce
this plan.

## 1. Decision

**Decision:** `STAGE43B2_EXTERNAL_FACTVER_EVAL_PLAN_PREPARED`

## 2. Stage43-B1 Inputs

Stage43-B1 produced two valid external fact-verification sources:

- VitaminC validation sample1000:
  - path: `data/stage43b1_vitaminc_validation_sample1000.jsonl`
  - decision: `STAGE43B1_FACTVER_ACQUISITION_READY`
  - accepted rows: 1000
  - rejected rows: 0
  - label counts: SUPPORT 500, REFUTE 355, NOT_ENTITLED 145
  - interpretation: in-family fact-verification external validation

- Climate-FEVER test sample1000:
  - path: `data/stage43b1_climate_fever_test_sample1000.jsonl`
  - decision: `STAGE43B1_FACTVER_ACQUISITION_READY`
  - accepted rows: 903
  - rejected rows: 97
  - label counts: SUPPORT 438, REFUTE 164, NOT_ENTITLED 301
  - rejected reason: `disputed_label_excluded` 97
  - interpretation: cross-domain fact-verification external validation

## 3. Evaluation Script

`scripts/evaluate_stage43_external_factver.py` evaluates Stage39-C
`safe_structured_v2` on Stage43-B1 JSONL files without training or tuning.
The script reads rows with `claim`, `evidence`, and gold `label`, obtains
base predictions from an existing model/export path when available, applies
the opt-in final-composer path only when requested, and writes JSON/Markdown
reports.

Required inputs:

- `--input-jsonl`
- `--output-json`
- `--output-md`
- `--run-name`

Optional inputs include `--max-rows`, `--composer-mode`,
`--prediction-source`, `--checkpoint`, `--device`, `--batch-size`,
`--max-length`, `--write-predictions-jsonl`, and `--allow-heuristic-only`.

## 4. Metrics And Safety Counters

The evaluator reports base and composed accuracy, macro-F1, per-label
precision/recall/F1, confusion matrices, label distributions, prediction
distributions, and base-to-composed deltas.

Safety counters include changed rows, change destinations, introduced unsafe
SUPPORT, introduced REFUTE-to-SUPPORT, introduced SUPPORT-to-REFUTE, total
wrong SUPPORT after composition, total SUPPORT-to-REFUTE after composition,
and Stage36 blocker / Stage37 recovery counts when available.

## 5. Decision Labels

- `STAGE43B2_EXTERNAL_FACTVER_PASS`
- `STAGE43B2_EXTERNAL_FACTVER_SAFE_BUT_NO_GAIN`
- `STAGE43B2_EXTERNAL_FACTVER_UNSAFE`
- `STAGE43B2_EXTERNAL_FACTVER_INCOMPLETE`

Heuristic-only output must always be `STAGE43B2_EXTERNAL_FACTVER_INCOMPLETE`
and must be described as diagnostic-only, not model evaluation.

## 6. Leakage Policy

The acquired data must remain external-evaluation-only. No threshold, model,
checkpoint, loss, composer behavior, or training decision may be selected
using Stage43-B2 outcomes.

Any failure on Climate-FEVER should be interpreted as a cross-domain
limitation, not as a training signal.

## 7. Guardrails

Stage43-B2 must not:

- train
- tune thresholds
- select checkpoints
- modify model behavior based on Stage43-B2
- use Stage43-B1 data for training or calibration
- silently fall back to heuristic-only model evaluation
- mark heuristic-only output as PASS
- alter Stage39-C behavior
- treat synthetic Stage34/35 as naturalistic external evidence

## 8. Recommendation

Run Stage43-B2 only as eval-only after a real model/export prediction path is
available for the Stage43-B1 JSONL files. Treat VitaminC as in-family
external fact-verification validation and Climate-FEVER as cross-domain
fact-verification validation.
