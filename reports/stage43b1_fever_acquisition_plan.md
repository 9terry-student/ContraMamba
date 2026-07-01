# Stage43-B1 FEVER/VitaminC HuggingFace Acquisition Plan

Preparation only. No acquisition script run, dataset download, model
training, model evaluation, external probe evaluation, Kaggle command, smoke
test, `py_compile`, help check, or local model execution was performed to
produce this plan.

## 1. Decision

**Decision:** `STAGE43B1_FEVER_ACQUISITION_PLAN_PREPARED`

## 2. Background

Stage43-A found no naturalistic external validation source inside the
repository:

- decision: `STAGE43A_EXTERNAL_VALIDATION_MANIFEST_AMBIGUOUS`
- 34 candidates scanned
- 0 clearly adaptable
- 10 ambiguous
- 24 not_adaptable
- no VitaminC/FEVER/RTE/MNLI/SNLI-style source file exists in the repo

Stage43-B0 then created a file-based converter:

- `scripts/convert_stage43_external_validation.py`
- `docs/stage43_external_validation_schema.md`
- `reports/stage43b0_external_intake_plan.md`
- `reports/stage43b0_external_intake_plan.json`

Stage43-B1 adds a FEVER/VitaminC-first HuggingFace acquisition layer using
`datasets.load_dataset`.

## 3. Source Priority

FEVER/VitaminC-style fact-verification sources are preferred over
RTE/MNLI/SNLI because they directly support
`SUPPORT`/`REFUTE`/`NOT_ENTITLED`-style claim/evidence validation.

RTE is not a primary test for `NOT_ENTITLED` because `not_entailment`
conflates contradiction with neutral or insufficient evidence. It must not be
treated as `REFUTE`.

MNLI/SNLI can be used later only as auxiliary NLI transfer diagnostics, not as
primary fact-verification validation.

## 4. Acquisition Script

`scripts/acquire_stage43_fever_external_validation.py` will pull a
user-specified HuggingFace dataset/config/split with `datasets.load_dataset`,
map FEVER/VitaminC-style rows into the Stage43 canonical external validation
JSONL schema, reject ambiguous rows, and write provenance reports.

Supported presets:

- `fever_claim_evidence`
- `vitaminc_claim_evidence`
- `auto_fever`
- `manual`

The script supports conservative evidence flattening for string, list/tuple,
dict, and structured FEVER-family evidence fields, and records the flattening
strategy in row metadata.

## 5. Label Policy

Default label mapping is FEVER/VitaminC-first:

- `SUPPORT`: `SUPPORTS`, `SUPPORT`, `supported`, `entailment`, `true`, `1`
- `REFUTE`: `REFUTES`, `REFUTE`, `refuted`, `contradiction`, `false`, `-1`
- `NOT_ENTITLED`: `NOT ENOUGH INFO`, `NEI`, `NOT_ENOUGH_INFO`, `not enough information`, `unknown`, `not_entitled`, `0`

If numeric labels are present, the acquisition script first attempts to resolve
them through HuggingFace `ClassLabel` names. Raw numeric mappings are not
blindly assumed unless an explicit `--label-map-json` override is supplied.

## 6. Stage43-B2 Gate

Stage43-B2 should run external final-composer evaluation only after a valid
FEVER/VitaminC acquisition report exists. This B1 layer does not modify model
code, training scripts, evaluator scripts, Stage33/36/37/39/40 logic, or
existing Stage39/40/41/42/43-A/43-B0 reports.

## 7. Leakage Policy

Acquired data must not be used for training, calibration, threshold selection,
checkpoint selection, loss design, or any other model-selection feedback loop.
It is external-evaluation-only.

## 8. Guardrails

Stage43-B1 must not:

- run the acquisition script during implementation
- download data during implementation
- run model training or evaluation
- create fake examples
- infer labels from model predictions
- treat RTE `not_entailment` as `REFUTE`
- treat synthetic Stage34/35 probes as naturalistic external evidence
- alter existing source reports or previous stage outputs

## 9. Recommendation

Use Stage43-B1 to acquire a genuine FEVER/VitaminC-style external validation
JSONL and provenance report through a future explicit user run. Proceed to
Stage43-B2 eval-only work only after the acquisition report indicates valid
accepted rows and the leakage policy remains intact.
