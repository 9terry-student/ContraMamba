# Stage43-B1 Targeted Fact-Verification Acquisition Plan

Preparation only. No acquisition script was run, no dataset was downloaded,
and no model training, model evaluation, external probe evaluation, Kaggle
command, smoke test, `py_compile`, help check, or local model execution was
performed to produce this plan.

## 1. Decision

**Decision:** `STAGE43B1_FACTVER_ACQUISITION_PLAN_PREPARED`

## 2. Background

Stage43-A found no naturalistic external validation source inside the
ContraMamba repository:

- decision: `STAGE43A_EXTERNAL_VALIDATION_MANIFEST_AMBIGUOUS`
- 34 candidates scanned
- 0 clearly adaptable
- 10 ambiguous
- 24 not_adaptable

Stage43-B0 created:

- `scripts/convert_stage43_external_validation.py`
- `docs/stage43_external_validation_schema.md`
- `reports/stage43b0_external_intake_plan.md`
- `reports/stage43b0_external_intake_plan.json`

## 3. EpistemicBERT Prior

Uploaded EpistemicBERT code confirms `tals/vitaminc` as the original main
fact-verification dataset loaded with `load_dataset("tals/vitaminc")`.
The confirmed main fields are `claim`, `evidence`, and `label`, with labels
`SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO`.

The uploaded EpistemicBERT code also confirms `climate_fever` as a prior
cross-domain fact-verification transfer dataset. For Climate-FEVER, claim is
read from `claim`, label from `claim_label` or `label`, and evidence from
`evidences` or `evidence`. `DISPUTED` is excluded for clean 3-way evaluation
and must not be mapped to `NOT_ENTITLED`.

## 4. Source Priority

VitaminC and Climate-FEVER are preferred over RTE/MNLI/SNLI because they
directly support `SUPPORT`/`REFUTE`/`NOT_ENTITLED`-style fact verification.

RTE is not a primary test because `not_entailment` conflates contradiction
and insufficient evidence.

MNLI/SNLI can be used later only as auxiliary NLI transfer diagnostics.

## 5. Acquisition Script

`scripts/acquire_stage43_factver_external_validation.py` is a targeted
HuggingFace acquisition script for:

- `vitaminc`, loading `tals/vitaminc`
- `climate_fever`, loading `climate_fever`

It converts rows into the Stage43 canonical external-validation JSONL schema,
writes rejected rows with reasons, and writes JSON/Markdown provenance
reports. It does not run ContraMamba evaluation.

## 6. Label And Evidence Policy

VitaminC label mapping:

- `SUPPORTS` -> `SUPPORT`
- `REFUTES` -> `REFUTE`
- `NOT ENOUGH INFO`, `NOT_ENOUGH_INFO`, `NEI` -> `NOT_ENTITLED`
- numeric labels first resolve through `dataset.features["label"].names`
- if names are unavailable, the EpistemicBERT-confirmed VitaminC fallback is
  `0 -> SUPPORT`, `1 -> REFUTE`, `2 -> NOT_ENTITLED`, and fallback use is
  recorded

Climate-FEVER label mapping:

- `SUPPORTS` or numeric `0` -> `SUPPORT`
- `REFUTES` or numeric `1` -> `REFUTE`
- `NOT_ENOUGH_INFO`, `NOT ENOUGH INFO`, `NEI`, or numeric `2` -> `NOT_ENTITLED`
- `DISPUTED` or numeric `3` -> reject with `disputed_label_excluded`

Evidence flattening supports strings, lists, dictionaries, and nested
structures. Climate-FEVER defaults to `--evidence-mode first`, with
`join_all` available for joining all flattened evidence segments.

## 7. Stage43-B2 Gate

Stage43-B2 should run ContraMamba external final-composer evaluation only
after valid VitaminC and/or Climate-FEVER acquisition reports exist.

## 8. Leakage Policy

Acquired data must not be used for training, calibration, threshold
selection, checkpoint selection, loss design, or any other model-selection
feedback loop. It is external-evaluation-only.

## 9. Guardrails

Stage43-B1 must not:

- run the acquisition script during implementation
- download data during implementation
- run model training or evaluation
- create fake examples
- infer labels from model predictions
- treat RTE/MNLI/SNLI as primary Stage43 external validation
- treat synthetic Stage34/35 as naturalistic external evidence
- alter existing source reports or previous stage outputs

## 10. Recommendation

Use the targeted Stage43-B1 script only when the user explicitly runs it
later. Proceed to Stage43-B2 eval-only work only after valid acquisition
reports exist and the leakage policy remains intact.
