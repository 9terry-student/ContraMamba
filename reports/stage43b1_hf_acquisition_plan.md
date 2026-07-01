# Stage43-B1 HuggingFace External Validation Acquisition Plan

Preparation only. No model training, evaluation, external probe evaluation,
Kaggle command, py_compile, smoke test, help check, or local model execution
was performed to produce this plan. The acquisition script described below
was not executed while writing this plan.

## 1. Decision

**Decision:** `STAGE43B1_HF_ACQUISITION_PLAN_PREPARED`

## 2. Background

Stage43-A ([`reports/stage43a_external_validation_manifest.json`](stage43a_external_validation_manifest.json),
[`reports/stage43a_external_validation_manifest.md`](stage43a_external_validation_manifest.md))
found no clearly adaptable naturalistic external validation source inside
the repository:

- decision: `STAGE43A_EXTERNAL_VALIDATION_MANIFEST_AMBIGUOUS`
- 34 candidate files scanned
- 0 clearly adaptable
- 10 ambiguous
- 24 not_adaptable
- no VitaminC/FEVER/RTE/MNLI/SNLI-style source file exists in the repo

Stage43-B0 responded to that gap by building a **file-based** intake
converter, [`scripts/convert_stage43_external_validation.py`](../scripts/convert_stage43_external_validation.py),
plus [`docs/stage43_external_validation_schema.md`](../docs/stage43_external_validation_schema.md)
and [`reports/stage43b0_external_intake_plan.md`](stage43b0_external_intake_plan.md) /
[`reports/stage43b0_external_intake_plan.json`](stage43b0_external_intake_plan.json).
That converter can only run once a naturalistic source file already exists
locally; it does not download anything.

The user clarified that prior EpistemicBERT/ContraMamba datasets were
usually pulled inside Kaggle via the standard HuggingFace pattern:

```python
from datasets import load_dataset
dataset = load_dataset(...)
```

Stage43-B1 closes this gap by adding a **reproducible HuggingFace
acquisition** script that pulls a user-specified dataset/config/split via
`datasets.load_dataset`, inspects its fields/labels, maps them into the
Stage43 canonical schema, and writes provenance reports -- without running
any model training, calibration, or evaluation.

## 3. Preferred Acquisition Targets

1. **VitaminC or FEVER-style fact verification**, if available on the Hub --
   the closest match to ContraMamba's SUPPORT/REFUTE/NOT_ENTITLED schema
   and to the fact-verification framing used elsewhere in this repo.
2. **RTE/GLUE**, as a weaker NLI transfer probe -- usable, but RTE's binary
   `entailment` / `not_entailment` labeling does not separate REFUTE from
   NOT_ENTITLED, so `not_entailment` is mapped to `NOT_ENTITLED` by default
   (never to `REFUTE`) unless a manual `--label-map-json` says otherwise.
3. **MNLI/SNLI**, as a broader three-way NLI source -- useful for volume,
   but its `entailment`/`neutral`/`contradiction` labels are less
   fact-verification-specific than VitaminC/FEVER, so results from this
   tier should be treated as a secondary signal.

## 4. Acquisition Script Scope

[`scripts/acquire_stage43_hf_external_validation.py`](../scripts/acquire_stage43_hf_external_validation.py)
implements:

- **Loading**: `from datasets import load_dataset`; calls
  `load_dataset(hf_dataset, hf_config, split=split, trust_remote_code=...)`
  when `--hf-config` is given, else
  `load_dataset(hf_dataset, split=split, trust_remote_code=...)`.
  `trust_remote_code` is only passed through when `--trust-remote-code` is
  explicitly supplied. The script is generic -- it does not hardcode any
  single dataset name.
- **Field detection**: explicit `--claim-field`/`--evidence-field`/
  `--label-field`/`--id-field` take precedence; `--auto-detect-fields`
  infers a mapping from alias lists (claim: `claim`, `hypothesis`,
  `statement`, `query`, `sentence2`, `premise2`; evidence: `evidence`,
  `premise`, `context`, `passage`, `text`, `sentence1`; label: `label`,
  `gold`, `gold_label`, `answer`, `verdict`, `relation`). If no confident
  mapping can be found, the script fails with
  `STAGE43B1_HF_EXTERNAL_ACQUISITION_NO_VALID_ROWS` and writes zero
  accepted rows rather than guessing.
- **ClassLabel resolution**: integer labels are resolved to string names
  via the dataset's `features[label_field].names` when available, before
  label-string mapping is applied.
- **Label mapping**: default mapping (see
  [`docs/stage43_external_validation_schema.md`](../docs/stage43_external_validation_schema.md))
  extended with an explicit `not_entailment -> NOT_ENTITLED` rule and a
  recorded risk note whenever that rule fires. `--label-map-json` (inline
  JSON or a file path) overrides the defaults entirely when supplied.
- **Rejection rules**: missing/empty claim or evidence, missing or
  unmapped label, text shorter than `--min-text-chars`, and (with
  `--dedupe`) duplicate claim/evidence/label triples. Rejected rows are
  written to `--rejected-jsonl` with row index, rejection reason, source
  label, and a compact original record.
- **Output schema**: each accepted row matches the canonical Stage43
  schema (`id`, `claim`, `evidence`, `label`, `source_dataset`,
  `source_label`, `stage43_split="external_validation"`, `metadata`, plus
  optional `original_id`, `source_config`, `source_split`).
- **`--dry-run`**: inspects and reports without writing the accepted
  output JSONL (the rejected JSONL and reports are still written so field/
  label mapping issues are visible before committing to a real pull).

## 5. Stage43-B2 Plan (Future, Not Executed Here)

Once a Stage43-B1 acquisition run produces a report with decision
`STAGE43B1_HF_EXTERNAL_ACQUISITION_READY` (or an acceptable
`..._LABEL_IMBALANCED` result the operator chooses to proceed with),
Stage43-B2 should run the existing external final-composer evaluation on
the acquired JSONL -- eval-only, without modifying model or training code.
Stage43-B2 must not:

- modify model code
- modify training scripts
- modify evaluator scripts
- modify Stage33/36/37/39/40 logic
- modify existing Stage39/40/41/42/43-A/43-B0 reports
- modify existing data/probe files
- use the acquired data for training, calibration, threshold selection,
  checkpoint selection, or loss design

## 6. Guardrails

Stage43-B1 (this preparation layer) and any future acquisition run must
not:

- download data at implementation time (this plan and script were written
  without invoking `load_dataset`)
- run the acquisition script as part of this task
- run model training or evaluation
- create fake examples
- infer labels from model predictions
- treat RTE `not_entailment` as `REFUTE` unless a manual label map says so
- treat synthetic Stage34/35 probes as naturalistic external evidence
- alter existing source reports or previous stage outputs

## 7. Leakage Policy

Any JSONL produced by `scripts/acquire_stage43_hf_external_validation.py`
is external-evaluation-only. It must never be used to update model
weights, select checkpoints, tune thresholds, or otherwise influence
training or calibration decisions. See
[`docs/stage43_external_validation_schema.md`](../docs/stage43_external_validation_schema.md)
for the full schema and label-mapping policy.

## 8. Recommendation

Run `scripts/acquire_stage43_hf_external_validation.py` explicitly (not as
part of this task) against a VitaminC/FEVER-style dataset first; fall back
to RTE/GLUE or MNLI/SNLI only if no fact-verification-specific dataset is
accessible. Review the resulting JSON/Markdown report -- especially
`risks`, `rejection_reason_counts`, and `accepted_label_counts` -- before
treating the output as ready for Stage43-B2 evaluation. No model,
training, or evaluator code was modified to produce this plan or the
acquisition script.

## 9. Artifacts

- Acquisition script: [`scripts/acquire_stage43_hf_external_validation.py`](../scripts/acquire_stage43_hf_external_validation.py)
- Prior converter (file-based): [`scripts/convert_stage43_external_validation.py`](../scripts/convert_stage43_external_validation.py)
- Schema document: [`docs/stage43_external_validation_schema.md`](../docs/stage43_external_validation_schema.md)
- This plan (Markdown): `reports/stage43b1_hf_acquisition_plan.md`
- This plan (JSON): `reports/stage43b1_hf_acquisition_plan.json`
