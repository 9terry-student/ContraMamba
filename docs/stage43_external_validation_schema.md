# Stage43 External Validation Schema

This document defines the canonical Stage43 external-validation intake
schema used by [`scripts/convert_stage43_external_validation.py`](../scripts/convert_stage43_external_validation.py).
It is a schema/policy document only; it does not itself contain data,
training logic, or evaluation logic.

## Background

Stage43-A ([`reports/stage43a_external_validation_manifest.json`](../reports/stage43a_external_validation_manifest.json))
scanned the repository for naturalistic external claim/evidence/label
sources and found none:

- decision: `STAGE43A_EXTERNAL_VALIDATION_MANIFEST_AMBIGUOUS`
- 34 candidate files scanned
- 0 clearly adaptable
- 10 ambiguous
- 24 not_adaptable
- no VitaminC/FEVER/RTE/MNLI/SNLI-style file exists in the repo
- all schema-plausible candidates are synthetic/controlled probes
  (Stage10/13/14/15/31/34/35, controlled_v5)

Stage43-B (external evaluation) therefore cannot honestly proceed until a
naturalistic external claim/evidence/label source is added to the
repository or supplied by the user. This Stage43-B0 layer defines the
schema and a converter so that, once such a source exists, it can be
converted into the ContraMamba schema without fabricating data and without
being used for training or calibration.

## Canonical output JSONL schema

Each row of a converted output file is a single JSON object on its own
line (`.jsonl`), with the following **required** fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable identifier for the converted row. Derived from `original_id` if present, otherwise a positional id within the conversion run. |
| `claim` | string | The claim/hypothesis text, taken verbatim from the source `--claim-field`. |
| `evidence` | string | The evidence/context text, taken verbatim from the source `--evidence-field`. |
| `label` | string | One of `SUPPORT`, `REFUTE`, `NOT_ENTITLED` (see label mapping below). |
| `source_dataset` | string | Name of the source dataset, from `--source-dataset`. |
| `source_label` | string | The original, unmapped label string as it appeared in the source file. |
| `stage43_split` | string | Defaults to `external_validation`. |
| `metadata` | object | Free-form object capturing any additional source fields not otherwise mapped (e.g. row provenance, raw source record). |

### Optional fields

These are included on a row when present/derivable from the source; they
are omitted (not set to null/empty) when not available:

- `original_id` -- the source row's own identifier field, if `--id-field` is given.
- `topic`
- `domain`
- `difficulty`
- `notes`

## Label mapping guidance

Source label strings are normalized (stripped, lowercased) and mapped to
one of the three ContraMamba labels:

| ContraMamba label | Accepted source values (case-insensitive) |
|---|---|
| `SUPPORT` | `SUPPORTS`, `SUPPORT`, `entailment`, `true`, `1` |
| `REFUTE` | `REFUTES`, `REFUTE`, `contradiction`, `false`, `-1` |
| `NOT_ENTITLED` | `NOT ENOUGH INFO`, `NEI`, `neutral`, `unknown`, `not_entitled`, `0` |

If `--allow-neutral-as-not-entitled` is enabled (default: true), `neutral`
maps to `NOT_ENTITLED` as shown above. If `--strict-labels` is set, only
the exact accepted values above are mapped; any other value is rejected
rather than guessed at.

Any source label value not in the tables above is **ambiguous** and the
row is written to the rejected-rows file, not the converted output.

## Strict rules

- Do not invent missing evidence. If a source row has no evidence text,
  reject the row -- never synthesize or backfill evidence.
- Do not infer gold labels from model predictions. The converter only
  reads the label field the caller specifies; it never runs a model or
  substitutes a predicted label for a missing gold label.
- Do not use Stage34/35 probes as naturalistic external validation. Those
  are synthetic/controlled artifacts (see Stage43-A manifest) and are out
  of scope for this converter's intended inputs.
- Do not use converted Stage43 files for training, calibration, threshold
  selection, checkpoint selection, or loss design. Converted output is
  external-evaluation-only.
- If a mapping is ambiguous (missing claim, missing evidence, missing
  label, or unmapped label value), the row is written to the rejected
  file together with a rejection reason -- it is never written to the
  accepted output.

## Leakage policy

Converted files produced by this schema/converter are for external
evaluation only. They must never be used to update model weights, select
checkpoints, tune thresholds, or otherwise influence training or
calibration decisions. See [`reports/stage43b0_external_intake_plan.md`](../reports/stage43b0_external_intake_plan.md)
for the full intake plan and preferred source order.
