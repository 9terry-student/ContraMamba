# Stage22 backbone audit notes

**Date:** 2026-06-24
**Audit script:** `scripts/audit_stage22_result_backbones.py`

---

## 1. Why the audit is needed

Stage22-A4c and A4e diagnostic sweeps were run with `backbone=dummy` to validate
plumbing: loss routing, CLI wiring, pair-contrastive encoding, schema normalization,
and metric reporting. These runs produced result JSON files containing OOD metric
fields (e.g. `frame_violation_prob` by Stage15 group) and dev metric fields
(`best_dev_metrics`, `best_dev_interventions`).

Before drawing any conclusion about whether pair-contrastive supervision improves
Stage15 OOD frame ranking on the real model, these result files must be classified
as either:
- **Plumbing validation** (dummy backbone, valid for confirming the code path works)
- **Claim-worthy evidence** (real Mamba backbone, valid for performance conclusions)

Without this audit, there is a risk of accidentally citing dummy-backbone OOD numbers
as model performance evidence in Stage22-B gate decisions or paper tables.

---

## 2. Difference between dummy smoke validation and real-backbone experiment

| Property | Dummy backbone | Real Mamba backbone |
|---|---|---|
| Representation capacity | Zero — random or constant features | Full Mamba encoder with trained weights |
| Text comprehension | None — ignores token content | Context-sensitive token representations |
| OOD metric meaning | Reflects loss weight tuning effects on random features | Reflects actual model behavior on Stage15 probe |
| Use case | Confirm CLI args, loss routing, encoding, and metric fields work | Confirm the mechanism improves OOD safety |
| Claim-worthy? | No | Yes (pending review) |

A dummy-backbone run where `pair_contrastive_frame_accuracy = 1.0` proves that the
pair-contrastive ranking loss can be satisfied by the optimizer on random features.
It does not prove that a real Mamba backbone would learn the same ranking from text.

A dummy-backbone OOD result where `fv_frame_location < fv_surface` does not mean the
real backbone will exhibit the same failure — the representation is different.
Equally, it does not mean the real backbone will succeed.

---

## 3. What must be rerun

Any result file classified as `needs_real_backbone_rerun` by the audit script must be
rerun with the real Mamba backbone before the corresponding config can be used for:
- Stage22-B gate decisions
- Paper table entries
- OOD ranking comparisons cited as model behavior

Priority rerun candidates from Stage22-A4c/A4e:
- All configs in `a4d_oodmatched_*`, `a4d_surface_*`, `a4d_temporal_*`
- All configs in `pair_all_*`, `pair_frame_*`, `pair_supportsafe_*`
- The `no_pair_frame_w0p05` baseline (used as the Stage22-A4 comparison reference)

Until real-backbone reruns are available, Stage22-A4 results should be described as:
"Under the dummy-backbone diagnostic setup, ..." not as "The model failed to ...".

---

## 4. What does not need rerun

The following artifact types are classified `implementation_only_ok` and require no
rerun regardless of when or how they were produced:

- **Prediction export files** (`*_preds.json`): raw per-record prediction exports,
  not performance summaries. Backbone type does not affect their schema validity.
- **Dataset generation outputs** (`stage22a4_pair_contrastive_*.json`,
  `stage22a4d_*.json`, etc.): these are data files, not experiment results.
- **Audit script outputs** (`stage22a4_frame_ood_alignment_*.json`): schema
  inspection artifacts, not model performance files.
- **Old stage10/12/13/16/17/18/19/20 results**: produced by earlier scripts with
  different backbone configurations; audit scope is Stage22 by default.

---

## 5. How to run the audit

```
python scripts/audit_stage22_result_backbones.py \
    --results-dir results \
    --glob "stage22*.json" \
    --output-csv results/stage22_backbone_audit.csv \
    --output-md  results/stage22_backbone_audit.md
```

To scan all stages (not just Stage22):
```
python scripts/audit_stage22_result_backbones.py \
    --results-dir results \
    --glob "*.json" \
    --output-csv results/all_stages_backbone_audit.csv \
    --output-md  results/all_stages_backbone_audit.md
```

Output files (`stage22_backbone_audit.csv` and `stage22_backbone_audit.md`) are
generated artifacts and should not be committed as authoritative records — rerun
the audit script whenever new result files are added.
