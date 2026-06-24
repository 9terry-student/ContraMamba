# Global backbone audit notes

**Date:** 2026-06-24
**Audit script:** `scripts/audit_result_backbones.py`
**Stage22-specific audit script:** `scripts/audit_stage22_result_backbones.py`

---

## 1. Why the Stage22-only audit is insufficient

The Stage22 backbone audit (`scripts/audit_stage22_result_backbones.py`) scanned only
`results/stage22*.json` and found:

- 142 files scanned
- 69 files with `backbone=dummy`
- 69 files with unknown backbone but OOD/performance metrics (likely OOD companion JSONs)
- 0 claim_candidate files with a real backbone

This result only covers Stage22 experiments. It does not answer whether earlier
Stage12–Stage21 result files were produced with a real Mamba backbone or with a dummy
backbone. If earlier stages were also dummy-only, then no experiment in this repository
has yet produced claim-worthy model performance evidence.

---

## 2. Why a global audit is needed

A full Stage22 publication or Kaggle submission claim requires at minimum:

1. **A real-backbone baseline** (Stage12 or equivalent) to establish that the model
   learns meaningful representations from text.
2. **Real-backbone comparative results** for any mechanism introduced in Stage15–Stage22
   (temporal comparator, predicate comparator, boundary head, frame violation head, pair
   contrastive loss) to show that the mechanism improves or preserves performance on the
   real model, not just the dummy scaffold.

If all result files show `backbone=dummy`, every published number is a plumbing-validation
artifact, not a model-performance claim. The global audit surfaces this status explicitly.

---

## 3. Difference between dummy smoke validation and real-backbone experiment

| Property | Dummy backbone | Real Mamba backbone |
|---|---|---|
| Text representation | Random or constant — ignores token content | Context-sensitive Mamba encoder with trained weights |
| OOD metric meaning | Reflects loss weight and optimizer effects on random features | Reflects actual model generalization on Stage15 probe |
| DEV accuracy | Can reach high values through label-frequency shortcuts | Reflects actual classification learning |
| Claim-worthy? | No — valid for plumbing validation only | Yes — pending review |

A dummy result where `overall_accuracy = 0.85` proves that the training loop runs and
the metric computation is correct, not that the model has learned to classify text.

---

## 4. What must be rerun

Any file classified as `needs_real_backbone_rerun` must be rerun with a real Mamba
backbone before the result can be cited as model performance. This includes:

- All Stage12–Stage13 training results if backbone was `dummy`
- All Stage16–Stage21 OOD sweep / penalty / calibration results if `backbone=dummy`
- All Stage22 diagnostic results (confirmed dummy from Stage22 audit)

Priority reruns (minimum viable set for a Stage22 paper claim):
1. **Stage12 clean baseline** (`stage12_v5_clean_retrain_seed*.json`) — needed to confirm
   the model learns at all with a real backbone
2. **Stage22-A4c/A4e representative configs** — the pair-contrastive diagnostic results
   need real-backbone reruns before any Stage22 mechanism claim

---

## 5. What does not need rerun

- `*_preds.json` prediction export files (data artifacts)
- Dataset generation outputs (`stage22a4*.json`, etc.)
- Audit script outputs (`*_backbone_audit.csv`, etc.)
- Stage17–Stage20 calibration overlay files (`stage17_*.json`, `stage18_*.json`,
  `stage19_*.json`, `stage20_*.json`) if they only contain `metadata + predictions` with
  no backbone or OOD metric fields — these are post-hoc prediction adjustments, not
  experiment results requiring backbone classification

---

## 6. How to interpret inferred OOD companion backbone

Many OOD evaluation files follow the pattern:

```
{config}_ood_seed{N}.json
```

These companion files hold `group_metrics`, `ood_group_metrics`, or OOD sweep results,
but they were generated alongside a main seed file that holds the `configuration` block
with the backbone field. Because they are written separately via `--output-ood-json`,
they historically lacked the configuration metadata.

After the Stage22 provenance logging fix (`results/stage22_backbone_audit_notes.md`),
new OOD JSONs produced by `train_controlled_v6b_minimal.py` include an `ood_provenance`
block containing backbone and all key config fields. Older OOD JSONs do not have this block.

For older OOD companions without `ood_provenance`, the global audit script infers the
backbone by searching for the paired main file:
- It removes `_ood` from the companion stem to find the candidate main file name.
- Example: `stage22a4e_a4d_oodmatched_w0p05_mini_ood_seed1.json`
  → candidate: `stage22a4e_a4d_oodmatched_w0p05_mini_seed1.json`
- If the candidate exists and its backbone is extractable, the companion is classified
  using the inferred value and flagged with `inferred_from` in the audit CSV.

An inferred backbone classification is clearly marked in both the CSV (`inferred_backbone`
column, non-empty `inferred_from` column) and the Markdown summary. It should be treated
as provisional: if the paired main file is itself misclassified, the inferred backbone
will be wrong. Always verify against the Kaggle run log when in doubt.

---

## 7. How to run the global audit

```
python scripts/audit_result_backbones.py \
    --results-dir results \
    --glob "*.json" \
    --output-csv results/global_backbone_audit.csv \
    --output-md  results/global_backbone_audit.md
```

The output files are generated artifacts. Re-run the script whenever new result files
are added. Do not commit the CSV/MD as authoritative records — they will become stale.

To audit only Stage22 (backward-compatible):
```
python scripts/audit_result_backbones.py \
    --results-dir results \
    --glob "stage22*.json" \
    --output-csv results/stage22_backbone_audit_v2.csv \
    --output-md  results/stage22_backbone_audit_v2.md
```
