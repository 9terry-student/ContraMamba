# Stage45-C Internal SUPPORT Entitlement Recovery Report

## Enabled

- Enabled: True
- Support recovery weight: 0.1
- Entitled NE penalty weight: 0.1
- Target label: `SUPPORT`
- Entitled labels: ['SUPPORT', 'REFUTE']

## Internal Training Label Counts

- SUPPORT: 356
- REFUTE: 364
- NOT_ENTITLED: 2160

## Loss Terms

- Active loss terms: ['support_recovery', 'entitled_ne_penalty']
- SUPPORT recovery loss mean: 0.0033116755075752735
- Entitled NE penalty loss mean: 0.001808105269446969

## Leakage Policy

{'scope': 'internal_controlled_training_split_only', 'stage43b1_files_read': False, 'external_examples_used': False, 'external_labels_or_metrics_used': False, 'vitaminc_used': False, 'climate_fever_used': False, 'used_for_threshold_selection': False, 'used_for_calibration': False, 'used_for_checkpoint_selection': False, 'used_dev_or_holdout_labels_in_loss': False}

## Recommendation

Stage45-C is an internal-only auxiliary diagnostic/training scaffold targeting SUPPORT under-recall and entitled-to-NOT_ENTITLED over-rejection observed in Stage45-B1 internal family holdouts. Treat any resulting improvement as an internal robustness signal only; it does not constitute external validation and must not be claimed as VitaminC/Climate-FEVER transfer success or naturalistic generalization without a new, separate held-out external evaluation.
