# Stage176-A SUPPORT-boundary attribution closure

**Official decision:** `STAGE176A_GLOBAL_CONSERVATIVE_SHIFT_WITH_USEFUL_FRAME_MISMATCH_CORRECTION_AND_SUPPORT_COLLATERAL_DAMAGE`

Stage176-A is closed as a clean-dev-only observational attribution. The deterministic seed-174 split contains 300 pair groups (240 train, 60 dev), 720 dev rows, 29 eligible dev SUPPORT anchor pairs, and zero train/dev overlap. Both selected checkpoints are epoch 20. Baseline Stage175-B is off; treatment uses `paraphrase_margin`, weight `0.05`, tolerance `0.10`. Stage174-C is off on both sides. Final classification comes from `output["logits"]`; no external evaluation, external labels, or time-swap data entered the analysis.

## Overall transition and correctness

The baseline produced 136 SUPPORT predictions and the treatment produced 97, a delta of -39. All prediction changes were `SUPPORT -> NOT_ENTITLED` (39); there were no changes involving REFUTE and no reverse transitions into SUPPORT.

Of those 39 rows, 25 were beneficial corrections of gold NOT_ENTITLED rows and 14 were harmful regressions of gold SUPPORT rows. The remaining rows comprise 611 unchanged-correct and 70 unchanged-incorrect cases, for a net correctness change of +11.

## Attribution

| Cohort | Rows | SUPPORT -> NOT_ENTITLED | Mean margin delta | Median margin delta | Negative rows | Accuracy delta |
|---|---:|---:|---:|---:|---:|---:|
| Eligible SUPPORT paraphrase | 29 | 1 | -0.228879 | -0.260825 | 27 | -0.034483 |
| Eligible SUPPORT canonical none | 29 | 6 | -0.211070 | -0.219160 | 29 | -0.206897 |
| Other gold SUPPORT | 31 | 7 | -0.210015 | -0.208368 | 31 | -0.225806 |
| Gold NOT_ENTITLED | 540 | 25 false SUPPORT removed | -0.119234 | — | — | +0.046296 |
| Gold REFUTE | 91 | 0 | +0.059465 | — | — | 0 |

Beneficial corrections concentrate in `location_swap` (13), `role_swap` (6), `title_name_swap` (3), `entity_swap` (2), and `event_swap` (1). Harmful regressions concentrate in `none` (6), `paraphrase` (1), and `polarity_flip` (7).

The evidence supports four descriptive findings: eligible paraphrase-local degradation, canonical reference drift, untargeted SUPPORT degradation, and a global conservative boundary shift. The treatment did remove useful frame-mismatch-family false SUPPORT predictions, but it also damaged canonical and polarity-flip SUPPORT rows. Relative paraphrase preservation therefore did not prevent global boundary movement.

## Interpretation and closure

This comparison is observational attribution between two selected checkpoints, not causal proof. It does not show that threshold adjustment, weight sweeping, tolerance sweeping, more seeds, or external tuning would resolve the tradeoff.

Stage175 weight/tolerance sweeps, three-seed expansion, and external evaluation remain closed; Stage175-B remains implementation-default-off. Stage176-B may proceed solely as a native structural separability audit. Stage177 is admissible only if at least one predeclared native structural signal separates the 25 beneficial corrections from the 14 harmful regressions under the fixed gate; exact gating thresholds, if warranted, belong to a later clean-only design audit.
