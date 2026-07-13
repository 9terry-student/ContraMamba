# Stage176-B native structural separability closure

**Official decision:** `STAGE176B_NO_ROBUST_NATIVE_STRUCTURAL_SEPARABILITY_SIGNAL`

Stage176-B is closed as a clean-dev-only, seed-174 observational audit of the selected epoch-20 baseline and treatment checkpoints. The comparison contains 25 beneficial corrections and 14 harmful regressions. It fitted no probe, searched no threshold, performed no calibration or checkpoint selection, and used no external data, external labels, or time-swap rows.

## Separability gate

The audited native signals were `frame_prob`, `predicate_coverage_prob`, `sufficiency_prob`, `entitlement_prob`, and `polarity_margin`; the fixed predeclared composites were `structural_minimum`, `structural_product`, and `entitlement_conditioned_product`. None was reconstructed from `output["logits"]`.

The Stage177 entry gate required a direction-adjusted AUROC of at least 0.75, a bootstrap 95% lower bound strictly above 0.50, and an absolute Cliff's delta or rank-biserial correlation of at least 0.50 in a baseline or treatment native-value view. Final SUPPORT margin was excluded. Zero signals qualified. The best direction-adjusted AUROC was approximately 0.634 and the best absolute effect size was approximately 0.269.

## Structural finding

All 25 beneficial corrections have gold final label NOT_ENTITLED and gold frame label 0. Their intervention distribution is `location_swap` 13, `role_swap` 6, `title_name_swap` 3, `entity_swap` 2, and `event_swap` 1. Their mean frame probability was 0.409046 at baseline and 0.358044 at treatment, a mean change of -0.051002.

All 14 harmful regressions have gold final label SUPPORT and gold frame label 1. Their intervention distribution is `none` 6, `paraphrase` 1, and `polarity_flip` 7. Their mean frame probability was 0.397612 at baseline and 0.349719 at treatment, a mean change of -0.047893.

Thus the gold frame label perfectly separates the two cohorts, while the native frame probability does not: both cohorts occupy nearly the same frame-score range and drift downward by nearly the same amount. Predicate coverage is nearly identical, sufficiency is saturated near 0.995 on both sides, and entitlement composites and polarity margin also fail to separate the cohorts.

## Established findings and closure

The final-logit conservative shift corrected 25 frame-mismatch false SUPPORT predictions but damaged 14 valid SUPPORT predictions. No native structural score robustly distinguishes those outcomes even though the gold frame label does. The current frame head therefore fails to express the relevant gold distinction on this hard subset.

The Stage175 final-logit auxiliary route and the post-hoc native gating route are closed. No Stage175 weight sweep, tolerance sweep, three-seed expansion, native-score threshold selection, or external tuning is authorized. The next admissible step is Stage177-A: a diagnostic-only frame-head hard-subset failure audit.

This result is observational and single-seed. It does not establish representation collapse, prescribe a threshold, or authorize a new loss. Those questions belong to the Stage177-A diagnostic.
