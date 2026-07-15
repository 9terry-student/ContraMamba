# Stage182-B compatible-positive margin-collapse closure

## Decision

`STAGE182B_COMPATIBLE_POSITIVE_MARGIN_COLLAPSE_SIGNAL`

Stage182-B establishes one narrow result: the frozen clean native-frame failure cohort has a scalar compatible-positive margin-collapse signal relative to matched clean controls. It does not establish a causal mechanism or authorize training.

## Closed evidence

- Clean native-frame failure candidates: 14; matched clean controls: 14.
- Native error direction: 13 compatible false negatives and 1 incompatible false positive.
- Candidate-minus-control frame-logit median: `-0.555523656308651`.
- Fixed-seed bootstrap 95% CI: `[-0.7878567576408386, -0.3871966600418091]`.
- Exact sign-test p: `0.0018310546875` (13 of 14 paired differences negative).
- Stage176 composition: 13 harmful regressions and 1 beneficial correction.
- Candidate families: `none=6`, `polarity_flip=6`, `paraphrase=1`, `location_swap=1`.

## Localization limits

The candidate and matched-control centroid-correct rates are both `1/14`. Consequently neither representation mislocalization nor readout alignment passed its gate. Projection evidence is available, but bias-specific decomposition is unavailable. Representation-movement magnitudes are available, but their direction is not. No polarity-conditioned causal claim is licensed.

The only confirmed result is the scalar compatible-positive margin-collapse signal. It must not be restated as evidence for classifier-bias dominance, a representation mechanism, or a polarity-specific cause.

## Provenance

- Scalar/diagnostic source: `embedded_stage180a_pass2`.
- Stage176 cohort source: `frozen_stage182a_cohort_fallback`.

## Authorization

Training, checkpoint forward, threshold fitting, calibration, relabeling, dataset mutation, external evaluation, and `time_swap` use remain unauthorized.

Authorized next stage:

`STAGE183_COMPATIBLE_FRAME_POSITIVE_PRESERVATION_DESIGN_AUDIT`
