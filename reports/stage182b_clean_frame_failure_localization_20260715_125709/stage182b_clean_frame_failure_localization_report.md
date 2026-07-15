# Stage182-B clean native-frame failure localization

**Decision:** `STAGE182B_COMPATIBLE_POSITIVE_MARGIN_COLLAPSE_SIGNAL`

**Artifact source mode:** `embedded_stage180a_pass2`

**Stage176 cohort source mode:** `frozen_stage182a_cohort_fallback`

## Artifact provenance

Stage180-A manifest SHA: `6ccb5b0eb8867eebcd83edbb51f30c162728ae07b804ef078ea8b44407d316a4`. Stage180-A Pass-2 packet SHA: `4fddd33bf7d9250804ab6b13bbd9deb4048daa92d48b390ae15068769e4cb82a`. Stage182-A candidate CSV SHA: `abda9d555404ff9cff62df895961f2c6d7c3055de8b27e738c582fc03a13fe2d`. Stage182-A decision: `STAGE182A_DATA_CONTAMINATION_CONFIRMED_AND_CLEAN_MODEL_FAILURE_SET_READY`. Frozen Stage176 transition SHA: `f89d555b9e2ebbaa7ae41c8fc7ef184e81cf1923cf5478c6d89376937a6f74ed`. Direct Stage176 runtime validation available: `False`.

Stage179-derived scalar values were frozen into the Stage180-A packet. Direct Stage179 runtime report provenance was unavailable. No scalar was reconstructed or recomputed. Stage176 cohort labels were frozen into Stage182-A outputs. Direct Stage176 runtime artifact was unavailable. No transition or cohort value was reconstructed from model outputs.

## Fixed clean cohorts

The analysis retained `14` Stage182-A clean model-failure candidates, `14` linked clean controls, `7` clean hard native-frame-correct references, and `30` clean control references. Every candidate is a hard row with a resolved schema, valid grammar and intervention contract, valid canonical control, and a native-frame label/prediction mismatch. Contaminated rows were excluded from every comparison.

Compatible false negatives: `13`. Incompatible false positives: `1`.

## Paired margin localization

The median candidate-minus-control frame-logit difference was `-0.555523656308651` (fixed-seed bootstrap 95% CI `[-0.7878567576408386, -0.3871966600418091]`); the two-sided exact sign-test p-value was `0.0018310546875`. These are descriptive localization statistics, not causal tests.

## Centroid and readout

Centroid/head subanalysis availability: `True`. Candidate centroid-correct rate was `0.07142857142857142` and matched-control centroid-correct rate was `0.07142857142857142`. Stage179 centroid outputs are gold-conditioned, leave-one-row-out, transductive diagnostics—not a deployable classifier or a Stage182 fitted probe.

Projection subanalysis availability: `True`. Bias-specific decomposition availability: `False`. A bias-specific conclusion is reported only when the stored projection reconstructs the native logit with an effectively constant inferred bias and maximum error at most `1e-5`.

Representation-movement subanalysis availability: `True`. Missing centroid, projection, or movement fields do not suppress scalar-margin analysis.

## Stratified evidence

Family results keep support below three descriptive only. Benjamini-Hochberg correction is applied only to family-specific sign tests. The `none`/`polarity_flip` comparison and harmful/beneficial cohort results are associations, not causal polarity-channel or training-mechanism findings. Representation movement uses only the frozen scalar magnitude and does not infer vector direction.

## Authorization and limitations

Authorized design-only route: `STAGE183_COMPATIBLE_FRAME_POSITIVE_PRESERVATION_DESIGN_AUDIT`. Training remains unauthorized. No annotation, model, checkpoint, dataset, label, generator, calibration, or threshold state was modified. Stage182-B distinguishes scalar-margin, centroid/representation, readout, and family-conditioned evidence but makes no causal mechanism claim.
