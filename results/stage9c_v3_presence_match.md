# Stage 9C Presence-vs-Match Gate Diagnostics

## Why Stage 9C was needed

Stage 9B thresholded downgrades cannot distinguish no-opportunity cases, no-signal gates, threshold artifacts, and confidently inverted gates. Stage 9C analyzes gate probabilities before routing and preserves the expected score direction.

## Main mechanism question: evidence presence or claim-evidence match?

The central contrast is whether gate probabilities fall only when evidence is absent or truncated, or also when evidence remains present but mismatches the claim.

## Evidence-presence diagnostic

| intervention | gold NOT_ENTITLED | pred NOT_ENTITLED | classifier error | mean sufficiency | mean entitlement | sufficiency state | entitlement state |
|---|---:|---:|---:|---:|---:|---|---|
| evidence_deletion | 60.0000 +/- 0.0000 | 60.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0006 +/- 0.0001 | 0.0005 +/- 0.0001 | no_opportunity:3 | no_opportunity:3 |
| evidence_truncation | 60.0000 +/- 0.0000 | 60.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0007 +/- 0.0005 | 0.0007 +/- 0.0004 | no_opportunity:3 | no_opportunity:3 |
| time_swap | 60.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 60.0000 +/- 0.0000 | 0.9902 +/- 0.0011 | 0.9090 +/- 0.0090 | confidently_inverted:3 | confidently_inverted:3 |

Low sufficiency/entitlement probabilities for deletion or truncation indicate evidence-presence sensitivity even when the classifier already predicts NOT_ENTITLED and no downgrade opportunity remains. High probabilities for a mismatched but present-evidence condition indicate a match-detection failure.

## Confidently inverted diagnostic

| intervention | false entitled | frame | predicate | sufficiency | entitlement | frame state | predicate state | sufficiency state | entitlement state |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| time_swap | 60.0000 +/- 0.0000 | 0.9243 +/- 0.0073 | 0.9925 +/- 0.0037 | 0.9902 +/- 0.0011 | 0.9090 +/- 0.0090 | confidently_inverted:3 | confidently_inverted:3 | confidently_inverted:3 | confidently_inverted:3 |
| entity_swap | 1.3333 +/- 2.3094 | 0.0173 +/- 0.0211 | 0.0324 +/- 0.0395 | 0.9992 +/- 0.0003 | 0.0124 +/- 0.0213 | correct_rejection:1;no_opportunity:2 | correct_rejection:1;no_opportunity:2 | confidently_inverted:1;no_opportunity:2 | correct_rejection:1;no_opportunity:2 |
| event_swap | 4.0000 +/- 2.6458 | 0.0509 +/- 0.0289 | 0.0681 +/- 0.0379 | 0.9990 +/- 0.0003 | 0.0412 +/- 0.0256 | correct_rejection:3 | correct_rejection:3 | confidently_inverted:3 | correct_rejection:3 |
| location_swap | 4.6667 +/- 2.0817 | 0.0597 +/- 0.0332 | 0.9656 +/- 0.0075 | 0.9963 +/- 0.0005 | 0.0577 +/- 0.0321 | correct_rejection:3 | confidently_inverted:3 | confidently_inverted:3 | correct_rejection:3 |
| role_swap | 3.6667 +/- 1.5275 | 0.0600 +/- 0.0220 | 0.9756 +/- 0.0140 | 0.9965 +/- 0.0004 | 0.0524 +/- 0.0167 | correct_rejection:3 | confidently_inverted:3 | confidently_inverted:3 | correct_rejection:3 |
| title_name_swap | 0.3333 +/- 0.5774 | 0.0083 +/- 0.0082 | 0.0206 +/- 0.0325 | 0.9991 +/- 0.0002 | 0.0044 +/- 0.0077 | correct_rejection:1;no_opportunity:2 | correct_rejection:1;no_opportunity:2 | confidently_inverted:1;no_opportunity:2 | correct_rejection:1;no_opportunity:2 |
| predicate_swap | 0.6667 +/- 0.5774 | 0.9361 +/- 0.0040 | 0.0181 +/- 0.0085 | 0.9986 +/- 0.0005 | 0.0158 +/- 0.0071 | confidently_inverted:2;no_opportunity:1 | correct_rejection:2;no_opportunity:1 | confidently_inverted:2;no_opportunity:1 | correct_rejection:2;no_opportunity:1 |

A confidently inverted state means gold NOT_ENTITLED cases receive high pass probabilities. If replicated across seeds, time_swap is the sharpest candidate for an auditor that detects evidence presence but endorses a mismatched claim-evidence relation.

Expected-direction signal for time_swap false-entitled cases:

| score | raw AUC | inverted AUC | mean pass probability on positives |
|---|---:|---:|---:|
| frame_fail_score | 0.5000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.9243 +/- 0.0073 |
| predicate_fail_score | 0.5000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.9925 +/- 0.0037 |
| sufficiency_fail_score | 0.5000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.9902 +/- 0.0011 |
| entitlement_fail_score | 0.5000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.9090 +/- 0.0090 |
| polarity_weak_score | 0.5000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |

## Polarity diagnostic

For polarity_flip, gold SUPPORT=29.6667 +/- 0.5774, gold REFUTE=30.3333 +/- 0.5774, pred SUPPORT=29.6667 +/- 0.5774, pred REFUTE=30.3333 +/- 0.5774, and classifier error=0.0000 +/- 0.0000. Polarity flips are entitled SUPPORT/REFUTE distinctions when gold NOT_ENTITLED is zero; they are not downgrade targets by default.

## Gate-correlation and independence diagnostic

Across all examples, mean absolute off-diagonal Pearson correlation is 0.3538 +/- 0.0056 and the maximum is 0.5910 +/- 0.0034. Mean absolute Spearman correlation is 0.4144 +/- 0.0196 and the maximum is 0.7744 +/- 0.0168.

Global pairwise gate correlations:

| pair | Pearson | Spearman |
|---|---:|---:|
| frame_prob / predicate_coverage_prob | 0.5620 +/- 0.0041 | 0.6292 +/- 0.0563 |
| frame_prob / sufficiency_prob | -0.1619 +/- 0.0087 | -0.5915 +/- 0.0654 |
| frame_prob / entitlement_prob | 0.5910 +/- 0.0034 | 0.5645 +/- 0.0166 |
| frame_prob / polarity_margin | -0.3818 +/- 0.0045 | -0.2074 +/- 0.0149 |
| predicate_coverage_prob / sufficiency_prob | -0.0582 +/- 0.0096 | -0.4620 +/- 0.0340 |
| predicate_coverage_prob / entitlement_prob | 0.5441 +/- 0.0052 | 0.7744 +/- 0.0168 |
| predicate_coverage_prob / polarity_margin | -0.2775 +/- 0.0309 | -0.0435 +/- 0.0583 |
| sufficiency_prob / entitlement_prob | 0.3748 +/- 0.0113 | -0.1911 +/- 0.0043 |
| sufficiency_prob / polarity_margin | 0.5587 +/- 0.0342 | 0.4164 +/- 0.0094 |
| entitlement_prob / polarity_margin | -0.0278 +/- 0.0149 | 0.2619 +/- 0.0149 |

High correlations would argue against describing the named heads as four independent mechanisms; they should instead be described as correlated gate heads over a shared representation.

## Paper implication

If the deletion/truncation versus time-swap contrast and expected-direction diagnostics replicate, the appropriate mechanism-level wording is: Structured entitlement heads detect evidence presence but fail to reliably detect claim-evidence match, with time_swap exposing a confidently inverted entitlement judgment.

This remains a controlled diagnostic. It does not establish the root cause of temporal failure, independent gate mechanisms, broad entitlement checking, or real-world factuality performance.
