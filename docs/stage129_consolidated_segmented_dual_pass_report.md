# Stage129 Consolidated Segmented Dual-Pass Report

Stage129 completed the Stage126-129 report chain with:

```text
STAGE129_CONSOLIDATED_REPORT_COMPLETE
```

The validated Stage126 comparison favored `segmented_dual_pass` over `full_evidence` for the controlled evaluation chain.

| configuration | accuracy | macro_f1 | false_entitlement_total | false_NE_total | polarity_error_total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage126 `segmented_dual_pass` | 0.997742 | 0.996390 | 14 | 0 | 0 |
| Stage126 `full_evidence` | 0.869194 | 0.753022 | 256 | 402 | 153 |

## Stage126 Conclusion

Stage126 showed that `segmented_dual_pass` is context-risk safe but neutral: it removed the large `full_evidence` degradation pattern without introducing false NOT_ENTITLED or polarity errors, but it still left a small residual false-entitlement cluster.

## Stage127 Conclusion

Stage127 localized the remaining residual failures to a narrow false-entitlement cluster rather than a broad context-risk regression. The chain should preserve this distinction: Stage127 supports targeted diagnosis, not a claim that all open-world entitlement failures are solved.

## Stage128-B Guard Result

Stage128-B checked an implementation-equivalent controlled location-slot guard. The guard result was:

| metric | value |
| --- | ---: |
| guard_applied_total | 14 |
| guard_applied_gold_NOT_ENTITLED | 14 |
| false_entitlement_before | 14 |
| false_entitlement_after | 0 |
| false_NE_after | 0 |
| polarity_error_after | 0 |

The Stage128 location-slot guard is controlled diagnostic/eval-only, disabled by default, and not used in training. It is not a general open-world NER or fact-verification solution. Its intended scope is only controlled `in <Location> during` slot mismatch diagnostics, where an exported `SUPPORT` prediction may be changed to `NOT_ENTITLED` when both controlled slots are present and unequal. It must not change `REFUTE` or `NOT_ENTITLED` predictions and must not use gold labels, predicted failure buckets, `stage122_variant`, `stage121_family`, diagnostic family metadata, or any label-derived field.

## Source Availability Caveat

During Stage129 consolidation, the current Kaggle runtime reported the Stage124-C and Stage125-C source directories as missing. This report preserves the validated Stage126-128 result chain and treats those older source directories as source-unavailable/current-runtime-missing rather than fabricating source paths.
