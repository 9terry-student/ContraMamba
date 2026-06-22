# Stage 5 Results: Final-Label Accuracy and Evidence-Entitlement Faithfulness

## Abstract-style summary

Stage 5 evaluates whether correct three-way predictions imply that ContraMamba's internal evidence-entitlement signals support those predictions. They do not necessarily do so. On a controlled intervention dataset, the strongest plain classifier reached a macro-F1 of 0.870 ± 0.006, yet was substantially less consistent under predicate and polarity interventions than intervention-aware variants. A conservative classifier-auditor router improved macro-F1 to 0.878 ± 0.009 while reducing entitled-output gate violations from 0.222 ± 0.066 to 0.000 ± 0.000 and eliminating the measured gap between output-level and internally faithful polarity reversals. These results provide controlled evidence that final-label correctness and internal evidence-entitlement faithfulness are empirically distinct. They do not yet establish improvements in open-domain hallucination, retrieval-augmented generation, or deployed language-model behavior.

## Experimental setup

Experiments use `controlled_v5_v2`, which contains 100 source pair groups with 13 records per group: an unmodified example and 12 controlled interventions. Splits are made by `pair_id`, preventing an original example and its interventions from crossing train and development partitions. Results are aggregated over three seeds and reported as mean ± sample standard deviation.

The model predicts `REFUTE`, `NOT_ENTITLED`, or `SUPPORT` and exposes frame compatibility, predicate coverage, sufficiency, entitlement, and polarity signals. Stage 5B compares single-model loss ablations. Stage 5C combines the strongest plain classifier with intervention-aware auditors. A conservative router retains an entitled classifier output only when the auditor's frame, predicate, sufficiency, and entitlement values all meet the 0.5 threshold; otherwise it returns `NOT_ENTITLED`.

We report three complementary evaluation families. Final-label macro-F1 measures classification quality. Pairwise intervention checks measure whether outputs change or remain stable under the intended controlled transformation. Internal-faithfulness checks measure whether an entitled output is supported by all gates and whether polarity-flip outputs agree with both gate states and the sign of the internal polarity margin.

## Stage 5B: single-model ablation

The no-intervention configuration was the strongest plain final-label classifier, with macro-F1 0.870 ± 0.006. Its intervention consistency was weaker, however: predicate-pair consistency was 0.750 ± 0.100 and polarity-flip consistency was 0.217 ± 0.202. Thus, high classification performance did not imply that the model responded correctly to targeted changes in predicate or polarity.

The full 4E objective showed the opposite profile. Its macro-F1 was lower at 0.820 ± 0.014, while predicate-pair and polarity-flip consistency increased to 0.950 ± 0.050 and 0.883 ± 0.076, respectively. The no-polarity-flip ablation offered the best single-model balance: macro-F1 0.850 ± 0.017, predicate-pair consistency 0.917 ± 0.076, and polarity-flip consistency 0.867 ± 0.058. Removing predicate contrast reduced predicate-pair consistency from 0.950 ± 0.050 to 0.817 ± 0.144, supporting the intended role of that loss within this controlled setting.

Together, the ablations reveal a trade-off rather than a uniformly dominant objective. Direct final-label supervision favors classification, whereas intervention-aware supervision more strongly constrains how predictions behave across controlled counterfactual pairs.

## Stage 5C: classifier-auditor router

Stage 5C tests whether the classification and consistency strengths can be combined without changing the underlying model architecture. The classifier-only system reproduced the no-intervention macro-F1 of 0.870 ± 0.006. Nevertheless, 0.222 ± 0.066 of its entitled outputs violated at least one internal gate, and its polarity output/internal gap was 0.250 ± 0.150. In other words, some output pairs exhibited the expected support/refute reversal even though the corresponding gates or polarity-margin signs did not internally justify that behavior.

The balanced-only auditor had the strongest output-level pairwise consistency: paraphrase preservation was 0.933 ± 0.029, predicate disentanglement was 0.917 ± 0.029, and polarity-flip consistency was 0.967 ± 0.029. Its macro-F1 was lower, at 0.850 ± 0.017. This again shows that pairwise consistency and aggregate label quality measure different properties.

The conservative balanced router achieved the strongest combined result. Its macro-F1 was 0.878 ± 0.009, slightly above classifier-only, while its entitled-output gate violation rate and polarity output/internal gap were both 0.000 ± 0.000. The conservative strict and dual-auditor routers likewise reduced gate violations and output/internal gaps to zero, although their macro-F1 values were 0.864 ± 0.011 and 0.871 ± 0.010. Across the three seeds, conservative routing therefore enforced the stated gate criterion while preserving competitive—and for the balanced router, modestly improved—final-label performance.

## Main empirical finding

The central result is that final-label correctness and internal evidence-entitlement faithfulness diverge under controlled interventions. Output-level success can conceal internal inconsistency: classifier-only produced strong macro-F1 and high output-level polarity-flip consistency, but its gate-violation rate and output/internal polarity gap remained nonzero. Conversely, intervention-aware models improved controlled consistency while sometimes reducing aggregate classification performance.

The router results show that this divergence is actionable. Treating intervention-aware gates as an auditor of a stronger classifier removed measured internal violations without requiring the auditor to replace the classifier's label prediction in every case. This is evidence for separating classification from entitlement auditing in the controlled Stage 5 setting, not evidence that the same gains will automatically transfer to unrestricted evidence or generated text.

## Failure-mode interpretation

The plain classifier's principal failure is not simply an incorrect final label. It can emit an entitled label despite a low frame, predicate, sufficiency, or entitlement score, or produce an output-level polarity reversal whose internal margin has the wrong sign. These cases are invisible to conventional accuracy and macro-F1.

Intervention-aware models exhibit a different failure mode. Their gate structure and pairwise behavior are more coherent, but conservative decisions can downgrade examples that the classifier would label correctly, reducing recall for entitled classes. The balanced-only model's high pairwise scores and lower macro-F1 illustrate this tension. Router performance depends on gate calibration and the fixed 0.5 threshold; a zero violation rate is partly guaranteed by the routing rule and should be interpreted together with final-label metrics, prediction distributions, and entitled-output counts.

## Paper table plan

The main paper should use two tables.

**Table 1: Stage 5B single-model ablation.** Report macro-F1 alongside paraphrase preservation, predicate-pair consistency, and polarity-flip consistency for `v2_no_intervention`, `v2_no_polarity_flip`, `v2_full4e`, and `v2_no_predicate_contrast`. This table should foreground the classification–consistency trade-off rather than ranking systems by macro-F1 alone.

**Table 2: Stage 5C classifier-auditor comparison.** Report macro-F1, entitled-output gate violation rate, polarity output consistency, polarity internal consistency, and the output/internal gap for classifier-only, balanced-only, strict-only, and the three conservative routers. The conservative balanced router should be highlighted as the best combined configuration, while balanced-only should be identified as the strongest output-level consistency model.

Seed-level values and the full set of pairwise checks should remain in an appendix table. In particular, entitled-output counts should accompany violation rates to make the effect of conservative downgrading visible.

## Limitations

These findings come from a template-controlled intervention dataset rather than naturally occurring evidence errors. The interventions are intentionally structured, and the number and diversity of source pair groups remain limited relative to benchmark or deployment-scale evaluation. Results may therefore reflect regularities in the controlled generation process.

The router uses a fixed threshold and derives zero gate violations by construction for retained entitled outputs. This establishes rule compliance, not calibrated confidence or causal use of each gate. The current experiments also do not test robustness to retrieved evidence, long contexts, domain shift, annotation noise, or unconstrained language generation. No claim is made here about general hallucination reduction, real-world RAG systems, or LLM deployment.

## Final takeaways

1. The best plain classifier achieved macro-F1 0.870 ± 0.006 but did not provide the strongest intervention consistency or internal faithfulness.
2. Intervention-aware objectives improved predicate and polarity consistency, with a measurable classification trade-off across single-model variants.
3. Output-level consistency alone was insufficient: classifier-only retained a 0.250 ± 0.150 polarity output/internal gap.
4. The conservative balanced router combined the strongest aggregate macro-F1, 0.878 ± 0.009, with zero measured entitled-output gate violations and zero polarity output/internal gap.
5. The evidence supports a controlled distinction between predicting the right label and internally justifying entitlement to that label. Broader generalization remains to be tested.
