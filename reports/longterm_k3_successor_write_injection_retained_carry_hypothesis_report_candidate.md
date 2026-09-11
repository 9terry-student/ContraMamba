# K3 Successor Hypothesis-Generation Report: Write Injection vs Retained-Contribution Carry

**Status:** READ-ONLY HYPOTHESIS-GENERATION / THEORY REPORT.

**Authority status:** NOT EXECUTION AUTHORITY.

**Preregistration status:** NOT A CONFIRMATORY PREREGISTRATION.

This report is derived only from the already-frozen K3 scientific artifacts and the read-only successor audit performed after K3 closure.

No new model execution, intervention, training, fine-tuning, or probe fitting is authorized by this report.

## 1. Governing closed evidence

K3 contradiction archive commit:

`99988e8d2a4c47d6bcaec1bf487aac777076fe2f`

K3 frozen scientific verdict:

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CONTRADICTED`

Frozen K3 block artifact:

`reports/longterm_k3_retention_write_causal_ac833fc_v1/block_metrics.jsonl`

SHA256:

`09ad9f2928947bdc5723cb536f028bc0a20ceba3310b0a1b71980a59b9636899`

K3 remained technically valid:

- natural structural replay = PASS_EXACT;
- sham replay = PASS_EXACT;
- GW pair collapse = PASS_EXACT;
- archived K2R baseline reproduction = PASS_EXACT;
- 1200 natural branch exact checks passed;
- 1200 sham branch exact checks passed;
- 600 GW pair exact checks passed.

Therefore the scientific contradiction is not attributed to replay/provenance failure.

## 2. Why a successor hypothesis is needed

The preregistered K3 hypothesis predicted:

- R/D local trajectory geometry: W-dominant;
- DISP/P net trajectory geometry: G-dominant.

That hypothesis was contradicted because:

- `DISP_SEL` was Holm-significant in the negative direction;
- `P_ATT` was Holm-significant in the negative direction;
- `P_SEL` was Holm-significant in the negative direction.

However, the K3 block artifact contains a stronger descriptive structural pattern that was not itself preregistered as the K3 confirmatory hypothesis:

**W equalization nearly eliminates pair-specific trajectory geometry across all four metric families, while G-coefficient equalization leaves the geometry approximately unchanged.**

Because this pattern was recognized after observing K3 outcomes, it is hypothesis-generating only.

It must not be promoted as confirmatory K3 evidence.

## 3. Read-only successor audit

The audit used exactly the 150 frozen K3 reciprocal blocks.

For each metric q in:

- R
- D
- DISP
- P

it compared absolute residual aligned geometry under:

- W_EQ
- G_EQ

relative to the same BASE block values.

The audit also compared absolute intervention attenuation:

`|ATT_W|`

versus:

`|ATT_G|`.

All exact-sign p-values below are exploratory diagnostics only.

They are not eligible for a future confirmatory claim on this same population.

## 4. R result

Median absolute BASE signal:

`2.7126813529900051`

Median absolute W_EQ residual:

`0.025188167531098316`

Median absolute G_EQ residual:

`2.7112701603518516`

Residual ratio:

- W_EQ / BASE = `0.0092853395786184414`
- G_EQ / BASE = `0.99947977943056299`

Among 137 nonzero-effective blocks:

- W_EQ had the smaller absolute residual in 136;
- G_EQ had the smaller residual in 1.

Absolute attenuation magnitude:

- `|ATT_W| > |ATT_G|` in 137/137 effective blocks.

BASE sign preservation:

- W_EQ = 64/137;
- G_EQ = 136/137.

Descriptive interpretation:

R pair-specific geometry is almost completely removed by W equalization while remaining almost unchanged under G-coefficient equalization.

## 5. D result

Median absolute BASE signal:

`0.047940305548892725`

Median absolute W_EQ residual:

`0.00066998981131954904`

Median absolute G_EQ residual:

`0.048922132878153779`

Residual ratio:

- W_EQ / BASE = `0.013975501483532446`
- G_EQ / BASE = `1.0204802059148272`

Among 137 effective blocks:

- W_EQ had the smaller absolute residual in 135;
- G_EQ had the smaller residual in 2.

Absolute attenuation magnitude:

- `|ATT_W| > |ATT_G|` in 137/137 effective blocks.

BASE sign preservation:

- W_EQ = 78/137;
- G_EQ = 137/137.

Descriptive interpretation:

D pair-specific geometry shows the same qualitative asymmetry as R.

## 6. DISP result

Median absolute BASE signal:

`6.1422237448719326`

Median absolute W_EQ residual:

`0.013665532616094822`

Median absolute G_EQ residual:

`6.1341981491482933`

Residual ratio:

- W_EQ / BASE = `0.0022248509959449147`
- G_EQ / BASE = `0.99869337294488181`

Among 137 effective blocks:

- W_EQ had the smaller absolute residual in 135;
- G_EQ had the smaller residual in 2.

Absolute attenuation magnitude:

- `|ATT_W| > |ATT_G|` in 136/137 effective blocks.

BASE sign preservation:

- W_EQ = 54/137;
- G_EQ = 137/137.

Descriptive interpretation:

The K3 G-dominant prediction failed not because DISP lacked a component-sensitive causal structure, but because W equalization removed substantially more pair-specific DISP geometry than G-coefficient equalization.

This is the opposite of the preregistered coefficient-specialization prediction.

## 7. P result

Median absolute BASE signal:

`0.01849793230917526`

Median absolute W_EQ residual:

`0.00014101246674011469`

Median absolute G_EQ residual:

`0.018634416272804238`

Residual ratio:

- W_EQ / BASE = `0.0076231475163399927`
- G_EQ / BASE = `1.007378336202543`

Among 137 effective blocks:

- W_EQ had the smaller absolute residual in 137;
- G_EQ had the smaller residual in 0.

Absolute attenuation magnitude:

- `|ATT_W| > |ATT_G|` in 136/137 effective blocks.

BASE sign preservation:

- W_EQ = 87/137;
- G_EQ = 137/137.

Descriptive interpretation:

P gives the strongest descriptive evidence that branch-specific write differences are tightly coupled to the observed pair-specific geometry under the current replay intervention.

## 8. Aggregate descriptive pattern

Across the 600 block-metric cells:

- W_EQ had the smaller absolute residual in 543;
- G_EQ had the smaller residual in 5;
- 52 were ties.

This 600-cell count is descriptive only.

Metric cells within a block are not independent replicates and must not be treated as N=600 inferential observations.

## 9. Critical conceptual correction

The original K3 structural equation was:

`S_t = G_t ⊙ S_(t-1) + W_t`.

K3 G_EQ equalized:

`G_t`

between correction and control branches.

It did **not** equalize the complete retained contribution:

`H_t = G_t ⊙ S_(t-1)`.

After branch divergence, correction and control generally have different:

`S_(t-1)`.

Therefore, even under identical midpoint G coefficients:

`Gbar_t ⊙ S_(t-1)^corr`

and:

`Gbar_t ⊙ S_(t-1)^ctrl`

can remain different.

Consequently, the descriptive finding:

`G_EQ approximately preserves pair-specific geometry`

does not imply:

`retained-state carry is causally irrelevant`.

It implies only that:

**branch-specific differences in the retention coefficient G are not necessary for preserving most of the observed pair-specific geometry under the frozen K3 intervention.**

This distinction is central.

## 10. Successor mechanistic hypothesis

The next hypothesis should not be the simplistic logical opposite of K3.

It should distinguish **divergence injection** from **history-dependent carry**.

Define:

`H_t = G_t ⊙ S_(t-1)`

as the full retained contribution.

The successor hypothesis candidate is:

### Write-Injection / Retained-Carry Hypothesis

1. Branch-specific differences in `W_t` are the primary source that injects new correction/control trajectory divergence after token divergence.
2. The retained contribution `H_t` transports, preserves, or reshapes divergence that has already entered the recurrent state.
3. Branch-specific variation in the coefficient `G_t` itself is not the primary source of the replicated pair-specific geometry.
4. Therefore coefficient equalization `G_EQ` can preserve most geometry even when retained-state carry remains mechanistically important.
5. A valid causal test must compare the direct write contribution `W_t` with the full retained contribution `H_t`, not merely compare W with the coefficient G.

This hypothesis is generated from K3 outcomes.

It is not established.

## 11. What K3 now supports and does not support

K3 plus the read-only audit support the following hypothesis-generating statement:

**Across the frozen layer-23 W=8 replay geometry, pair-specific correction/control structure is highly sensitive to branch-specific write equalization and largely insensitive to branch-specific retention-coefficient equalization.**

They do not yet support:

- a confirmatory W-dominant mechanism;
- a claim that retained-state carry is unimportant;
- a claim that H is weaker than W;
- a claim that W alone generates the full trajectory geometry;
- a task-decision causal claim;
- an authorization/entitlement causal claim;
- external-distribution generalization.

## 12. Required successor experimental correction

A future confirmatory causal experiment should use a new claim-disjoint population and prospectively distinguish:

`W_t`

from:

`H_t = G_t ⊙ S_(t-1)`.

The old 300 K3 items cannot be reused to confirm the successor hypothesis because the hypothesis was generated from their outcomes.

A future experiment must therefore:

- use a new claim-disjoint population;
- preregister the structural contribution intervention before outcome inspection;
- freeze the population recipe before execution;
- preserve layer 23 and W=8 unless a new scientific rationale is prospectively frozen;
- retain exact replay/provenance gates;
- use a new confirmatory family defined before execution;
- avoid reusing the exploratory p-values in this report as confirmatory evidence.

## 13. Candidate next-stage name

Recommended successor stage:

`K3C — Native Recurrence Contribution Decomposition`

Scientific question:

**Is pair-specific layer-23 trajectory geometry causally sourced primarily by direct write injection W, while the full retained contribution H mainly carries already-created divergence?**

K3C is a new experiment.

It is not a K3 rescue.

## 14. K4 boundary

K4 remains blocked as a downstream validation of the contradicted W-local/G-net K3 mechanism.

A future K4 decision-space study may be considered only after its rationale is independently frozen.

K3C, if pursued, remains a recurrence-mechanism study and does not automatically authorize K4.

## 15. Current authority state

`K3_CLOSED = YES`

`K3_PREREGISTERED_SPECIALIZATION_CONTRADICTED = YES`

`K3_SUCCESSOR_AUDIT_COMPLETE = YES`

`K3_SUCCESSOR_HYPOTHESIS_GENERATED = YES`

`K3C_PREREGISTRATION_AUTHORIZED_TO_DRAFT = YES`

`K3C_EXECUTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

No new scientific execution is authorized by this report.

## 16. Final interpretation

The K3 contradiction should not be treated as a dead end.

It materially sharpens the mechanism question.

The evidence no longer favors a decomposition in which local geometry is written by W while net geometry is generated by branch-specific G coefficients.

Instead, the frozen data motivate a more structurally faithful distinction:

**direct write injection versus history-dependent retained contribution.**

That successor hypothesis now requires a new prospective test.
