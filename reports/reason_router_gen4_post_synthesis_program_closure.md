# ContraMamba Gen4 Post-Synthesis Program Closure
## Behavioral Relevance, Natural-Language Transfer, Steering Utility, Numerical Robustness, and Site Specificity

### Status

`STATIC_POST_SYNTHESIS_PROGRAM_CLOSURE`

This report closes the five-experiment program defined in:

`reports/reason_router_gen4_post_synthesis_research_program.md`

It synthesizes only already frozen repository evidence through:

`d411b1b25f1112260dcf0a875e23b83f33735a53`

It introduces:

- no new model execution;
- no training or backward pass;
- no new response population;
- no new primary endpoint;
- no new p-value;
- no alternative-tail test;
- no multiplicity-family change;
- no layer, token, epsilon, plane, control, cohort, or label rescue;
- no reinterpretation of failed primary results.

The purpose of this document is to freeze the scientific outcome of the completed
post-synthesis sequence:

1. behavioral bridge;
2. natural-language external causal transfer;
3. causal-atlas-guided steering;
4. small-epsilon robustness;
5. one-shot adjacent-site specificity.

The pre-program mechanistic starting point remains:

`CORE-STABLE / RESIDUAL-PLASTIC STRENGTHENED THROUGH MAMBA-1.4B`

with the exact prior meaning that a scale-local dominant causal role recurs while the
surrounding residual realization reorganizes across scale.

---

## 1. Program-level outcome

The five experiments do not collapse to a single uniformly positive result.

The frozen outcome is:

`MECHANISTIC_VALIDITY_AND_SPECIFICITY_SUPPORTED_WITH_SCALE_DEPENDENT_BEHAVIORAL_RELEVANCE_AND_NO_ESTABLISHED_STEERING_UTILITY`

The evidence supports all of the following narrow statements:

1. the internal causal geometry is not explained solely by the original
   `epsilon=0.025` choice;
2. the canonical Mamba-1.4B site carries a stronger frozen rank-aligned core signal
   than the single prospectively fixed adjacent `+1` site;
3. causal effects transfer to a natural-language gold-evidence setting at the tested
   130M and 370M scales;
4. downstream behavioral relevance is scale-dependent rather than uniformly recurring;
5. the tested fixed-mirror steering intervention did not establish practical utility.

The evidence does **not** support replacing these distinct conclusions with a stronger
single claim such as:

- universal behavioral recurrence;
- monotonic scaling of downstream usefulness;
- a universally useful causal steering direction;
- a global layer optimum;
- architecture-independent universality;
- benchmark superiority.

Experiments 4 and 5 are robustness/specificity results only.

They do not rescue failures in Experiments 1–3.

---

## 2. Frozen evidence chain

### 2.1 Program definition

Program:

`reports/reason_router_gen4_post_synthesis_research_program.md`

Frozen pre-program synthesis:

`reports/reason_router_gen4_core_stable_residual_plastic_cross_scale_synthesis.md`

### 2.2 Experiment 1 — behavioral bridge

Frozen primary artifact:

`reports/reason_router_gen4_mamba370m14b_behavioral_bridge_runs/g4k-mamba370m14b-behavioral-bridge-xg1-4801-5100-2gpu-2b41f28-retry1/behavioral_bridge_analysis.json`

Execution HEAD:

`2b41f28577a59e46e2748a40db2037f1c8ab7ffa`

Frozen scientific conclusion:

`SCALE_SPECIFIC_BEHAVIORAL_BRIDGE_ONLY`

### 2.3 Experiment 2 — natural-language external causal transfer

Frozen primary result family:

`reports/reason_router_gen4_averitec_external_transfer_runs/g4k-averitec-external-transfer-130m370m-dev462-c5b470f-postcheck-recovery2/external_transfer_analysis.json`

Frozen static post-result interpretation:

`reports/reason_router_gen4_averitec_external_transfer_static_post_result_analysis.json`

Frozen source conclusion:

`CROSS_SCALE_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_THROUGH_370M_SUPPORTED`

### 2.4 Experiment 3 — causal-atlas-guided steering

Frozen analysis:

`reports/reason_router_gen4_averitec_370m_fixed_mirror_steering_analysis_328ae53/steering_analysis.json`

Frozen result:

`AVERITEC_370M_FIXED_MIRROR_P3_STEERING_NOT_ESTABLISHED`

### 2.5 Experiment 4 — small-epsilon robustness

Frozen analysis:

`reports/reason_router_gen4_small_epsilon_robustness_analysis_199fd28/small_epsilon_robustness_analysis.json`

Frozen result:

`SMALL_EPSILON_P3_SPECTRAL_DOMINANCE_PRESERVED`

### 2.6 Experiment 5 — one-shot adjacent-site specificity

Frozen final analysis:

`reports/reason_router_gen4_mamba14b_adjacent_site_specificity_analysis_runs/g4k-mamba14b-adjacent-site-specificity-analysis-xg1-5101-5400-5d651f6/adjacent_site_specificity_analysis.json`

Frozen final report:

`reports/reason_router_gen4_mamba14b_adjacent_site_specificity_analysis_runs/g4k-mamba14b-adjacent-site-specificity-analysis-xg1-5101-5400-5d651f6/adjacent_site_specificity_analysis.md`

Final artifact freeze commit:

`d411b1b25f1112260dcf0a875e23b83f33735a53`

Frozen scientific conclusion:

`MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_SUPPORTED`

---

## 3. Experiment 1 — scale-dependent behavioral bridge

### 3.1 Frozen design question

The behavioral bridge asked whether the already frozen internal causal core changes the
model's downstream three-way decision margin relative to a response-blind geometric
control.

The two new scale-local tests formed one Holm-controlled family:

- Mamba-370M dominant candidate `P3` vs control `P5`;
- Mamba-1.4B dominant candidate `P5` vs control `P4`.

Population:

`xg1_fact_4801..xg1_fact_5100`

with:

`N=300`

at each scale.

### 3.2 Mamba-370M result

Primary endpoint:

`D_BEH = M_dominant_restored - M_dominant_control`

Frozen statistics:

- `mean(D_BEH) = +0.0011206856990853946`;
- `sd = 0.0033338468073894592`;
- `t(299) = 5.822356821643221`;
- raw one-sided `p = 7.478590243607131e-09`;
- Holm-adjusted `p = 1.4957180487214262e-08`;
- Cohen's `dz = 0.3361539278293768`;
- positive fraction `= 0.58`.

Therefore the 370M behavioral bridge was supported within the frozen two-scale family.

### 3.3 Mamba-1.4B result

Frozen statistics:

- `mean(D_BEH) = -0.002012885312239329`;
- `sd = 0.005109305587665866`;
- `t(299) = -6.823666290433015`;
- raw one-sided `p = 0.9999999999753244`;
- Holm-adjusted `p = 0.9999999999753244`;
- Cohen's `dz = -0.3939645569641676`;
- positive fraction `= 0.37`.

Therefore the 1.4B behavioral bridge was not supported.

### 3.4 Experiment 1 conclusion

Cross-scale behavioral bridge through 1.4B:

`NOT ESTABLISHED`

Scale-specific conclusion:

`SCALE_SPECIFIC_BEHAVIORAL_BRIDGE_ONLY`

The supported 370M result does not rescue the 1.4B failure.

The earlier historical behavioral result is context only and is not part of this new
two-scale inferential family.

The correct interpretation is:

> Internal causal relevance reaches the downstream decision margin at Mamba-370M
> under the frozen bridge procedure, but the same type of positive bridge does not
> recur at Mamba-1.4B.

This rules out a simple monotonic claim that increasing scale preserves or strengthens
behavioral usefulness of the identified internal causal core.

---

## 4. Experiment 2 — natural-language external causal transfer

### 4.1 Frozen question

This experiment tested whether a frozen ContraMamba causal intervention transfers from
synthetic structured inputs to a natural-language AVeriTeC gold-evidence setting.

The analysis is a three-class compatibility study, not a full four-class AVeriTeC
leaderboard evaluation.

It does not test retrieval competence.

### 4.2 Mamba-130M

Frozen primary context:

- `N = 462`;
- `mean(D_EXT) = +0.00023104868881158465`;
- Cohen's `dz = 0.11964886938686847`;
- Holm-adjusted `p = 0.010430704961748994`;
- supported after Holm correction: `true`.

### 4.3 Mamba-370M

Frozen primary context:

- `N = 462`;
- `mean(D_EXT) = +4.55500221672976e-05`;
- Cohen's `dz = 0.10631034549402116`;
- Holm-adjusted `p = 0.01138150035747162`;
- supported after Holm correction: `true`.

### 4.4 Experiment 2 conclusion

Frozen source conclusion:

`CROSS_SCALE_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_THROUGH_370M_SUPPORTED`

This establishes a narrow external-validity result:

> At the tested 130M and 370M scales, the frozen causal intervention produced a
> positive correct-class-margin transfer effect on the frozen AVeriTeC gold-evidence
> cohort relative to its matched control.

The effect sizes are small.

The static post-result analysis explicitly does not support:

- benchmark-accuracy improvement;
- uniform transfer across source labels;
- unique mediation;
- a four-class AVeriTeC claim;
- retrieval competence;
- a Mamba-1.4B external-transfer claim.

No new p-values were added by the static post-result analysis.

---

## 5. Experiment 3 — fixed-mirror steering utility not established

### 5.1 Frozen question

The steering experiment asked whether the frozen Mamba-370M P3 causal atlas could be
converted into useful intervention under a pre-specified utility/harm rule on a fresh,
deduplicated AVeriTeC train-derived three-class gold-evidence cohort.

Population:

`N = 2799`

with:

- native incorrect: `2468`;
- native correct: `331`.

### 5.2 Frozen outcome

Corrections:

`0`

Damages:

`0`

Net correct-count change:

`0`

Net accuracy change:

`0.0`

All native predictions remained unchanged under both the fixed P3 mirror steering and
the P5 matched control.

The preservation gate passed because damage rate was zero, but the utility gate could
not pass:

- `C > D`: false;
- positive discordance count: false;
- significance gate: false;
- primary test estimable: false;
- new primary p-value added: `0`.

### 5.3 Experiment 3 conclusion

Frozen result:

`AVERITEC_370M_FIXED_MIRROR_P3_STEERING_NOT_ESTABLISHED`

The correct negative interpretation is:

> The specific fixed mirror extrapolation did not establish useful steering under the
> preregistered utility/harm rule.

This does not negate the causal mechanism or the natural-language transfer result.

It shows that explanatory causal structure and useful control are distinct empirical
questions.

Experiments 4 and 5 do not rescue this failure.

---

## 6. Experiment 4 — small-epsilon robustness

### 6.1 Frozen question

The robustness experiment tested whether the previously observed P3 spectral
organization was an artifact of the single finite-difference scale:

`epsilon = 0.025`.

The fixed scale set was:

- reference: `0.025`;
- new: `0.0125`;
- new: `0.00625`.

No epsilon selection or post-result expansion was allowed.

### 6.2 Frozen spectral result

At `epsilon=0.0125`:

- unique spectral dominant candidate: `P3`;
- `mean C(P3) - mean C(P5) = +2.524775165858401e-08`.

At `epsilon=0.00625`:

- unique spectral dominant candidate: `P3`;
- `mean C(P3) - mean C(P5) = +2.4240054982786555e-08`.

Both fixed new-epsilon gates passed.

Normalized mean-profile similarities were:

- cosine(`0.0125`, `0.00625`) = `0.9999457044844027`;
- cosine(`0.025`, `0.00625`) = `0.9999793916161879`;
- cosine(`0.025`, `0.0125`) = `0.9999495217216572`.

No inferential p-values were added.

### 6.3 Experiment 4 conclusion

Frozen result:

`SMALL_EPSILON_P3_SPECTRAL_DOMINANCE_PRESERVED`

Allowed interpretation:

> Across the preregistered smaller finite-difference scales `0.0125` and `0.00625`
> on the frozen XG1 population, the P3 spectral dominant candidate and positive
> P3-minus-P5 mean contribution contrast were preserved.

This does **not** establish:

- an `epsilon -> 0` limit;
- exact finite-difference reconstruction;
- causal additivity;
- plane independence;
- an optimal epsilon;
- steering utility.

---

## 7. Experiment 5 — one-shot adjacent-site specificity

### 7.1 Frozen question

The final experiment tested whether the Mamba-1.4B canonical site was stronger than
exactly one architecture-predeclared neighboring site under matched response
semantics.

Canonical triplet:

`(33,34,35)`

Adjacent `+1` triplet:

`(34,35,36)`

Fresh population:

`xg1_fact_5101..xg1_fact_5400`

with:

`N = 300`.

Fixed rank-aligned candidate:

`P5`

Canonical response-blind control:

`P4`

Adjacent geometry-only response-blind control:

`P4`

No second adjacent site, alternate direction, layer sweep, epsilon sweep, token sweep,
response-based control selection, or rescue was executed.

### 7.2 Adjacent geometry

Adjacent strong dimension:

`1205`

Adjacent geometry-only `lambda_plus` profile:

| plane | lambda_plus |
|---|---:|
| `P1` | `0.9622631061969398` |
| `P2` | `0.9733292938755923` |
| `P3` | `0.9881970133174621` |
| `P4` | `0.9947030656854544` |
| `P5` | `0.9977081088488762` |

The unique response-blind control excluding fixed P5 was therefore P4.

### 7.3 Fresh paired response

Definitions:

`D_CAN = Q_restored,canonical(P5) - Q_control,canonical(P4)`

`D_ADJ = Q_restored,adjacent(P5) - Q_control,adjacent(P4)`

`S = D_CAN - D_ADJ`

Frozen descriptive results:

| endpoint | mean | SD | Cohen's dz | positive fraction |
|---|---:|---:|---:|---:|
| `D_CAN` | `9.102783197942576e-09` | `9.931025875528005e-09` | `0.9166004914329766` | `0.7766666666666666` |
| `D_ADJ` | `-1.551069575441687e-09` | `9.295405983276345e-10` | `-1.6686410235682707` | `0.006666666666666667` |
| `S` | `1.0653852773384264e-08` | `9.998440714495731e-09` | `1.0655514272278794` | `0.81` |

Frozen 95% t-CI for `S`:

`[9.517845212840007e-09, 1.1789860333928522e-08]`

### 7.4 Primary inference

Exactly one inferential p-value was computed.

Test:

one-sample Student t-test on paired `S`

with:

- null: `E[S] <= 0`;
- alternative: `E[S] > 0`;
- `t(299) = 18.455892100362185`;
- one-sided `p = 1.3359277323396141e-51`.

Decision gates:

1. `mean(D_CAN) > 0`: PASS;
2. `mean(S) > 0` and `p < 0.05`: PASS.

### 7.5 Experiment 5 conclusion

Frozen scientific conclusion:

`MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_SUPPORTED`

Allowed claim:

> On the prospectively frozen Mamba-1.4B XG1 `5101..5400` cohort, the canonical
> `(33,34,35)` homologous site carried a stronger positive rank-aligned P5 core signal
> than the single architecture-predeclared adjacent `+1` site `(34,35,36)` under the
> matched frozen measurement procedure.

This does **not** establish:

- uniqueness across all layers;
- a global layer optimum;
- absence of causal signal at every other layer;
- semantic identity of P5 across sites;
- architectural universality;
- downstream behavioral usefulness;
- steering utility.

---

## 8. Integrated scientific synthesis

### 8.1 Mechanistic reality is better supported than practical utility

The post-synthesis program separates three questions that should not be conflated:

1. is the internal causal structure reproducible and specific?
2. does it affect external or downstream decision quantities?
3. can it be converted into useful control?

The frozen evidence answers them differently.

For internal mechanistic validity:

- prior cross-scale work supports recurrence of a scale-local dominant causal role;
- Experiment 4 supports robustness to smaller predeclared finite-difference scales;
- Experiment 5 supports one-shot spatial specificity against the fixed adjacent site.

For downstream and external relevance:

- Experiment 1 supports a behavioral bridge at 370M but not 1.4B;
- Experiment 2 supports natural-language correct-class-margin transfer through 370M.

For intervention utility:

- Experiment 3 does not establish useful fixed-mirror steering.

Therefore the strongest current evidence is for a **real, structured, locally specific
causal mechanism**, while behavioral expression and practical controllability are less
uniform.

### 8.2 Scale does not imply monotonic behavioral usefulness

The 1.4B evidence is particularly important because two results coexist:

- the internal 1.4B core is strongly supported and spatially specific;
- the new 1.4B downstream behavioral bridge is negative and not supported.

This combination rules out the interpretation that a stronger or clearly detectable
internal causal structure must automatically produce a stronger positive downstream
behavioral effect at larger scale.

The project should therefore preserve the distinction between:

`internal causal role`

and:

`downstream behavioral utility`.

### 8.3 Natural-language transfer does not imply steering success

Experiment 2 and Experiment 3 also form a deliberate contrast.

At 370M:

- natural-language margin-level causal transfer is supported;
- fixed-mirror steering produces zero corrections and zero damages.

Thus a mechanism can transfer to a natural-language decision quantity without the
tested intervention rule being strong enough to change discrete predictions.

The appropriate conclusion is not that the causal atlas is useless.

It is that:

> The specific preregistered fixed-mirror steering policy did not establish useful
> control.

A different steering question would require a new prospective design rather than
post-hoc tuning of this failed result.

### 8.4 Robustness and specificity strengthen the mechanistic claim, not failed utility claims

Experiment 4 blocks the narrow objection that the observed spectral organization is
specific to one finite-difference scale.

Experiment 5 blocks the narrow objection that the Mamba-1.4B core effect is merely a
generic response at the immediately adjacent predeclared site.

These results strengthen the **mechanistic** interpretation.

They do not change:

- the failed Mamba-1.4B behavioral bridge;
- the failed fixed-mirror steering utility result;
- any historical failed criterion from the earlier program.

---

## 9. Final claim boundary

The completed evidence supports the following narrow synthesis:

> Across the tested ContraMamba native-Mamba settings, a scale-local dominant causal
> role remains mechanistically robust and locally specific under the frozen procedures.
> Its downstream behavioral expression is scale-dependent, and natural-language
> margin-level transfer does not by itself imply useful steering control.

The project may additionally state, with the relevant scope attached:

- **mechanistic recurrence:** supported through Mamba-1.4B under scale-local rank
  identities;
- **small-epsilon robustness:** supported for the preregistered `0.0125` and `0.00625`
  scales in the tested robustness experiment;
- **one-shot site specificity:** supported for canonical `(33,34,35)` versus the single
  predeclared adjacent `(34,35,36)` site at Mamba-1.4B;
- **behavioral bridge:** supported at Mamba-370M but not Mamba-1.4B;
- **natural-language causal transfer:** supported at the tested Mamba-130M and
  Mamba-370M scales on the frozen three-class AVeriTeC gold-evidence study;
- **fixed-mirror steering utility:** not established.

The completed evidence does **not** establish:

- a universal plane identity;
- a universal layer;
- a global layer optimum;
- architecture-independent universality;
- transformer generalization;
- monotonic scaling of causal or behavioral effect magnitude;
- benchmark superiority;
- full four-class AVeriTeC performance;
- retrieval competence;
- useful steering in general;
- an optimal steering magnitude;
- an `epsilon -> 0` convergence theorem;
- exact causal additivity;
- uniqueness across all layers;
- rescue of any failed primary endpoint.

---

## 10. Program closure status

Experiment 1 — behavioral bridge:

`CLOSED — SCALE_SPECIFIC_BEHAVIORAL_BRIDGE_ONLY`

Experiment 2 — AVeriTeC natural-language external causal transfer:

`CLOSED — CROSS_SCALE_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_THROUGH_370M_SUPPORTED`

Experiment 3 — causal-atlas-guided steering:

`CLOSED — AVERITEC_370M_FIXED_MIRROR_P3_STEERING_NOT_ESTABLISHED`

Experiment 4 — small-epsilon robustness:

`CLOSED — SMALL_EPSILON_P3_SPECTRAL_DOMINANCE_PRESERVED`

Experiment 5 — one-shot adjacent-site specificity:

`CLOSED — MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_SUPPORTED`

Program-wide new inference introduced by this closure report:

`NONE`

Program-wide new model execution introduced by this closure report:

`NONE`

Failed results rescued:

`NO`

Post-synthesis five-experiment program:

`CLOSED`

---

## 11. Research-state handoff

The branch state used to construct this closure report was:

`gen4-mamba370m-core-replication`

with frozen evidence through:

`d411b1b25f1112260dcf0a875e23b83f33735a53`

Any future scientific extension should begin from a new prospective question.

It should not reopen, tune, subset, or rescue the completed five-experiment family.

A future project may investigate new questions such as intervention-policy design or
cross-architecture transfer, but those would be new studies rather than continuations
of unfinished work in this closed program.
