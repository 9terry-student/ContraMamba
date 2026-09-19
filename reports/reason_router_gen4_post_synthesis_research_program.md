# ContraMamba Post-Synthesis Research Program
## Behavioral Bridge → Natural-Language External Transfer → Steering → Robustness → Site Specificity

### Status

`RESEARCH_DIRECTION_AND_SEQUENCE_ONLY`

This document is a durable research-program handoff for the work that follows the
completed Gen4 native-Mamba cross-scale synthesis.

It is **not** an execution authority by itself.

It does not authorize:

- training;
- evaluation;
- Kaggle/GPU execution;
- new response inspection;
- model selection;
- hyperparameter search;
- layer/token/epsilon sweeps;
- post-hoc rescue.

Its purpose is to preserve the intended scientific sequence across future chats and
prevent the project from drifting into routine tuning or from forgetting why each
next experiment exists.

---

## 0. Frozen starting point

Branch at program start:

`gen4-mamba370m-core-replication`

Frozen synthesis anchor:

`eecb268b77b946990774cbaccb265090c4e13d68`

Frozen synthesis report:

`reports/reason_router_gen4_core_stable_residual_plastic_cross_scale_synthesis.md`

Current narrow scientific synthesis:

`CORE-STABLE / RESIDUAL-PLASTIC STRENGTHENED THROUGH MAMBA-1.4B`

The exact meaning is:

- a **scale-local dominant causal role** recurs under the frozen procedure;
- the principal-plane rank implementing that role is not fixed across scales;
- the surrounding residual remains structured but reorganizes across scale in signed
  effects, coefficient-mass distribution, and generator-family coupling.

Do not reinterpret this as:

- semantic identity of same-numbered planes across backbones;
- a universal scaling law;
- architecture-independent universality;
- benchmark improvement;
- behavioral usefulness;
- natural-language transfer.

Those are exactly the gaps targeted by the next program.

Historical boundary that must remain frozen:

The preregistered 370M joint residual-recurrence criterion failed and remains failed.
Nothing in the later 1.4B program rescues or rewrites that historical result.

---

## 1. Research priority table

| Priority | Experiment | Main objection blocked | Information value | Cost / risk |
|---|---|---|---|---|
| 1 | 370M + 1.4B behavioral bridge | “Q is only an internal diagnostic” | Very high | Medium |
| 2 | AVeriTeC natural-language external causal transfer | “The mechanism is synthetic-only” | Highest | Medium–high |
| 3 | Causal-atlas-guided steering | “The analysis is descriptive but not useful” | High | Medium |
| 4 | Small-`epsilon` robustness | “The result is specific to `epsilon=0.025`” | Medium | Low |
| 5 | One-shot adjacent-site specificity | “The result is an accident of one layer/site” | Medium | Medium–high |

The execution order is the priority order unless an earlier experiment is blocked by a
hard feasibility defect.

Routine convenience is not a reason to reorder the program.

---

# Experiment 1 — 370M + 1.4B behavioral bridge

## 1.1 Scientific purpose

The current strongest causal measurements use the internal susceptibility endpoint
`Q`.

The first remaining objection is therefore:

> The intervention changes an internal state-space diagnostic, but that change may not
> matter for the model's downstream decision.

This experiment connects the frozen internal causal mechanism to the final task logits.

The earlier seed181 behavioral restoration bridge already established the feasibility
of this type of test on the earlier scale.

Frozen prior design:

`reports/reason_router_gen4_seed181_behavioral_restoration_bridge_design.md`

Frozen prior result:

`reports/reason_router_gen4_seed181_behavioral_restoration_bridge_analysis_retry2.json`

Historical result:

- `N = 300`;
- `mean(D_BEH) = +0.008332191656033197`;
- one-sided `p = 2.454643852170648e-17`;
- result:
  `SEED181_BEHAVIORAL_RESTORATION_BRIDGE_SUPPORTED`.

This historical result is context only. It is not part of the new inferential family.

## 1.2 New scales

The new bridge targets exactly:

- Mamba-370M;
- Mamba-1.4B.

Use each scale's already frozen **scale-local** core objects.

Mamba-370M:

- dominant candidate: `P3`;
- response-blind control: `P5`.

Mamba-1.4B:

- dominant candidate: `P5`;
- response-blind control: `P4`.

Do not impose one scale's rank identity on the other.

## 1.3 Behavioral intervention semantics

The default design should inherit the earlier behavioral bridge logic:

- full-model forward;
- intervention at the already frozen causal site for that scale;
- same target-token semantics used by the frozen causal measurement;
- downstream computation continues through the frozen task heads;
- evaluate final three-way logits.

Candidate behavioral conditions:

1. native;
2. dominant component neutralized;
3. exact dominant-component restoration;
4. response-blind matched-control replacement.

No alternative plane or control may be selected from behavioral responses.

The exact implementation must first prove that the 370M and 1.4B downstream
checkpoints, label ordering, head reconstruction, and intervention path preserve the
same task semantics.

## 1.4 Candidate primary endpoint

For a row with frozen correct class `y`:

`m = z_y - max_{c != y} z_c`.

For source pair `i`, use a predeclared pair-average margin `M_i` over the same
behaviorally interpretable six-cell rows used by the earlier bridge if static audit
confirms those row semantics remain valid.

Primary scale-local contrast:

`D_BEH,s,i = M_restored,s,i - M_control,s,i`.

The point is not to maximize accuracy.

The point is to test whether the already frozen internal causal core has a downstream
decision effect relative to its response-blind geometric control.

## 1.5 Inferential family

The intended new family contains exactly two pre-specified primary tests:

- 370M `D_BEH`;
- 1.4B `D_BEH`.

Preferred family control:

- one-sided one-sample Student t-tests;
- positive alternative;
- two primary p-values;
- Holm family-wise correction across the two new scales.

Interpretation:

- both pass after correction:
  `CROSS_SCALE_BEHAVIORAL_BRIDGE_THROUGH_1.4B_SUPPORTED`;
- exactly one passes:
  only that scale receives a behavioral bridge claim;
- neither passes:
  the new cross-scale behavioral bridge is not established.

The historical earlier-scale behavioral result must not rescue a failure.

## 1.6 Fresh-population rule

Do not choose the next XG1 numeric range by assumption.

Before materialization, perform a repository-wide occupancy audit of all frozen XG1
ranges.

Then allocate fresh disjoint cohorts prospectively.

No response inspection is permitted during range allocation or structural build.

## 1.7 First subphase after this program document

`EXPERIMENT_1_STATIC_FEASIBILITY_AUDIT`

Read-only / CPU-only.

Must establish:

1. exact frozen 370M downstream checkpoint and SHA;
2. exact frozen 1.4B downstream checkpoint and SHA;
3. final task class order and head compatibility at both scales;
4. intervention path compatibility with full downstream forward;
5. frozen local dominant/control geometry identities;
6. exact behavioral row semantics to reuse;
7. global XG1 occupancy and candidate fresh ranges;
8. expected forward budget;
9. whether one common implementation can serve both scales without changing science.

No GPU execution is authorized during this subphase.

---

# Experiment 2 — AVeriTeC natural-language external causal transfer

## 2.1 Scientific purpose

The highest-value external-validity objection is:

> The mechanism may exist only in ContraMamba's synthetic structured XG1 language.

AVeriTeC is a real-world claim-verification dataset built from naturally occurring
claims and web evidence.

Its evidence representation includes natural-language questions/answers and textual
justification, and the benchmark uses four verdict classes:

- Supported;
- Refuted;
- Not Enough Evidence;
- Conflicting Evidence/Cherrypicking.

This is intentionally much farther from the synthetic six-cell generator than the
existing XG1 studies.

## 2.2 First external-transfer target: gold-evidence causal transfer

Do **not** begin with web retrieval.

Retrieval quality would add a major confound:

`mechanism transfer × retrieval failure`.

The first AVeriTeC experiment should therefore use the dataset's frozen annotated
evidence / question-answer representation so that the first question is narrowly:

> Does the frozen ContraMamba causal mechanism transfer to natural-language
> claim/evidence inputs when retrieval is held fixed?

Only after this is answered should open-web retrieval be considered.

## 2.3 Label-space boundary

ContraMamba's frozen task is three-way:

`REFUTE, NOT_ENTITLED, SUPPORT`.

AVeriTeC is four-way.

Therefore the project must **not** silently force the fourth AVeriTeC class into an
existing ContraMamba class.

The first transfer study should prospectively choose one of two scientifically clean
options before model responses are inspected:

### Preferred initial option

A three-label compatibility cohort containing only:

- Supported;
- Refuted;
- Not Enough Evidence.

The fourth AVeriTeC label is excluded by a dataset-semantic rule fixed before
inference.

This is a **three-class natural-language causal transfer study**, not a full AVeriTeC
leaderboard claim.

### Later option

A separate four-class adaptation/training project.

That is a different scientific question and must not be mixed into the first frozen
external causal transfer.

## 2.4 Primary mechanism question

The preferred first external test should target the strongest frozen scale, Mamba-1.4B,
and ask whether the frozen scale-local core intervention changes an external
natural-language decision endpoint in the predicted beneficial direction relative to
the response-blind control.

The exact natural-language formatting, token anchor, and behavioral endpoint must be
frozen before response inspection.

No prompt-format search, anchor search, layer search, or plane reselection is permitted
on the evaluation cohort.

## 2.5 Minimum staging

1. dataset/license/schema audit;
2. deterministic natural-language adapter;
3. label-compatibility rule;
4. token-anchor feasibility;
5. response-blind structural holdout;
6. implementation validation;
7. one prospective external causal run.

The first study should use gold evidence, not retrieval.

---

# Experiment 3 — causal-atlas-guided steering

## 3.1 Scientific purpose

Objection:

> Even if the mechanism is real, the causal atlas is only explanatory and has no
> practical control value.

This experiment asks whether the frozen causal atlas can guide a pre-specified
intervention that improves downstream decision behavior without unacceptable harm.

## 3.2 Dependency

Do not design steering from internal `Q` alone.

Steering begins only after Experiment 1 has determined whether the core intervention
has a reproducible downstream behavioral bridge at 370M / 1.4B.

Experiment 2 may additionally inform whether a steering target should remain synthetic
or extend to natural language.

## 3.3 Design principles

The steering vector must come from already frozen causal geometry.

Do not train a new direction on the evaluation responses.

Do not search over:

- plane identity;
- layer;
- token;
- intervention sign;
- intervention magnitude;
- trigger threshold

on the final evaluation cohort.

If a gating rule or magnitude requires calibration, use a separate calibration
population and freeze it before evaluation.

## 3.4 Required primary utility/harm structure

A useful steering experiment must include both:

- benefit on a prospectively defined target population;
- harm / preservation on a prospectively defined non-target population.

A steering method that improves selected errors while broadly degrading preserved
examples does not establish useful control.

Candidate outputs:

- correct-class margin change;
- correction rate on target errors;
- damage rate on previously correct examples;
- net accuracy change;
- class-specific harm;
- abstention / entitlement shifts.

The final primary endpoint and harm gate must be frozen before execution.

---

# Experiment 4 — small-`epsilon` robustness

## 4.1 Scientific purpose

Objection:

> The causal geometry may be an artifact of the single finite-difference scale
> `epsilon = 0.025`.

The project already has a finite-`epsilon=0.025` five-plane reconstruction artifact
with high but non-exact reconstruction fidelity.

This next experiment tests whether the core geometry remains qualitatively stable at
smaller intervention scales.

## 4.2 No epsilon sweep

This is **not** an epsilon optimization experiment.

Use a small fixed predeclared set derived algebraically from the existing value, for
example:

- `0.025` as frozen reference;
- `0.0125`;
- `0.00625`.

The exact set must be frozen before new responses.

No additional epsilon may be added after inspecting results.

## 4.3 Preferred outputs

Descriptive robustness should include:

- scale-local dominant candidate stability;
- signed per-plane profile stability;
- normalized profile similarity;
- finite-difference magnitude convergence;
- core-control contrast direction;
- numerical signal-to-noise / degeneracy diagnostics.

Prefer descriptive robustness first.

Do not multiply the main scientific claim with a large family of new p-values unless a
specific inferential robustness question is prospectively justified.

---

# Experiment 5 — one-shot adjacent-site specificity

## 5.1 Scientific purpose

Objection:

> The result may be an accidental effect of one chosen layer triplet.

This study is deliberately last because it is the easiest to turn into a layer sweep.

The experiment must remain **one-shot**.

## 5.2 No layer sweep

Current homologous 48-layer mapping:

- source block `33`;
- target residual layer `34`;
- intervention layer `35`.

The adjacent-site study must freeze exactly one neighboring homologous triplet by an
architecture-only rule before response inspection.

Candidate rule:

shift the full triplet by exactly one block in one predeclared direction, preserving
the same relative offsets.

The exact direction (`+1` or `-1`) must be selected from static architectural and
instrumentation feasibility, not causal response.

No second adjacent site may be added as rescue.

## 5.3 Preferred specificity endpoint

Use a fresh same-pair cohort and measure both:

- canonical frozen site;
- one frozen adjacent site.

A prospective paired contrast is preferable to comparing results from unrelated
historical populations.

The question is whether the canonical site carries stronger causal signal than the one
predeclared adjacent site under otherwise matched measurement semantics.

The exact geometry-reconstruction and comparability rules must be resolved statically
before execution.

---

# 6. Program-wide scientific rules

These rules apply to all five experiments.

## 6.1 Frozen evidence is not tuning data

Completed 130M, 370M, and 1.4B response artifacts may motivate hypotheses and define
existing frozen objects.

They may not be used to tune a new evaluation cohort after results are observed.

## 6.2 Fresh cohorts before responses

Any new confirmatory population must be:

1. generated / selected structurally;
2. audited for overlap;
3. frozen;
4. only then exposed to model responses.

## 6.3 Scale-local plane identities

Never treat the same plane number across backbones as established semantic identity.

Use terms such as:

- scale-local dominant rank;
- rank-aligned profile;
- local residual plane.

## 6.4 Separate four validity layers

Always distinguish:

1. code correctness;
2. execution success;
3. artifact/provenance validity;
4. scientific conclusion.

A successful run alone does not establish a scientific claim.

## 6.5 No rescue

A failed primary endpoint is not rescued by:

- another plane;
- another layer;
- another token;
- another epsilon;
- another tail;
- another row subset;
- another label mapping;
- another external dataset slice.

A scientifically different follow-up requires a new prospective question.

## 6.6 Minimal documentation

Do not create routine authority/spec documents for every workflow step.

Create a new scientific design only when needed to freeze a genuinely new response
question or endpoint.

This program document should remain the durable sequencing / handoff document.

---

# 7. Stop/go logic across experiments

## After Experiment 1

If the 370M + 1.4B behavioral bridge is supported:

- internal causal recurrence is connected to downstream decision behavior;
- proceed to AVeriTeC external transfer with stronger motivation.

If only one new scale supports the bridge:

- record a scale-specific behavioral bridge;
- still proceed to AVeriTeC, but do not claim cross-scale behavioral recurrence.

If neither supports the bridge:

- do not abandon the mechanistic result;
- narrow interpretation to internal causal dynamics;
- AVeriTeC becomes a higher-risk external falsification attempt rather than a direct
  behavioral extension.

## After Experiment 2

If natural-language transfer is supported:

- steering may target the external natural-language setting.

If not:

- steering remains within the native synthetic task;
- do not rescue by prompt search or retrieval tricks.

## Experiments 4 and 5

These are robustness/specificity studies.

They do not rescue failures in Experiments 1–3.

---

# 8. New-chat resume protocol

When continuing in a new chat, do **not** ask the user to reconstruct the research
state from memory.

Read this file first:

`reports/reason_router_gen4_post_synthesis_research_program.md`

Then read:

`reports/reason_router_gen4_core_stable_residual_plastic_cross_scale_synthesis.md`

Then request / inspect:

`cm context`

unless exact current repository state is already available.

Program priority remains:

1. 370M + 1.4B behavioral bridge;
2. AVeriTeC natural-language external causal transfer;
3. causal-atlas-guided steering;
4. small-`epsilon` robustness;
5. one-shot adjacent-site specificity.

Do not use Codex for this research line.

Do not start Kaggle merely because a new chat begins.

The next step immediately after this program document is frozen is:

`EXPERIMENT_1_STATIC_FEASIBILITY_AUDIT`

with no GPU execution.

That audit should resolve exact checkpoints, full-forward behavioral compatibility,
fresh-population occupancy, and implementation reuse for 370M and 1.4B.

---

## Final program objective

The next research program is not another search for a stronger internal effect.

It is a sequence designed to progressively answer:

1. **Does the internal causal mechanism affect the model's actual decision?**
2. **Does it survive transfer from synthetic structure to real natural language?**
3. **Can the frozen causal atlas be used for controlled beneficial intervention?**
4. **Is the mechanism robust to smaller finite-difference scale?**
5. **Is it specific to the identified causal site rather than a generic layer effect?**

This sequence converts the current mechanistic result from an internal cross-scale
observation into a progressively stronger test of behavioral relevance, external
validity, utility, numerical robustness, and spatial specificity.
