# ContraMamba Experiment 2 Static Feasibility Audit
## AVeriTeC Gold-Evidence Natural-Language External Causal Transfer

### Status

`STATIC_FEASIBILITY_PASS_WITH_PRE_RESPONSE_TOKEN_GATE`

This document records the static feasibility decision for Experiment 2 of the
post-synthesis research program.

Current ContraMamba anchor:

`1d460f63d07f0c4d8dbeed580fcdd21fd082d5ca`

Current completed Experiment 1 conclusion:

`SCALE_SPECIFIC_BEHAVIORAL_BRIDGE_ONLY`

with:

- Mamba-370M behavioral bridge: supported after the frozen two-test Holm family;
- Mamba-1.4B behavioral bridge: not supported;
- cross-scale behavioral bridge through 1.4B: not established.

This audit performs no model forward, no checkpoint load, no CUDA execution, no
training, and no new statistical inference.

It does not authorize GPU execution by itself.

---

## 1. Scientific question

Experiment 2 addresses the external-validity objection:

> The causal mechanism may be specific to the synthetic XG1 language and may not
> survive on naturally occurring textual claims and evidence.

The first external study will therefore ask:

> Does the already frozen ContraMamba causal core that has a validated downstream
> behavioral effect at Mamba-370M remain behaviorally causal on natural-language
> AVeriTeC claim/evidence inputs when retrieval is removed as a confound?

This is not a benchmark-leaderboard study.

It is a causal-transfer study.

---

## 2. Why Mamba-370M is now the primary scale

The original post-synthesis program named Mamba-1.4B as the preferred initial
external-transfer target before Experiment 1 had been executed.

Experiment 1 subsequently produced a scale-specific behavioral result:

### Mamba-370M

Frozen local causal object:

- selected dominant plane: `P3`;
- response-blind control: `P5`.

Fresh behavioral bridge:

- `mean(D_BEH) = +0.0011206856990853946`;
- raw one-sided `p = 7.478590243607131e-09`;
- Holm-adjusted `p = 1.4957180487214262e-08`;
- behavioral bridge supported: `true`.

### Mamba-1.4B

Frozen local causal object:

- selected dominant plane: `P5`;
- response-blind control: `P4`.

Fresh behavioral bridge:

- `mean(D_BEH) = -0.002012885312239329`;
- raw one-sided `p = 0.9999999999753244`;
- Holm-adjusted `p = 0.9999999999753244`;
- behavioral bridge supported: `false`.

Therefore the first AVeriTeC causal-transfer test will use:

`MAMBA-370M ONLY`

Reason:

A 370M external test asks whether an already behaviorally validated synthetic causal
mechanism transfers to natural language.

A 1.4B external test at this point would confound two questions:

1. natural-language external transfer; and
2. the already observed failure of the 1.4B synthetic behavioral bridge.

Mamba-1.4B is not included as a second primary, secondary rescue, or alternative scale
in this experiment.

A later 1.4B natural-language study would be a different prospective scientific
question.

---

## 3. External dataset identity

Dataset:

`AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web`

Official repository:

`MichSchli/AVeriTeC`

Pinned upstream repository commit observed by this audit:

`7c62d1ec8df3fb560d6efe2b85fa191135636f81`

Official development file:

`data/dev.json`

Git blob identity:

`40974243267f395dc583d805d10f043812419249`

Repository-reported byte count:

`1785475`

License recorded by the official repository:

`Creative Commons Attribution-NonCommercial 4.0 International`

Any derived ContraMamba artifact must retain dataset attribution and the upstream
identity.

The implementation stage must compute and freeze the exact SHA256 of the downloaded
pinned source bytes before deriving the evaluation cohort.

No moving-branch `main` content may be used as the final scientific source identity.

---

## 4. Dataset schema relevant to this experiment

Each AVeriTeC example contains, among other metadata:

- natural-language `claim`;
- four-way verdict `label`;
- textual `justification`;
- a list of fact-checking `questions`;
- one or more annotated `answers` for a question;
- answer type;
- evidence source URL / cached URL;
- source medium.

The official baseline is retrieval-based, but the dataset itself already contains
annotated gold question-answer evidence.

Experiment 2 intentionally removes retrieval from the first test.

Therefore:

`RETRIEVAL_EXECUTED = FALSE`

The model will receive only frozen annotated gold evidence.

---

## 5. Label-space compatibility rule

ContraMamba final class order is frozen as:

1. `REFUTE = 0`
2. `NOT_ENTITLED = 1`
3. `SUPPORT = 2`

AVeriTeC contains four labels:

- `Supported`;
- `Refuted`;
- `Not Enough Evidence`;
- `Conflicting Evidence/Cherrypicking`.

The first external causal-transfer cohort uses the exact prospective mapping:

| AVeriTeC label | ContraMamba class |
|---|---|
| `Refuted` | `REFUTE = 0` |
| `Not Enough Evidence` | `NOT_ENTITLED = 1` |
| `Supported` | `SUPPORT = 2` |
| `Conflicting Evidence/Cherrypicking` | excluded |

The conflicting-evidence class is excluded because ContraMamba has no frozen
one-to-one fourth decision class.

It must not be silently collapsed into any three-way class.

Published AVeriTeC development-set statistics report:

- Supported: `122`;
- Refuted: `305`;
- Not Enough Evidence: `35`;
- Conflicting Evidence/Cherrypicking: `38`;
- total: `500`.

Therefore the expected three-class-compatible cohort size is:

`462`

The response-blind builder must independently validate these counts against the exact
pinned source bytes.

If the exact pinned file does not match the expected label inventory, materialization
must fail closed.

No label-balanced subsampling will be performed.

The external study targets the natural compatible-label development distribution,
not an artificially balanced benchmark.

---

## 6. Deterministic gold-evidence serialization

For each included claim, use the original AVeriTeC question order and answer order.

Construct evidence text deterministically by flattening each gold question-answer item
as:

`Question: <question>\nAnswer: <answer>`

and joining items in source order with:

`\n\n`

If a question contains multiple answers, repeat the question once per answer while
preserving answer order.

No evidence document retrieval is performed.

No question generation is performed.

No LLM summarization is performed.

No use is made of the AVeriTeC textual `justification` field in the model input.

Reason:

The justification often directly verbalizes the final verdict reasoning and would
create an unnecessary label-leakage risk.

The annotated question-answer evidence is the external input.

---

## 7. Frozen input encoding

Use the same frozen active feature contract as the existing ContraMamba downstream
model:

`claim[:63] + EOS(0) + evidence[:64]`

with:

- maximum length: `128`;
- claim budget: `63` tokens;
- one EOS separator;
- evidence budget: `64` tokens;
- no prompt-format search;
- no adaptive evidence reordering;
- no response-dependent truncation rule.

AVeriTeC examples can be longer than this budget.

That is not hidden by this experiment.

The transfer claim is explicitly scoped to:

> AVeriTeC gold question-answer evidence under the frozen ContraMamba 63/64 active
> encoding contract.

The structural materialization must record, before model response:

- raw claim token count;
- consumed claim token count;
- raw evidence token count;
- consumed evidence token count;
- claim truncation indicator;
- evidence truncation indicator.

Truncation statistics are descriptive and response-blind.

They may not be used to select a favorable subset after model execution.

---

## 8. Token-anchor problem and resolution

### 8.1 Why the synthetic anchor cannot be reused literally

The existing Gen4 causal protocol uses an XG1 generator-defined semantic anchor:

`A_IDENTITY`

The AVeriTeC natural-language examples do not contain that synthetic generator span.

Therefore a literal `A_IDENTITY` transfer is impossible without inserting synthetic
content or searching for a new semantic anchor.

Neither is desirable.

### 8.2 Frozen external anchor

Use the already existing serialized claim/evidence separator as the response-blind
external anchor:

`A_CLAIM_EVIDENCE_BOUNDARY`

Definition:

- let `c` be the number of consumed claim tokens after the frozen 63-token cap;
- the EOS separator is at absolute token index `c`;
- define the external anchor index as `a = c`;
- preserve the existing intervention offset:
  `target_token = a + 2`.

Because evidence begins at `a + 1`, this places the intervention at the second
consumed evidence token.

This rule is:

- deterministic;
- dataset-agnostic;
- independent of labels beyond the predeclared compatibility filter;
- independent of model response;
- free of anchor search.

### 8.3 Interpretation boundary

This is **not** a claim that the AVeriTeC boundary token is semantically identical to
the synthetic `A_IDENTITY` span.

The external experiment is a transported-coordinate test:

> Does the frozen 370M causal plane remain behaviorally functional when transported
> from the synthetic generator coordinate to a fixed, response-blind natural-language
> claim/evidence boundary coordinate?

This is a stronger domain-shift test, but a narrower semantic claim.

If it fails, do not search another token anchor on the same evaluation population.

---

## 9. Mandatory pre-response token gate

Before any checkpoint load or model forward, the exact pinned AVeriTeC development
bytes must be materialized and tokenized with the exact frozen Mamba-370M tokenizer.

For every one of the expected `462` compatible examples require:

1. non-empty consumed claim span;
2. non-empty consumed evidence span;
3. at least two consumed evidence tokens;
4. `target_token = EOS_boundary + 2` is inside the attended sequence;
5. exact deterministic re-materialization;
6. no response or endpoint fields exist.

Target verdict:

`PASS_462_OF_462`

If fewer than 462 examples satisfy the frozen boundary rule:

`BLOCKED_EXTERNAL_TOKEN_GATE`

and no model execution is allowed.

Do not silently drop ineligible examples and continue.

A changed cohort or changed anchor would require a new pre-response scientific
decision.

---

## 10. Frozen Mamba-370M causal object

Use the already frozen 370M model and geometry only.

Checkpoint:

`reports/reason_router_gen4_mamba370m_core_replication_checkpoint_compact/seed181/G3-GROUP-D-HALF/selected_downstream_checkpoint.pt`

Checkpoint SHA256:

`9d8e3db22af4636938679aac6a8a97dd45344937d434fab29eac2ddc41a52a72`

Backbone:

`state-spaces/mamba-370m-hf`

Revision:

`589179554943157be31701edd8b4558889276674`

Frozen causal object:

- dominant plane: `P3`;
- response-blind control: `P5`;
- intervention layer: `35`;
- existing target offset: `+2`.

No plane, layer, direction, control, or intervention sign may be reselected from
AVeriTeC responses.

---

## 11. External conditions

Use exactly three full-model conditions per compatible claim:

1. `native`
2. `dominant_neutralized`
3. `dominant_control`

`dominant_control` uses the same frozen matched-control construction as the completed
370M behavioral bridge:

- subtract the native P3 component;
- insert the P5 control component using the native P3 `(a,b)` coefficients.

No `dominant_restored` forward is needed.

The completed behavioral protocol established that exact restoration has zero net
correction and is state-identical to native at the intervention coordinate.

Therefore the external primary contrast may use native directly without spending a
redundant full-model forward.

Expected forward budget if the token gate passes `462/462`:

`462 × 3 = 1386 full-model forwards`

No training.

No backward pass.

---

## 12. Primary endpoint

For compatible AVeriTeC example `i` with mapped correct class `y_i`, define final
three-way correct-class margin:

`M_i = z[y_i] - max_{c != y_i} z[c]`

Primary external causal effect:

`D_EXT,i = M_native,i - M_control,i`

Primary hypothesis:

`H0: E[D_EXT] <= 0`

versus:

`H1: E[D_EXT] > 0`

Pre-specified inference:

- one one-sample Student t-test;
- one-sided `greater`;
- family alpha `0.05`;
- exactly `1` primary p-value.

Primary support requires both:

- `mean(D_EXT) > 0`;
- `p < 0.05`.

Result labels:

- pass:
  `AVERITEC_GOLD_EVIDENCE_370M_CAUSAL_TRANSFER_SUPPORTED`
- fail:
  `AVERITEC_GOLD_EVIDENCE_370M_CAUSAL_TRANSFER_NOT_ESTABLISHED`

There is no second scale in this primary family.

---

## 13. Secondary descriptive outputs

No additional inferential p-values.

Record:

- native accuracy over the compatible cohort;
- control accuracy;
- neutralized accuracy;
- `mean(native - neutralized margin)`;
- prediction flip rate versus native for neutralized and control;
- per-label mean `D_EXT`;
- per-label native accuracy;
- fraction of `D_EXT > 0`;
- claim/evidence truncation rates;
- raw and consumed token-length summaries;
- correlation between `D_EXT` and native correct-class margin;
- response-blind anchor-position distribution.

These are descriptive only.

No per-label significance tests.

---

## 14. What this experiment can establish

If supported, the narrow claim is:

> The frozen Mamba-370M causal plane that has a validated synthetic downstream
> behavioral effect also changes correct-class decision margin in the predicted
> direction on real-world AVeriTeC natural-language claim/gold-evidence inputs under a
> pre-specified boundary-anchored transport protocol.

This would directly weaken the objection that the measured mechanism is purely
synthetic-language-specific.

---

## 15. What this experiment cannot establish

Even if supported, it does not establish:

- full AVeriTeC benchmark performance;
- four-class AVeriTeC competence;
- retrieval competence;
- web-search competence;
- full-evidence-context competence beyond the frozen 128-token input;
- semantic identity between `A_IDENTITY` and the external boundary anchor;
- Mamba-1.4B external causal transfer;
- cross-scale natural-language recurrence;
- architecture-independent transfer;
- universal fact-checking causality.

A failure also does not erase the completed synthetic 370M causal result.

It would mean the frozen boundary-anchored natural-language transfer claim was not
established.

No prompt, anchor, plane, layer, label-map, or subset rescue is allowed on the same
evaluation population.

---

## 16. Dataset/license/provenance requirements

The implementation must freeze:

- upstream repository:
  `MichSchli/AVeriTeC`;
- upstream commit:
  `7c62d1ec8df3fb560d6efe2b85fa191135636f81`;
- upstream development-file Git blob:
  `40974243267f395dc583d805d10f043812419249`;
- downloaded source-file SHA256;
- downloaded source byte count;
- derived compatible-cohort SHA256;
- deterministic adapter source identity;
- license:
  `CC BY-NC 4.0`;
- AVeriTeC paper citation metadata.

No external URLs or source documents need to be scraped for this first test.

---

## 17. Static feasibility verdict

Official gold-evidence dataset exists:

`PASS`

Natural-language claim field exists:

`PASS`

Gold question-answer evidence exists:

`PASS`

Three-way label compatibility can be defined without collapsing the fourth class:

`PASS`

ContraMamba 370M three-way downstream head matches the required mapped output space:

`PASS`

Frozen behaviorally validated 370M causal object exists:

`PASS`

Full-model intervention path exists:

`PASS`

Response-blind external anchor can be defined without semantic-anchor search:

`PASS — A_CLAIM_EVIDENCE_BOUNDARY`

Retrieval required:

`NO`

Training required:

`NO`

Backward pass required:

`NO`

GPU required for next step:

`NO`

Remaining pre-response requirement:

`EXACT DATASET PROVISIONING + 462/462 TOKEN GATE`

### Final audit label

`PASS_READY_FOR_AVERITEC_RESPONSE_BLIND_ADAPTER_AND_TOKEN_GATE_IMPLEMENTATION`

---

## 18. Next implementation scope

Implement exactly:

1. pinned AVeriTeC dev downloader/authenticator;
2. deterministic three-label compatible-cohort builder;
3. deterministic gold-QA evidence serializer;
4. frozen 63/64 tokenizer encoder;
5. `A_CLAIM_EVIDENCE_BOUNDARY` token gate;
6. structural/token manifest;
7. tests for source identity, label mapping, fourth-class exclusion, deterministic
   materialization, truncation accounting, and 462/462 fail-closed behavior.

Do **not** implement or execute model inference in the same step.

The model runner and one-test analyzer are authorized only after the structural/token
gate itself is frozen and passes.

No additional authority/spec document is required unless that CPU-only gate exposes a
genuine scientific ambiguity.
