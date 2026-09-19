# ContraMamba Gen4 — Pre-emission Unsupported-Commitment Precursor Design

## Status

- Purpose: preregister the first bounded pre-emission generative precursor experiment before any generation response is inspected.
- Scientific evidence created by this document: none.
- Training: prohibited.
- Generation/evaluation execution: not authorized by this document.
- Model/layer/plane selection reopening: prohibited.
- This design temporarily precedes the previously planned post-synthesis steering experiment by explicit current research instruction.
- Broad “hallucination precursor” wording is not licensed by this first experiment. The first admissible claim is narrower: **pre-emission precursor of an unsupported decisive commitment under a fixed finite generation grammar**.

## 1. Scientific question

Primary question:

> Can the already-frozen Mamba-370M P3 causal coordinate contain prefix-only information about a future unsupported decisive commitment before that commitment is emitted?

The intended evidence chain is:

1. **Temporal precedence** — a frozen internal signal differs before the unsupported commitment is emitted.
2. **Prospective information** — a prefix-only frozen signal carries information about whether an unsupported commitment occurs within a prespecified future horizon.
3. **Causal prevention** — a prefix-only intervention on frozen P3, applied before commitment emission under a separately frozen trigger rule, changes the later unsupported-commitment rate or commitment margin.

Only the conjunction of these three stages can support a bounded precursor claim.

## 2. Existing objects that remain frozen

### Model scale

Primary model only:

- `state-spaces/mamba-370m-hf`
- exact repository-pinned revision already used by the 370M line:
  `589179554943157be31701edd8b4558889276674`

No 130M or 1.4B inclusion in the first precursor experiment.

### Existing causal plane

- selected plane: `P3`
- response-blind control plane: `P5`
- original intervention block: `35`

The P3/P5 selection is not reopened.

### Existing late localization evidence

Existing stagewise work established:

- early readable divergence immediately after the block-35 intervention;
- late aggregate behavioral consolidation around block 47.

This motivates a two-site design without a layer sweep.

### Site lock

- `L_early = 35`
- `L_late = 47`

No blocks 36–46 are scanned for a “best” precursor.

Block 42 is not part of the first experiment.

## 3. Critical cross-layer restriction

The frozen P3 plane is defined in the **block-35 intervention-space representation** used by the existing 370M causal intervention machinery.

Therefore:

- P3 may be measured directly at block 35.
- P3 must **not** be projected naively onto block 47.
- Block 47 is a late propagation/readout site, not a second P3 coordinate system.
- Any late-site observable must be defined in block-47-native coordinates or by a frozen downstream/LM readout applied to the captured block-47 hidden state.

This prevents an invalid cross-layer coordinate identification.

## 4. Why unrestricted free-form hallucination is not the first experiment

For unrestricted natural-language generation, the exact “first unsupported token” is not generally fail-closed because:

- factual commitment is often span-level rather than single-token;
- a prefix can remain semantically ambiguous until later continuation;
- token-level unsupportedness may require an external semantic judge;
- introducing an LLM judge or a newly trained hallucination classifier would add a new learned decision system before the precursor hypothesis itself is tested.

Therefore the first experiment uses a **finite, explicitly parsed commitment grammar**.

A later free-form replication may broaden the claim if the bounded first experiment succeeds.

## 5. Natural-language grounding population

Reuse the already frozen AVeriTeC compatible gold-evidence cohort:

- source: frozen AVeriTeC dev artifact already used in external transfer;
- compatible rows: `N = 462`;
- labels:
  - `REFUTE`
  - `NOT_ENTITLED`
  - `SUPPORT`;
- existing claim/evidence serialization and exact tokenizer bytes remain the source input identity.

No subgroup rescue, new AVeriTeC label family, or new item selection based on precursor outcomes is permitted.

## 6. Generation model reconstruction

The generation implementation may instantiate the causal-LM wrapper from the **same exact pinned snapshot** used by the 370M line.

It must fail closed unless all of the following hold:

1. exact snapshot revision matches the frozen revision;
2. all already-frozen snapshot file byte sizes and SHA256 values match;
3. causal-LM loading reports no random initialization and no unexpected/missing pretrained weights relevant to generation;
4. the causal-LM backbone state is byte/canonical-state equivalent to the already-frozen 370M backbone;
5. the existing frozen 370M backbone canonical state hash is reproduced;
6. tokenizer byte identities match the already-frozen 370M tokenizer;
7. no training or weight update occurs.

The causal LM head is used only as the frozen pretrained generation readout. It is not fine-tuned.

## 7. Commitment grammar

The first experiment must use exactly three canonical commitment alternatives corresponding to:

- `REFUTE`
- `NOT_ENTITLED`
- `SUPPORT`

The exact surface strings and token sequences must be chosen **before any generation response is inspected**.

A CPU/tokenizer-only grammar gate must verify:

1. all three alternatives tokenize deterministically under the exact frozen tokenizer;
2. the alternatives form a finite prefix tree;
3. commitment identity becomes uniquely decidable at a deterministic token position;
4. no alternative is a complete prefix of another;
5. the online parser can determine commitment identity from emitted tokens only;
6. no future token is required to decide that the commitment has occurred.

If this gate cannot be satisfied with the proposed strings, the strings may be changed only using tokenizer structure, not model-generation responses.

Generation is constrained to this finite grammar for the first experiment.

## 8. Operational definition of `t*`

Let generated token time begin after the fixed claim/evidence prompt.

For each generated sequence, define:

`t*` = the first emitted token index at which the online finite-state parser can uniquely determine that the generated commitment is one of the two decisive classes (`SUPPORT` or `REFUTE`).

`NOT_ENTITLED` is an abstention/non-decisive outcome for the primary unsupported-commitment event.

The parser must use only tokens emitted up to the current step.

### Unsupported commitment

An emitted decisive commitment is **unsupported** when:

- gold = `NOT_ENTITLED` and emitted commitment is `SUPPORT` or `REFUTE`; or
- gold = `SUPPORT` and emitted commitment is `REFUTE`; or
- gold = `REFUTE` and emitted commitment is `SUPPORT`.

### Supported decisive control

A decisive commitment is **supported** when:

- gold = `SUPPORT` and emitted commitment is `SUPPORT`; or
- gold = `REFUTE` and emitted commitment is `REFUTE`.

A generated `NOT_ENTITLED` when the gold label is decisive is an abstention/error but is **not** counted as an unsupported decisive commitment in the primary event definition.

Malformed generation is impossible under a correctly implemented finite grammar; if the parser and decoder disagree, execution fails closed.

## 9. Pre-emission alignment

The primary temporal window is:

- `t* - 4`
- `t* - 3`
- `t* - 2`
- `t* - 1`

For every stored internal observation at time `t`:

- the model input may contain only the prompt and generated tokens up to `t`;
- no token at `t+1` or later may participate in the forward state used for that observation;
- future outcome labels may be attached only after generation completes;
- the feature extraction path must not branch on whether the later commitment will be supported or unsupported.

Examples lacking four valid prefix states before `t*` are ineligible by a rule frozen before outcome inspection.

## 10. Early-site observable: block 35 P3

At block 35, reuse the already-frozen intervention-space construction.

For prefix time `t`, capture the exact strong-channel intervention vector `h_t` before any new intervention.

Project onto the frozen P3 basis:

- coefficient `a_t`
- coefficient `b_t`
- frozen plane component
- `P3_COMPONENT_L2_t`

Primary scalar for the first prospective-information test:

`Z35_t = P3_COMPONENT_L2_t`

Rationale:

- no new learned probe;
- no fitted direction;
- no response-dependent orientation;
- exactly the existing frozen P3 plane.

The raw two coefficients may be retained as descriptive diagnostics, but no post-hoc combination of `a_t` and `b_t` may replace the preregistered scalar after outcomes are seen.

P5 may be retained only as the already-frozen response-blind control coordinate.

## 11. Late-site observable: block 47

Do **not** project P3 onto block 47.

At `post_block_47`, capture the native hidden state for the same prefix.

The primary late-site readout is a **frozen causal-LM commitment margin** obtained by applying only the model’s frozen terminal normalization and LM head to the captured post-block-47 hidden representation for the current next-token decision.

No recurrent block after 47 exists.

The exact margin must be defined from the finite commitment grammar before generation responses are inspected.

This late-site quantity is a readout/consolidation observable, not a new causal plane.

## 12. Stage A — Temporal precedence

Goal:

Determine whether the already-frozen early P3 signal differs before unsupported decisive commitment.

Primary comparison:

- unsupported decisive commitments
vs
- supported decisive commitments

at each fixed relative time:

- `-4`
- `-3`
- `-2`
- `-1`

Primary signal:

- block35 `Z35_t`

Late diagnostic:

- block47 frozen LM commitment margin.

No training.

No learned classifier.

No layer sweep.

No arbitrary token scan outside the four preregistered offsets.

The statistical family and multiplicity rule must be frozen in the implementation contract before scientific execution.

## 13. Stage B — Prospective information

Stage B is run only after Stage A artifacts are frozen and interpreted.

The question is prospective:

> From prefix-only internal information at time `t`, is there information about an unsupported decisive commitment within the next `k` emitted tokens?

The first predictor remains the frozen scalar:

`Z35_t`

No new hallucination classifier is trained.

The first predictive evaluation should use a rank/discrimination statistic that does not require fitting a high-capacity predictor, such as a preregistered AUROC of `Z35_t` for the future-event label.

If a calibrated probability model is later desired, it requires a separate calibration protocol and cannot replace the first no-training test.

## 14. Stage C — Causal prevention

Stage C is run only after Stages A and B are frozen.

The intervention is the existing block35 P3 neutralization mechanism.

A valid prevention experiment must satisfy:

- intervention occurs before commitment emission;
- the trigger uses prefix-only information;
- the trigger rule is frozen on a calibration partition or is a fixed time rule independent of future outcome;
- no `t*` from the untreated future may be used online to decide when to intervene;
- P5 remains the response-blind control intervention;
- supported-commitment behavior is monitored to detect nonspecific suppression.

Primary causal endpoints:

1. change in unsupported decisive-commitment rate;
2. change in frozen decisive-token commitment margin.

A reduction in generation length or indiscriminate suppression is not sufficient evidence of mechanism-specific prevention.

## 15. Partition discipline

Before Stage A generation outcomes are inspected, the 462-row cohort must be deterministically partitioned into:

- design/calibration partition;
- confirmatory partition.

The partition must depend only on pre-existing stable row identity and gold label.

It must not depend on:

- generated text;
- P3 values;
- LM margins;
- correctness;
- confidence;
- future precursor outcomes.

Any threshold needed for Stage C must be fixed using only the design/calibration partition and then evaluated on the untouched confirmatory partition.

## 16. Decoding discipline

The generation rule must be deterministic.

Default:

- greedy constrained decoding within the finite commitment grammar;
- sampling disabled;
- temperature-based sampling disabled;
- beam search disabled unless a later design explicitly freezes it before outcomes.

The first experiment is about internal precursor structure, not decoding hyperparameter sensitivity.

## 17. Provenance requirements

Every scientific run must bind:

- full git commit SHA;
- exact model revision;
- exact tokenizer hashes;
- exact model snapshot hashes;
- reconstructed causal-LM backbone canonical hash;
- exact P3/P5 basis identities;
- exact commitment grammar and token sequences;
- exact AVeriTeC cohort artifact identities;
- deterministic partition identity;
- exact decoding configuration;
- exact command SHA256;
- output artifact SHA256 values.

Any mismatch fails closed.

## 18. Prohibited moves

Before the first confirmatory result is frozen, do not:

- train a hallucination classifier;
- train or fine-tune the LM;
- scan layers 35–47 for the best signal;
- project the block35 P3 basis directly onto block47;
- add block42 after seeing response;
- choose commitment strings based on model outputs;
- choose a predictive threshold on confirmatory outcomes;
- rescue a negative result with AVeriTeC subgroups;
- add 1.4B;
- replace the frozen P3 plane;
- optimize generation accuracy;
- tune epsilon/steering strength;
- call a finite-grammar result a general free-form hallucination precursor.

## 19. Stagewise claim ladder

### After Stage A only

Allowed claim, if supported:

`PRE_EMISSION_TEMPORAL_PRECEDENCE_OBSERVED`

Not allowed:

`HALLUCINATION_PRECURSOR_ESTABLISHED`

### After Stages A + B

Allowed claim, if both are supported:

`PREFIX_ONLY_FROZEN_P3_SIGNAL_PRECEDES_AND_PREDICTS_UNSUPPORTED_COMMITMENT`

Still not sufficient for a causal precursor claim.

### After Stages A + B + C

Allowed bounded claim, if all are supported with valid provenance:

`FROZEN_P3_PRECEDES_PREDICTS_AND_CAUSALLY_MODULATES_UNSUPPORTED_GENERATIVE_COMMITMENT`

Even then, the first experiment remains bounded to the finite commitment grammar.

A later unrestricted free-form generation replication is required before broad “hallucination precursor” wording is defensible.

## 20. Relationship to Experiment 3

The previously planned post-synthesis steering experiment remains deferred.

Order is now:

1. pre-emission unsupported-commitment precursor program;
2. freeze and interpret its result;
3. then resume Experiment 3 / steering.

This ordering must not be silently reversed.
