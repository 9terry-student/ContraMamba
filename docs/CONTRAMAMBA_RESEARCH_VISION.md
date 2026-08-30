# ContraMamba Long-Term Research Vision

**Status:** LONG-TERM RESEARCH VISION / NON-AUTHORITY
**Purpose:** Preserve the durable research philosophy, long-horizon architectural direction, and scientific guardrails of ContraMamba across semesters, papers, model generations, and chat sessions.
**Authority boundary:** This document does **not** authorize implementation, training, evaluation, promotion, Kaggle execution, or scientific claims. Current user instruction, stage-specific authority/spec/report/manifest, executed evidence, repository policy, and provenance requirements take precedence.

---

## 1. Originating research question

ContraMamba is motivated by a structural question about modern neural reasoning systems:

> Is it always desirable to let a single globally shared latent computation absorb multiple semantically distinct reasoning operations and allow the end-task objective to freely repurpose those representations?

The project does **not** assume that GPT, Transformers, Mamba, or end-to-end learning are categorically wrong. The narrower hypothesis is that some reasoning and authorization tasks may benefit from explicitly structured semantic computation instead of relying entirely on implicit structure inside one shared latent state.

The long-term target is therefore not merely a better classifier. It is an architecture in which the model's information flow, semantic roles, optimization rights, and final decision process can be separately specified and experimentally tested.

---

## 2. Core ContraMamba philosophy

ContraMamba treats reasoning as a sequence of semantically owned computations rather than a single undifferentiated decision state.

The current semantic skeleton is:

```text
Frame
  -> Predicate
  -> Sufficiency
  -> Polarity
  -> Authorization Decision
```

The intended meanings are:

- **Frame:** whether claim and evidence refer to a compatible entity/event/context frame.
- **Predicate:** whether the relevant relation, action, or proposition is actually covered.
- **Sufficiency:** whether the available evidence is sufficient for the requested conclusion.
- **Polarity:** whether authorized evidence supports or refutes the claim.
- **Authorization Decision:** whether the system is entitled to emit SUPPORT/REFUTE or must return NOT_ENTITLED.

The architecture should preserve these meanings strongly enough that they can be tested by interventions, not merely named after the fact.

---

## 3. Three long-term research axes

### Axis I — Structured Decision

The final decision should be composed from explicit semantic factors instead of being delegated entirely to an unconstrained final classifier.

Current research elements include:

- explicit product entitlement;
- conditional first-blocker reason routing;
- reason-specific supervision;
- explicit REFUTE / NOT_ENTITLED / SUPPORT collapse.

The point is not to force one immutable formula forever. The durable requirement is that decision composition remains semantically inspectable and experimentally identifiable.

### Axis II — Gradient Ownership

Forward access to information and backward authority to modify a module are different questions.

A downstream module may be allowed to read an upstream semantic state without automatically receiving the right to reshape the upstream module through its loss.

This motivates explicit control over:

- which loss owns each semantic module;
- which downstream gradients are allowed upstream;
- when gradient sharing is beneficial;
- when gradient sharing produces semantic contamination or shortcut behavior.

The current `joint` versus `explicit_local` comparison is the first controlled test of this axis, not its final form.

### Axis III — Information / State Ownership

The long-term ContraMamba architecture may extend semantic structure into the backbone itself.

The central question becomes:

> What information is stored in which state, which semantic computation may read it, and under what topology may states influence one another?

Potential future directions include:

- claim/evidence state separation;
- semantic state channels;
- semantic selective state updates;
- constrained inter-state communication;
- structured state-space dynamics inside Mamba-like backbones.

This axis is intentionally **not** part of the current A0–A3 experiment.

---

## 4. The two-graph principle

A mature ContraMamba should distinguish two graphs.

### Information Graph — `G_I`

`G_I` specifies forward information flow:

- which state/module can read which upstream representation;
- which semantic channels may communicate;
- which information may reach the final decision.

### Gradient Graph — `G_G`

`G_G` specifies backward modification authority:

- which objective may update which parameters;
- which downstream losses may cross semantic boundaries;
- which gradient edges are blocked, attenuated, or conditionally opened.

The graphs need not be identical.

For example:

```text
Frame state  ------->  Predicate module     # information allowed
Predicate loss --X-->  Frame parameters      # modification authority denied
```

The current `explicit_local` mechanism is an early implementation of this principle: forward values remain available while selected downstream gradients are detached.

---

## 5. Current generation and its deliberate limitations

The current controlled generation is **not** the final ContraMamba architecture.

Its deliberate properties include:

- shared Mamba backbone representation;
- claim/evidence span masks rather than fully separate backbone state streams;
- explicit downstream Frame / Predicate / Sufficiency / Polarity modules;
- structured final decision;
- frozen backbone for the current causal experiment;
- binary gradient-ownership intervention (`joint` versus `explicit_local`).

Therefore the current generation primarily tests:

> What changes when semantic decision structure and gradient ownership are controlled while the shared backbone is held fixed?

It does **not** yet test the stronger claim that backbone-level semantic state ownership is superior to globally shared latent processing.

---

## 6. Role of A0–A3

A0–A3 are causal reference arms, not four candidate final products.

```text
                     Joint ownership      Explicit-local ownership
Explicit product           A0                        A2
First-blocker + reason      A1                        A3
```

Their purpose is to establish a clean coordinate system for later development.

They are intended to reveal:

- whether structured first-blocker/reason supervision changes behavior;
- whether gradient ownership changes behavior;
- whether the two interact;
- which failure modes emerge under each configuration;
- which future architectural revision is scientifically justified.

A0–A3 should remain stable until their authorized evidence is collected. Future model improvements should be motivated by observed failures rather than inserted prematurely into this reference experiment.

---

## 7. Long-term architectural trajectory

The following is a research trajectory, **not an implementation commitment**. Each step requires evidence and separate authority.

### Generation 1 — Controlled reference

```text
Structured semantic heads
+ explicit decision composition
+ binary gradient ownership
```

Goal: identify the first reproducible causal effects.

### Generation 2 — Continuous gradient ownership

Replace the binary choice with controlled downstream-gradient strength.

Conceptually:

```text
z_down = stopgrad(z) + lambda * (z - stopgrad(z))
```

where `lambda = 0` reproduces full local isolation and `lambda = 1` reproduces joint ownership.

Research question:

> Is there a measurable trade-off frontier between semantic specialization and end-task adaptation?

### Generation 3 — Edge-specific ownership

Allow different information/gradient edges to have different ownership policies, for example:

- Frame -> Predicate;
- Frame -> Sufficiency;
- Predicate -> Sufficiency;
- semantic primitives -> final decision.

Goal: identify which downstream gradient paths are helpful and which are destructive.

### Generation 4 — Adaptive ownership

If justified by prior evidence, ownership may become conditional on training state or gradient compatibility.

Possible signals include:

- local/downstream gradient cosine similarity;
- conflict frequency;
- uncertainty;
- semantic calibration error;
- intervention consistency.

Any adaptive mechanism must be evaluated against simple fixed controls to avoid disguising unconstrained end-to-end optimization as semantic ownership.

### Generation 5 — Structured reason calibration

If the first-blocker router is limited by global calibration capacity, extend it while preserving reason semantics.

Potential mechanisms include:

- reason-specific temperatures;
- bounded residual reason biases;
- constrained conditional calibration;
- uncertainty-aware authorization calibration.

The router must not silently collapse into a generic unconstrained classifier.

### Generation 6 — Structured state-space backbone

Move semantic ownership into the encoder/state-space computation itself.

Potential research directions:

1. **Claim/evidence state separation**
   Maintain distinct state streams before controlled comparison or interaction.

2. **Semantic state channels**
   Maintain states associated with Frame, Predicate, Sufficiency, and Polarity rather than relying on one undifferentiated hidden state.

3. **Selective semantic state update**
   Investigate whether different tokens or evidence events should update different semantic states.

4. **Constrained communication topology**
   Prevent semantic channels from becoming a fully mixed latent representation unless evidence shows that such communication is beneficial.

5. **Separate information and gradient topology inside the backbone**
   Extend the `G_I` / `G_G` distinction below the downstream heads.

This generation would test the project's strongest long-term hypothesis about structured latent computation.

---

## 8. What would constitute a meaningful general result

The project should avoid framing its objective as merely “ContraMamba beats baseline by N F1 points.”

A stronger result would establish a reproducible principle such as:

> End-task supervision can produce correct predictions while repurposing semantically intended modules, and explicit control of information and gradient ownership can improve semantic faithfulness, robustness, calibration, or generalization.

The claim should only be broadened beyond ContraMamba when replicated across appropriate settings.

Possible generalization axes include:

- multiple random seeds;
- multiple datasets or intervention families;
- natural or semi-natural fact-verification settings;
- alternative backbones, including Transformer-family models where appropriate;
- other modular/hierarchical decision tasks.

---

## 9. Long-term success criteria

A mature ContraMamba should be judged on more than aggregate accuracy.

### Predictive

The model should remain competitive on the actual task rather than obtaining semantic cleanliness by sacrificing useful decision quality without justification.

### Semantic

The intended modules/states should respond to the phenomena they claim to represent and resist systematic repurposing by unrelated objectives.

### Causal

Architectural claims should be supported by controlled interventions, ablations, or gradient/information-flow evidence that identifies why a component matters.

### Generalizable

The core mechanism should survive at least some changes in seed, data distribution, task, or backbone before being claimed as a broad learning principle.

### Calibrated

Authorization confidence, uncertainty, and reason confidence should correspond meaningfully to observed error and failure structure where calibration is part of the claim.

---

## 10. Scientific discipline and guardrails

Long-term development must not become unconstrained architecture accumulation.

### Preserve causal baselines

Every major generation should retain a well-defined reference model so that improvements remain attributable.

### Evidence before complexity

Do not add continuous ownership, adaptive routing, structured backbone channels, or calibration machinery merely because they appear plausible. Add them to resolve a documented failure or answer a distinct research question.

### Reject algebraic non-novelty

If a proposed mechanism is algebraically or functionally equivalent to an existing one under the relevant objective, treat that as evidence and do not preserve the proposal as novelty.

### Separate prediction quality from semantic correctness

A model can be accurate for the wrong internal reason. Both outcomes should be measured when relevant.

### Separate execution success from scientific conclusion

Successful training, valid artifacts, and reproducible metrics are necessary but do not themselves establish the research claim.

### Do not protect favored hypotheses

If executed evidence contradicts a favored explanation, revise or discard the explanation rather than modifying the experiment post hoc to preserve it.

### Keep authority explicit

This vision document never overrides current execution authority. Each implementation, training run, evaluation, recovery attempt, and promotion decision must remain separately authorized and provenance-bound.

---

## 11. Relationship between URP, papers, and the model

ContraMamba is a long-term research program rather than a project that must terminate at one semester or one publication.

```text
URP completion != ContraMamba completion
paper submission != ContraMamba completion
paper acceptance != ContraMamba completion
```

A paper should be treated as a defensible snapshot of one mature scientific question and its evidence.

A model generation should be frozen when needed for reproducibility, while later generations may continue to address new failures and broader research questions.

This allows a sequence such as:

```text
mechanism identification
-> bounded model revision
-> generalization
-> paper snapshot
-> next architectural generation
-> new scientific question
```

without rewriting the history of earlier experiments.

---

## 12. Repository-memory principle

Long-term continuity must not depend on any single chat session remembering the project perfectly.

The intended memory architecture is:

```text
repository = canonical long-term memory
chat       = current reasoning/workflow interface
handoff    = bootstrap pointer back into repository state
```

A future research session should reconstruct the project from:

1. `cm context`;
2. current research state;
3. active/latest applicable authority;
4. artifact index and executed reports;
5. this long-term research vision;
6. latest weekly/semester research record.

This document provides the durable **why** and long-horizon direction. Stage authorities and executed artifacts provide the current **what is allowed** and **what has actually been shown**.

---

## 13. Interpretation rule for future controllers

When reading this document in a future session:

- preserve the core research questions and terminology unless later evidence explicitly supersedes them;
- do not treat speculative future generations as already selected designs;
- do not authorize implementation or execution from this document alone;
- use current evidence to decide which long-term branch, if any, is scientifically justified;
- preserve negative findings and rejected mechanisms as part of the research lineage;
- keep the large architectural question visible even when the current workstream is deliberately narrow.

The enduring question is:

> Can neural reasoning systems benefit from explicitly owned semantic states, explicitly structured decision flow, and explicitly controlled learning-signal ownership, rather than relying entirely on a globally shared latent computation to discover all of those structures implicitly?

That question is broader than the current A0–A3 experiment, broader than one URP semester, and broader than any single ContraMamba release.
