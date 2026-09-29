# ContraMamba: Evidence-Entitlement Modeling for Claim-Evidence Verification

ContraMamba studies whether a claim-evidence verifier is not only correct at the final-label level, but also internally entitled to make that decision from the supplied evidence.

The central distinction is:

> A model can predict the correct final label without having an internally faithful evidence-entitlement path for that label.

ContraMamba therefore separates final judgment from intermediate epistemic signals such as frame compatibility, predicate coverage, evidence sufficiency, authorization, and polarity. Confidence, entropy, or softmax calibration alone are not treated as evidence of epistemic entitlement.

---

## Current status

This README has been reconciled to the current research frontier through the five-scale paper lineage and the independent K-series evidence lineage.

Current evidence parents:

```text
FIVE_SCALE_PAPER_PARENT
b4d20e3fe61ec97af35a699b4bd454a5bd0eecb7

K_SERIES_PARENT
5f5f4d6a80085ad8baf43445475d1c3049535c22
```

Current development branch: `gen5-causal-role-state-ownership`.

Repository-index note: this README describes the current research frontier. The
corresponding code and evidence remain on their named research branches unless
explicitly merged; README synchronization alone does not imply branch-content merge.

| Surface | Status |
|---|---|
| Gen4 standalone native-Mamba / NAME mechanistic branch | **CLOSED / FROZEN** |
| K-series broad precursor line | **NOT SUPPORTED / PARKED** |
| K0-RVG directional-alignment causal contribution | **SUPPORTED IN BOUNDED 130M SETTING** |
| Gen4 × K convergence program | **COMPLETED EVIDENCE PARENT** |
| Five-scale causal-role recurrence program | **COMPLETED EVIDENCE PARENT / PAPER LINEAGE** |
| Fixed plane identity as cross-scale invariant | **NOT SUPPORTED** |
| Exact cross-scale geometric invariance | **NOT ESTABLISHED** |
| Universal scale-independent behavioral relevance | **NOT ESTABLISHED** |
| Fixed-mirror steering utility | **NOT ESTABLISHED** |
| Gen5 development scale | **MAMBA-130M** |
| Gen5 Phase 0 | **FROZEN AT `25a4206`** |
| Gen5 Phase 1 state-update ownership causal design | **FROZEN ON CURRENT GEN5 COMMIT** |
| Gen5 implementation / training / evaluation | **NOT AUTHORIZED** |

The older pre-D A-series material remains below as historical context; it no longer defines the repository's current research frontier.

The active forward question is:

> Can recurrent-state ownership grounded in an already interventionally validated causal role preserve that role better than a matched ownership-null construction?

The first Gen5 success criterion is causal-role integrity, not task-score improvement.

---

## Research thesis

The long-term question is broader than one classifier or one ablation:

> Can neural reasoning systems benefit from explicitly structured semantic roles, structured decision flow, and controlled learning-signal ownership instead of relying entirely on one unconstrained latent computation to discover all of those structures implicitly?

The current semantic skeleton is:

```text
Frame
  -> Predicate
  -> Sufficiency
  -> Polarity
  -> Authorization Decision
```

The project distinguishes three graphs:

```text
G_I = representation / information communication graph
G_G = gradient modification-authority graph
G_D = structured decision / authorization graph
```

The graphs need not be identical. Forward read permission does not automatically imply backward modification permission, and logical decision order does not automatically imply neural owner-to-owner communication.

The durable research documents are:

- [`docs/CONTRAMAMBA_RESEARCH_VISION.md`](docs/CONTRAMAMBA_RESEARCH_VISION.md)
- [`docs/CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md`](docs/CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md)

They are long-term research context, not execution authority.

---

## Historical controlled-generation snapshot (A0-A3)

The completed A0-A3 experiment deliberately held the Mamba encoder frozen. It therefore tested downstream semantic decision structure and a binary gradient-ownership intervention while the base representation was held fixed.

```text
Frozen Mamba encoder
        |
        v
shared hidden representation
        |
        +-- Frame
        +-- Predicate
        +-- Sufficiency
        +-- Polarity
                |
                v
      structured authorization/router
                |
                v
      REFUTE / NOT_ENTITLED / SUPPORT
```

The current generation does **not** establish whether backbone-level semantic-state ownership is beneficial. Encoder/state-space ownership remains a later research axis.

### A0-A3 matrix

| Arm | Router | Gradient ownership | Reason-loss weight |
|---|---|---|---:|
| A0 | `explicit_product` | `joint` | 0 |
| A1 | `conditional_first_blocker` | `joint` | `0.6273209029272248` |
| A2 | `explicit_product` | `explicit_local` | 0 |
| A3 | `conditional_first_blocker` | `explicit_local` | `0.6273209029272248` |

Frozen reason order:

```text
FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED
```

Secondary reasons are multi-label diagnostics only. They are not external classes and are not duplicated into the training loss.

The external final label space remains:

```text
REFUTE / NOT_ENTITLED / SUPPORT
```

The final 3-way CE is router-only. Under `explicit_local`, the final CE receives detached F/P/S and polarity inputs while authorized local losses retain their local owners.

---

## Controlled data and Seed8192 split

The controlled corpus contains:

```text
300 pair groups
3,600 examples
12 intervention families
```

The historical Seed174 split was found incompatible with A1/A3 dev-polarity binary readiness. It was replaced by the frozen Seed8192 pair split.

Seed8192:

```text
train pairs: 240
dev pairs:    60
train rows:  2880
dev rows:     720
```

The split is pair-grouped, so an original pair and its interventions do not cross train/dev partitions.

The resolved common reason-loss weight for A1/A3 is:

```text
0.6273209029272248
```

It was derived from the authorized pooled calibration procedure rather than dev performance, A0 predictions, checkpoint selection, or a downstream hyperparameter sweep.

---

## A0 N=3 baseline result

A0 was rerun cleanly under Seed8192 for seeds 180/181/182 and validated before the factorial comparison.

Mean descriptive performance:

| Metric | Mean |
|---|---:|
| Accuracy | `0.903241` |
| Macro-F1 | `0.803636` |
| NOT_ENTITLED F1 | `0.938313` |
| REFUTE F1 | `1.000000` |
| SUPPORT F1 | `0.472596` |

A0 showed a reproducible structural failure stronger than seed variation:

- errors were concentrated in SUPPORT versus NOT_ENTITLED;
- REFUTE remained perfect in all three A0 seeds;
- many gold SUPPORT rows were blocked as FRAME;
- PREDICATE reason ownership leaked strongly toward FRAME;
- sufficiency and polarity were not the primary A0 failure.

The A0 result justified the matched A1/A2/A3 mechanism experiment. It did not itself establish a causal mechanism.

---

## Seed8192 A0-A3 factorial result

All 12 cells — A0/A1/A2/A3 across seeds 180/181/182 — use aligned 720-row dev populations with matched stable IDs and gold labels.

### N=3 descriptive aggregate

| Arm | Macro-F1 mean | Accuracy mean | NOT_ENTITLED F1 mean | REFUTE F1 mean | SUPPORT F1 mean |
|---|---:|---:|---:|---:|---:|
| A0 | `0.803636` | `0.903241` | `0.938313` | `1.000000` | `0.472596` |
| A1 | `0.800010` | `0.906019` | `0.941072` | `0.992424` | `0.466536` |
| A2 | `0.668195` | `0.790741` | `0.870855` | `0.783787` | `0.349942` |
| A3 | `0.615573` | `0.807407` | `0.909011` | `0.635080` | `0.302627` |

Matched conclusions:

```text
A1 - A0:
mixed / seed-dependent
mean macro-F1 change approximately neutral

A2 - A0:
descriptively harmful in all three seeds

A3 - A1:
descriptively harmful in all three seeds

A3 - A2:
mixed / seed-dependent
conditional routing does not reproducibly rescue explicit_local

factorial interaction:
heterogeneous
no stable beneficial interaction
```

No tested changed arm is promoted by this factorial.

---

## Integrated failure localization

The pre-D integrated matched-row analysis used the frozen 12-cell prediction exports and did not train a model, run inference, or load checkpoints.

### C1 — A0 -> A2

This contrast isolates `explicit_local` under the explicit-product router.

Across the three seeds:

```text
repaired rows: 34
broken rows:  277
```

The largest loss is broad NOT_ENTITLED leakage, but REFUTE/SUPPORT errors are predominantly cross-polarity. C1 does not expose row-level polarity vectors for A0/A2, so the highest supported localization is:

```text
LEVEL_2_AUTHORIZATION_ASSOCIATED
```

This is descriptive localization, not a causal attribution.

### C2 — A1 -> A3

This contrast isolates `explicit_local` under the conditional-first-blocker router.

Previously correct REFUTE rows newly broken by A3:

```text
seed180: 49
seed181: 43
seed182: 45
total:   137
```

Destinations:

```text
REFUTE -> SUPPORT:      100 / 137
REFUTE -> NOT_ENTITLED: 37 / 137
```

The dominant failure is therefore REFUTE-to-SUPPORT rather than only an authorization rejection.

Twenty-four broken REFUTE stable IDs recur in all three seeds. Recurrent failures show a stronger adverse polarity and final REFUTE-vs-SUPPORT margin signature than REFUTE rows retained correct in all three seeds, while earlier q/reason signals do not provide one uniform separator.

Highest supported localization for the dominant C2 REFUTE-to-SUPPORT failure:

```text
LEVEL_1_POLARITY_ASSOCIATED
```

Again, this is descriptive, not causal.

### What is and is not localized

The integrated analysis supports:

- hard `explicit_local` damages final-class discrimination under both router settings;
- REFUTE is the clearest recurrent failure, especially C2 REFUTE -> SUPPORT;
- SUPPORT shows a smaller mirror-like R/S discrimination failure;
- C2 supplies direct polarity-associated evidence;
- C1 supplies broader authorization-associated evidence;
- recurrent C2 REFUTE failures have a stable stronger polarity/final signature.

The analysis does **not** support:

- one single common upstream q edge as the cause of both C1 and C2;
- a pure F/P/S earlier-gating explanation;
- a stable beneficial router x ownership interaction;
- a causal polarity-head diagnosis from association alone.

The strongest bounded interpretation is that the current binary hard isolation creates an **over-isolation-like failure pattern**: downstream coordination is impaired, while the evidence does not justify declaring the broader Gradient Ownership research axis false.

---

## Historical O-series context

The O-series asks whether epistemic-risk or insufficiency-sensitive signals already exist in native Mamba dynamics before constructing stronger architectural ownership.

The long-term research order is:

```text
observe existing dynamics
-> test precursor predictability
-> introduce architectural structure only if justified
```

or, equivalently:

```text
discover
-> disentangle
-> control
```

### O0b

The matched-control O0b study found a **narrow sufficiency-sensitive precursor clue** in native Mamba hidden-state proxies under the frozen matched-control design.

This does not establish a hallucination detector, causal mechanism, statistical significance, population generalization, or optimal anchor/layer.

### O0c

O0c directly instrumented native selective-SSM recurrent state trajectories with validated provenance and complete native-state capture.

The broad precursor hypothesis was not supported across the pre-registered broad anchor/layer pattern. A strong terminal-localized recurrent-state separation was observed, but it was heterogeneous across pairs and is not promoted post hoc into a broad precursor claim.

Current bounded O0c conclusion:

```text
BROAD_NATIVE_PRECURSOR_NOT_SUPPORTED
TERMINAL_LOCALIZED_RECURRENT_STATE_SEPARATION_OBSERVED
```

O0c is scientifically closed/parked at this stage. No O1 promotion is implied.

### O-series <-> A-series connection

The current working synthesis is intentionally a hypothesis, not an established claim:

```text
O-series:
Where and how does insufficiency-sensitive information appear?

A-series:
How should explicit semantic decision structure and learning-signal authority use that information?
```

O0b suggests that useful information can exist in shared hidden representations, while O0c does not support a simple broad localization in native recurrent states. The A-factorial independently shows that hard downstream gradient isolation harms task discrimination. Together, these results motivate testing controlled rather than absolute learning-signal ownership without assuming that useful semantic information must live in completely isolated latent streams.

---

## Long-term architecture trajectory

The original Research Vision used prospective Generation 1-6 labels written before the later Gen4, K-series, convergence, and five-scale evidence existed. Those labels remain historical planning context but no longer define forward generation numbering.

Current evidence-aware lineage:

```text
Generation 1
  -> Generation 2
  -> Generation 3
  -> Generation 4 standalone
       reason-router / native-Mamba mechanistic program
       through Phase F and NAME branch closure

K-series standalone
       native-state dynamics
       K0-RVG localization
       directional-alignment causal falsification
             |
             +----> Gen4 × K convergence program
                       transport / specificity
                       necessity / restoration
                       structured residual causality
                              |
                              v
                    Cross-scale / five-scale
                    publication program
                              |
                              v
Generation 5
       Causal-Role-Grounded State Ownership
```

K-series, Gen4 × K convergence, and the cross-scale/five-scale paper are scientific programs, not extra architecture-generation numbers.

The old prospective labels `Generation 5 = structured reason calibration` and `Generation 6 = structured state-space backbone` are not reserved forward identities. Structured reason calibration remains an optional module; state-space ownership remains central but must inherit the mechanistic evidence accumulated after that roadmap was written.

The initial Gen5 development scale is Mamba-130M because it has the deepest overlapping causal evidence. This does not make 130M coordinates universal.

The Gen5 invariant candidate is `operational causal role`, not principal-plane number, fixed direction, fixed channel identity, or universal coordinates.

Gen5 must preserve:

`causal role != geometric realization != objective-conditioned functional readout != behavioral utility`

and:

`mechanistic validity != steering utility`

The first Gen5 question is causal-role preservation under a minimal ownership intervention. Cross-scale generality, behavioral improvement, semantic-state identity, and confident-error prediction remain separate later questions.

---

## Pre-D next-design conclusion

The last pre-D integrated analysis recommends exactly one next design direction:

```text
D1 = continuous / partial gradient ownership
```

Conceptually:

```text
z_down = stopgrad(z) + lambda * (z - stopgrad(z))
```

with the boundary cases:

```text
lambda = 0 -> current hard explicit_local
lambda = 1 -> joint ownership
```

Why D1 before edge-specific ownership:

- hard isolation is harmful under both router settings;
- C1 and C2 localize differently at the exported-signal level;
- dominant C2 REFUTE -> SUPPORT failures are polarity-associated;
- C1 is broader and only authorization-associated at the supported evidence level;
- no single q/reason edge consistently explains the dominant recurrent failure;
- therefore immediately selecting one edge for D2 would exceed the evidence.

The pre-D analysis does **not** select a numeric lambda. It does not authorize a sweep, training run, implementation, Kaggle action, or promotion.

The intended falsification principle for a later authorized D1 experiment is that an intermediate ownership setting must reduce newly broken REFUTE rows relative to hard `explicit_local` without compensating deterioration versus A0 in final discrimination and macro-F1. Otherwise the simple hard-boundary explanation is weakened for this setting.

---

## Historical empirical lineage

The active research state above supersedes the old README status that treated A0/A1/A2/A3 as not yet executed. Earlier empirical results remain historical context.

### Stage71 historical empirical primary

The earlier Stage71 recovery remained the historical empirical primary of the pre-URP lineage:

| Metric | Clean controlled dev |
|---|---:|
| Accuracy | `0.975` |
| Macro-F1 | `0.964` |
| NOT_ENTITLED predictions | `522` |
| REFUTE predictions | `90` |
| SUPPORT predictions | `108` |

Its Stage73 VitaminC diagnostic remained weak externally and did not establish a solved hallucination-control model.

### Stage99-Stage106

The Stage99-Stage106 branch tested support-floor bridge and threshold/routing recovery ideas. It produced useful diagnostics but no promotable candidate.

Frozen branch conclusion:

```text
STAGE106_KEEP_STAGE71_PRIMARY_CLOSE_STAGE99_TO_STAGE105_BRANCH
```

This branch motivated a move away from repeated bridge/threshold repair toward mechanism-level routing and ownership experiments.

### Stage26-H1

Stage26-H1 remains historical evidence that entitlement geometry matters: treating entitlement as an additive final feature caused SUPPORT collapse, while restoring entitlement as a gate over polarity energies recovered 3-way decision behavior.

### Stage7

Stage7 remains historical evidence that explicit entitlement auditing can expose and constrain failure modes hidden by flat final-label performance.

Historical results are context. They do not override the current Seed8192 evidence.

---

## Reproducibility and authority boundary

ContraMamba separates:

1. code correctness;
2. execution success;
3. artifact/provenance validity;
4. scientific conclusion.

A successful run alone does not establish a scientific claim.

For scientific runs, provenance should connect:

```text
run name
-> frozen authority / execution commit
-> exact command and command SHA256
-> input identities
-> seed / split / arm
-> runtime report
-> prediction artifacts
-> selected checkpoint
-> artifact hashes
-> collection handoff
-> local import audit
-> validated analysis
```

Raw runtime outputs are immutable evidence. Failed runs, blocked executions, provenance failures, non-promoted candidates, and falsified hypotheses are retained when scientifically relevant.

README text is documentation only. It does not authorize implementation, training, evaluation, Kaggle/GPU use, checkpoint loading, inference, promotion, or scientific execution.

---

## Repository structure

| Path | Purpose |
|---|---|
| `src/contramamba/` | ContraMamba models, heads, labels, and losses |
| `scripts/` | Controlled-data builders, trainers, evaluators, diagnostics, and report writers |
| `data/` | Controlled intervention and long-term matched-control datasets |
| `experiments/` | Stage plans and experiment notes |
| `results/` | Seed-level and aggregate result material |
| `docs/` | Architecture, long-term research vision, hypothesis map, and paper-oriented documentation |
| `tests/` | Unit, validation, training-smoke, and reporting tests |
| `reports/` | Frozen specifications, authorities, manifests, imported evidence, validation records, and scientific analyses |

---

## Historical pre-D research claim

The strongest pre-D claim is deliberately bounded:

> In the frozen-encoder Seed8192 factorial, replacing joint downstream learning-signal ownership with the current hard `explicit_local` intervention reproducibly degrades final task performance under both tested router settings. The clearest recurrent degradation is REFUTE/SUPPORT discrimination, especially a polarity-associated REFUTE-to-SUPPORT failure under the conditional-first-blocker router. The exported evidence does not localize one single common upstream edge, so the result argues against the tested hard binary isolation, not against the broader principle of controlled Gradient Ownership.

The router-only A1 comparison is mixed and near-neutral on the mean scale. No tested A1/A2/A3 configuration is promoted as a superior model by the completed factorial.

The broader long-term question remains open:

> How much backward modification authority should downstream objectives have over semantically structured computation, and at which edges or representation levels?

---

## Next milestone

Phase 0 is frozen at commit `25a4206`.

Frozen Phase 0 artifact:

`reports/reason_router_gen5_causal_role_grounded_state_ownership_phase0_scientific_spec_candidate.md`

The initial Gen5 development model is Mamba-130M.

Phase 1 selects `STATE_UPDATE_AUTHORITY` as the first Gen5 ownership dimension
and freezes write protection as the minimal ownership semantic. It explicitly
forbids direct promotion of PP3 or the K-series alignment direction into a
permanent owner identity.

The active next milestone is:

`GEN5_PHASE1B_NATIVE_UPDATE_ROLE_BRIDGE_SPECIFICATION`

Phase 1B must prospectively define the bridge from the validated layer-17
causal-role program to a native layer-22 write/post-state realization before any
trainable Gen5 ownership implementation is authorized.

Phase 1 must choose exactly one minimal ownership dimension for the first identifiable causal test:

1. state-update authority;
2. information-flow authority;
3. gradient ownership.

No Gen5 architecture implementation, training, evaluation, Kaggle execution, owner-count search, layer search, plane search, or channel search is authorized by the current README or Phase 0 candidate.

Historical pre-D conclusions remain valid within their original scope.

The operating rule remains:

```text
Measure first.
Explain second.
Modify third.
```
