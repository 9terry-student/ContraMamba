# ContraMamba: Evidence-Entitlement Modeling for Claim-Evidence Verification

ContraMamba studies whether a claim-evidence verifier is not only correct at the final-label level, but also internally entitled to make that decision from the supplied evidence.

The central distinction is:

> A model can predict the correct final label without having an internally faithful evidence-entitlement path for that label.

ContraMamba therefore separates final judgment from intermediate epistemic signals such as frame compatibility, predicate coverage, evidence sufficiency, authorization, and polarity. Confidence, entropy, or softmax calibration alone are not treated as evidence of epistemic entitlement.

---

## Current status

This README is a milestone snapshot of the research state **through the last pre-D-series integrated failure analysis**:

```text
PRE-D SNAPSHOT CUTOFF
5347c9e179e8365b591fe62884f8291b30e7f7ff
Freeze Seed8192 integrated factorial failure analysis
```

D-series implementation, execution, or scientific results are intentionally outside this snapshot. The next design direction is recorded only as a pre-D research conclusion; this README does not authorize that work.

Current active research line:

```text
ContraMamba Reason-Preserving Authorization Router
P3-W7 / Seed8192 A0-A3 factorial lineage
```

Current state:

| Surface | Status |
|---|---|
| Canonical P4-L artifact/provenance lineage | **CLOSED / ESTABLISHED** |
| Seed8192 A0 N=3 baseline | **EXECUTED + VALIDATED** |
| Seed8192 reason-loss calibration | **RESOLVED** |
| Seed8192 A1/A2/A3 factorial, seeds 180/181/182 | **EXECUTED + IMPORTED + VALIDATED** |
| Factorial scientific interpretation | **FROZEN** |
| Integrated matched-row failure localization | **FROZEN** |
| Current tested `explicit_local` ownership | **DESCRIPTIVELY HARMFUL** |
| Conditional-first-blocker router-only effect | **MIXED / SEED-DEPENDENT** |
| Stable beneficial factorial interaction | **NOT SUPPORTED** |
| Next pre-D design direction | **D1 continuous/partial gradient ownership** |
| D-series execution/results in this README | **NOT INCLUDED** |

Key repository identities:

| Milestone | Commit |
|---|---|
| A0 N=3 validated-evidence analysis | `dd183f59f4040405c178da193fe99c7c7f3ef57f` |
| A1/A2/A3 factorial execution commit | `3a76c6cd3f6bd8b011317f37938677822ce9191d` |
| Factorial validated-evidence analysis | `6dcef9520af2cb88691628a77b72f3fdd7042cd8` |
| Factorial scientific interpretation | `0894a921bf7ed69151722e3ce2691eb49bf4f40f` |
| Integrated factorial failure localization | `5347c9e179e8365b591fe62884f8291b30e7f7ff` |
| Long-term O0b scientific interpretation | `f1dc559d546d20611d66b27684bbfa0f02afa696` |
| Validated O0c native-state results | `ff2fb076f6e66a34a632515bb8502d8b1c90ad7f` |
| Long-term research vision | `bca6db6de2e1bb5d1b81188b61b2023be20eadd3` |
| Long-term hypothesis map | `56bf9e7dca92d1d7e61ab153038a68aeb21c4017` |

These are repository authority/provenance/evidence identities, not model checkpoint identities.

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

## Current controlled generation

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

## Long-term O-series context

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

The durable Research Vision already separates the current binary experiment from later generations.

```text
Generation 1
binary gradient ownership
joint <-> explicit_local

Generation 2
continuous / partial gradient ownership

Generation 3
edge-specific ownership

Generation 4
adaptive ownership, only if justified

Generation 5
structured reason calibration, only if justified

Generation 6
structured state-space backbone / semantic-state ownership
```

The Hypothesis Map also pre-registers `Over-Isolation` as a failure mode:

> semantic locality improves but capability collapses.

The current A2/A3 result is consistent with an over-isolation-like pattern, but full `Over-Isolation` classification still requires care because semantic locality itself is not uniformly established as improved.

The current result therefore narrows the next question rather than ending the long-term program.

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

## Current research claim

The strongest pre-D claim is deliberately bounded:

> In the frozen-encoder Seed8192 factorial, replacing joint downstream learning-signal ownership with the current hard `explicit_local` intervention reproducibly degrades final task performance under both tested router settings. The clearest recurrent degradation is REFUTE/SUPPORT discrimination, especially a polarity-associated REFUTE-to-SUPPORT failure under the conditional-first-blocker router. The exported evidence does not localize one single common upstream edge, so the result argues against the tested hard binary isolation, not against the broader principle of controlled Gradient Ownership.

The router-only A1 comparison is mixed and near-neutral on the mean scale. No tested A1/A2/A3 configuration is promoted as a superior model by the completed factorial.

The broader long-term question remains open:

> How much backward modification authority should downstream objectives have over semantically structured computation, and at which edges or representation levels?

---

## Next milestone after this snapshot

The evidence-supported next design question is **D1 continuous/partial gradient ownership**.

This README intentionally stops before D-series implementation/execution status. Any D-series design, implementation, training, evaluation, Kaggle run, checkpoint use, or scientific conclusion requires its own applicable authority and provenance.

The operating rule remains:

```text
Measure first.
Explain second.
Modify third.
```
