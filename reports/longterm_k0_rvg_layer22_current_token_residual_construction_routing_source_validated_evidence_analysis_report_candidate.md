# ContraMamba K0-RVG Layer-22 Current-Token Residual-Construction Routing-Source
## Validated Evidence Analysis Report Candidate

## 1. Status

**Stage:** validated K0 observational/algebraic scientific evidence.

**Scientific boundary:**

`R21 -> RMSNorm21 -> Mixer21 -> Y21`

`R22 = R21 + Y21`

followed by the already-frozen parent residual-source map:

`ΔR22`
→ layer22 parent residual-source RMSNorm map
→ fixed `W_H22`
→ frozen strong/weak partition.

**Static design freeze:**

`d6c5827e5fb6309fb03ebec3c8fffacd9bfdb633`

**Implementation / execution commit:**

`ab3c852b3655b587a7476ea1baca3bcc469e6a9b`

**Runner:**

`scripts/longterm_k0_rvg_layer22_current_token_residual_construction_routing_source_audit.py`

**Runner SHA256:**

`973692b8002572e0b229fb5ecf288457b063bae4043ccfeba56fa0c6e7448300`

**Run directory:**

`reports/longterm_k0_rvg_layer22_current_token_residual_construction_routing_source_ab3c852_v1`

**Artifact validation marker:**

`PASS_LAYER22_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_ROUTING_SOURCE_ARTIFACT_VALIDATION`

This report interprets only the validated artifacts from the bounded common-330, current-token `k=2` run.

It does not introduce new model execution, training, tokenizer work, task evaluation, causal intervention, learned geometry, post-hoc channel selection, or K1 evidence.

---

## 2. Scientific question

The parent RMSNorm routing-source stage established that the distinctive layer22 residual-source routing phenotype was already present in the raw layer22 residual-stream difference `ΔR22`.

The unresolved question was therefore:

> How is that parent residual-source strong-positive / weak-negative routing signature constructed at the immediately preceding residual-addition boundary?

The exact boundary is:

`R22 = R21 + Y21`.

The scientific decomposition asks whether the parent residual-source routing phenotype is:

1. inherited primarily from incoming `ΔR21`;
2. produced primarily by layer21 mixer update `ΔY21`;
3. generated primarily by their downstream vector interaction;
4. genuinely mixed.

---

## 3. Parent phenotype being decomposed

The immediate parent validated residual-source contrasts were:

### All channels

`-1.1700471099810297`

### Strong partition

`+1.7958155601281673`

### Weak partition

`-2.965862670109197`

The qualitative parent phenotype is therefore:

- strong:
  positive;
- weak:
  negative.

The present stage must reproduce those exact parent totals before interpreting source terms.

It does.

---

## 4. Exact algebra

For one item and role:

`ΔR21 = R21_m - R21_s`

`ΔY21 = Y21_m - Y21_s`

`ΔR22 = R22_m - R22_s`.

Because runtime residual addition occurs in float32, the exact difference identity uses an explicit numerical bridge:

`ΔR22
 = ΔR21
 + ΔY21
 + Q_add_eps`.

The fixed parent layer22 residual-source map is:

`L_R22(v)
 = gamma22 ⊙ (s_bar22 v)`.

Therefore:

`Q_in
 = L_R22(ΔR21)`

`Q_update
 = L_R22(ΔY21)`

`Q_add_eps_scaled
 = L_R22(Q_add_eps)`

and:

`Q_R22
 = Q_in
 + Q_update
 + Q_add_eps_scaled`.

After fixed layer22 hidden in-projection:

`H_in = W_H22 Q_in`

`H_update = W_H22 Q_update`

`H_eps = W_H22 Q_add_eps_scaled`

`H_R22 = W_H22 Q_R22`.

With the parent denominator:

`D_X = ||ΔX22||²`,

the exact channel energy decomposition is:

`e_R22
 = e_in
 + e_update
 + e_cross
 + e_eps`

where:

`e_cross
 = 2 H_in H_update / D_X`

and the numerical bridge includes all terms involving `H_eps`.

For paired corr/ctrl items:

`D_R22
 = D_in
 + D_update
 + D_cross
 + D_eps`.

This identity is evaluated for all `1536` frozen hidden output channels and then summed over the preregistered all/strong/weak partitions.

---

## 5. Execution validity

The full bounded run completed with:

`PASS_LAYER22_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_ROUTING_SOURCE_EXECUTION`

and:

- model forward count:
  `1344`;
- common cohort:
  `330`;
- source block:
  `21`;
- target residual layer:
  `22`;
- relative coordinate:
  `k=2`;
- strong channels:
  `240`;
- weak channels:
  `1296`;
- equal channels:
  `0`.

No scientific conclusion is based solely on the execution marker.

---

## 6. Independent artifact validation

A separate read-only validator was run outside the scientific runner.

Validation passed with:

`PASS_LAYER22_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_ROUTING_SOURCE_ARTIFACT_VALIDATION`.

The validator independently checked:

- exact five-file artifact set;
- no `.partial` sibling;
- runtime HEAD;
- frozen runner SHA and blob;
- frozen design identity;
- immediate parent provenance;
- handoff/checkpoint/encoder/Mamba provenance;
- `330` item rows;
- `1536` channel rows;
- `1536` cumulative rows;
- output hashes;
- per-item closures;
- parent scalar bridges;
- parent all/strong/weak residual-source reproduction;
- item→summary aggregate bridge;
- channel→item aggregate bridge;
- fixed-kernel-rank cumulative reconstruction;
- forbidden-action flags.

The validation therefore supports artifact/provenance validity independently of runner self-report.

---

## 7. Validated artifact identities

### Item metrics

`063bf6355795c20adde263d8ceb8868be4e5cb42fe5c5fd46da8219cf5b58124`

### Channel summary

`96a1308a5e384e59cd08e4df6e25d406ff6899e663326ecaff2c1702c8adcdfc`

### Fixed kernel-rank cumulative

`3f7be599d6f505a6f2e679bb24920f7ca09b8577f73289cc457a1138b4547f5d`

### Summary

`364d3a961cc103c2760ea7d70a5547af52502d7e5eadcd2c6421e08ee4893ee0`

### Execution manifest

`55f992bddc7bdc232ce5a8c2d6555feb82026e6f14bb86424eed23d369368f4b`

---

## 8. Numerical validity

The validated maxima are:

### Branch residual-add reconstruction

`0.0`

against tolerance:

`1e-7`.

### Difference-relative residual-add diagnostic

`6.175659960095624e-07`.

This quantity was preregistered as diagnostic-only because it is cancellation-sensitive.

It is not a blocking gate.

### Explicit residual-add error-difference identity

`0.0`

against maximum absolute tolerance:

`5e-12`.

### Channel source closure

`4.218847493575595e-15`

against maximum absolute tolerance:

`5e-12`.

### Parent residual-source reproduction

`0.0`.

The numerical bridge is therefore explicitly represented and negligible at the aggregate scientific scale.

---

## 9. Exact layer-boundary identity

The run verified:

`layer21_r22_equals_parent_layer22_rms_input = True`.

This is important because the present stage is not comparing approximately corresponding tensors.

It decomposes the exact float32 layer21 block output that becomes the already-validated layer22 RMSNorm input.

Therefore the upstream scientific chain is continuous:

`R21 + Y21`
→ exact runtime `R22`
→ parent layer22 RMSNorm residual-source term
→ fixed `W_H22`
→ frozen strong/weak routing phenotype.

---

## 10. Parent residual-source reproduction

The new run reproduced the frozen parent residual-source totals exactly.

### All channels

Observed:

`-1.1700471099810297`

Frozen parent:

`-1.1700471099810297`.

### Strong partition

Observed:

`+1.7958155601281673`

Frozen parent:

`+1.7958155601281673`.

### Weak partition

Observed:

`-2.965862670109197`

Frozen parent:

`-2.965862670109197`.

Maximum parent reproduction residual:

`0.0`.

The source decomposition therefore acts on the exact parent scientific quantity rather than a surrogate.

---

## 11. Main source decomposition

### 11.1 All channels

Parent total:

`-1.1700471099810297`

Incoming residual:

`-0.2341372085181502`

Layer21 update:

`-0.3658600939090227`

Incoming×update interaction:

`-0.5700497992643802`

Numerical bridge:

`-8.289476701898003e-09`

Largest single scientific component by absolute magnitude:

**interaction**.

Absolute component shares:

- incoming:
  `0.20010921485199543`
  ≈ `20.01%`;
- update:
  `0.3126883445872145`
  ≈ `31.27%`;
- interaction:
  `0.4872024334760526`
  ≈ `48.72%`;
- numerical bridge:
  `7.084737555595007e-09`.

At the all-channel net level, interaction is the largest individual contribution.

---

## 12. Strong partition

Parent total:

`+1.7958155601281673`

Incoming residual:

`+0.9173468619072845`

Layer21 update:

`+0.5378577660286331`

Incoming×update interaction:

`+0.34061093337907916`

Numerical bridge:

`-1.1868295104541888e-09`

Largest single scientific component:

**incoming residual**.

Absolute component shares:

- incoming:
  `0.5108246531894901`
  ≈ `51.08%`;
- update:
  `0.29950612816792876`
  ≈ `29.95%`;
- interaction:
  `0.189669217981695`
  ≈ `18.97%`;
- numerical bridge:
  `6.608860816423522e-10`.

All three scientific components have the **same positive sign** as the parent strong contrast.

This is coherent accumulation, not cancellation.

---

## 13. Weak partition

Parent total:

`-2.965862670109197`

Incoming residual:

`-1.1514840704254348`

Layer21 update:

`-0.903717859937656`

Incoming×update interaction:

`-0.9106607326434594`

Numerical bridge:

`-7.1026471914438266e-09`

Largest single scientific component:

**incoming residual**.

Absolute component shares:

- incoming:
  `0.38824591645136397`
  ≈ `38.82%`;
- update:
  `0.3047065762840539`
  ≈ `30.47%`;
- interaction:
  `0.30704750486978233`
  ≈ `30.70%`;
- numerical bridge:
  `2.3947997535510706e-09`.

All three scientific components have the **same negative sign** as the parent weak contrast.

Again, this is coherent accumulation rather than cancellation.

---

## 14. Sign-structure result

The validated summary reports:

`incoming_carries_parent_sign_structure = True`

`update_carries_parent_sign_structure = True`

`interaction_carries_parent_sign_structure = True`.

Under the preregistered sign criterion, each scientific component independently has:

- strong:
  `> 0`;
- weak:
  `< 0`.

Therefore the distinctive partition sign structure is **not exclusive to one term**.

This directly rules out a clean single-source interpretation.

---

## 15. Scientific outcome classification

The correct classification is:

# Outcome D: mixed

A more precise description is:

# incoming-residual-led partition signature with substantial layer21-update and additive-interaction contributions

This wording is chosen for the following reasons.

### Why not Outcome A alone

Incoming residual is the largest **single** component in both partition-resolved strong and weak contrasts.

However:

- strong incoming share is only about `51.1%`;
- weak incoming share is only about `38.8%`;
- layer21 update and interaction each independently carry the same strong+/weak− sign structure;
- all-channel net is interaction-dominant.

Therefore the parent phenotype cannot be described as simply inherited from `ΔR21`.

### Why not Outcome B

The layer21 mixer update is substantial:

- strong:
  about `30.0%`;
- weak:
  about `30.5%`;
- all:
  about `31.3%`.

But it is not the largest single component in the distinctive strong/weak partitions.

The phenotype is therefore not primarily newly produced by `ΔY21`.

### Why not Outcome C

Interaction is large and is the largest all-channel component.

But:

- strong interaction share is only about `19.0%`;
- weak interaction share is about `30.7%`;
- both incoming and update independently already carry the correct strong+/weak− signs.

Therefore the phenotype does not require interaction to create its qualitative partition structure.

### Why Outcome D

The strong+/weak− routing phenotype is distributed coherently across all three scientific terms.

No one term uniquely creates the sign structure.

The most discriminative partition-resolved observation is that incoming residual is the largest single component in both strong and weak partitions, but the remaining update and interaction terms are too large and too directionally consistent to be treated as minor corrections.

---

## 16. Coherent accumulation, not compensating cancellation

This stage differs importantly from the preceding RMSNorm decomposition.

At the RMSNorm boundary, substantial components opposed and canceled one another.

At the present residual-addition boundary:

### Strong

`+0.9173`
+
`+0.5379`
+
`+0.3406`
≈
`+1.7958`.

### Weak

`-1.1515`
+
`-0.9037`
+
`-0.9107`
≈
`-2.9659`.

The scientific components reinforce the same directional phenotype.

Thus the residual-addition boundary is best understood as a **coherent accumulation boundary** for the partition signature.

It is not a stage where one term creates the phenotype by canceling an opposing source.

---

## 17. Incoming residual interpretation

The incoming layer21 residual component is:

### Strong

`+0.9173468619072845`

### Weak

`-1.1514840704254348`.

Therefore the strong-positive / weak-negative signature is already present **before the layer21 mixer update is added**.

This establishes an upstream-presence fact:

> At the immediate layer21→22 residual-addition boundary, the incoming residual stream already carries the same qualitative partition routing signature later observed in the full `ΔR22` residual-source term.

This is observational/algebraic.

It does not establish that the incoming residual is causally necessary or sufficient.

---

## 18. Layer21 update interpretation

The layer21 mixer update contributes:

### Strong

`+0.5378577660286331`

### Weak

`-0.903717859937656`.

It therefore also independently carries the parent qualitative sign structure.

The layer21 mixer update is not merely isotropic magnitude added on top of a pre-existing residual pattern.

It is directionally aligned with the same downstream strong+/weak− partition phenotype.

However, because incoming residual remains the largest single partition-resolved component and the overall result is mixed, this stage does not justify entering layer21 mixer internals as the sole next branch.

---

## 19. Interaction interpretation

The additive interaction contributes:

### Strong

`+0.34061093337907916`

### Weak

`-0.9106607326434594`.

It also carries the parent sign structure.

At all-channel level it is the largest component:

`-0.5700497992643802`.

This means the fixed downstream map does not treat incoming and update sources as independent energies.

Their vector alignment materially changes the resulting channel energy.

However, interaction does not create the qualitative sign structure from scratch, because incoming and update terms each already carry it independently.

---

## 20. Numerical bridge interpretation

The aggregate numerical bridge is:

### All

`-8.289476701898003e-09`

### Strong

`-1.1868295104541888e-09`

### Weak

`-7.1026471914438266e-09`.

These values are negligible relative to the scientific components.

The explicit float32 execution bridge therefore closes the algebra without becoming a scientific explanation.

No numerical-artifact blocker is present.

---

## 21. What is now localized

The current validated chain is:

`ΔR21`
plus
`ΔY21`
plus
their interaction
→ exact runtime `ΔR22`
→ layer22 RMSNorm mixed transformation
→ `ΔX22`
→ fixed `W_H22` directional routing
→ corr-enriched strong-kernel channel exposure
→ selective lag0 convolution transfer
→ downstream U/write/state separation.

The new localization adds:

> The layer22 residual-source strong+/weak− phenotype is already present in incoming `ΔR21`, while layer21 mixer update and incoming×update interaction coherently reinforce the same partition direction.

This is narrower and stronger than merely observing that `||ΔR22||` is large.

---

## 22. Relation to the previous RMSNorm result

The previous validated RMSNorm stage found:

- raw residual source:
  strong positive / weak negative;
- RMS scale term:
  negative in both partitions;
- residual×scale interaction:
  positive in both;
- substantial cancellation and reshaping.

The current stage moves one exact boundary upstream and shows that the raw residual source itself is constructed differently:

- incoming residual:
  strong positive / weak negative;
- layer21 update:
  strong positive / weak negative;
- incoming×update interaction:
  strong positive / weak negative.

Therefore:

- residual addition:
  coherent same-sign accumulation;
- layer22 RMSNorm:
  mixed reshaping/cancellation.

These are distinct algebraic regimes.

---

## 23. What the result does not establish

This evidence does not establish:

- causal necessity of `ΔR21`;
- causal sufficiency of `ΔR21`;
- causal necessity of `ΔY21`;
- causal sufficiency of `ΔY21`;
- semantic meaning of individual hidden dimensions;
- performance improvement;
- intervention effect;
- generalization across layers, models, seeds, tasks, or checkpoints;
- K1 claims.

No intervention was run.

No training was run.

No task head or logits were read.

No tokenizer was invoked.

No learned geometry was introduced.

---

## 24. Code correctness, execution, artifact validity, scientific conclusion

These layers remain separate.

### 24.1 Code correctness

Static preflight passed before implementation freeze.

The implementation was frozen at:

`ab3c852b3655b587a7476ea1baca3bcc469e6a9b`.

### 24.2 Execution success

Full execution completed with:

`PASS_LAYER22_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_ROUTING_SOURCE_EXECUTION`.

### 24.3 Artifact/provenance validity

Independent artifact validation completed with:

`PASS_LAYER22_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_ROUTING_SOURCE_ARTIFACT_VALIDATION`.

### 24.4 Scientific conclusion

Only after the preceding layers passed is the present Outcome D interpretation accepted.

---

## 25. Validated scientific conclusion

The validated conclusion is:

> **At common-330 current-token `k=2`, the layer22 raw-residual strong-positive / weak-negative routing phenotype is not generated by a single source at the `R22 = R21 + Y21` boundary. Incoming `ΔR21`, layer21 mixer update `ΔY21`, and their downstream additive interaction all independently carry the same partition sign structure and coherently reinforce it. Incoming residual is the largest single component in both strong and weak partitions, while interaction is largest at the all-channel net level. The correct classification is therefore Outcome D: mixed, specifically an incoming-residual-led partition signature with substantial layer21-update and additive-interaction contributions.**

This conclusion is observational/algebraic, not causal.

---

## 26. Quantitative compact summary

| Partition | Parent total | Incoming | Update | Interaction | Numerical bridge | Largest single scientific component |
|---|---:|---:|---:|---:|---:|---|
| all | -1.170047110 | -0.234137209 | -0.365860094 | -0.570049799 | -8.29e-09 | interaction |
| strong | +1.795815560 | +0.917346862 | +0.537857766 | +0.340610933 | -1.19e-09 | incoming |
| weak | -2.965862670 | -1.151484070 | -0.903717860 | -0.910660733 | -7.10e-09 | incoming |

Absolute scientific component shares:

| Partition | Incoming | Update | Interaction |
|---|---:|---:|---:|
| all | 20.01% | 31.27% | 48.72% |
| strong | 51.08% | 29.95% | 18.97% |
| weak | 38.82% | 30.47% | 30.70% |

---

## 27. Next K0 boundary

The current static design preregistered that for a mixed result the next stage should follow the largest scientifically relevant unresolved branch while preserving the exact partition-resolved sign structure.

For the distinctive parent phenotype, the scientifically discriminative object is the **strong-positive / weak-negative partition signature**, not the all-channel scalar alone.

Incoming residual is the largest single component in both:

- strong;
- weak.

It also independently carries the exact parent sign structure.

Therefore the next K0 question, after this evidence is frozen, should move one residual boundary upstream:

> How is incoming layer21 current-token residual difference `ΔR21` constructed at the immediately preceding residual-addition boundary?

Subject to source authentication, the expected exact form is:

`R21 = R20 + Y20`.

That equation must be authenticated before a new static design is frozen.

The next stage must not yet enter K1.

It must not reopen layer22 RMSNorm, `W_H22`, lag0 convolution, or tokenizer provenance.

---

## 28. Freeze recommendation

The validated run artifacts and this report should be frozen together.

Intended evidence-freeze scope:

1. item metrics JSONL;
2. channel summary JSONL;
3. fixed kernel-rank cumulative JSONL;
4. summary JSON;
5. execution manifest JSON;
6. this validated evidence analysis report.

No unrelated K1 files belong in the freeze.

---

## 29. Final status

**Execution:** PASS.

**Independent artifact validation:** PASS.

**Parent reproduction:** exact.

**Numerical bridge:** negligible.

**Scientific classification:** Outcome D — mixed.

**Refined interpretation:** incoming-residual-led partition signature with substantial layer21-update and additive-interaction contributions.

**Next unresolved K0 boundary after evidence freeze:** upstream construction of `ΔR21`.

**K1:** not authorized.
