# ContraMamba K0-RVG Layer-21 Current-Token Residual-Construction Routing-Source
## Validated Evidence Analysis Report Candidate

## 1. Status

**Stage:** validated K0 observational/algebraic scientific evidence.

**Static design freeze:**

`a1dc479b25fb0b8277ff33102773ef2e05f5fc67`

**Implementation / execution commit:**

`845827d3d99de5fdf5409b901b7039197cf1c08e`

**Runner:**

`scripts/longterm_k0_rvg_layer21_current_token_residual_construction_routing_source_audit.py`

**Runner SHA256:**

`788cd3b64883d2d4f8733f0787e0a44de25952e14cf48251b288ac36354388b8`

**Run directory:**

`reports/longterm_k0_rvg_layer21_current_token_residual_construction_routing_source_845827d_v1`

**Execution marker:**

`PASS_LAYER21_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_ROUTING_SOURCE_EXECUTION`

**Independent artifact-validation marker:**

`PASS_LAYER21_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_ROUTING_SOURCE_ARTIFACT_VALIDATION`

This report interprets only the validated common-330, current-token `k=2` artifacts.

It introduces no new model execution, tokenizer work, training, logits, task evaluation, learned geometry, causal intervention, post-hoc search, or K1 evidence.

---

## 2. Scientific question

The immediate parent evidence established that the incoming layer21 residual contribution `ΔR21` independently carries the same layer22 strong-positive / weak-negative routing signature.

The present stage asks:

> How is that already-validated incoming `ΔR21` routing source constructed at the exact previous residual-addition boundary `R21 = R20 + Y20`?

The frozen target quantity is not a new layer20/layer21 normalization.

It is the exact parent incoming-source routing quantity:

`D_in21`

under the same frozen layer22 map:

`s_bar22 · gamma22 · W_H22 / ||ΔX22||²`.

The purpose is therefore to determine whether the parent incoming-layer21 routing signature is:

1. inherited from `ΔR20`;
2. produced by layer20 mixer update `ΔY20`;
3. generated mainly by the additive `R20×Y20` interaction under the fixed downstream map;
4. genuinely mixed.

---

## 3. Frozen parent target

The parent incoming-source population means were:

### All channels

`-0.2341372085181502`

### Strong partition

`+0.9173468619072845`

### Weak partition

`-1.1514840704254348`

This is the exact target reproduced and decomposed in the current stage.

The qualitative parent sign structure is:

- strong:
  positive;
- weak:
  negative.

---

## 4. Exact upstream boundary

The authenticated candidate boundary is:

`R20 -> RMSNorm20 -> Mixer20 -> Y20`

followed by:

`R21 = R20 + Y20`.

The validated execution confirmed:

`layer20_r21_equals_parent_layer21_input = True`.

Therefore the layer20 block output is exactly the same float32 tensor used as the immediate parent's layer21 input.

No approximate cross-stage alignment is involved.

---

## 5. Exact algebra

For one item and role:

`ΔR20 = R20_m - R20_s`

`ΔY20 = Y20_m - Y20_s`

`ΔR21 = R21_m - R21_s`.

Because residual addition executes in float32, define:

`Q_add20_eps
 = epsilon20_m - epsilon20_s`

with:

`epsilon20_b
 = R21_b - (R20_b + Y20_b)`.

Then:

`ΔR21
 = ΔR20
 + ΔY20
 + Q_add20_eps`.

Under the frozen layer22 parent map:

`L_parent(v)
 = gamma22 ⊙ (s_bar22 v)`,

define:

`Q_r20
 = L_parent(ΔR20)`

`Q_y20
 = L_parent(ΔY20)`

`Q_eps20
 = L_parent(Q_add20_eps)`

and:

`Q_in21
 = Q_r20
 + Q_y20
 + Q_eps20`.

After the frozen layer22 hidden projection:

`H_r20 = W_H22 Q_r20`

`H_y20 = W_H22 Q_y20`

`H_eps20 = W_H22 Q_eps20`

`H_in21 = W_H22 Q_in21`.

Using the unchanged parent denominator:

`D_X22
 = ||ΔX22||²`,

the channel energy identity is:

`e_in21
 = e_r20
 + e_y20
 + e_ry20
 + e_eps20`

where:

`e_ry20
 = 2 H_r20 H_y20 / D_X22`.

Paired corr/ctrl contrasts satisfy:

`D_in21
 = D_r20
 + D_y20
 + D_ry20
 + D_eps20`.

This is the scientific decomposition evaluated across all frozen `1536` channels.

---

## 6. Execution validity

The full run completed with:

`model_forward_count = 1344`.

The validated scope remained:

- common DDSSSSS items:
  `330`;
- source block:
  `20`;
- target residual layer:
  `21`;
- downstream parent-map layer:
  `22`;
- relative coordinate:
  `k=2`;
- strong partition:
  `240`;
- weak partition:
  `1296`;
- equal:
  `0`;
- lag0 kernel RMS:
  `0.24383223809052498`.

No scientific conclusion is based solely on execution success.

---

## 7. Independent artifact validation

Independent artifact validation passed.

The validator did not import or execute the scientific runner.

It independently checked:

- exact five-file artifact set;
- no `.partial` sibling;
- runtime HEAD and branch;
- static-design identity;
- runner SHA256 and blob;
- frozen parent evidence identities;
- manifest provenance;
- output file hashes;
- `330` item rows;
- `1536` channel rows;
- `1536` fixed-rank cumulative rows;
- item source closures;
- parent item-level `delta_in_*` reproduction;
- channel-level frozen `mean_d_in` reproduction;
- cumulative frozen `cumulative_mean_d_in` reproduction;
- item↔channel aggregate bridge;
- channel↔cumulative recurrence;
- summary↔item aggregate bridge;
- forbidden-action flags.

This closes execution, artifact validity, and provenance separately from scientific interpretation.

---

## 8. Validated artifact identities

### Item metrics

`58be748a721be37971f45bb4f5a2996f240fe91a0cd6ba99f33c0985b7538470`

### Channel summary

`23f19995bf06d7fd10048e13411cbd0d4ec75676a99ae9d7032889f24fcb6537`

### Fixed kernel-rank cumulative

`81845540f4da7d5a43e1dcfa7893177a9ad88b619de8dff4a89e1be10477be4a`

### Summary

`d65af91038d05bf3c92533ff8b48a0d495bd862dcbb779813bf56d337ab1855e`

### Execution manifest

`5268703eb069ad2d385bf15a2f9456adfd583a9fae5fda97c0e646ee05cc2300`

---

## 9. Numerical validity

Validated maxima:

### Branch residual-add reconstruction

`0.0`

against:

`1e-7`.

### Difference-relative diagnostic

`4.924787218651847e-07`.

This was preregistered as diagnostic-only because it is cancellation-sensitive.

### Explicit residual-add error-difference identity

`0.0`

against:

`5e-12`.

### Maximum channel source closure

`2.3522850334245504e-15`.

### Maximum parent item incoming-source reproduction residual

`0.0`.

### Maximum parent channel `mean_d_in` reproduction residual

`0.0`.

### Maximum parent cumulative `mean_d_in` reproduction residual

`0.0`.

The parent scientific quantity is therefore reproduced exactly at item, channel, and cumulative levels.

The numerical bridge is negligible relative to scientific components.

---

## 10. Main decomposition: all channels

Parent incoming-source total:

`-0.2341372085181502`.

Components:

### `R20` source

`-0.7142140003946674`

### layer20 mixer update `Y20`

`-1.2219687049262402`

### `R20×Y20` interaction

`+1.7020454571466457`

### numerical bridge

`+3.965611148813729e-08`

Largest single scientific component by absolute magnitude:

**interaction**.

Absolute component mass:

`3.6382282021236647`.

Absolute shares:

- `R20`:
  `19.630819198690577%`;
- `Y20`:
  `33.58691750597081%`;
- interaction:
  `46.78226220535444%`;
- numerical bridge:
  approximately `1.09e-6%`.

The all-channel parent total is therefore a small residual after strong cancellation.

The two direct source energies are negative in the corr-minus-ctrl contrast, while the interaction is strongly positive and offsets most of their combined magnitude.

---

## 11. Strong partition

Parent strong total:

`+0.9173468619072845`.

Components:

### `R20`

`+0.10667586921656537`

### `Y20`

`+0.050076264450526`

### interaction

`+0.760594723140676`

### numerical bridge

`+5.0995171374531194e-09`

Largest single scientific component:

**interaction**.

Absolute shares:

- `R20`:
  `11.628738664322918%`;
- `Y20`:
  `5.458814602189937%`;
- interaction:
  `82.91244617758867%`;
- numerical bridge:
  negligible.

All three scientific terms are positive in the strong partition.

However, the interaction accounts for the overwhelming majority of the strong positive total.

Thus the strong side of the parent phenotype is **interaction-dominated**.

---

## 12. Weak partition

Parent weak total:

`-1.1514840704254348`.

Components:

### `R20`

`-0.8208898696112327`

### `Y20`

`-1.2720449693767664`

### interaction

`+0.9414507340059698`

### numerical bridge

`+3.455659435068417e-08`

Largest single scientific component:

**Y20**.

Absolute shares:

- `R20`:
  `27.05291863923244%`;
- `Y20`:
  `41.92100589363113%`;
- interaction:
  `31.026074328303116%`;
- numerical bridge:
  negligible.

The weak negative phenotype is therefore not interaction-driven.

Instead:

- `R20` contributes substantial negative routing contrast;
- `Y20` contributes an even larger negative routing contrast;
- their positive interaction cancels a large portion of those negative direct-source terms;
- the net nevertheless remains negative.

Thus the weak side is **direct-source negative with strong positive interaction cancellation**.

---

## 13. Parent sign-structure carriers

Validated flags:

`r20_carries_parent_sign_structure = True`

`y20_carries_parent_sign_structure = True`

`interaction_carries_parent_sign_structure = False`.

Under the preregistered criterion, a component carries the parent qualitative sign structure only if:

- strong aggregate is positive;
- weak aggregate is negative.

### `R20`

strong:

`+0.10667586921656537`

weak:

`-0.8208898696112327`.

Therefore `R20` carries the parent sign structure.

### `Y20`

strong:

`+0.050076264450526`

weak:

`-1.2720449693767664`.

Therefore `Y20` also carries the parent sign structure.

### Interaction

strong:

`+0.760594723140676`

weak:

`+0.9414507340059698`.

The interaction is positive in both partitions.

Therefore it does **not** independently carry the strong+/weak− parent sign structure.

---

## 14. Scientific outcome classification

The validated outcome is:

# Outcome D: mixed

A more precise characterization is:

# partition-asymmetric mixed construction with strong-side interaction dominance and weak-side direct-source negativity under positive interaction cancellation

This classification is required because no single scientific component describes both partitions.

---

## 15. Why not Outcome A

Outcome A would require the upstream incoming `R20` residual to provide the primary partition-resolved account.

It does carry the parent sign structure.

However:

- strong `R20` contribution is only:
  `+0.10667586921656537`;
- strong absolute share is only:
  `11.63%`;
- strong interaction is:
  `+0.760594723140676`;
- interaction accounts for:
  `82.91%`
  of strong absolute component mass.

Therefore the strong phenotype cannot be described as primarily inherited from `R20`.

Outcome A is rejected.

---

## 16. Why not Outcome B

Outcome B would require layer20 mixer update `Y20` to be the primary partition-resolved carrier.

`Y20` does carry the parent sign structure and is the largest weak negative component.

However:

- strong `Y20` contribution is only:
  `+0.050076264450526`;
- strong absolute share is:
  `5.46%`;
- strong positive routing is overwhelmingly interaction-driven.

Therefore `Y20` is important but cannot alone describe the complete strong+/weak− phenotype.

Outcome B alone is rejected.

---

## 17. Why not Outcome C

Outcome C would require interaction to provide the primary descriptive account of the parent phenotype.

Interaction is:

- largest all-channel component;
- dominant strong component.

But interaction is:

- positive in strong;
- positive in weak.

It therefore does not independently carry the parent strong+/weak− sign structure.

In weak channels it acts in the opposite direction to the net parent phenotype by canceling direct-source negativity.

Outcome C alone is rejected.

---

## 18. Why Outcome D

The parent phenotype is built differently in its two frozen partitions.

### Strong side

The positive total is primarily generated by positive `R20×Y20` interaction.

### Weak side

The negative total is primarily generated by negative direct-source terms, especially `Y20`, with `R20` also substantial, while positive interaction partially cancels them.

Hence the mechanism is partition-asymmetric.

The same algebraic components play different roles across the frozen downstream partitions.

This is a genuine mixed result, not merely a near-tie among similar same-sign components.

---

## 19. Strong-side interpretation

The strong total is:

`+0.9173468619072845`.

Interaction alone contributes:

`+0.760594723140676`.

This means approximately `82.9%` of absolute strong scientific component mass is interaction.

The direct source terms together contribute only:

`+0.15675213366709137`.

Therefore:

> The validated strong-positive incoming-layer21 routing phenotype is mostly not present as separate `R20` or `Y20` routing energy. It emerges from their joint alignment under the frozen downstream layer22 map.

This remains algebraic/observational.

It is not a causal interaction claim.

---

## 20. Weak-side interpretation

The weak total is:

`-1.1514840704254348`.

Direct source terms sum to:

`-2.092934838987999`.

Interaction contributes:

`+0.9414507340059698`.

Therefore the weak total is the residual after substantial positive interaction cancellation.

The largest individual weak component is:

`Y20 = -1.2720449693767664`.

Thus:

> Weak-channel suppression is already present in direct upstream sources, especially the layer20 mixer update, while their interaction counteracts rather than creates that suppression.

---

## 21. All-channel cancellation

All-channel direct sources sum to:

`-1.9361827053209076`.

Interaction contributes:

`+1.7020454571466457`.

The net is only:

`-0.2341372085181502`.

Thus about most of the direct negative magnitude is cancelled by positive interaction.

The all-channel scalar alone would therefore obscure the partition-resolved structure.

This validates the decision to preserve the preregistered strong/weak partition rather than interpreting only the total.

---

## 22. Relation to the immediate parent stage

The parent `R22 = R21 + Y21` stage found coherent same-sign accumulation:

- incoming `R21`:
  strong+/weak−;
- layer21 update:
  strong+/weak−;
- interaction:
  strong+/weak−.

The present `R21 = R20 + Y20` stage is qualitatively different.

Here:

- `R20`:
  strong+/weak−;
- `Y20`:
  strong+/weak−;
- interaction:
  strong+/weak+.

Thus one boundary upstream, the system transitions from:

**partition-asymmetric cancellation**

to the downstream parent boundary's:

**coherent same-sign accumulation**.

This is an important localization result.

---

## 23. Integrated current chain

The validated K0 chain is now:

`ΔR20`
+
`ΔY20`
+
their interaction
→ `ΔR21`
→ `ΔR21 + ΔY21`
→ `ΔR22`
→ layer22 RMSNorm mixed reshaping
→ `ΔX22`
→ fixed `W_H22` directional routing
→ corr-enriched strong-kernel channel exposure
→ selective lag0 convolution transfer
→ downstream U/write/state separation.

The newly localized stage shows:

- strong routing positivity at `R21` is mainly interaction-generated;
- weak routing negativity is carried mainly by direct sources and partially cancelled by interaction.

---

## 24. What this result does not establish

This evidence does not establish:

- causal necessity of `R20`;
- causal sufficiency of `R20`;
- causal necessity of `Y20`;
- causal sufficiency of `Y20`;
- causal interaction;
- semantic meaning of hidden coordinates;
- task performance effect;
- model-general behavior;
- cross-seed generality;
- K1 claims.

No intervention was executed.

No tokenizer was invoked.

No logits or task heads were read.

No training occurred.

No PCA/SVD or learned geometry was introduced.

---

## 25. Code correctness, execution, artifact validity, science

These remain separate.

### Code correctness

Static preflight passed before implementation freeze.

Implementation freeze:

`845827d3d99de5fdf5409b901b7039197cf1c08e`.

### Execution success

Full execution:

`PASS_LAYER21_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_ROUTING_SOURCE_EXECUTION`.

### Artifact/provenance validity

Independent validation:

`PASS_LAYER21_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_ROUTING_SOURCE_ARTIFACT_VALIDATION`.

### Scientific conclusion

Only after those gates passed is Outcome D accepted.

---

## 26. Validated scientific conclusion

The validated conclusion is:

> **At common-330 current-token `k=2`, the frozen incoming-layer21 routing source `D_in21` is constructed by a partition-asymmetric mixed mechanism at `R21 = R20 + Y20`. The strong-positive component is overwhelmingly dominated by positive `R20×Y20` interaction (`+0.76059`, ~82.9% of strong absolute scientific component mass), whereas the weak-negative component is driven by negative direct-source terms, especially `Y20` (`-1.27204`) and also `R20` (`-0.82089`), with a large positive interaction (`+0.94145`) partially cancelling that suppression. `R20` and `Y20` each independently preserve the parent strong+/weak− sign structure; the interaction does not, because it is positive in both partitions. The correct classification is therefore Outcome D: mixed, specifically partition-asymmetric mixed construction with strong-side interaction dominance and weak-side direct-source negativity under positive interaction cancellation.**

This conclusion is observational/algebraic, not causal.

---

## 27. Compact quantitative summary

| Partition | Parent total | R20 | Y20 | Interaction | Numerical bridge | Largest component |
|---|---:|---:|---:|---:|---:|---|
| all | -0.234137209 | -0.714214000 | -1.221968705 | +1.702045457 | +3.97e-08 | interaction |
| strong | +0.917346862 | +0.106675869 | +0.050076264 | +0.760594723 | +5.10e-09 | interaction |
| weak | -1.151484070 | -0.820889870 | -1.272044969 | +0.941450734 | +3.46e-08 | Y20 |

Absolute component shares:

| Partition | R20 | Y20 | Interaction |
|---|---:|---:|---:|
| all | 19.63% | 33.59% | 46.78% |
| strong | 11.63% | 5.46% | 82.91% |
| weak | 27.05% | 41.92% | 31.03% |

Sign-structure carrier:

| Component | Strong sign | Weak sign | Carries parent strong+/weak− structure |
|---|---:|---:|---|
| R20 | + | - | yes |
| Y20 | + | - | yes |
| interaction | + | + | no |

---

## 28. Next K0 scientific branch after evidence freeze

The current result creates a genuine partition-dependent divergence.

For the original strong-kernel routing question, the most distinctive unresolved fact is:

> Why is the `R20×Y20` interaction so strongly positive in the frozen strong partition?

This interaction contributes approximately `82.9%` of strong absolute scientific component mass and is the dominant generator of the strong-positive side of the phenotype.

The next K0 stage should therefore **not automatically peel to `R20 = R19 + Y19`**.

Instead, after this evidence is frozen, the narrowest next scientific question should localize the exact strong-partition interaction geometry under the already-frozen layer22 map.

The target quantity is:

`D_ry20,strong`

derived from:

`2 H_r20 H_y20 / ||ΔX22||²`.

A future static design should determine whether the strong interaction advantage is associated primarily with:

- larger role-dependent `H_r20` magnitude;
- larger role-dependent `H_y20` magnitude;
- stronger same-sign/channelwise alignment between them;
- or a mixed magnitude×alignment structure,

without learned projections or causal claims.

The weak-side `Y20` source remains a valid secondary branch, but it is not the first branch because the long-running K0 chain is specifically localizing the corr-enriched **strong-kernel exposure** mechanism.

No K1 transition is authorized.

---

## 29. Freeze recommendation

Freeze exactly:

1. item metrics JSONL;
2. channel summary JSONL;
3. fixed kernel-rank cumulative JSONL;
4. summary JSON;
5. execution manifest JSON;
6. this validated evidence analysis report.

Do not include the unrelated K1 files.

---

## 30. Final status

**Execution:** PASS.

**Independent artifact validation:** PASS.

**Parent item reproduction:** exact.

**Parent channel reproduction:** exact.

**Parent cumulative reproduction:** exact.

**Numerical bridge:** negligible.

**Scientific classification:** Outcome D — mixed.

**Refined interpretation:** partition-asymmetric mixed construction with strong-side interaction dominance and weak-side direct-source negativity under positive interaction cancellation.

**Next K0 target after evidence freeze:** strong-partition `R20×Y20` interaction geometry under the frozen downstream map.

**K1:** not authorized.
