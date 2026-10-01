# Gen5 Optimization-Path Bypass Stage D
## Downstream Image Equivalence Interpretation Report

### Status

RESULT=PASS_GEN5_STAGE_D_VALIDATED_DOWNSTREAM_IMAGE_EVIDENCE

Execution commit:

`76bef94f642e9c69355bcf0d336a79fff6372266`

Implementation freeze:

`fde945d6266649bfeea9de7f2b9b37b226cae94f`

Run:

`gen5-staged-downstream-image-equivalence-76bef94-r1`

Execution command SHA256:

`3392b9561a235daacee44c3698180c7d2a30cdf416ed1f02e606092ecccfcc2e`

Imported handoff ZIP SHA256:

`cf0e1b571c179b6ee651a01f968a43857d20affbbebd0d446d7fd3e2017079bb`

Imported artifact hashes:

- downstream_image_summary.json:
  `99232e57b26fbffad7e315ef719d15c063e8229e3900af994af4e3466647d842`
- downstream_image_rows.jsonl:
  `0ee29f0eecec9cca825a417230e8c0dadad5ae34df0bd0c73bb3ff37ae3b6ee6`
- run_provenance.json:
  `7b9912bd7a47685d9b3b9450f04f93370bee3b0ac8ea6124a52a4fa8089ab72c`

Execution boundaries:

- 9 cells
- 480 frozen Phase3A dev stressor-domain rows per cell
- 4320 row records
- two independent Tesla T4 workers
- no DDP
- no backward
- no optimizer construction
- no optimizer step
- no training
- no confirmatory 9601..9900 access
- no scientific p-values

## Scientific question

Stage D asks whether the causally validated native R22 write subspace and
the learned final-B write subspace, despite occupying different coordinates
in layer-22 write space, acquire similar downstream functional images.

This is distinct from asking whether the actually learned perturbation uses
R22.

Two objects therefore remain separate:

1. realized learned-perturbation equivalence;
2. local downstream subspace-image equivalence.

## Part 1 — realized learned perturbation

A read-only analysis of the frozen Stage B row-level outputs gave:

- mean `||Delta_R22|| / ||Delta_FULL||`
  = `0.00217506665507`;
- mean relative difference between FULL and R22_REMOVED
  = `0.000302850086818`;
- mean cosine between FULL and R22_REMOVED
  = `0.999999968421`;
- mean nonlinear interaction relative to FULL
  = `0.00212956162378`;
- FULL versus R22_REMOVED prediction agreement
  = `1.0`.

Therefore the actually learned task perturbation is almost entirely preserved
after removal of its R22 component.

This establishes realized functional bypass around R22 in the frozen Stage B
setting.

It does not by itself establish equality of the R22 and learned-B downstream
subspace images.

## Part 2 — downstream image geometry

### Input write-space geometry

Across the nine Phase3A cells, mean R22 versus final-B write-space affinity was:

`0.000263514298234`

The two rank-2 write planes are therefore nearly orthogonal.

Cell principal angles were approximately:

- first angle: `88.46` to `88.95` degrees;
- second angle: `89.80` to `89.92` degrees.

Thus downstream convergence cannot be attributed to the two planes already
being similar at the layer-22 write surface.

### Final backbone hidden surface

Across all 4320 rows:

- effective-rank pair:
  `2x2` on `4320 / 4320`;
- mean affinity:
  `0.160218728268`;
- median affinity:
  `0.165547149232`;
- mean principal angle 1:
  `58.06652384` degrees;
- mean principal angle 2:
  `81.06413442` degrees.

This is substantially more aligned than input write space but remains far
from rank-2 functional equivalence.

### Task-representation surface

Across all 4320 rows:

- effective-rank pair:
  `2x2` on `4320 / 4320`;
- mean affinity:
  `0.316143053645`;
- median affinity:
  `0.34275718893`;
- mean principal angle 1:
  `41.35264831` degrees;
- mean principal angle 2:
  `76.87937456` degrees.

The R22 and learned-B functional images become more similar in the
task-relevant representation than at the final backbone hidden surface.

They nevertheless remain distinct rank-2 planes.

### Decision-primitives surface

Across all 4320 rows:

- effective-rank pair:
  `2x2` on `4320 / 4320`;
- mean affinity:
  `0.710102532662`;
- median affinity:
  `0.697912048897`;
- mean principal angle 1:
  `1.82944211` degrees;
- median principal angle 1:
  `0.99956238` degrees;
- mean principal angle 2:
  `50.88231580` degrees.

The first downstream functional direction becomes almost shared, while the
second direction remains materially different.

Therefore Stage D does not establish complete rank-2 downstream equivalence.

Instead it establishes strong anisotropic functional convergence:
one direction becomes nearly common while the second remains separated.

### Centered logits

Centered three-way logits had:

- mean affinity:
  `0.999246764136`;
- median affinity:
  `0.999998704787`.

These values are descriptive only.

Because the centered three-class logit surface is intrinsically
low-dimensional, it is not used as the primary rank-2 equivalence verdict.

## Rank-collapse check

All three primary downstream surfaces retained effective-rank pair `2x2` on
all `4320 / 4320` rows.

Therefore the observed downstream convergence is not explained by collapse of
both image planes to one-dimensional or zero-dimensional responses under the
implemented rank criterion.

This specifically argues against a simple rank-collapse form of downstream
Jacobian degeneracy.

It does not establish that every broader form of downstream Jacobian
degeneracy is absent.

## Pressure dependence

Pressure means were highly similar.

Mean input affinity:

- P0: `0.000259439022244`
- PR: `0.000255973094285`
- PC: `0.000275130778173`

Mean final-hidden affinity:

- P0: `0.159999712926`
- PR: `0.16172308028`
- PC: `0.158933391597`

Mean task-representation affinity:

- P0: `0.31311315709`
- PR: `0.308398582456`
- PC: `0.326917421388`

Mean decision-primitives affinity:

- P0: `0.708970262889`
- PR: `0.714295769033`
- PC: `0.707041566064`

Thus the downstream convergence pattern is not specific to one Phase3A
pressure condition.

## Finite-difference scale audit

Primary radius:

`0.025`

Audit radius:

`0.05`

Frozen audit rows:

`32 per cell`

Primary surfaces showed strong directional stability.

R22:

- final hidden:
  mean relative difference `0.000844097333027`,
  maximum `0.00180975166627`,
  minimum cosine `0.999998364003`;
- task representation:
  mean relative difference `0.00464741387515`,
  maximum `0.0397668795063`,
  minimum cosine `0.999276485719`;
- decision primitives:
  mean relative difference `0.00421615894393`,
  maximum `0.0986190260629`,
  minimum cosine `0.999307612656`.

Final-B:

- final hidden:
  mean relative difference `0.00202196725238`,
  maximum `0.00248423650161`,
  minimum cosine `0.999996928312`;
- task representation:
  mean relative difference `0.00172084491385`,
  maximum `0.0137574426017`,
  minimum cosine `0.999907621303`;
- decision primitives:
  mean relative difference `0.000525408534382`,
  maximum `0.00426332812891`,
  minimum cosine `0.999997818747`.

The worst relative deviation previously observed over all diagnostic surfaces
was on centered logits, which are not a primary equivalence endpoint.

The primary subspace-image direction estimates are sufficiently stable for
the bounded Stage D interpretation.

## Stage D conclusion

The data do not support the claim that R22 and final-B are globally identical
downstream functional subspaces.

They support a narrower mechanism:

1. R22 and final-B are nearly orthogonal in layer-22 write space.
2. The learned task perturbation itself is almost completely preserved without
   its R22 component.
3. Despite their write-space separation, R22 and final-B downstream image
   geometry becomes progressively more similar through:
   - final backbone hidden states;
   - task representation;
   - decision primitives.
4. At the decision-primitives surface, one functional direction is nearly
   shared while the second remains substantially distinct.
5. This convergence occurs without effective-rank collapse.

Bounded Stage D decision:

`GEN5_STAGE_D_PARTIAL_ANISOTROPIC_DOWNSTREAM_FUNCTIONAL_SUBSTITUTABILITY_SUPPORTED`

This is partial functional substitutability, not complete rank-2 functional
equivalence.

## Stage A–D synthesis

The validated sequence is:

- native R22 is locally causally important in the frozen Mamba computation;
- Stage A shows learned correction geometry belongs to a structured but
  seed-sensitive family of alternate solutions;
- Stage B shows essentially all realized learned task benefit is carried
  outside R22;
- Stage C shows the initial task-loss gradient already bypasses R22 rather
  than first demanding R22 and subsequently rerouting;
- Stage D shows those nearly orthogonal learned write directions can converge
  toward overlapping downstream task-functional coordinates.

The resulting bounded mechanism is:

`NATIVE_CAUSAL_IMPORTANCE_DOES_NOT_IMPLY_OPTIMIZATION_PRIVILEGE`

More specifically, in this frozen Gen5 setting the optimizer can select a
write-space route almost orthogonal to a native-causal subspace while still
recovering overlapping downstream task-relevant functional directions.

## Hypothesis status after Stage D

### H1 — gradient misalignment

Supported.

Stage C directly showed that the initial task-loss-driven B gradient largely
bypasses R22.

### H2 — functional substitutability

Partially supported.

The realized learned perturbation is preserved without R22, and the
downstream subspace images strongly converge.

Complete two-dimensional downstream equivalence is not established.

### H3 — loss shortcut

Not isolated.

No Stage A–D experiment uniquely identifies a specific shortcut in the task
loss.

### H4 — distributed / alternate bypass

Strongly supported in the bounded setting.

Stages A–D consistently show alternate optimization routes that avoid direct
occupation of R22 while retaining downstream task function.

### H5 — downstream Jacobian degeneracy

Simple effective-rank collapse is not supported.

Broader Jacobian degeneracy remains unestablished rather than ruled out.

### H6 — native-computation-specific importance

Consistent with the evidence, but not independently established.

The current experiments show that native causal importance and task
optimization privilege diverge; they do not by themselves prove a general
native-versus-task objective theorem.

## Scope boundary

These results apply to:

- the frozen native Mamba-130M model and checkpoint lineage;
- layer 22;
- the validated R22 subspace;
- the frozen Gen5 correction architecture;
- the frozen Phase3A seeds and pressures;
- the frozen Phase3A dev stressor-domain population.

They do not establish:

- that R22 represents reasoning in general;
- that every optimizer avoids every native-causal subspace;
- that R22 and final-B are globally functionally identical;
- that downstream Jacobians are generally degenerate;
- that the same mechanism holds across models, layers, tasks, or training
  objectives without further evidence.
