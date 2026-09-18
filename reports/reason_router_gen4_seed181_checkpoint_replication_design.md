# Gen4 independent checkpoint/seed replication — seed181 core design

## Scope

This is a checkpoint-controlled replication of the completed seed180 native-state geometry/restoration result.

- reference checkpoint: seed180 / G3-GROUP-D-HALF
- replication checkpoint: seed181 / G3-GROUP-D-HALF
- replication checkpoint SHA256: `afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f`
- replication checkpoint selected epoch: `19`
- architecture/arm/tokenizer/model snapshot/intervention layer/token semantics/epsilon are held fixed.
- only the training seed/checkpoint is changed.

This design does not authorize training or checkpoint mutation.

## Isolation

Preparation branch:

`g4k-seed181-checkpoint-replication-prep`

Preparation base:

`fb6498e52d0410c64d458896b555f4cbdbf5e407`

Historical seed180 runners and artifacts are read-only dependencies. The replication implementation must add new files rather than modifying the historical experiment implementation.

## Stage A — seed181 checkpoint-specific geometry extraction

Reuse the frozen XG2 and XG4 `301..600` populations and the same native-state construction, but load the exact seed181 checkpoint.

For each family, collect the 300 checkpoint-specific `alignment_delta_h` vectors using four baseline forwards per pair.

Forward budget:

- XG2: 1,200
- XG4: 1,200
- Stage A total: 2,400

No response-based family-subspace probe is required to construct the geometry.

## Stage B — exact seed181 principal geometry

CPU-only deterministic construction:

1. normalize the 300 `alignment_delta_h` vectors per family;
2. reconstruct the top-5 second-moment eigenspace for XG2 and XG4;
3. compute the five principal pairs from the SVD of `B2^T B4`;
4. form the exact positive/negative projector-contrast modes in each principal plane.

Model forwards: `0`.

### Response-blind homolog rule

The seed181 causal target is selected before any seed181 causal response is inspected.

Let `H180` be the frozen seed180 PP3 projector and `Pi_k,181` each seed181 principal plane projector. Define

`overlap_k = tr(H180 Pi_k,181)`.

The seed181 **PP3-matched plane** is the unique argmax of `overlap_k`. Ties within numerical tolerance fail closed. The overlap magnitude is reported descriptively; the argmax alone is not treated as proof of geometric identity across checkpoints.

The matched geometric control is the non-matched principal plane with the largest `sin(theta_k)`; ties within numerical tolerance fail closed.

No response magnitude, p-value, restoration result, or finite-epsilon reconstruction result may enter either selection rule.

## Stage C — finite-epsilon principal reconstruction

Use XG1 `901..1200` and the seed181 principal modes.

For each item, directly probe all ten seed181 principal contrast directions at epsilon `0.025` and reconstruct

`Q_principal = (1/5) sum_k sin(theta_k) [J(z_k+)^2 - J(z_k-)^2]`.

The comparison target `Q_native` is the exact-native/restored condition measured in Stage D below, so no additional original-basis native condition is required.

Forward budget: `300 * 10 directions * 4 = 12,000`.

This is a descriptive finite-epsilon reconstruction check. No binary reconstruction threshold or confirmatory p-value is introduced.

## Stage D — homolog restoration replication

Use the same XG1 `901..1200` items intentionally, because the scientific variable under replication is checkpoint identity rather than dataset identity.

For the seed181 PP3-matched plane `H` with orthonormal basis `(h+, h-)`, let native coefficients be

`a=<x,h+>`, `b=<x,h->`.

Let the response-blind matched control plane have orthonormal basis `(c+,c-)` and define coefficient transfer using the same `(a,b)`.

Conditions:

1. `homolog_neutralized`: remove the native homolog component;
2. `homolog_restored`: exact native restoration, equal to the original state;
3. `matched_control_replacement`: insert the same coefficients into the response-blind control plane.

Under each condition, compute the broad seed181 checkpoint-specific XG2-vs-XG4 susceptibility endpoint using the seed181 top-5 bases.

Primary endpoint:

`D_SUF = Q_restored - Q_control`.

Prospectively inherited inference from the original restoration experiment:

- H0: `mean(D_SUF) <= 0`
- H1: `mean(D_SUF) > 0`
- one-sample Student t-test, one-sided
- N=300, df=299, alpha=0.05
- no multiplicity correction because there is exactly one inherited confirmatory causal endpoint.

Positive causal-replication gates inherit the original restoration logic:

1. `mean(Q_restored) > 0`
2. `mean(S_homolog) > 0`
3. `mean(D_SUF) > 0`
4. one-sided p < 0.05

Forward budget: `300 * 3 conditions * 10 directions * 4 = 36,000`.

## Total scientific forward budget

- checkpoint-specific geometry extraction: 2,400
- direct principal reconstruction: 12,000
- restoration replication: 36,000
- total: **50,400 scientific forwards**

No training, backward, logits/task-head evaluation, epsilon sweep, layer sweep, token sweep, checkpoint sweep, rescue selection, or post-outcome plane selection is allowed.

## Interpretation boundary

A positive causal result supports replication of the local restoration-sufficiency signature on the response-blind seed181 plane most geometrically aligned with seed180 PP3. The reported projector overlap separately quantifies geometric correspondence; a positive causal result does not by itself establish checkpoint-invariant geometric identity.

It does not establish model-family universality, scale universality, behavioral sufficiency, or checkpoint-independent equality of plane indices/effect sizes.
