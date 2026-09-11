# Generation-3 edge-specific ownership first-pass validated-evidence analysis

```text
VERDICT = PASS
PHASE = GEN3_FIRST_PASS_READ_ONLY_VALIDATED_EVIDENCE_ANALYSIS
PRIMARY_INTERPRETATION = DISTRIBUTED_OWNERSHIP_DEPENDENCE
SECONDARY_INTERPRETATION = GLOBAL_CUMULATIVE_OR_INTERACTION_COMPONENT_REMAINS_UNRESOLVED
LOCALIZED_DOWNSTREAM_ADAPTATION_DEPENDENCE = NOT_SUPPORTED
PAIRWISE_EXECUTION_AUTHORIZED_BY_THIS_REPORT = NO
TRAINING_EVALUATION_INFERENCE_CHECKPOINT_LOADING = NOT_PERFORMED
```

## 1. Scope and evidence gate

This report analyzes the completed Generation-3 first-pass matrix only:

- 10 single-edge-half arms `G3-G1-HALF` through `G3-G10-HALF`
- training seeds `180, 181, 182`
- split seed `8192`
- fixed `lambda_probe = 0.5`
- exactly one attenuated edge per arm and nine edges at `1.0`
- frozen Mamba encoder
- `explicit_product`
- `reason_loss_weight = 0`
- historical JOINT/A0 and GLOBAL-HALF/D1 references reused under the previously verified endpoint-equivalence contract.

No checkpoint was opened or deserialized. No new training, evaluation, inference, lambda sweep, pairwise experiment, adaptive ownership experiment, K-series work, or implementation change was performed.

### Evidence packages

The Gen3 analysis package contains 30 runs and 120 non-checkpoint evidence files. Every one of the 120 internal files matches its manifest SHA256 and byte size.

The locally reported outer Gen3 transport ZIP SHA256 was:

`1af505452114d4dcad9d2c25d8e61a64f96f5b53872478faeb5afd763c9860ac`

The uploaded transport container has a different outer ZIP byte hash, but all 120 manifest-bound internal evidence files verify exactly. The outer transport ZIP is therefore treated only as a carrier; scientific admission is based on the verified inner evidence identities.

The A0/D1 reference package verifies exactly:

`538db8b3c146f027ab546c902df8b94f8e04ec2c3e711c9882f910f2c4d23a7a`

All 18 reference files match their manifest SHA256 and byte size.

### Configuration/provenance reconciliation

All 30 Gen3 run provenances reconcile to:

- scientific implementation commit `7e3b50b411a2e6556d0b81435cfe3b1a46236339`
- clean source worktree
- `gradient_ownership_mode = edge_specific`
- exact arm ID
- exact seed
- exact split seed `8192`
- exact ten-entry edge map
- exactly one named edge at `.5`
- all remaining edges at `1.0`
- `reason_router_mode = explicit_product`
- `reason_loss_weight = 0`
- `freeze_encoder = true`
- `frame_downstream_gradient_mode = joint`

All 36 prediction exports (30 Gen3 + 3 A0 + 3 D1) contain the same 720 stable IDs. Per-ID gold label, claim, evidence, intervention identity, pair/source IDs, structural targets, primary reason, and polarity target reconcile exactly.

## 2. Aggregate task-quality result

Three-seed means are shown below. Standard deviation is the sample SD across seeds.

| arm | conceptual edge | accuracy mean ± SD | Δaccuracy vs A0 | macro-F1 mean ± SD | Δmacro-F1 vs A0 |
|---|---|---:|---:|---:|---:|
| A0 / JOINT | reference | .903241 ± .005782 | — | .803636 ± .010359 | — |
| G1 | F→P | .905093 ± .005782 | +.001852 | .799712 ± .004731 | -.003925 |
| G2 | F→S | .897685 ± .002122 | -.005556 | .799780 ± .004737 | -.003856 |
| G3 | P→S | .902315 ± .002891 | -.000926 | .794762 ± .004726 | -.008875 |
| G4 | F→Q | .897222 ± .007349 | -.006019 | .794567 ± .007278 | -.009069 |
| G5 | P→Q | .900463 ± .009454 | -.002778 | .801051 ± .004934 | -.002585 |
| G6 | S→Q | .883333 ± .016839 | -.019907 | .795672 ± .008605 | -.007964 |
| G7 | F→D | .888889 ± .018056 | -.014352 | .798769 ± .002194 | -.004867 |
| G8 | P→D | .898148 ± .008929 | -.005093 | .796083 ± .006359 | -.007554 |
| G9 | S→D | .902315 ± .003208 | -.000926 | .792740 ± .005206 | -.010896 |
| G10 | Q→D | .901852 ± .003208 | -.001389 | .798910 ± .004415 | -.004726 |
| D1 / GLOBAL-HALF | all ten edges .5 | .863889 ± .014096 | -.039352 | .778960 ± .004836 | -.024677 |

No single-edge arm reproduces the GLOBAL-HALF degradation.

G6 (`S_TO_Q`) has the strongest reproducible accuracy loss: all three seeds are below A0, with mean Δaccuracy `-.019907`. Its macro-F1 is also below A0 in all three seeds.

G8 (`P_TO_D`) is below A0 on both accuracy and macro-F1 in all three seeds, but the magnitude is smaller.

G4 (`F_TO_Q`) is below A0 on macro-F1 in all three seeds, again with a smaller magnitude.

G9 has the lowest three-seed mean macro-F1 among single-edge arms, but it is not reproducibly harmful across seeds: seed181 is above A0 on both accuracy and macro-F1. Therefore G9 is not a valid localized winner.

The seed180 numerical G5 high score does not reproduce as a unique or stable effect and must not be interpreted as a best edge.

## 3. Class-specific behavior

Three-seed mean class-F1 deltas relative to A0:

| arm | ΔNE F1 | ΔREFUTE F1 | ΔSUPPORT F1 |
|---|---:|---:|---:|
| G1 | +.001504 | .000000 | -.013277 |
| G2 | -.003833 | .000000 | -.007735 |
| G3 | -.000282 | .000000 | -.026342 |
| G4 | -.003657 | -.001883 | -.021669 |
| G5 | -.001911 | .000000 | -.005844 |
| G6 | -.014581 | .000000 | -.009311 |
| G7 | -.010600 | .000000 | -.004002 |
| G8 | -.003356 | .000000 | -.019305 |
| G9 | +.000377 | -.003766 | -.029300 |
| G10 | -.000779 | .000000 | -.013400 |
| D1 | -.027660 | -.003788 | -.042583 |

The edge effects are heterogeneous rather than one-dimensional.

- G6 and G7 are dominated by NOT_ENTITLED/authorization-side degradation.
- G9 is dominated by SUPPORT degradation and is the only single-edge arm with a repeated REFUTE-F1 deficit, but its aggregate task effect is seed-inconsistent.
- G4 and G8 show smaller mixed downstream effects.
- The GLOBAL-HALF phenotype is larger than every individual effect on both NE and SUPPORT.

This class structure is incompatible with a single universal edge explanation.

## 4. Exact JOINT/A0 matched-row break/repair analysis

A “break” is a row correct under A0 but wrong under the compared arm. A “repair” is a row wrong under A0 but correct under the compared arm.

Aggregated over seeds 180/181/182:

| arm | A0-correct breaks | A0-wrong repairs | net repairs-breaks | NE breaks | REFUTE breaks | SUPPORT breaks |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 14 | 18 | +4 | 7 | 0 | 7 |
| G2 | 34 | 22 | -12 | 28 | 0 | 6 |
| G3 | 18 | 16 | -2 | 8 | 0 | 10 |
| G4 | 31 | 18 | -13 | 23 | 1 | 7 |
| G5 | 14 | 8 | -6 | 11 | 0 | 3 |
| G6 | 63 | 20 | -43 | 62 | 0 | 1 |
| G7 | 48 | 17 | -31 | 47 | 0 | 1 |
| G8 | 23 | 12 | -11 | 17 | 0 | 6 |
| G9 | 19 | 17 | -2 | 7 | 2 | 10 |
| G10 | 19 | 16 | -3 | 13 | 0 | 6 |
| D1 | 111 | 26 | -85 | 107 | 2 | 2 |

G6 supplies the strongest row-level single-edge authorization-side signature: 62 of its 63 A0-correct breaks are NOT_ENTITLED rows. G7 is similar but smaller.

This still does not reproduce D1. D1 has 107 NOT_ENTITLED breaks, whereas no single edge approaches that population.

### D1-break overlap with the union of all single-edge arms

| seed | D1 A0-correct breaks | D1 breaks reproduced by ≥1 single-edge arm | D1 breaks absent from every single-edge arm |
|---:|---:|---:|---:|
| 180 | 37 | 16 | 21 |
| 181 | 23 | 16 | 7 |
| 182 | 51 | 28 | 23 |
| total | 111 | 60 | 51 |

Thus 60/111 D1 breaks are directionally represented somewhere in the single-edge matrix, while 51/111 appear only when all ten edges are attenuated together.

This is direct evidence for both:

1. real distributed single-edge sensitivity; and
2. an additional global cumulative/non-additive component that is not reducible to any observed single edge.

The second point does not identify an interaction pair.

## 5. Cross-seed row reproducibility

Same-edge A0-correct breaks recurring in at least two seeds are not concentrated in one unique edge.

Notable counts:

- G6: 15 recurrent stable IDs, 31 recurrent break occurrences
- G2: 6 recurrent stable IDs, 13 occurrences
- G7: 6 recurrent stable IDs, 12 occurrences
- G4: 2 recurrent stable IDs, 5 occurrences
- G8: 2 recurrent stable IDs, 4 occurrences
- G9: no same-row break recurring in ≥2 seeds

G6 therefore has the clearest repeated row-level sensitivity, especially on NOT_ENTITLED rows, but it is still only a partial component of the global effect.

## 6. Frozen seed182 residual check

The four pre-specified D1 seed182 residuals were compared directly against all ten single-edge arms.

| stable ID | gold | A0 | D1 | single-edge result |
|---|---|---|---|---|
| `generated_fact_157__none` | REFUTE | REFUTE | SUPPORT | only G9 reproduces SUPPORT; G1–G8/G10 are REFUTE |
| `generated_fact_166__paraphrase` | REFUTE | REFUTE | SUPPORT | no single edge reproduces the D1 failure |
| `generated_fact_152__polarity_flip` | SUPPORT | SUPPORT | NOT_ENTITLED | G1–G5/G8–G10 reproduce NE; G6/G7 recover SUPPORT |
| `generated_fact_285__polarity_flip` | SUPPORT | SUPPORT | NOT_ENTITLED | G1–G4/G8–G10 reproduce NE; G5/G6/G7 recover SUPPORT |

The two REFUTE residuals do not share a single responsible edge. The two SUPPORT→NOT_ENTITLED residuals are broadly distributed and are explicitly repaired under G6/G7 despite G6/G7 being the strongest aggregate NE-sensitive edges.

This again rejects a one-edge localization.

## 7. Predefined-category decision

### `LOCALIZED_DOWNSTREAM_ADAPTATION_DEPENDENCE`

**NOT SUPPORTED.**

No single-edge-half arm reproducibly shows a degradation comparable to GLOBAL-HALF and no one arm explains the D1 class or matched-row phenotype.

### `DISTRIBUTED_OWNERSHIP_DEPENDENCE`

**SUPPORTED — PRIMARY FIRST-PASS INTERPRETATION.**

Several single-edge attenuations produce reproducible but smaller harmful effects, with distinct class/row signatures. G6 is strongest on repeated authorization-side damage; G8 and G4 provide smaller repeated task-quality damage; additional effects occur on other edges.

### `CUMULATIVE_OR_INTERACTION_DEPENDENCE`

**PARTIALLY INDICATED BUT NOT IDENTIFIED AS THE SOLE PRIMARY CATEGORY.**

GLOBAL-HALF is materially worse than every single-edge arm, and 51/111 D1 A0-correct breaks are absent from every single-edge condition. Therefore a cumulative/non-additive component remains. However, the individual arms are not all approximately null: several show reproducible smaller harm. The evidence therefore fits distributed dependence with an unresolved cumulative/interaction component better than a pure cumulative-only interpretation.

### `OWNERSHIP_TOLERANT_EDGE`

No edge is promoted to a universal tolerant-edge claim from these three seeds. G1/G5 are weak in aggregate, but each has seed-specific degradation; formal universal tolerance is not established.

### `GEN3_SINGLE_EDGE_PREMISE_UNSUPPORTED`

**NO.**

The single-edge interventions produce interpretable, edge-dependent, class-dependent differences, and the historical references reconcile. What is unsupported is the stronger premise that one edge alone accounts for GLOBAL-HALF.

## 8. Scientific conclusion

The first-pass Gen3 result is:

```text
PRIMARY = DISTRIBUTED_OWNERSHIP_DEPENDENCE

STRONGEST_REPEATABLE_SINGLE_EDGE_SIGNATURE
    = G6 / S_TO_Q
    = authorization-side / NOT_ENTITLED sensitivity

SECONDARY_REPEATABLE_EFFECTS
    = G4 / F_TO_Q
    = G8 / P_TO_D
    = plus smaller or seed-heterogeneous effects on other edges

SINGLE_EDGE_EXPLANATION_OF_GLOBAL_HALF
    = REJECTED

GLOBAL_EFFECT_FULLY_EXPLAINED_BY_OBSERVED_SINGLE_EDGE_BREAKS
    = NO

D1_A0_CORRECT_BREAKS_TOTAL
    = 111

D1_BREAKS_SEEN_IN_AT_LEAST_ONE_SINGLE_EDGE_ARM
    = 60

D1_BREAKS_SEEN_IN_NO_SINGLE_EDGE_ARM
    = 51
```

The scientifically bounded statement is that downstream adaptation rights are not localized to one semantic gradient ingress. Multiple edges contribute smaller, heterogeneous sensitivities, while simultaneous attenuation creates additional degradation not visible under any one-edge perturbation.

This does **not** establish parameter-level orthogonality, a unique causal path through model parameters, semantic state ownership, native Mamba recurrent-state mechanism, an optimal lambda, a production architecture, or a specific interacting edge pair.

## 9. Pairwise/K-series boundary

This analysis does **not** authorize pairwise execution.

The first pass demonstrates an unresolved cumulative/non-additive component, but it does not identify which pair, if any, is mechanistically privileged. Selecting the two numerically worst arms post hoc would violate the frozen edge-identity-only first-pass discipline and would turn the study into best-arm-driven search.

A future pairwise design, if pursued, requires a separate bounded authority with an independently justified selection rule.

K-series remains separate and is neither supported nor authorized by this Gen3 result.

## 10. Repository disposition

Recommended next repository action:

1. freeze this read-only validated-evidence analysis as the Gen3 first-pass report;
2. review the imported Gen3 evidence scope with `cm ship`;
3. stage only the intended report and admissible non-checkpoint evidence files after explicit review;
4. do not stage `selected_checkpoint.pt` merely because it was imported;
5. do not begin new training/evaluation before the first-pass evidence/report freeze is complete.

No files were staged, committed, pushed, reset, cleaned, or deleted by this analysis.
