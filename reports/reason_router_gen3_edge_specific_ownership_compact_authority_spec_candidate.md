# Generation-3 edge-specific gradient-ownership compact authority

```text
STATUS = AUTHORITY_CANDIDATE / DESIGN_AND_FUTURE_IMPLEMENTATION_EXECUTION_SCOPE
PHASE = GEN3_COMPACT_DESIGN_IMPLEMENTATION_EXECUTION_AUTHORITY_AUTHORING
SCIENTIFIC_EXECUTION = NOT_AUTHORIZED
IMPLEMENTATION = NOT_YET_AUTHORIZED_BY_THIS_RECORD_ALONE; FUTURE BOUNDED SCOPE DEFINED
FROZEN_D1_IMPLEMENTATION = 1abe147484e2155b19d3c973175628d337807501
FROZEN_D1_EVIDENCE = 76eb5b0926aecc8e4afc45a0df09dc83edd7e3d4
FROZEN_D1_SEED182_RESIDUAL_LOCALIZATION = a8e68bb6fb111242628dda2c290983096ca120ca
TOPOLOGY_AUDIT = reports/reason_router_gen3_gradient_topology_minimal_design_audit_candidate.md
TOPOLOGY_AUDIT_SHA256 = 975650b61ca1356ac83c33607668e82d4c6cbdaf8602388262188e7f514fa1fa
```

## 1. Decision, question, and boundary

This is the sole initial Generation-3 authority for the compact single-edge study. It freezes one question: **given that D1 global partial ownership at lambda=.5 substantially repaired hard isolation but remained below joint ownership, which individual downstream semantic gradient edges require full downstream adaptation rights?** Equivalently: does changing exactly one D1-derived owner-to-recipient edge from 1.0 to .5 reproduce any component of the observed GLOBAL-HALF degradation?

The independent variable is **EDGE IDENTITY ONLY**. `lambda_probe = 0.5` is fixed and is the already executed, provenance-bearing D1 interior attenuation. It is neither an optimum, a tuned value, a sweep value, nor a trade-off frontier claim. No other lambda, pairwise attenuation, adaptive ownership, or residual-anecdote-driven edge selection is authorized.

This authority is deliberately compact: routine transitions from implementation to verification to the exact future commands require no new Gen3 design/implementation/execution/analysis authority if this design remains unchanged. It does not itself start implementation, tests, training, evaluation, checkpoint loading, Kaggle work, commit, or push.

## 2. Frozen topology and scientific unit

`F`, `P`, `S`, `Q`, and `D` mean Frame, Predicate, Sufficiency, Polarity, and the structured explicit-product final Decision/final three-way CE. An edge is a direct semantic-owner-to-recipient **gradient ingress**. A tensor occurrence is a recipient call argument, not a source allocation.

|Edge ID|Conceptual edge|Recipient-specific tensor occurrences|Count|
|---|---|---|---:|
|`F_TO_P` (G1)|F -> P|`claim_frame_state`, `evidence_frame_state`, `frame_pair_repr`, `frame_prob`|4|
|`F_TO_S` (G2)|F -> S|`frame_pair_repr`, `frame_prob`|2|
|`P_TO_S` (G3)|P -> S|`predicate_pair_repr`, `predicate_coverage_prob`|2|
|`F_TO_Q` (G4)|F -> Q|`frame_pair_repr`|1|
|`P_TO_Q` (G5)|P -> Q|`predicate_pair_repr`|1|
|`S_TO_Q` (G6)|S -> Q|`sufficiency_repr`|1|
|`F_TO_D` (G7)|F -> D|`frame_prob`|1|
|`P_TO_D` (G8)|P -> D|`predicate_coverage_prob`|1|
|`S_TO_D` (G9)|S -> D|`sufficiency_prob`|1|
|`Q_TO_D` (G10)|Q -> D|`positive_energy`, `negative_energy`|2|
|**Total**|**10 conceptual direct edges**|**all listed occurrences**|**16**|

All occurrences for one conceptual edge receive the same lambda. Occurrences are never split into separate scientific arms. Reused raw owner tensors must receive recipient-specific aliases: for example, changing F->S must not change F->P, F->Q, or F->D merely because a source tensor is shared. Raw owner tensors remain available to owner-local losses.

The following are explicitly not edges in this study: shared frozen-backbone `token_states`; `final_router_inputs_detached`/`local_to_final_router_detached` duplicate observability booleans; inactive auxiliary, bridge, ranking, intervention, EMA, or teacher paths; and the legacy FrameGate hook. The D1 topology audit establishes that the legacy `frame_downstream_gradient_mode=frame_local_only` is inactive/incompatible here and that the backbone is frozen.

## 3. Required edge semantics

For each recipient-specific occurrence on edge `e`, use:

```text
partial_grad(z, lambda_e) = z.detach() + lambda_e * (z - z.detach())
```

It must equal `z` in the forward pass exactly, for every permitted lambda. Its backward derivative across that recipient ingress is `lambda_e`. Edge attenuation therefore affects only the named recipient path; it must not alter forward values, targets, decision composition, labels, source data, split semantics, or owner-local paths.

The intended local-gradient accounting is:

```text
dL_total/dz_owner = dL_owner_local/dz_owner
                  + sum_e lambda_e * dL_downstream_e/dz_owner
```

The sum is over separately controlled downstream recipient paths. This scalar identity specifies edge-wise gradient scaling; it does **not** alone prove parameter-level orthogonality, a unique whole-network pathway, semantic state ownership, or a native Mamba mechanism. Multi-hop coupling remains: a downstream loss may reach an earlier owner through more than one direct ingress. A single-edge arm identifies attenuation of its named direct ingress in an otherwise joint graph, not removal of every downstream influence on that owner.

## 4. Frozen configuration schema and fail-closed rules

Future implementation must expose exactly one explicit ten-entry map, conceptually:

```text
gradient_ownership_mode = "edge_specific"
edge_gradient_lambdas = {
  "F_TO_P": <float>, "F_TO_S": <float>, "P_TO_S": <float>,
  "F_TO_Q": <float>, "P_TO_Q": <float>, "S_TO_Q": <float>,
  "F_TO_D": <float>, "P_TO_D": <float>, "S_TO_D": <float>,
  "Q_TO_D": <float>
}
```

The exact CLI/config spelling may be chosen from the existing trainer only if it remains minimal, unambiguous, and serializes this exact schema. The implementation must fail closed for a missing, extra, or duplicate edge; non-finite lambda; lambda outside `[0,1]`; edge-specific configuration combined with an incompatible legacy ownership mode; or any silent fallback to a global lambda. Legacy `joint`, `explicit_local`, and D1 `partial` modes must retain their exact existing behavior; this study must not refactor those paths merely for uniformity.

## 5. First-pass experiment matrix

Every new arm has exactly one `.5` edge and nine `1.0` edges. The references are included as controls, not permission to rerun them.

|Arm ID|G1|G2|G3|G4|G5|G6|G7|G8|G9|G10|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`G3-JOINT` (REFERENCE J)|1|1|1|1|1|1|1|1|1|1|
|`G2-GLOBAL-HALF` (REFERENCE H; existing D1)|.5|.5|.5|.5|.5|.5|.5|.5|.5|.5|
|`G3-G1-HALF`|.5|1|1|1|1|1|1|1|1|1|
|`G3-G2-HALF`|1|.5|1|1|1|1|1|1|1|1|
|`G3-G3-HALF`|1|1|.5|1|1|1|1|1|1|1|
|`G3-G4-HALF`|1|1|1|.5|1|1|1|1|1|1|
|`G3-G5-HALF`|1|1|1|1|.5|1|1|1|1|1|
|`G3-G6-HALF`|1|1|1|1|1|.5|1|1|1|1|
|`G3-G7-HALF`|1|1|1|1|1|1|.5|1|1|1|
|`G3-G8-HALF`|1|1|1|1|1|1|1|.5|1|1|
|`G3-G9-HALF`|1|1|1|1|1|1|1|1|.5|1|
|`G3-G10-HALF`|1|1|1|1|1|1|1|1|1|.5|

## 6. Fixed coordinate system and references

Preserve the D1 coordinate system exactly: `router=explicit_product`; `reason_loss_weight=0`; `split_seed=8192`; training seeds `180, 181, 182`; same frozen Mamba encoder policy; same dataset and splits; same reason order `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`; diagnostic-only secondary reasons; same final three-way semantics; and `frame_downstream_gradient_mode=joint`, unless the executed D1 contract proves another setting is necessary for exact reproduction. Conditional-first-blocker arms are excluded from the Gen3 first pass.

Existing JOINT/A0 and GLOBAL-HALF/D1 evidence are **CONDITIONALLY_ELIGIBLE_FOR_REUSE**, not automatically reused. Reuse requires later implementation verification that:

1. the all-ones Gen3 vector is exactly forward/backward equivalent to the executed joint endpoint under the controlled harness;
2. the all-.5 Gen3 vector is exactly forward/backward equivalent to executed D1 partial lambda=.5;
3. dataset, split, seed, backbone freeze, router, loss composition, and every other scientific control reconcile exactly; and
4. records distinguish historical reference evidence from newly run Gen3 arms.

If either endpoint fails equivalence, reuse is **BLOCKED** and the controller must stop before scientific execution. Do not silently rerun a reference.

## 7. Future minimal implementation and verification scope

The implementation scope is derived from frozen D1 and is limited to:

- `src/contramamba/modeling_v6b_minimal.py`: edge-map validation, recipient-specific aliases, edge operator, and configuration witness.
- `scripts/train_controlled_v6b_minimal.py`: bounded CLI/config resolution, model forwarding, and unambiguous provenance/run identity serialization.
- `tests/test_reason_router_p2_contract.py`: the focused edge-specific contract tests.

Add a file only if existing provenance/run tooling strictly requires it; otherwise stop and report scope mismatch. No unrelated model behavior may change. Since loss/gradient semantics change, one independent verifier must inspect the implementation and pass before commit/freeze.

Required later verification is:

1. **Forward identity:** identical weights, inputs, and RNG yield the same pre-existing forward values for every allowed edge vector.
2. **Joint endpoint:** all ten `1.0` values match legacy joint forward and backward behavior.
3. **Global-half endpoint:** all ten `.5` values match legacy D1 partial lambda=.5 forward and backward behavior.
4. **Single-edge isolation:** for each `Gi=.5`, all other edges `=1`, only the downstream gradient contribution traversing `Gi` is scaled.
5. **Recipient-specific aliasing:** changing one recipient edge does not attenuate another path from the same owner tensor.
6. **Local-loss preservation:** owner-local gradients are unchanged as downstream edge lambdas change.
7. **Provenance:** every run records mode `edge_specific`, all ten lambdas, arm ID, seed, split seed, implementation commit, and relevant legacy global-ownership fields without contradiction.

## 8. Execution gate and prescribed analysis

Scientific execution remains **NOT AUTHORIZED**. It can begin only after implementation completion; all required tests; independent verifier PASS; manual commit and push; known exact implementation commit; and controller confirmation of reference endpoint reuse. There is no authorization to train/evaluate merely because a static test passes.

Future analysis must separately report aggregate task quality, class-specific behavior, matched-row break/repair structure, provenance validity, and bounded causal interpretation. No promotion may rest on one seed, one class, one residual stable ID, best-arm selection, or post-hoc lambda changes. The seed182 residual report is contextual only and must not select edges.

Predefined interpretations are:

|Category|Predefined evidence meaning|
|---|---|
|`LOCALIZED_DOWNSTREAM_ADAPTATION_DEPENDENCE`|One single-edge-half arm reproducibly has substantial degradation relative to JOINT and directionally explains GLOBAL-HALF.|
|`DISTRIBUTED_OWNERSHIP_DEPENDENCE`|Several single-edge arms have reproducible smaller harmful effects.|
|`CUMULATIVE_OR_INTERACTION_DEPENDENCE`|Individual arms are approximately null/minor while GLOBAL-HALF remains substantially harmful.|
|`OWNERSHIP_TOLERANT_EDGE`|An edge can be attenuated without meaningful degradation, or with bounded improvement.|
|`GEN3_SINGLE_EDGE_PREMISE_UNSUPPORTED`|D1 reference effect does not reconcile/reproduce, or perturbations have no interpretable relation to the established global result.|

Do not infer that stronger semantic ownership is beneficial without semantic-locality evidence. Do not claim an optimal lambda, universal harmful/helpful edge, parameter-level causal pathway beyond the controlled intervention, state ownership, native Mamba mechanism, or production readiness.

## 9. Pairwise and K-series boundaries

Pairwise interaction experiments are **NOT AUTHORIZED**. They are a new design question only after first-pass evidence supports `CUMULATIVE_OR_INTERACTION_DEPENDENCE` or another comparably specific reason; that result does not identify an interaction pair.

K-series is a separate parallel mechanistic-discovery line. K-series evidence neither authorizes nor is consumed as Gen3 causal evidence, and this authority does not modify it. Conversely, Gen3 evidence does not authorize K2/K3 and must not be interpreted as native recurrent-state mechanism evidence.

## 10. Authority provenance and exclusions

This record was reconciled against `docs/CONTRAMAMBA_RESEARCH_VISION.md`; frozen implementation `1abe147484e2155b19d3c973175628d337807501`; frozen D1 evidence `76eb5b0926aecc8e4afc45a0df09dc83edd7e3d4`; frozen seed182 localization `a8e68bb6fb111242628dda2c290983096ca120ca`; and the named topology audit with its stated SHA256. The topology audit establishes the 10-edge/16-occurrence direct-ingress mapping, active-versus-inactive D1 paths, recipient-specific alias requirement, and absence of a presently justified pairwise design.

This authority neither changes existing topology-audit bytes nor authorizes source/test/data/checkpoint mutation, checkpoint loading, training, evaluation, Kaggle, commit, push, lambda selection, pairwise execution, adaptive ownership, conditional-first-blocker Gen3 arms, or K-series work.
