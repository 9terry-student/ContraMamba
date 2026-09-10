# Generation-3 gradient-topology and minimal-design audit

```text
VERDICT = PASS
PHASE = GEN3_READ_ONLY_GRADIENT_TOPOLOGY_AND_MINIMAL_DESIGN_AUDIT
AUTHORITY = current user instruction
EXECUTED_D1_IMPLEMENTATION_BASIS = 1abe147484e2155b19d3c973175628d337807501
FROZEN_D1_EVIDENCE = 76eb5b0926aecc8e4afc45a0df09dc83edd7e3d4
FROZEN_D1_RESIDUAL_LOCALIZATION = a8e68bb6fb111242628dda2c290983096ca120ca
TRAINING_EVALUATION_INFERENCE_CHECKPOINT_LOADING = NOT_PERFORMED
```

## 1. Decision

The provisional decomposition is **correct** if an edge means a direct semantic-owner-to-recipient gradient ingress. The four D1 boundary groups are implementation groups, not four Gen3 edges. They expand to **10 conceptual owner edges** and **16 concrete tensor occurrences**. Each direct edge can be set to lambda=.5 while the other nine are lambda=1, with no forward-value change.

No static redundancy eliminates a direct probe. The minimum first pass is JOINT, the existing GLOBAL-HALF/D1 control, and ten one-edge-half arms. Pairwise interaction is **not justified now**: each direct edge is independently realizable; multi-hop gradients constrain interpretation but do not prevent identification of a direct-edge intervention.

`F`, `P`, `S`, `Q`, and `D` denote FrameGate, predicate coverage, sufficiency, polarity energy, and the explicit-product final decision/final CE. An occurrence is a consumer argument at a call site, not a source allocation.

## 2. Exact executed D1 topology

|D1 boundary group|Conceptual edge|Concrete source tensor occurrences at the recipient|count|D1 lambda=.5 operation|
|---|---|---|---:|---|
|B1 frame -> predicate|F -> P|`claim_frame_state`; `evidence_frame_state`; `frame_pair_repr`; `frame_prob`|4|partial alias at each predicate-call argument|
|B2 frame/predicate -> sufficiency|F -> S|`frame_pair_repr`; `frame_prob`|2|partial alias at each sufficiency-call argument|
|B2 frame/predicate -> sufficiency|P -> S|`predicate_pair_repr`; `predicate_coverage_prob`|2|partial alias at each sufficiency-call argument|
|B3 frame/predicate/sufficiency -> polarity|F -> Q|`frame_pair_repr`|1|partial alias at polarity-call argument|
|B3 frame/predicate/sufficiency -> polarity|P -> Q|`predicate_pair_repr`|1|partial alias at polarity-call argument|
|B3 frame/predicate/sufficiency -> polarity|S -> Q|`sufficiency_repr`|1|partial alias at polarity-call argument|
|B4 local -> decision/final CE|F -> D|`frame_prob`|1|partial alias at decision-head argument|
|B4 local -> decision/final CE|P -> D|`predicate_coverage_prob`|1|partial alias at decision-head argument|
|B4 local -> decision/final CE|S -> D|`sufficiency_prob`|1|partial alias at decision-head argument|
|B4 local -> decision/final CE|Q -> D|`positive_energy`; `negative_energy`|2|partial aliases at both decision-head arguments|
|**Total**|**10**|**all listed occurrences**|**16**|**one shared lambda=.5 in D1**|

Thus B1=4, B2=4, B3=3, B4=5. A repeated raw source is still recipient-specific: `frame_pair_repr` occurs in F->P/F->S/F->Q, `frame_prob` in F->P/F->S/F->D, and `predicate_pair_repr` in P->S/P->Q. Those independent call arguments must not be collapsed due to shared origin.

## 3. Boundary, edge, occurrence, and non-edge distinction

A **boundary group** is a source-level conditional-expression cluster (B1--B4). A **conceptual edge** is all direct input occurrences from one owner to one immediate recipient; its occurrences move together in the corresponding Gen3 arm. A **tensor occurrence** is the actual recipient argument where an alias is applied.

`final_router_inputs_detached` is not a fifth boundary. It is a duplicate observability/configuration Boolean for the same `explicit_local` state as `local_to_final_router_detached`; it creates no detach, alias, or additional route.

Shared `token_states` is not F->P or another semantic-owner edge. FrameGate and predicate head each read it directly from the shared backbone; a predicate-loss path via it does not traverse FrameGate. The P2/D1 contract freezes the backbone, so it is not a trainable D1 ownership edge.

## 4. Active-versus-inactive path audit

|Path/concern|Executed D1 status|Topology result|
|---|---|---|
|Final 3-way CE through `output['logits']` / explicit product|Active|the five B4 occurrences are the only direct owner ingress to D from final CE|
|Frame, predicate, sufficiency BCE; authorized polarity CE|Active and raw|local owner tensors are raw; an edge partial alias does not scale local-owner gradients|
|Primary-reason CE/reason router|Inactive: `explicit_product`, `reason_loss_weight=0`|not counted|
|Auxiliary boundary/frame-violation/predicate-isolation/preservation/temporal/adapter/channel/slot/v7 losses|Inactive or forbidden by P2 resolver|not counted|
|Temporal/predicate comparators and final penalties|Inactive/forbidden; scales zero|not counted|
|Ranking/intervention/pairwise/bridge objectives|Inactive or forbidden|not counted|
|Legacy `frame_downstream_gradient_mode=frame_local_only`|Inactive; P2 requires `joint`|not a D1/Gen3 boundary|
|EMA/teacher observer|Inactive/forbidden|no edge|
|Shared backbone/token states|Backbone frozen|no trainable owner edge|

The trainer computes local BCE from raw `frame_logit`, `predicate_coverage_logit`, and `sufficiency_logit`, and local polarity CE from raw `negative_energy`/`positive_energy`. The D1 test contract specifically verifies local-plus-downstream, rather than half of the total owner gradient. Local losses would remain unscaled under an edge-specific partial intervention.

## 5. Independent intervention versus multi-hop coupling

`partial_grad(z, lambda) = stopgrad(z) + lambda * (z - stopgrad(z))` preserves the forward value and has derivative lambda with respect to `z`. Replacing only one edge's listed occurrences by lambda=.5 while every other occurrence remains raw/lambda=1 is therefore forward-invariant and changes that direct ingress independently.

No two proposed direct edges share a recipient call argument; recipient-specific aliases can be created for every reused source. No edge probe requires changing a loss, target, label, or decision formula. Hence no algebraic/redundant coupling eliminates a probe.

There is non-redundant **multi-hop** coupling: an S loss can reach F directly by F->S and indirectly by P->S followed through P's F->P inputs. Q likewise has direct and inherited upstream paths. A one-edge arm identifies attenuation of that named direct ingress in an otherwise-joint graph, not all downstream influence on the source owner. This is an interpretation limit, not single-edge non-identifiability.

## 6. Minimum Gen3 first-pass matrix

Each new arm is .5 on exactly one conceptual edge and 1 on all others. No lambda sweep is proposed. Frozen A2 remains endpoint context and need not be rerun in this first pass.

|Arm|F->P|F->S|P->S|F->Q|P->Q|S->Q|F->D|P->D|S->D|Q->D|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|JOINT / A0|1|1|1|1|1|1|1|1|1|1|
|GLOBAL-HALF / D1|.5|.5|.5|.5|.5|.5|.5|.5|.5|.5|
|G3-FP-HALF|.5|1|1|1|1|1|1|1|1|1|
|G3-FS-HALF|1|.5|1|1|1|1|1|1|1|1|
|G3-PS-HALF|1|1|.5|1|1|1|1|1|1|1|
|G3-FQ-HALF|1|1|1|.5|1|1|1|1|1|1|
|G3-PQ-HALF|1|1|1|1|.5|1|1|1|1|1|
|G3-SQ-HALF|1|1|1|1|1|.5|1|1|1|1|
|G3-FD-HALF|1|1|1|1|1|1|.5|1|1|1|
|G3-PD-HALF|1|1|1|1|1|1|1|.5|1|1|
|G3-SD-HALF|1|1|1|1|1|1|1|1|.5|1|
|G3-QD-HALF|1|1|1|1|1|1|1|1|1|.5|

The ten probes cannot be reduced without assuming an untested owner grouping. Q->D correctly moves both energies together: they jointly constitute the one polarity-owner-to-decision edge.

## 7. Predefined interpretation and interaction decision

|Observed pattern|Interpretation|
|---|---|
|One edge-half arm reproducibly improves specified outcomes versus JOINT, with GLOBAL-HALF directionally consistent|localized harmful **direct** ownership edge; not total transitive influence|
|Several edge-half arms have comparable non-isolated effects|distributed dependence|
|No individual arm explains GLOBAL-HALF but the combined effect is larger|cumulative/interaction dependence; then consider a prespecified pairwise follow-up|
|An edge-half arm is neutral or harms versus JOINT|harmless or beneficial direct ownership boundary at lambda=.5|
|No reproducible single-edge pattern and GLOBAL-HALF is not decomposed descriptively|Gen3 single-edge premise unsupported at this granularity|

**Pairwise interaction decision: NO.** Static topology allows independent recipient-specific aliases and has no algebraically inseparable pair. The frozen residual report also does not identify a common causal edge. Pairwise execution requires a later, evidence-triggered authority after a cumulative/interaction signature.

## 8. Explicit non-claims

This audit does not authorize or perform training, evaluation, inference, checkpoint loading, Gen3 implementation, source/test modification, optimal-lambda selection, a lambda curve, pairwise execution, statistical significance, backbone state/information ownership, or a causal conclusion from D1 outcomes. It does not claim the four D1 residuals have one edge cause.

## 9. Sources and final-state record

Read: `docs/CONTRAMAMBA_RESEARCH_VISION.md`; the named D1 implementation/evidence/residual commits; `src/contramamba/modeling_v6b_minimal.py`; `scripts/train_controlled_v6b_minimal.py`; and `tests/test_reason_router_p2_contract.py`. The three primary implementation/test files match commit `1abe147484e2155b19d3c973175628d337807501` byte-for-byte in this checkout.

Modified: this report only. No source, test, data, checkpoint, or existing report artifact was modified. File bytes/SHA256 and final git status/diff checks follow the write.
