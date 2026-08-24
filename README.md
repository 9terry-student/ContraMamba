# ContraMamba: Evidence-Entitlement Modeling for Claim-Evidence Verification

ContraMamba studies whether a claim-evidence verifier is not only correct at the final-label level, but also internally entitled to make that decision from the supplied evidence.

The central distinction is:

> A model can predict the correct final label without having an internally faithful evidence-entitlement path for that label.

ContraMamba therefore separates final judgment from intermediate epistemic signals such as frame compatibility, predicate coverage, evidence sufficiency, entitlement, and polarity. This is not a generic uncertainty-estimation project: confidence, entropy, and softmax calibration are not treated as sufficient evidence of epistemic entitlement.

---

## Current status

Active research line:

```text
ContraMamba Reason-Preserving Authorization Router
P2 / P3-W6-F2 lineage
```

P4-Z README status rewrite:

```text
COMPLETE / FROZEN
freeze: 5e345d065cc2fc274d7e11906a41169064823684
```

Current documentation-planning milestone:

```text
P3-W6-F2-P4-AA
15-week research-credit roadmap update
```

The P4 canonical lineage recovery and validation phase is complete. Canonical P4-L artifact/provenance integrity is **ESTABLISHED** through the frozen P4-V closure lineage.

Frozen current-line identities:

| Surface | Status | Identity |
|---|---|---|
| P4 lineage closure evidence freeze | established | `69c7d7b142171c8a3b21c0984b2b3162da04fe77` |
| P4-X trainer-rebind authority freeze | frozen | `dfe6926ff2ceb6280ed899d514854c531e3af6c1` |
| P4-X trainer-rebind implementation freeze | frozen | `8f6defacc1995f263c97000fe43f6034b1ce9324` |
| P4-Y trainer-rebind validation evidence freeze | frozen | `81f52d0e6b84f50f211b93c35999adb7b975b06f` |

These commit identities are repository authority/provenance identities. They are **not** model checkpoint identities.

Current dispositions:

| Question | Disposition |
|---|---|
| Canonical artifact/provenance integrity | **ESTABLISHED** via P4-V lineage |
| Bounded trainer-rebind code correctness | **ESTABLISHED** via P4-X/P4-Y |
| Trainer execution success | **NOT_ESTABLISHED** |
| Scientific effectiveness | **NOT_ESTABLISHED** |
| Actual reason-router research execution | **NOT_ESTABLISHED** |
| A0 / training / evaluation / Kaggle / GPU | **NOT YET AUTHORIZED** by P4-AA |

After P4-AA is independently verified and frozen, the next stage is creation, independent verification, and freeze of the P3-W7-A0 research-credit execution authority. A0 is not authorized by this README.

---

## 15-week research-credit roadmap

Planning context:

```text
duration: 15 weeks
context: 학부연구생 연구학점제 / research-credit project
active topic: ContraMamba Reason-Preserving Authorization Router
stage: P3-W6-F2-P4-AA
```

This roadmap is a planning layer, not execution authority. Exact stages and experiments may change in response to validated evidence, and every actual training, evaluation, Kaggle, or GPU operation still requires a separately frozen execution authority. README roadmap changes do not retroactively change frozen authority, and a README plan cannot authorize training, evaluation, Kaggle/GPU use, or scientific conclusions. Scientific conclusions require validated experimental evidence.

Planning/authority boundary:

| Layer | Role |
|---|---|
| Roadmap | Planning only |
| Frozen stage authority | Controls what may be implemented or executed |
| Runtime evidence | Establishes whether execution occurred successfully |
| Validated imported evidence | Basis for scientific interpretation |
| README alone | Does not authorize experiments |

Conceptual namespace:

```text
Research-credit project
└── P3
    ├── W7 - A0 baseline establishment
    ├── later workstream(s) - matched mechanism comparison
    └── later workstream(s) - evidence-driven follow-up / replication
```

P3-W7 is the first research-credit experimental workstream, not the entire 15-week project namespace. P3-W8 may become a matched mechanism comparison workstream, and P3-W9 may become a follow-up or replication workstream, but those names are anticipated only: not reserved, not frozen, and subject to later authority and evidence.

### Week-level plan

Week boundaries are planning targets, not immutable deadlines.

| Weeks | Planning target |
|---|---|
| 1-2 | A0 baseline authority and execution preparation: create/freeze P3-W7-A0 authority, verify exact execution commit and commands, prepare provenance-valid Kaggle execution, and begin A0 only after authority freeze. |
| 2-3 | A0 three-seed baseline execution and provenance collection for seeds 180/181/182, one seed at a time, with collection/import and artifact/provenance validation. Failed runs may be retained when scientifically or provenance useful. These executions have not yet occurred. |
| 3-4 | A0 baseline validation and analysis: reconcile the three seeds, validate immutable same-seed references, document baseline behavior, and avoid causal mechanism claims from A0. |
| 4-5 | Next-arm authority resolution: resolve only evidence-supported A1/A2/A3 execution prerequisites, confirm reason-loss authority, confirm the matched comparison contract, and do not automatically authorize all arms. |
| 5-8 | Conditional matched mechanism experiments, potentially A1/A2/A3, only if separately authorized and prerequisite gates pass. This block may be delayed, narrowed, reordered, or stopped based on evidence. |
| 8-9 | A0-A3 comparison / E0-related validation as separately authorized: matched seed comparison, gradient-ownership interpretation, reason-router effect analysis, and E0 only under explicit applicable authority. These experiments are not guaranteed. |
| 9-11 | Failure analysis and evidence-driven follow-up: inspect failure modes, determine whether bounded additional ablations are scientifically justified, and reject unsupported ad hoc hyperparameter search. This does not pre-authorize new experiments. |
| 11-12 | Replication / robustness phase if warranted: replicate key results if scientifically warranted, prioritize reproducibility over breadth, and do not invent a fixed replication design now. |
| 13 | Final provenance and evidence audit: ensure the authority -> command -> run -> artifact -> import chain is complete, reconcile missing/failed runs, and freeze validated experiment evidence. |
| 14 | Research-credit report preparation: methods, experimental design, provenance, figures/tables, results, limitations, and failed hypotheses or negative results where relevant. |
| 15 | Final synthesis and presentation/submission preparation: scientific interpretation, limitations, reproducibility statement, presentation/report material, and clear separation between established findings and open hypotheses. |

### Adaptive roadmap rules

The roadmap may change if a provenance blocker appears, an execution authority cannot be frozen, A0 fails, artifact integrity cannot be established, a scientific prerequisite remains unresolved, a proposed comparison becomes invalid, a result eliminates the need for a planned experiment, or a new bounded experiment becomes scientifically necessary.

Roadmap changes must be documented, preserve frozen historical evidence, avoid rewriting prior authorities, avoid reinterpreting failed runs as successful, avoid silently moving experimental goals, and receive new authority where execution scope changes. A README commit is not required for every small operational adjustment; README should represent meaningful milestone-level changes.

### Research-credit provenance policy

From A0 onward, every experimental run should retain provenance connecting:

```text
run name
-> authority freeze full SHA
-> execution HEAD
-> exact command / command SHA256
-> input dataset identity
-> sidecar/provenance identity
-> seed / split seed / arm
-> runtime report
-> prediction artifacts
-> selected checkpoint
-> artifact SHA256
-> collection handoff
-> local import audit
-> subsequent analysis
```

Raw runtime outputs are immutable evidence and must not be manually edited for presentation. Cleaned summaries and figures belong in separate analysis artifacts. Failed runs, blocked executions, provenance failures, negative results, failed hypotheses, and non-promoted candidates should be preserved when scientifically relevant. A zero exit code does not alone establish a valid result; scientific interpretation requires provenance-valid imported evidence.

### Artifact organization policy

Research-credit work should be separated by workstream, not placed under one permanent P3-W7 namespace. Historical P3-W6-F2/P4 artifacts remain in their existing paths and must not be moved or renamed.

Examples:

| Workstream | Example namespace |
|---|---|
| P3-W7 A0 baseline | `reports/reason_router_p2_p3w7_*` |
| Later formally authorized workstream | `reports/reason_router_p2_p3w8_*` |
| Later formally authorized workstream | `reports/reason_router_p2_p3w9_*` |

The P3-W8/P3-W9 examples are illustrative only and do not reserve or freeze future stage names. New workstreams receive new namespaces, raw run directories remain isolated by arm and seed, and analysis/result artifacts remain separate from raw runtime outputs. This README task does not create future directories.

### A0 role

At current HEAD:

| Surface | Status |
|---|---|
| A0 execution authority | **NOT YET FROZEN** |
| A0 execution | **NOT YET PERFORMED** |
| A0 training | **NOT YET AUTHORIZED** by P4-AA |
| A0 clean-dev evaluation | **NOT YET AUTHORIZED** by P4-AA |
| Kaggle/GPU | **NOT YET AUTHORIZED** by P4-AA |

A0 is the `explicit_product` baseline with joint ownership, reason-specific primary CE inactive / effective reason weight 0, and the role of creating same-seed reference evidence for later comparison. A0 is not evidence that first-blocker routing works, not evidence that reason supervision works, not evidence that explicit-local ownership works, and not sufficient for a scientific conclusion.

A1/A2/A3 remain future, conditional, separately authorized planned comparison arms. Their exact configuration may depend on later frozen scientific authority, and this roadmap does not release them. E0 remains an algebraic-equivalence check retained in the research design; this README does not state that E0 has run, passed, or will definitely run in a particular week during the research-credit phase.

### Intended final deliverables

Intended research-credit deliverables include a provenance-valid experiment record, A0 baseline evidence and any separately authorized comparisons, aggregate results, failure analysis, figures/tables, reproducibility notes, limitations, and final research-credit report/presentation material. These are intended deliverables, not claims that they already exist.

---

## Historical empirical primary

The latest established empirical primary from the earlier experimental lineage remains:

```text
Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery
```

Stage71 is historical empirical evidence, not the active mechanism-development stage. The active research line is now the reason-preserving authorization router.

### Stage71 historical metrics

Clean controlled development performance:

| Metric | Value |
|---|---:|
| Accuracy | 0.975 |
| Macro-F1 | 0.964 |
| Prediction count: NOT_ENTITLED | 522 |
| Prediction count: REFUTE | 90 |
| Prediction count: SUPPORT | 108 |

External VitaminC diagnostic for the Stage71 primary was run in Stage73.

| Metric | Value |
|---|---:|
| Accuracy | 0.353 |
| Macro-F1 | 0.326 |
| SUPPORT recall | 0.432 |
| REFUTE recall | 0.203 |
| SUPPORT predictions | 393 |
| REFUTE predictions | 219 |
| NOT_ENTITLED predictions | 388 |
| False NOT_ENTITLED total | 323 |
| False entitlement total | 80 |

Stage71 is not considered a solved hallucination-control model. It remains the latest established empirical primary from the earlier experimental lineage because later bridge/threshold branches did not pass promotion criteria.

---

## Why the research direction changed

The Stage99-Stage106 diagnostic branch tested whether additional support-floor bridges or post-hoc threshold/routing policies could recover suppressed SUPPORT decisions without breaking clean development behavior. The branch produced useful diagnostic evidence, but no candidate passed promotion criteria.

The Stage99-Stage106 branch conclusion was:

```text
STAGE106_KEEP_STAGE71_PRIMARY_CLOSE_STAGE99_TO_STAGE105_BRANCH
```

The reason-router line was then motivated by an explicit mechanism audit. The proposed sum of ordered masses was rejected because it is algebraically equivalent to the existing product composer in both forward and backward behavior. Because it did not create a causal mechanism change, it was rejected.

The current candidate mechanism family is:

1. Conditional First-Blocker Reason Router
2. Reason-Specific Supervision
3. Explicit Gradient Ownership

This mechanism is specified, implemented, and boundedly validated for code correctness. It has not been experimentally established as improving model quality or causal effectiveness.

---

## Reason semantics

The frozen internal primary first-blocker reason order is:

```text
FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED
```

This is an internal reason order for the reason-router line. Secondary reasons remain multi-label diagnostics only. They are not external classes, and they are not duplicated into the training loss.

The external final label space remains:

```text
REFUTE / NOT_ENTITLED / SUPPORT
```

No additional reason classes are introduced by the current authority.

---

## Gradient and loss ownership

The core frozen P2 ownership contract is:

- final 3-way CE is router-only;
- on explicit-local/A3 paths, F/P/S and polarity inputs to that final CE are detached;
- local reason-specific losses retain their authorized local owners;
- A1 joint ownership semantics are preserved;
- EMA is observer/baseline only, not a teacher, loss target, or novelty mechanism.

This is not an "all heads are detached" rule. Detach is specific to the final CE and explicit-local ownership paths, while authorized local losses still train their local components.

---

## A0-A3 and E0 status

The retained experiment matrix includes:

- A0
- A1
- A2
- A3
- E0 algebraic-equivalence check

These semantics are preserved in code/tests. P4-X and P4-Y did **not** execute these research experiments. Passing unit/static validation is not A0 execution evidence, does not authorize A0, and does not establish scientific effectiveness.

No numerical A0-A3/E0 results are established by P4-X/P4-Y.

---

## Canonical trainer binding

The active reason-router trainer entry point is:

```text
scripts/train_controlled_v6b_minimal.py
```

It is now rebound to the canonical P4-L reason-router lineage with fail-closed identity, provenance, and join validation. This README intentionally does not provide a training command because training is not yet authorized.

Canonical P4-L runtime directory identity:

```text
reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458
```

The canonical sidecar bytes are not required to be present in this local checkout for the README rewrite. The tracked P4-V closure result records the canonical P4-L path, hashes, and count reconciliation.

Trainer binding requirements:

- canonical universe: 3600 source rows;
- trainer joins source `id` to sidecar `row_id`;
- the join must be exact one-to-one and fail closed;
- runtime identity/provenance mismatch must fail before training;
- canonical artifacts must not be silently rebuilt or regenerated by the trainer.

Reason-supervision disposition:

| Field | Count |
|---|---:|
| `p2_reason_supervision_eligible = true` | 1769 |
| `p2_reason_supervision_eligible = false` | 1831 |
| `integrity_status = ELIGIBLE` | 1769 |
| `integrity_status = INELIGIBLE` | 1562 |
| `integrity_status = UNRESOLVED` | 269 |

Positive-margin compatibility surface:

| Field | Count |
|---|---:|
| `eligible_for_positive_margin = true` | 724 |
| `eligible_for_positive_margin = false` | 2876 |

Positive-margin compatibility is a separate compatibility surface. It is not reason-supervision admission. `integrity_status` is not converted into a target.

---

## Current validation evidence

Bounded P4-X/P4-Y validation evidence:

| Validation surface | Result |
|---|---|
| P4-X focused tests | 25 passed, 1 skipped |
| P2 contract | 32 passed |
| P4-L builder contract | 20 passed, 1 skipped |
| `py_compile` | PASS |

These results establish bounded trainer-rebind code correctness. They do not establish trainer runtime success, model performance, causal effectiveness, or scientific success.

Authority artifacts:

- [P4-X trainer-rebind authority spec](reports/reason_router_p2_p3w6f2_p4x_trainer_rebind_authority_spec_candidate.md)
- [P4-Y validation result](reports/reason_router_p2_p3w6f2_p4y_trainer_rebind_validation_8f6defacc1995f263c97000fe43f6034b1ce9324/p3w6f2_p4y_trainer_rebind_validation_result_candidate.json)
- [P4-V canonical P4-L closure evidence](reports/reason_router_p2_p3w6f2_p4v_canonical_sidecar_path_provenance_schema_correction_execution_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4v_canonical_sidecar_path_provenance_schema_correction_result.json)

---

## Controlled clean data

The historical controlled clean dataset for the earlier empirical lineage is:

```text
data/controlled_v5_v3_without_time_swap.jsonl
```

The earlier corrupted `time_swap` family is excluded from main classification training.

The clean controlled data contains:

```text
12 intervention types
300 pair groups
3,600 examples
```

Label distribution:

```text
NOT_ENTITLED: 2700
REFUTE:        450
SUPPORT:       450
```

Splits are performed by `pair_id`, preventing an original pair and its interventions from crossing train/development partitions.

---

## Stage99-Stage106 diagnostic branch

Stage99-Stage106 tested whether observed SUPPORT suppression could be repaired by bridge data or threshold/routing diagnostics.

| Run | Validity | Acc | Macro-F1 | SUPPORT recall | REFUTE recall | SUPPORT pred | REFUTE pred | NE pred | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Stage73 / Stage71 historical empirical primary | PRIMARY | 0.353 | 0.326 | 0.432 | 0.203 | 393 | 219 | 388 | KEEP_PRIMARY |
| Stage92C / support-preserving bridge | external diagnostic, rejected | 0.351 | 0.331 | 0.402 | 0.234 | 365 | 223 | 412 | NEAR_MISS_NOT_PROMOTED |
| Stage97C / half anti-NE bridge | external diagnostic, rejected | 0.355 | 0.343 | 0.348 | 0.318 | 310 | 296 | 394 | TOPLINE_WIN_BUT_SUPPORT_SUPPRESSED |
| Stage99C / support-floor micro bridge | external diagnostic, rejected | 0.344 | 0.335 | 0.316 | - | 278 | 314 | 408 | REJECT_SUPPORT_SUPPRESSION_WORSE |
| Stage102 / external-tuned threshold diagnostic | diagnostic only, not promotable | 0.384 | 0.350 | 0.480 | 0.234 | 440 | 223 | 337 | POSITIVE_SIGNAL_BUT_NOT_PROMOTABLE |
| Stage105 / portable clean-delta diagnostic | diagnostic only, not promotable | 0.353 | 0.332 | 0.406 | 0.234 | 367 | 223 | 410 | WEAK_OR_NEGATIVE |

Key historical findings:

1. Bridge append is not the right next mechanism. Stage88, Stage92, Stage95, Stage97, and Stage99 show that synthetic bridge data can move REFUTE/macro-F1, but often suppresses SUPPORT entitlement.
2. External threshold sweep revealed a latent SUPPORT signal, but Stage102 selected its threshold using VitaminC external labels and is diagnostic only.
3. Clean-only threshold selection was underdetermined because clean development already had perfect SUPPORT and REFUTE recall.
4. Stage104B found no strict nontrivial clean-safe threshold candidate.
5. Stage105's portable clean delta was weak and remained below Stage73 on SUPPORT recall.

This branch is closed as historical diagnostic evidence. It is not current reason-router execution evidence.

---

## Historical Stage26-H1 result

Stage26-H1 studied a repaired v7 hierarchical model using a real Mamba backbone. It preserved the v7 hierarchical channel architecture but replaced an unstable raw-additive final decision with a v6B-style softplus/multiplicative final-decision geometry.

The key finding from that phase was:

> The v7 hierarchy itself was not the main failure. The failure came from the final-decision geometry. Treating entitlement as an additive logit feature caused SUPPORT collapse. Restoring entitlement as a gate over polarity energies recovered 3-way judgment behavior.

The H1 final decision used nonnegative polarity energies and entitlement-gated final logits:

```python
positive_energy = softplus(support_polarity_logit)
negative_energy = softplus(refute_polarity_logit)

support_score = entitlement_for_decision * positive_energy
refute_score = entitlement_for_decision * negative_energy
ne_score = ne_bias + alpha * (1 - entitlement_for_decision)

final_logits = [refute_score, ne_score, support_score]
```

Stage26-H1 remains important historical evidence for the project's core thesis that entitlement must function as a gate rather than as an additive feature. It is not the active mechanism-development status.

---

## Historical Stage7 result

Earlier Stage7 experiments used a classifier-auditor router over the v5 pipeline.

The main Stage7 system was ContraMamba-CAR at threshold 0.5, using `v3_no_intervention` as the classifier and `v3_no_polarity_flip` as the balanced entitlement auditor.

| Accuracy | Macro-F1 | NOT_ENTITLED F1 | REFUTE F1 | SUPPORT F1 | Gate violation rate | Output/internal gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.929 +/- 0.003 | 0.906 +/- 0.005 | 0.952 +/- 0.002 | 1.000 +/- 0.000 | 0.765 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

Stage7 remains historical evidence that explicit entitlement auditing can expose and constrain failures hidden by flat final-label performance.

---

## Repository structure

| Path | Purpose |
|---|---|
| `src/contramamba/` | ContraMamba models, heads, labels, and losses |
| `scripts/` | Controlled-data builders, training utilities, evaluators, and report writers |
| `data/` | Controlled intervention datasets |
| `experiments/` | Stage plans and experiment notes |
| `results/` | Seed-level and aggregate reports |
| `docs/` | Architecture and paper-oriented results documentation |
| `tests/` | Unit, validation, training-smoke, and reporting tests |
| `reports/` | Frozen specifications, manifests, validation records, and lineage evidence |

---

## Reproducibility boundary

External diagnostics are treated as diagnostics only. They are not used for training, threshold tuning, candidate selection, or promotion unless an explicit cleanly separated protocol authorizes that use.

For the active reason-router line, trainer execution must validate canonical identity, provenance, and the exact source `id` to sidecar `row_id` join before training. Any mismatch must fail closed. Canonical P4-L artifacts must not be silently rebuilt or regenerated by trainer code.

Basic local test commands and training commands are intentionally omitted here because P4-AA is a README planning update stage, and A0/training/evaluation are not yet authorized.

---

## Limitations

Current evidence must be read in separate layers:

- Stage71 is the latest established empirical primary from the earlier lineage, not proof that the new reason-router mechanism works.
- Stage99-Stage106 show bridge/threshold limitations and diagnostic latent SUPPORT signal, not a promotable new primary.
- P4-V establishes canonical P4-L artifact/provenance integrity, not trainer execution.
- P4-X/P4-Y establish bounded trainer-rebind code correctness, not model performance or scientific effectiveness.
- The reason-router mechanism is specified and implemented but has not yet been experimentally established.
- The upcoming "학부연구생 연구학점제" / research-credit experimental phase is workflow context only, not an institutional scientific endorsement.

---

## Current research claim

The strongest current claim is conservative:

> Evidence-entitlement modeling exposes failure modes that flat 3-way claim-evidence performance can hide. Earlier bridge and threshold work showed that SUPPORT information can exist latently while clean-only recovery remains unsafe or underdetermined, motivating a mechanism-level routing change.

Current mechanism status:

```text
specified + implemented + boundedly validated for code correctness
not trainer-executed
not experimentally established
not scientifically concluded
```

The first-blocker reason router has not been shown to improve model quality. P4-X validates bounded code correctness; P4-Y freezes validation evidence. Neither validates causal effectiveness, model performance, A0 success, or research-hypothesis success.

---

## Next workflow milestone

Current milestone:

```text
P3-W6-F2-P4-AA
15-week research-credit roadmap update
```

P4-Z README status rewrite is complete/frozen at:

```text
5e345d065cc2fc274d7e11906a41169064823684
```

After P4-AA is independently verified and frozen, the next stage is creation, independent verification, and freeze of the P3-W7-A0 research-credit execution authority for the "학부연구생 연구학점제" / research-credit experimental phase.

A0 is not yet authorized. Training, evaluation, Kaggle execution, and GPU execution are not authorized by this README.
