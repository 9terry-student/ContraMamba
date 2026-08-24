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

Current README milestone:

```text
P3-W6-F2-P4-Z
README status rewrite only
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
| A0 / training / evaluation | **NOT YET AUTHORIZED** at this README rewrite stage |

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

Basic local test commands and training commands are intentionally omitted here because P4-Z is a README rewrite stage, and A0/training/evaluation are not yet authorized.

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
README rewrite / P4-Z
```

After this README rewrite is independently verified and frozen, the next stage is creation and freeze of the research-credit A0 execution authority for the "학부연구생 연구학점제" / research-credit experimental phase.

A0 is not yet authorized. Training, evaluation, Kaggle execution, and GPU execution are not authorized by this README.
