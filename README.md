# ContraMamba: Evidence-Entitlement Modeling for Claim-Evidence Verification

ContraMamba studies whether a claim-evidence verifier is not only correct at the final-label level, but also internally entitled to make that decision from the supplied evidence.

The central distinction is:

> A model can predict the correct final label without having an internally faithful evidence-entitlement path for that label.

ContraMamba therefore separates final judgment into intermediate epistemic signals:

- frame compatibility
- predicate coverage
- evidence sufficiency
- entitlement
- polarity

The current system predicts:

```text
REFUTE / NOT_ENTITLED / SUPPORT
```

while exposing internal channel diagnostics for controlled intervention analysis.

This is not a generic uncertainty-estimation project. Confidence, entropy, or softmax calibration are not treated as sufficient evidence of epistemic entitlement. ContraMamba directly evaluates whether the model's internal decision path respects the evidence structure.

---

## Current status

The current valid primary model is:

```text
Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery
```

Stage71 remains the primary after the Stage99-Stage106 diagnostic branch.

The current branch conclusion is:

```text
STAGE106_KEEP_STAGE71_PRIMARY_CLOSE_STAGE99_TO_STAGE105_BRANCH
```

Stage99-Stage105 tested whether additional support-floor bridges or post-hoc threshold/routing policies could recover suppressed SUPPORT decisions without breaking clean development behavior. The branch produced useful diagnostic evidence, but no candidate passed promotion criteria.

The main conclusion from Stage99-Stage106 is:

> Synthetic bridge append strategies can improve REFUTE recall or external macro-F1, but repeatedly suppress SUPPORT entitlement. External-tuned threshold diagnostics reveal a recoverable latent SUPPORT signal in Stage92C probabilities, but clean-only threshold selection does not produce a strict nontrivial safe candidate. The valid primary remains Stage71.

This branch is therefore closed. Future work should not continue broad synthetic bridge append search. The next research direction is a new entitlement-first routing mechanism, not more data append or external-threshold tuning.

---

## Current architecture

The current primary pipeline is a Mamba-based claim-evidence verifier with explicit epistemic channels:

```text
claim + evidence
      |
Mamba encoder
      |
Frame / predicate / sufficiency / polarity / entitlement heads
      |
Final routing
      |
REFUTE / NOT_ENTITLED / SUPPORT
```

Conceptually, ContraMamba treats the final label as the result of two separable questions:

```text
1. Is the model entitled to make a truth-polarity judgment from this evidence?
2. If entitled, does the evidence support or refute the claim?
```

This differs from a flat 3-way classifier:

```text
flat verifier:
    y = softmax(encoder(claim, evidence))

ContraMamba-style verifier:
    representation = Mamba(claim, evidence)
    frame_state = FrameChannel(representation)
    predicate_state = PredicateChannel(representation)
    sufficiency_state = SufficiencyChannel(representation)
    entitlement_state = EntitlementHead(representation)
    polarity_state = PolarityHead(representation)
    final_label = FinalRouting(entitlement_state, polarity_state, channel diagnostics)
```

The goal is not merely to predict SUPPORT / REFUTE / NOT_ENTITLED, but to preserve the epistemic role of entitlement:

```text
SUPPORT and REFUTE should only be available when the evidence entitles a polarity decision.
NOT_ENTITLED should not merely be the residual class of low confidence.
```

---

## Controlled clean data

The current controlled clean dataset is:

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

Controlled intervention families include:

- none
- paraphrase
- entity_swap
- event_swap
- location_swap
- role_swap
- title_name_swap
- predicate_swap
- evidence_deletion
- evidence_truncation
- irrelevant_evidence
- polarity_flip

Splits are performed by `pair_id`, preventing an original pair and its interventions from crossing train/development partitions.

---

## Current primary result

### Stage71 primary

Stage71 is the current valid primary model.

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

Stage71 is not considered a solved hallucination-control model. It is the best valid primary checkpoint after later branches failed promotion criteria.

---

## Stage99-Stage106 diagnostic branch

Stage99-Stage106 tested whether the observed SUPPORT suppression could be repaired by bridge data or threshold/routing diagnostics.

### Branch comparison

| Run | Validity | Acc | Macro-F1 | SUPPORT recall | REFUTE recall | SUPPORT pred | REFUTE pred | NE pred | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Stage73 / Stage71 current primary | PRIMARY | 0.353 | 0.326 | 0.432 | 0.203 | 393 | 219 | 388 | KEEP_PRIMARY |
| Stage92C / support-preserving bridge | external diagnostic, rejected | 0.351 | 0.331 | 0.402 | 0.234 | 365 | 223 | 412 | NEAR_MISS_NOT_PROMOTED |
| Stage97C / half anti-NE bridge | external diagnostic, rejected | 0.355 | 0.343 | 0.348 | 0.318 | 310 | 296 | 394 | TOPLINE_WIN_BUT_SUPPORT_SUPPRESSED |
| Stage99C / support-floor micro bridge | external diagnostic, rejected | 0.344 | 0.335 | 0.316 | - | 278 | 314 | 408 | REJECT_SUPPORT_SUPPRESSION_WORSE |
| Stage102 / external-tuned threshold diagnostic | diagnostic only, not promotable | 0.384 | 0.350 | 0.480 | 0.234 | 440 | 223 | 337 | POSITIVE_SIGNAL_BUT_NOT_PROMOTABLE |
| Stage105 / portable clean-delta diagnostic | diagnostic only, not promotable | 0.353 | 0.332 | 0.406 | 0.234 | 367 | 223 | 410 | WEAK_OR_NEGATIVE |

### Key findings

1. **Bridge append is not the right next mechanism.**  
   Stage88, Stage92, Stage95, Stage97, and Stage99 show that synthetic bridge data can move REFUTE/macro-F1, but often suppresses SUPPORT entitlement.

2. **External threshold sweep reveals latent SUPPORT signal.**  
   Stage102 showed that Stage92C probabilities contain recoverable SUPPORT evidence. The best external-tuned diagnostic policy improved external accuracy, macro-F1, SUPPORT recall, and false-NE count.

3. **External-tuned thresholds are not promotable.**  
   Stage102 selected its threshold using VitaminC external labels. It is diagnostic evidence, not a valid deployment rule.

4. **Clean-only threshold selection is underdetermined.**  
   Stage104 showed that clean development already has perfect SUPPORT and REFUTE recall, making clean-only threshold selection mostly no-op or underdetermined.

5. **No strict nontrivial clean-safe threshold was found.**  
   Stage104B filtered no-op policies and found no strict nontrivial clean-safe threshold candidate.

6. **Portable clean delta is weak.**  
   Stage105 applied a small clean-derived NE-to-SUPPORT delta to external predictions. It slightly improved Stage92C but remained below Stage73 on SUPPORT recall.

Final branch decision:

```text
STAGE106_KEEP_STAGE71_PRIMARY_CLOSE_STAGE99_TO_STAGE105_BRANCH
```

---

## Historical Stage26-H1 result

Earlier README versions described Stage26-H1 as the mainline. That is now historical context, not the current primary.

Stage26-H1 studied a repaired v7 hierarchical model using a real Mamba backbone. It preserved the v7 hierarchical channel architecture but replaced an unstable raw-additive final decision with a v6B-style softplus/multiplicative final-decision geometry.

The key finding from that phase was:

> The v7 hierarchy itself was not the main failure. The failure came from the final-decision geometry. Treating entitlement as an additive logit feature caused SUPPORT collapse. Restoring entitlement as a gate over polarity energies recovered 3-way judgment behavior.

The H1 final decision used nonnegative polarity energies and entitlement-gated final logits:

```python
positive_energy = softplus(support_polarity_logit)
negative_energy = softplus(refute_polarity_logit)

support_score = entitlement_for_decision * positive_energy
refute_score  = entitlement_for_decision * negative_energy
ne_score      = ne_bias + alpha * (1 - entitlement_for_decision)

final_logits = [refute_score, ne_score, support_score]
```

Stage26-H1 remains important historical evidence for the project's core thesis:

```text
entitlement must function as a gate rather than as an additive feature.
```

However, the current primary and current stopping point are Stage71 and Stage106.

---

## Historical Stage7 result

Earlier Stage7 experiments used a classifier-auditor router over the v5 pipeline.

The main Stage7 system was ContraMamba-CAR at threshold 0.5, using `v3_no_intervention` as the classifier and `v3_no_polarity_flip` as the balanced entitlement auditor.

| Accuracy | Macro-F1 | NOT_ENTITLED F1 | REFUTE F1 | SUPPORT F1 | Gate violation rate | Output/internal gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.929 +/- 0.003 | 0.906 +/- 0.005 | 0.952 +/- 0.002 | 1.000 +/- 0.000 | 0.765 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

Stage7 remains an important historical result, but the current research direction has moved toward hierarchical entitlement-first routing.

---

## Next direction: ContraMamba-vNext

Stage106 closes the bridge/threshold branch. The next stage is not more synthetic bridge data and not external threshold tuning.

The next direction is:

```text
ContraMamba-vNext:
state-space epistemic judgment controller, not a flat 3-way verifier
```

The target is to make final routing structurally entitlement-first.

Planned vNext direction:

```text
Frame state first
Predicate coverage second
Sufficiency / entitlement third
Polarity only after entitlement
Final routing last
```

Minimal future output schema:

```text
frame_state:
    same_frame / frame_mismatch / frame_uncertain

predicate_state:
    covered / partially_covered / not_covered

sufficiency_state:
    sufficient / insufficient / ambiguous

polarity_state:
    support / refute / mixed_or_uncertain

final_label:
    SUPPORT / REFUTE / NOT_ENTITLED
```

Deferred future schema:

```text
not_entitled_reason:
    frame_mismatch
    predicate_not_covered
    insufficient_evidence
    ambiguity
    novelty_or_ood

action_state:
    answer
    abstain
    ask_back
    retrieve_more
    mark_unsupported
    revise_answer
```

The immediate next work is a report-only Stage107 scope lock and then a static repo audit before any implementation patch.

Planned next stages:

| Stage | Type | Goal |
|---|---|---|
| Stage107 | report-only scope lock | Define vNext scope and prevent overbroad implementation |
| Stage108 | static repo audit | Inspect existing v7/vNext code and decide salvage vs new file |
| Stage109 | minimal code patch | Implement or salvage minimal entitlement-first composition |
| Stage110 | plumbing validation | Validate imports/config/help without training |
| Stage111 | clean-dev preservation | Run clean train/dev only if plumbing passes |
| Stage112 | external diagnostic | Run external diagnostic only if clean gate passes |

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

---

## Reproducibility

Basic local validation:

```bash
pip install -e .
pytest
```

The main current training entry point is:

```text
scripts/train_controlled_v6b_minimal.py
```

Current main clean data:

```text
data/controlled_v5_v3_without_time_swap.jsonl
```

Current primary:

```text
Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery
```

Current closed diagnostic branch:

```text
Stage99-Stage106
```

External diagnostics are treated as diagnostics only. They are not used for training, threshold tuning, or promotion unless explicitly separated by a clean-only selection protocol and validated under anti-leakage constraints.

---

## Limitations

Current evidence is still primarily from controlled intervention data and diagnostic external evaluation.

Important limitations:

- ContraMamba is not yet a general-purpose hallucination detector.
- The current primary is a claim-evidence verifier, not a deployed RAG controller.
- External VitaminC diagnostics are used for analysis, not tuning.
- Stage102 showed that external-label thresholding can recover latent SUPPORT signal, but that result is not promotable.
- Stage104/104B showed that clean-only threshold selection is underdetermined when clean SUPPORT/REFUTE recall is saturated.
- Stage105 showed that a portable clean-derived threshold delta is too weak to replace Stage71.
- Synthetic bridge append search has reached a stopping point for the current architecture.
- The next required advance is mechanism-level: entitlement-first routing, not more bridge data.

---

## Current research claim

The strongest current claim is:

> Evidence-entitlement modeling exposes failure modes that flat 3-way claim-evidence classifiers hide. In the current ContraMamba branch, later bridge and threshold interventions reveal that SUPPORT information can exist latently in the model probabilities, but final routing cannot safely recover it under clean-only constraints. The current bottleneck is therefore not simply data coverage or uncertainty calibration, but the structure of entitlement-first final decision-making.

The current project direction is to turn this diagnosis into ContraMamba-vNext: a state-space epistemic judgment controller with explicit, hierarchical entitlement routing.