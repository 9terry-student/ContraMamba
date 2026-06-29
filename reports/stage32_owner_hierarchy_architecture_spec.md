# Stage32 Owner-Hierarchy Architecture Specification

## Purpose
Stage32 documents the structural redesign required after Stage31 Coverage/Entailment diagnostics. The conclusion from Stage31 is that directional Coverage/Entailment cannot be safely handled by the current v7 proxy stack, by cap/gamma/boost patching, or by a small post-hoc diagnostic head.

The next architecture should be flat-arbiter-less, not decision-less.

Denied path:
`Mamba Encoder -> multiple heads -> learned black-box arbiter -> final label`

Approved path:
`Mamba Encoder -> explicit owners -> rule-structured composer -> final label`

## Stage31 Evidence
| Stage | Variant | Key Metrics | Failure Pattern | Interpretation |
|---|---|---|---|---|
| Stage31-B | current proxy stack | total_accuracy=0.445; macro_f1=0.3607 | support_entailment_predicted_ne=61; coverage_failure_predicted_support=6; refute_case_predicted_support=5; refute_case_predicted_ne=27 | Conservative but under-structured; suppresses many overclaims but fails to preserve valid SUPPORT entailment. |
| Stage31-C | 4-class diagnostic head | alignment=0.445 | support=39/80; overclaim=44/80; refute=6/40; CONTRADICTS_REFUTE->ENTAILS_SUPPORT=20/40; OVERCLAIM_NOT_ENTITLED->ENTAILS_SUPPORT=27/80 | Partial signal exists but unsafe for composer integration. |
| Stage31-C2 | 3-class hard contrast | total_accuracy=0.450; macro_f1=0.3535; alignment=0.365 | support=42/80; overclaim=14/80; refute=17/40; overclaim_misread_as_entails_support=32; refute_misread_as_entails_support=17 | Hard contrast did not solve the problem; overclaim ownership collapsed. |
| Stage31-C3 | raw_pair access | total_accuracy=0.420; macro_f1=0.3096; alignment=0.400 | support=46/80; overclaim=18/80; refute=16/40; refute_to_entails=15; overclaim_to_entails=34 | Raw pair access does not make the diagnostic head stable enough. |
| Stage31-C3 | hybrid access | total_accuracy=0.460; macro_f1=0.3743; alignment=0.340 | support=31/80; overclaim=26/80; refute=11/40; refute_to_entails=17; overclaim_to_entails=23 | Hybrid access does not make the diagnostic head safe; directional alignment worsens. |

## Decision
| Path | Decision |
|---|---|
| More caps | DENY |
| Gamma tuning | DENY |
| NE boost | DENY |
| Head-only composer wiring | DENY |
| Flat learned arbiter | DENY |
| Owner hierarchy | APPROVE |
| Explicit Coverage/Entailment owner | APPROVE |
| Rule-structured composer | APPROVE |

## Proposed Architecture
Text diagram:

`Mamba Encoder`
-> `Hard Core Validity Owner`
-> `Coverage / Entailment Owner`
-> `Residual Adjudication Owner`
-> `ANI Diagnostic Readout`
-> `Polarity Owner`
-> `Rule-Structured Composer`
-> `Final Label`

## Owner Responsibilities
| Owner | Input | Output | Authority | Failure Mode Owned |
|---|---|---|---|---|
| Hard Core Validity Owner | Encoded claim/evidence target, event, relation, entity, location, and temporal anchors | hard_core_valid, hard_core_failure_type, target/event anchor confidence | Hard failure cannot be rescued by later owners; may block SUPPORT/REFUTE when judgeable target identity or event anchor is broken | wrong person, wrong organization, wrong event, core location mismatch, core temporal mismatch |
| Coverage / Entailment Owner | Hard-Core-valid pair state plus direct directional scope/coverage features | coverage_relation, coverage_status, coverage_entitlement_state, contradiction_license | Overclaim routes to NOT_ENTITLED unless a stronger contradiction applies; entailment-preserving relation preserves entitlement; contradiction routes toward REFUTE | all->some SUPPORT, some->all NOT_ENTITLED, specific->general SUPPORT, general->specific NOT_ENTITLED, whole->part SUPPORT, part->whole NOT_ENTITLED, only->base SUPPORT, also->only NOT_ENTITLED, none->some REFUTE |
| Residual Adjudication Owner | Cases not cleanly resolved by Hard Core or Coverage/Entailment | residual_status, residual_reason, unresolved_relation_confidence | Produces residual state for ANI and composer; should not override Hard Core or straightforward Coverage/Entailment | borderline mismatch, underspecification, paraphrase ambiguity, unresolved relation gaps |
| ANI Diagnostic Readout | Owner states from Hard Core, Coverage/Entailment, Residual, and Polarity | novelty, ambiguity, ignorance diagnostics | Explanation layer only; not primary final-label controller | ignorance from missing coverage/insufficient entitlement, ambiguity from unresolved residual readings, novelty from unsupported added information |
| Polarity Owner | Entitled or contradiction-licensed pair state | polarity_state, support_score, refute_score, contradiction_strength | Decides SUPPORT vs REFUTE only after entitlement is alive or contradiction is explicitly licensed; cannot rescue failed Hard Core or turn overclaim into SUPPORT | explicit support/refute polarity after ownership routing |
| Rule-Structured Composer | All owner states and calibrated owner confidences | final_label, decision_trace, blocking_owner | Applies explicit owner priority and authority rules; replaces black-box final MLP arbitration | final decision ownership and conflict resolution |

## Coverage / Entailment Directional Relations
| Relation | Outcome |
|---|---|
| all -> some | SUPPORT |
| some -> all | NOT_ENTITLED |
| specific -> general | SUPPORT |
| general -> specific | NOT_ENTITLED |
| whole -> part | SUPPORT |
| part -> whole | NOT_ENTITLED |
| only -> base | SUPPORT |
| also -> only | NOT_ENTITLED |
| none -> some | REFUTE |

Coverage/Entailment needs direct structured access, potentially including symbolic/scope features or explicit contrastive operators. It is not Frame, and it should not be simulated by cap stacking.

## Composer Priority Rules
1. Hard Core failure blocks unsupported final SUPPORT and cannot be rescued by later owners.
2. If Hard Core passes and Coverage/Entailment identifies overclaim, route to NOT_ENTITLED unless a stronger contradiction owner explicitly applies.
3. If Hard Core passes and Coverage/Entailment identifies entailment-preserving relation, preserve the SUPPORT entitlement path and prevent unnecessary NOT_ENTITLED collapse.
4. If Coverage/Entailment identifies contradiction, route toward REFUTE through polarity/contradiction handling.
5. Residual unresolved cases route to NOT_ENTITLED or ANI-informed uncertainty, without overriding Hard Core or clear Coverage/Entailment.
6. Polarity decides SUPPORT/REFUTE only after entitlement is established or contradiction is explicitly licensed.
7. Composer may use calibrated owner confidence, but owner priority and authority must be explicit.
8. No black-box final MLP arbiter and no learned all-head mixing as the primary final decision.

## Core Design Principles
1. Entitlement before polarity.
2. Single Owner Rule: each failure type should have a primary owner.
3. Hard Core failures cannot be rescued.
4. Coverage/Entailment is not Frame.
5. Residual Adjudication is not cap stacking.
6. ANI is diagnostic, not primary decision logic.
7. No flat learned arbiter.
8. Composer is structured, not black-box.
9. Coverage/Entailment needs direct structured access.
10. Stage31 probe remains diagnostic-only.

## Implementation Roadmap
| Stage | Goal | Final Prediction Impact |
|---|---|---|
| Stage32-A | Introduce owner state dataclass / structured output schema | No change |
| Stage32-B | Implement Hard Core + Coverage/Entailment owner interfaces | No direct final-label replacement |
| Stage32-C | Implement rule-structured composer in shadow mode | No change unless explicitly enabled later |
| Stage32-D | Evaluate shadow composer on Stage31 probe and controlled dev | Diagnostic only |
| Stage32-E | Consider replacing v7 final decision only after shadow evidence passes | Conditional |

## Leakage Policy
The Stage31 probe remains diagnostic-only. It must not be used for training, calibration, threshold selection, checkpoint selection, model selection, or auxiliary loss construction.

## Risks
| Risk | Mitigation |
|---|---|
| Stage31 probe is templated and diagnostic, not a broad benchmark | Use it as a targeted structural diagnostic, not as a sole benchmark. |
| Explicit owner hierarchy may initially reduce aggregate dev accuracy | Deploy first in shadow mode and compare decision traces. |
| Symbolic/scope features may be brittle | Keep features auditable and pair them with structured contrastive operators. |
| Training leakage from Stage31 probe | Keep Stage31 strictly diagnostic-only for all Stage32 work. |

## Lock
Stage32 proceeds as an owner-hierarchy redesign with an explicit Coverage/Entailment owner and a rule-structured composer. Flat learned arbitration and further cap/head-only patching are denied.
