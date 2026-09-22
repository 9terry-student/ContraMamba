# ContraMamba Gen4 Introduction Manuscript Draft v1

## Status

- Status: STATIC MANUSCRIPT INTRODUCTION DRAFT
- Evidence HEAD: `59cd24a7e7fc3104a092e8ddd0c160f9eeb297a6`
- Literature audit date: 2026-09-22
- Scientific execution: CLOSED
- New scientific claims: NONE
- New statistical tests: NONE
- New p-values: NONE
- Priority claim: NONE

This draft implements the frozen central-question and current-literature
positioning without expanding the scientific claim set.

---

## 1. Introduction

Mechanistic analysis often asks whether an internal computation identified in
one model also appears in another. But recurrence alone leaves a more precise
question unresolved: what exactly is preserved when a causal role reappears?
A recurring role might retain the same geometric realization, the same
downstream functional alignment, both, or neither. Distinguishing these
possibilities matters whenever mechanistic correspondences are transferred
across checkpoints, scales, or intervention settings.

Mamba provides a useful setting for this question. Selective state-space
models replace Transformer attention with input-dependent state-space
dynamics while retaining competitive sequence-modeling performance
[gu2023mamba]. Their internal mechanisms have already been studied with causal
tracing and editing [sharma2024locating], token- and layer-level knockout
analysis [endy2025knockout], architecture-specific relevance propagation
[jafari2024mambalrp], and activation-subspace interventions
[mohan2026subspace]. We therefore do not ask whether Mamba contains
interpretable or causally manipulable internal structure.

Related work also shows why recurrence should not automatically be equated
with invariance. Cross-architecture studies have found substantial
mechanistic similarities alongside architecture-specific differences
[wang2025universality], and causal comparisons show that similar behavior can
be implemented by different internal retrieval mechanisms
[arora2025mechanistic]. In Mamba specifically, causal localization can be
non-unique and intervention-surface dependent [jiang2026circuit]. Mechanistic
scaling has also been studied directly in associative-recall settings
[koren2026recall]. These results establish important ingredients, but they do
not determine whether recurrence of a causally validated role across
checkpoints requires preservation of either its geometric realization or its
downstream task alignment.

We test these two implications directly in frozen pretrained Mamba checkpoints
at 130M, 370M, and 1.4B parameters. At each checkpoint, the internal geometry
is reconstructed independently rather than transferred from a smaller model.
The recurring object is therefore defined functionally: a scale-local
component that plays the dominant causal role relative to a prospectively
frozen response-blind control. Principal-plane rank is checkpoint-local and is
not treated as a semantic identifier shared across models.

The first result separates causal-role recurrence from geometric invariance.
At Mamba-130M, the starting causal component is prospectively validated by
external-generator transport, a hard response-blind geometric control,
matched-control necessity, restoration, and finite-perturbation robustness.
Applying the same scientific procedure at larger checkpoints again identifies
a dominant causal role, but not a fixed coordinate realization: the selected
rank changes and the surrounding residual mechanism reorganizes in signed
effects, coefficient-mass distribution, dominant residual rank, and
generator-family coupling. We summarize this bounded pattern as
**core-stable / residual-plastic**. At Mamba-1.4B, a prospectively fixed
cross-block transport experiment further shows that geometric reorientation
is functionally relevant: transporting the canonical subspace shifts the
adjacent intervention response toward the canonical direction, but does not
restore the canonical positive effect.

The second result separates causal-role recurrence from downstream functional
invariance. We measure the local directional readout of each checkpoint's
independently frozen selected-versus-control causal displacement into the
correct-class task margin. In the primary prospectively matched comparison
over 300 source pairs, Mamba-370M has positive mean readout alignment while
Mamba-1.4B has negative mean alignment. The paired scale contrast is strongly
positive (`t(299)=10.77`, one-sided `p=2.22e-23`), and both pre-specified sign
gates pass. Mamba-130M supplies a separately frozen positive contextual point,
but we do not combine the three checkpoints into a model-size regression or
claim a parameter-count threshold.

Together, these results answer the paper's central question: across the
tested Mamba checkpoints, recurrence of a scale-local causal role preserves
neither a fixed geometric realization nor a guaranteed downstream readout
orientation. The contribution is not a new interpretability primitive, a new
form of Mamba steering, or a generic demonstration that representation and
function can differ. It is the prospective empirical separation, within one
frozen causal program, of three properties that are easy to conflate:
causal-role recurrence, geometric realization, and downstream task alignment.

Our two headline contributions are:

1. **Causal-role recurrence without fixed geometric realization.** Across the
   tested checkpoints, a scale-local dominant causal role recurs under the
   same reconstruction and matched-control procedure, while its surrounding
   geometric realization reorganizes. Cross-block transport at 1.4B shows that
   this reorientation contributes causally to local functional change without
   fully accounting for it.

2. **Downstream readout reversal despite causal-role recurrence.** In the
   prospectively matched Mamba-370M-versus-Mamba-1.4B comparison, the recurrent
   causal role couples to the downstream task margin with opposite mean local
   orientation. Thus internal mechanistic recurrence does not guarantee stable
   downstream functional alignment.

Additional experiments define the scope of these conclusions rather than a
third headline contribution. Direct behavioral intervention establishes
positive downstream coupling at 370M while the same pre-specified positive
bridge is not established at 1.4B. The frozen causal intervention also
transfers at the margin level to natural-language AVeriTeC gold-evidence
inputs at 130M and 370M, whereas a preregistered fixed-mirror steering rule
does not produce useful prediction-level control. These results distinguish
mechanistic validity, downstream alignment, external causal transfer, and
control utility as separate empirical properties.

The scope is deliberately limited. Our inference concerns fixed pretrained
Mamba checkpoints and item-level populations rather than a population of
independently trained model seeds. Residual cross-checkpoint comparisons are
descriptive, the 1.4B site-specificity test uses one prospectively fixed
adjacent site, and the AVeriTeC study uses gold evidence rather than retrieval.
We therefore do not claim a universal scaling law, a phase transition,
architecture-independent universality, complete mediation by geometric
transport, or general steering failure.

The practical implication is correspondingly bounded: when a causal role
appears to recur after a checkpoint or scale change, intervention geometry and
downstream alignment should be revalidated rather than assumed to transfer
from role recurrence alone.

---

## Internal citation-key mapping

- `[gu2023mamba]` -> Gu and Dao, Mamba.
- `[sharma2024locating]` -> Sharma, Atkinson, Bau, COLM 2024.
- `[jafari2024mambalrp]` -> Jafari et al., NeurIPS 2024.
- `[wang2025universality]` -> Wang et al., ICLR 2025.
- `[endy2025knockout]` -> Endy et al., ACL 2025.
- `[arora2025mechanistic]` -> Arora et al., arXiv:2505.15105.
- `[mohan2026subspace]` -> Mohan et al., ICML 2026.
- `[jiang2026circuit]` -> Jiang and Zhang, arXiv:2606.00930.
- `[koren2026recall]` -> Koren et al., arXiv:2609.07681.

This mapping is an internal drafting aid and should not appear in the final
typeset manuscript.

## Introduction claim boundaries

- No first-ever priority claim.
- No same-numbered-plane semantic identity.
- No universal scaling law or model-size threshold.
- No inference over independently trained model seeds.
- No architecture-independent universality.
- No complete geometric mediation claim.
- No general steering-failure claim.
- No AVeriTeC benchmark-superiority claim.

`INTRODUCTION_MANUSCRIPT_DRAFT_V1_READY = YES`
