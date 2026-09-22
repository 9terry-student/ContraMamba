# ContraMamba Gen4 Related Work Manuscript Draft v1

## Status

- Status: STATIC MANUSCRIPT RELATED-WORK DRAFT
- Evidence HEAD: `e1b3cf48db498f9f0162e266a10257a171ba6f18`
- Literature audit date: 2026-09-22
- Scientific execution: CLOSED
- New scientific claims: NONE
- Priority claim: NONE

---

## Related Work

### Mechanistic analysis of Mamba

Prior work has already established that Mamba admits detailed mechanistic
analysis and causal intervention. Sharma et al. use causal tracing,
interchange-style interventions, information-flow analysis, and model editing
to localize factual associations in Mamba, showing that factual recall can be
assigned to specific token and layer locations despite the architecture's
differences from Transformers [sharma2024locating]. Endy et al. subsequently
adapt attention-knockout-style analysis to Mamba-1 and Mamba-2 to trace factual
information flow across tokens and layers [endy2025knockout]. Complementary
work on MambaLRP develops architecture-specific relevance propagation for
selective state-space models [jafari2024mambalrp].

These studies establish that Mamba can be localized, intervened on, and
explained. Our question begins after that point. We do not ask whether Mamba
contains an interpretable or causal internal structure, but what is preserved
when an analogous causal role recurs across pretrained checkpoints.

### Mechanistic similarity across models and architectures

Mechanistic universality work asks whether different models implement similar
internal computations. Wang et al. compare Transformers and Mambas using
interpretable features and circuit-level analysis, reporting substantial
feature similarity and structurally analogous induction circuits alongside
architecture-specific differences [wang2025universality]. Arora et al. use
causal interventions to compare retrieval mechanisms across Transformers and
state-space models, emphasizing that similar task behavior can conceal
different internal algorithms [arora2025mechanistic].

Our study is complementary to this line of work. Rather than deciding whether
two architectures or checkpoints instantiate a similar mechanism in the
aggregate, we condition on recurrence of a causally validated scale-local role
and separate two stronger implications: whether that recurrence preserves its
geometric realization and whether it preserves downstream task alignment.

### Activation subspaces, steering, and causal non-uniqueness

Recent work has also made activation subspaces explicit intervention objects in
state-space models. Mohan et al. identify activation subspace bottlenecks in
Mamba-family SSMs and manipulate them with test-time steering interventions
[mohan2026subspace]. Our use of low-dimensional causal geometry therefore is
not intended as a novelty claim by itself.

A particularly close conceptual boundary is provided by Jiang and Zhang, who
show that causal localization of a Mamba-2 state-sink phenomenon is non-unique:
different unit sets can support similar causal effects, representational
similarity need not track causal function, and conclusions depend on the
intervention surface [jiang2026circuit]. This makes generic
`representation != function` an insufficient novelty statement for our work.

The distinction tested here is cross-checkpoint and relational. We ask whether
recurrence of a causal role entails preservation of its geometry or downstream
functional orientation. The frozen experiments independently reconstruct the
geometry at each checkpoint, causally test cross-block transport, and then
evaluate a prospectively matched downstream readout endpoint.

### Scaling and checkpoint variation in Mamba

Mamba mechanisms have also been studied as a function of architecture and
capacity. Koren et al. derive and empirically validate mechanistic scaling laws
for associative recall, relating Mamba recall to an internal hashing algorithm
and to model dimensions [koren2026recall]. Such work establishes that scaling
can be studied mechanistically rather than only through aggregate performance.

Our evidence does not define a model-size scaling law. We analyze three fixed
pretrained checkpoints, and the principal cross-checkpoint inference is the
matched 370M-versus-1.4B item comparison. The 130M result is a separately
frozen contextual point. We therefore do not estimate a monotonic trend, a
parameter-count threshold, or a model-population effect across random seeds.

### Position of the present work

Taken together, prior work establishes causal localization in Mamba,
information-flow analysis, cross-architecture mechanistic similarity,
activation-subspace steering, causal-localization non-uniqueness, and
mechanistic scaling. These ingredients leave a narrower question unresolved:
what must remain invariant when a causally validated role itself recurs?

Across the tested Mamba checkpoints, our frozen program separates causal-role
recurrence from both geometric invariance and downstream functional
invariance. The contribution is therefore not a new interpretability primitive
but an empirical separation of three properties within one causal program:

`CAUSAL ROLE RECURRENCE`

does not guarantee

`FIXED GEOMETRIC REALIZATION`

and does not guarantee

`STABLE DOWNSTREAM READOUT ORIENTATION`.

This positioning is intentionally bounded to the tested Mamba checkpoints and
does not assert a first-ever priority claim or an architecture-independent
principle.

---

## Internal citation-key mapping

- `[sharma2024locating]` -> Sharma, Atkinson, Bau, COLM 2024.
- `[jafari2024mambalrp]` -> Jafari et al., NeurIPS 2024.
- `[wang2025universality]` -> Wang et al., ICLR 2025.
- `[endy2025knockout]` -> Endy et al., ACL 2025.
- `[arora2025mechanistic]` -> Arora et al., arXiv:2505.15105.
- `[mohan2026subspace]` -> Mohan et al., ICML 2026.
- `[jiang2026circuit]` -> Jiang and Zhang, arXiv:2606.00930.
- `[koren2026recall]` -> Koren et al., arXiv:2609.07681.

This mapping is a drafting aid and will be replaced by manuscript bibliography
commands in the final typeset paper.

`RELATED_WORK_MANUSCRIPT_DRAFT_V1_READY = YES`
