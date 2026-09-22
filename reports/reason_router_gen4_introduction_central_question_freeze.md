# ContraMamba Gen4 Introduction Central-Question Freeze

## Status

- Status: STATIC MANUSCRIPT POSITIONING
- Evidence HEAD: `de81f26fa514d49eabeba4ad567f2e8274254554`
- Scientific execution: CLOSED
- New scientific claims: NONE
- New statistical tests: NONE
- New p-values: NONE

This artifact fixes the logical framing for the first page of the manuscript.
It does not authorize additional experiments and does not expand the frozen
scientific claim set.

## 1. Central question

The paper is not organized around the question:

`Can Mamba contain an interpretable causal subspace?`

Prior work already establishes related forms of causal localization,
mechanistic interpretation, activation-subspace intervention, and
representation/function dissociation.

The paper instead asks:

> If an internal causal role recurs across model checkpoints, what else must be
> preserved?

We separate two candidate implications.

### Implication I — geometric invariance

A recurring causal role might be expected to retain a stable geometric
realization.

The frozen evidence tests:

`CAUSAL ROLE RECURRENCE`
`=>?`
`FIXED GEOMETRIC REALIZATION`

Result:

`NOT SUPPORTED`.

The dominant causal role recurs under the frozen reconstruction procedure, but
the scale-local plane identity and surrounding residual realization reorganize.

### Implication II — downstream functional invariance

Even if internal geometry reorganizes, a recurring causal role might still be
expected to retain the same downstream task alignment.

The frozen evidence tests:

`CAUSAL ROLE RECURRENCE`
`=>?`
`STABLE DOWNSTREAM READOUT ORIENTATION`

Result:

`NOT SUPPORTED`.

In the prospectively matched Mamba-370M-versus-Mamba-1.4B comparison, the
scale-local directional readout has opposite mean alignment.

## 2. Paper-level logical structure

The paper should therefore be read as testing the following implication chain:

`MECHANISTIC RECURRENCE`
does not guarantee
`GEOMETRIC INVARIANCE`,
and does not guarantee
`DOWNSTREAM FUNCTIONAL INVARIANCE`.

This is the manuscript's primary conceptual contribution.

The novelty is not any single ingredient in isolation.

It is the empirical separation of:

1. recurrence of a causally validated internal role;
2. preservation of its geometric realization;
3. preservation of its downstream task alignment.

These properties are measured within one frozen causal program rather than
inferred by combining unrelated studies.

## 3. First-page Introduction paragraph draft

Mechanistic analyses often ask whether an internal computation discovered in
one model recurs in another. Recurrence, however, leaves a more basic question
unresolved: what exactly is preserved when a causal role reappears? A recurring
role could retain its geometric realization, its downstream functional
alignment, both, or neither. Existing work on Mamba and state-space models has
established that internal mechanisms can be causally localized and manipulated,
while broader mechanistic studies have examined representational similarity and
the non-uniqueness of causal localization. These results do not determine
whether recurrence of a causally validated role across model checkpoints
implies geometric or downstream functional invariance. We test these
implications directly in pretrained Mamba models at 130M, 370M, and 1.4B
parameters. Across the tested checkpoints, a scale-local dominant causal role
recurs, but its surrounding geometric realization reorganizes; moreover, in a
prospectively matched 370M-versus-1.4B comparison, downstream readout alignment
reverses sign. Thus mechanistic recurrence alone is insufficient to infer either
fixed coordinates or stable downstream functional alignment.

## 4. One-sentence gap statement

> Prior work asks whether mechanisms recur; we ask what recurrence actually
> preserves.

This sentence may be used as an internal writing guide.

It should not be presented as a literal claim that no prior paper has ever
asked a related question unless a separate exhaustive literature review
supports that priority claim.

## 5. One-sentence answer

> Across the tested Mamba checkpoints, recurrence of a scale-local causal role
> preserves neither a fixed geometric realization nor a guaranteed downstream
> readout orientation.

The scope phrase `across the tested Mamba checkpoints` is mandatory.

## 6. How each Results section answers the central question

### Section 3.1

Establishes that the starting object is a prospectively validated local causal
role rather than a purely correlational geometric direction.

### Section 3.2

Tests Implication I at the cross-checkpoint level.

Answer:

`CORE-STABLE / RESIDUAL-PLASTIC`.

The role recurs, while rank identity and residual realization reorganize.

### Section 3.3

Provides causal support that geometric reorientation is functionally relevant.

Cross-block transport moves the response toward the canonical effect but does
not fully recover it.

This section does not claim complete mediation.

### Section 3.4

Tests Implication II.

The matched 370M-versus-1.4B readout comparison shows opposite mean alignment,
with the frozen paired inference providing the principal statistical endpoint.

### Sections 3.5–3.6

Establish downstream consequences and boundaries.

They are validation/boundary evidence, not independent headline novelty claims.

## 7. Closest-prior positioning rule

The Introduction must explicitly concede that prior literature already covers
important ingredients, including:

- causal localization and intervention in Mamba;
- mechanistic comparison across models or architectures;
- activation-subspace interpretation and steering;
- representation/function or localization non-uniqueness;
- scaling-related analysis of Mamba mechanisms.

Do not argue novelty by claiming these ingredients are absent.

Instead state the narrower gap:

> The unresolved issue is whether recurrence of a causally validated role
> entails preservation of its geometry or downstream task alignment.

The manuscript's evidence addresses that relation.

## 8. Novelty-defense paragraph logic

If a reviewer interprets the work as a collection of known ingredients, the
paper-facing response is:

1. causal localization is not claimed as novel;
2. geometric non-identity alone is not claimed as novel;
3. downstream intervention or steering alone is not claimed as novel;
4. the paper prospectively separates three properties that are often conflated:
   causal-role recurrence, geometric realization, and downstream alignment;
5. the matched readout reversal supplies a direct counterexample to the
   implication that internal mechanistic recurrence guarantees stable
   downstream functional orientation.

Do not claim that this establishes a universal theorem.

## 9. Impact implication

The bounded practical implication is:

> Intervention directions or mechanistic correspondences should be revalidated
> after checkpoint or scale changes rather than transferred solely because an
> analogous internal causal role appears to recur.

This is a motivation for revalidation, not evidence that model editing or
steering generally fails.

Do not frame the result as a general safety guarantee or a general safety
failure.

## 10. Required scope boundaries

The Introduction must not imply:

- a universal scaling law;
- a parameter-count phase transition;
- a universal zero-crossing threshold;
- semantic identity of same-numbered planes;
- generalization over independently trained model seeds;
- architecture-independent universality;
- first mechanistic interpretation of Mamba;
- first causal intervention in Mamba;
- first representation/function dissociation;
- complete explanation of geometric reorganization;
- general steering failure.

## 11. Manuscript writing rule

Every headline Results paragraph should be traceable to one of two questions:

1. Does causal-role recurrence preserve geometric realization?
2. Does causal-role recurrence preserve downstream functional alignment?

Material that answers neither question should be treated as validation,
boundary evidence, method, or appendix material rather than elevated to an
additional headline contribution.

`INTRODUCTION_CENTRAL_QUESTION_FROZEN = YES`
