# ContraMamba Gen4 Abstract Manuscript Draft v1

## Status

- Status: STATIC FINAL-ABSTRACT DRAFT
- Evidence HEAD: `f5c5c636bbbd427d7ccb9869f83f827e6da8f0cf`
- Scientific execution: CLOSED
- New scientific claims: NONE
- New statistical tests: NONE
- New p-values: NONE
- Priority claim: NONE

## Abstract

Mechanistic analyses often ask whether an internal causal computation recurs
across model checkpoints, but recurrence does not specify what else is
preserved. We separate three properties—causal-role recurrence, geometric
realization, and downstream functional alignment—in frozen pretrained Mamba
checkpoints at 130M, 370M, and 1.4B parameters using a common causal
reconstruction and intervention program. A scale-local dominant causal role
recurs across the tested checkpoints, while its local realization reorganizes:
the selected principal-plane rank and surrounding residual structure are not
preserved as fixed coordinates. At Mamba-1.4B, transporting the canonical
subspace across an adjacent block shifts the intervention response toward the
canonical effect but does not restore its positive response, showing that
geometric reorientation contributes causally to local functional change without
fully accounting for it. Downstream functional alignment also changes. In a
prospectively matched comparison over 300 source pairs, the independently
frozen selected-versus-control displacement has positive mean task-readout
alignment at Mamba-370M and negative mean alignment at Mamba-1.4B, supporting
an orientation reversal despite recurrence of the internal causal role. Direct
behavioral interventions likewise establish a positive downstream bridge at
370M while the same pre-specified positive bridge is not established at 1.4B.
Together, these results show that, across the tested Mamba checkpoints,
recurrence of a causal role does not guarantee either a fixed geometric
realization or stable downstream functional alignment. Mechanistic
correspondences and intervention directions should therefore be revalidated
after checkpoint or scale changes rather than transferred from role recurrence
alone.

## Scope checks

- Two headline results only: geometry and downstream readout.
- 370M-versus-1.4B inference explicitly identified as matched item-level comparison.
- 130M is contextual rather than part of a three-point scaling inference.
- Cross-block transport described as causal contribution, not complete mediation.
- No universal scaling law, phase transition, architecture-wide universality, or seed-population claim.
- Behavioral bridge retained as supporting consequence rather than a third novelty contribution.

Abstract word count: 234

`ABSTRACT_MANUSCRIPT_DRAFT_V1_READY = YES`
