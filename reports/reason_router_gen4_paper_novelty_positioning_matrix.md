# ContraMamba Gen4 Paper Novelty and Positioning Matrix

## 0. Status

- Status: STATIC LITERATURE POSITIONING / PAPER NOVELTY FREEZE CANDIDATE
- Evidence HEAD: `684d69eb777e0ff868bf7b41d3e55ca23bd805ca`
- Branch: `gen4-mamba370m-core-replication`
- Literature audit date: 2026-09-22
- New training: NO
- New model forward/backward: NO
- New scientific execution: NO
- New p-values: 0
- Scientific claim set source:
  `reports/reason_router_gen4_final_paper_claim_disposition.md`

This report positions the already-frozen Gen4 paper claims against the closest
identified literature. It does not create new scientific evidence and does not
authorize additional experiments.

Novelty labels:

- `NOVEL_CANDIDATE`: no directly matching claim was identified in the targeted
  closest-prior audit; this is not a first-ever priority claim.
- `EXTENDS`: closest prior work establishes a related phenomenon, while the
  frozen ContraMamba result adds a materially different axis or causal relation.
- `CONFIRMS`: the result is useful evidence but the conceptual point is already
  represented in prior work.
- `NOT_NOVEL`: do not position as a contribution.
- `DROP`: excluded by the frozen paper claim disposition.

## 1. Closest prior-work families

The closest literature families identified for paper positioning are:

1. Mamba factual causal tracing and factual-association editing.
   These works establish that internal Mamba computations can be causally
   localized and intervened on.

2. Mamba information-flow / knockout analyses.
   These establish token- and layer-localized factual information flow in Mamba.

3. Transformer-Mamba mechanistic universality and representation alignment.
   These investigate shared features or circuits across architectures.

4. SSM activation-subspace interpretation and steering.
   These establish that behaviorally relevant activation subspaces can be found
   and manipulated in state-space models.

5. Mamba causal-localization non-uniqueness.
   Recent work shows that representational signatures need not uniquely
   determine causal function and that causal conclusions can depend on the
   intervention surface.

6. Mamba scaling analyses.
   These investigate algorithmic/capacity changes with model or state scale,
   especially in associative-recall settings.

The current paper must not claim priority for mechanistic interpretation of
Mamba, causal intervention in Mamba, causal subspaces in SSMs, representation-
function dissociation in general, or mechanistic comparison across model scale.

## 2. Claim-by-claim novelty matrix

| ID | Frozen claim | Positioning | Paper action |
|---|---|---|---|
| C1 | A scale-local dominant causal role recurs across tested native-Mamba scales while the surrounding residual realization reorganizes. | `EXTENDS` / CORE NOVELTY AXIS | Closest universality work asks whether features/circuits recur. ContraMamba distinguishes recurrence of a causally validated role from preservation of its residual/geometric realization. Do not claim identical coordinates or plane identities across scale. |
| C2 | 130M and 370M readout means are positive while 1.4B is negative. | `NOVEL_CANDIDATE` | Use as evidence that recurrent internal causal structure need not preserve downstream task-readout orientation with scale. Do not claim a universal scaling law or phase transition. |
| C3 | Mamba-130M has positive local readout alignment. | `EXTENDS` | Supporting scale point. Do not sell the existence of a local Mamba readout by itself as novel. |
| C4 | Matched 370M vs 1.4B readout alignment reverses sign. | `NOVEL_CANDIDATE` / CORE NOVELTY AXIS | Central cross-scale result. Position as downstream orientation reversal despite recurrence of an internal causal role. |
| C5 | Behavioral relevance recurs uniformly with scale. | `DROP` | Not supported by frozen evidence. |
| C6 | Internal causal relevance reaches the downstream decision margin at 370M, while the same positive bridge is not established at 1.4B. | `EXTENDS` / CORE NOVELTY AXIS | Extends representation-function and causal-localization work by showing scale-dependent downstream behavioral coupling under the same research program. |
| C7 | The frozen causal intervention transfers to AVeriTeC gold-evidence inputs at 130M and 370M. | `EXTENDS` | External-validity evidence. Keep narrow gold-evidence / three-class scope. Not a benchmark-superiority claim. |
| C8 | Fixed-mirror intervention provides useful steering. | `DROP` | Steering utility was not established. Prior SSM steering work also means steering itself is not a novelty claim. |
| C9 | Explanatory causal structure and useful control are distinct empirical questions. | `CONFIRMS` | Use as synthesis/boundary statement, not a priority claim. |
| C10 | The canonical 1.4B site is stronger than one prospectively fixed adjacent +1 site. | `CONFIRMS` / `EXTENDS` | Mechanism-specific specificity validation. Not a global layer-optimum claim. |
| C11 | Factor-2 is unexplained nonlinear amplification. | `DROP` | Root cause is frozen D-half gradient ownership. |
| C12 | Factor-2 is a fused CUDA backward defect. | `DROP` | Independent slow/reference backend rules this out. |
| C13 | Locally, `D_BEH(alpha) ~= alpha * Delta_L_forward`. | `NOT_NOVEL` | Correct first-order interpretation and consistency result, not scientific novelty. |
| C14 | Same-numbered principal planes are semantically identical across scale. | `DROP` | Unsupported. Plane ranks remain scale-local. |
| C15 | A universal monotonic scaling law exists in readout magnitude or usefulness. | `DROP` | Unsupported and unnecessary. |
| C16 | ContraMamba establishes architecture-independent causal universality. | `DROP` | Outside evidence scope and not novel as a research question. |

## 3. Primary novelty spine

The paper should not be positioned as:

`Mamba contains interpretable causal subspaces.`

That territory is already occupied by prior mechanistic and SSM-subspace work.

The defensible paper-level novelty is the conjunction:

`causal role recurrence`
does not imply
`fixed geometric realization`,
which does not imply
`stable downstream readout orientation`.

In compact form:

`STABLE CAUSAL ROLE / PLASTIC GEOMETRY / SCALE-DEPENDENT READOUT`

The empirical components are:

1. A scale-local dominant causal role recurs across the tested Mamba scales.

2. Its residual/geometric realization reorganizes rather than preserving a
   fixed coordinate identity.

3. Cross-block transport causally accounts for part, but not all, of local
   functional change.

4. Downstream local readout alignment is positive at smaller tested scales but
   negative at Mamba-1.4B under the matched comparison.

5. Direct behavioral relevance is therefore scale-dependent rather than a
   guaranteed consequence of internal causal recurrence.

## 4. Closest-prior boundary

A particularly close conceptual prior is recent work showing that a
representational signature need not uniquely identify a causal locus in Mamba
and that causal sets may be intervention-surface dependent.

Therefore the present paper must not use

`representation != function`

as its standalone novelty statement.

The additional contribution is the cross-scale structure:

- a causal role can recur;
- its geometric realization can reorganize;
- downstream readout orientation can reverse;
- cross-block geometric transport only partially accounts for local functional
  change.

This is a more specific claim than generic representation-function
dissociation.

## 5. Claims that must not appear as novelty claims

Do not claim:

1. first mechanistic interpretation of Mamba;
2. first causal intervention in Mamba;
3. first causal subspace in Mamba or SSMs;
4. first demonstration that representation and function differ;
5. universal Mamba mechanism across scale;
6. universal semantic identity of same-numbered planes;
7. universal monotonic scaling of causal effect;
8. general inability to steer Mamba;
9. architecture-independent universality;
10. first-ever priority for any current result without a separate exhaustive
    literature audit.

## 6. Recommended contribution structure

The manuscript should use three principal contribution bullets.

### Contribution 1 — Causal role versus geometry

Across the tested Mamba scales, a scale-local dominant causal role recurs under
the frozen reconstruction procedure, while its residual and geometric
realization reorganizes.

### Contribution 2 — Geometry versus downstream function

Despite recurrence of the internal causal role, local downstream task-readout
alignment changes with scale and reverses sign in the matched 370M-versus-1.4B
comparison. Cross-block transport explains part, but not all, of the associated
local functional change.

### Contribution 3 — Mechanism versus consequence

The identified structure has measurable downstream behavioral and external
fact-verification consequences, but causal validity does not by itself imply
uniform behavioral alignment or useful fixed-intervention control.

Contribution 3 is a synthesis contribution, not a first-ever conceptual
priority claim.

## 7. Title positioning

The previous title candidate

`Causal Roles Without Fixed Coordinates in Mamba`

remains defensible but emphasizes only the first half of the final story.

A stronger working title is:

`Stable Causal Roles, Plastic Geometry, and Scale-Dependent Readout in Mamba`

This title directly exposes the three frozen scientific axes and avoids
implying priority for generic representation-function dissociation.

Title and abstract remain manuscript-level wording choices and may be revised
without reopening scientific execution.

## 8. Paper stop rule

This literature-positioning step does not reopen experiments.

`NEW_SCIENTIFIC_EXECUTION = CLOSED`

Permitted next work:

- manuscript title selection;
- contribution wording;
- abstract drafting;
- Results-first manuscript outline;
- figures/tables from frozen evidence;
- Related Work and citations;
- reproducibility packaging.

`PAPER_NOVELTY_POSITIONING_READY_FOR_FREEZE = YES`
