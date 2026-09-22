# ContraMamba Gen4 Manuscript Construction Spine

## Status

- Status: STATIC MANUSCRIPT CONSTRUCTION
- Revision base HEAD: `e7466c6`
- Scientific execution: CLOSED
- Claim source:
  `reports/reason_router_gen4_final_paper_claim_disposition.md`
- Novelty source:
  `reports/reason_router_gen4_paper_novelty_positioning_matrix.md`

This artifact fixes the working manuscript structure. It creates no new
scientific claim and authorizes no new experiment.

## Working title

Causal Role Recurrence with Geometric Reorganization and Readout Reversal in Mamba

## Paper thesis

Across the tested Mamba checkpoints, recurrence of a scale-local causal role
does not imply preservation of its geometric realization or downstream
task-readout orientation.

The paper's central empirical structure is:

`CAUSAL ROLE RECURRENCE`
does not imply
`FIXED GEOMETRIC REALIZATION`,
which does not imply
`STABLE DOWNSTREAM READOUT ORIENTATION`.

Cross-block geometric transport accounts for part, but not all, of local
functional change.

The paper does not claim a universal scaling law, a scale threshold, or
model-population generalization across random training seeds.

## Abstract v2

Mechanistic structure that recurs across model checkpoints need not retain
either a fixed geometric realization or stable downstream functional alignment.
We study this separation in pretrained Mamba state-space models at 130M, 370M,
and 1.4B parameters using a frozen causal reconstruction and intervention
program. Across the tested checkpoints, a scale-local dominant causal role
recurs, while the surrounding residual mechanism reorganizes in its signed
effects, coefficient-mass distribution, and functional coupling rather than
preserving a common coordinate identity. At 1.4B, transporting the selected
local subspace across an adjacent block shifts the intervention response toward
the canonical effect but does not recover it, showing that cross-block
geometric reorientation contributes to, but does not fully determine, local
functional change. We then connect these internal mechanisms to downstream task
readout. After accounting for the model's explicit gradient-ownership
semantics, local forward-equivalent readout alignment is positive at 130M and
370M but negative at 1.4B. In a prospectively matched comparison over 300 paired items, Mamba-370M and
Mamba-1.4B exhibit opposite mean readout alignment. Direct behavioral interventions likewise show positive downstream
coupling at 370M while the same positive bridge is not established at 1.4B.
Together, these results show that recurrence of an internal causal role does
not guarantee preservation of either its geometric realization or its
downstream functional alignment across the tested Mamba checkpoints.

## Core contributions

The paper has two headline scientific contributions.

### Core Contribution 1 — Causal role recurrence without fixed geometric realization

Across the tested Mamba checkpoints, a scale-local dominant causal role recurs
under the same frozen reconstruction and matched-control procedure.

The recurring object is the causal role under that procedure, not a fixed
principal-plane rank or coordinate identity.

At the same time, the surrounding residual realization reorganizes in:

- signed local effects;
- coefficient-mass distribution;
- dominant residual rank;
- generator-family coupling.

Cross-block transport at 1.4B further shows that geometric reorientation
causally contributes to local functional change without completely explaining
it.

### Core Contribution 2 — Downstream readout reversal despite causal-role recurrence

Despite recurrence of the internal causal role, downstream local task-readout
alignment is not preserved.

The frozen three-checkpoint pattern is:

- Mamba-130M: positive mean readout alignment;
- Mamba-370M: positive mean readout alignment;
- Mamba-1.4B: negative mean readout alignment.

The principal inferential comparison is not a three-point scaling trend.

It is the prospectively matched Mamba-370M-versus-Mamba-1.4B comparison on the
same 300-item population.

Frozen owned-gradient statistics:

- mean `Delta_L_370M = +0.0005089563518854174`;
- mean `Delta_L_1.4B = -0.0011954475058862238`;
- mean paired difference
  `Delta_L_370M - Delta_L_1.4B = +0.001704403857771641`;
- paired `t(299) = 10.771333954019786`;
- one-sided `p = 2.2238330789610916e-23`.

Under the frozen `G3-GROUP-D-HALF` ownership correction:

- forward-equivalent 370M mean:
  `+0.001017912703770835`;
- forward-equivalent 1.4B mean:
  `-0.002390895011772448`;
- forward-equivalent paired mean difference:
  `+0.003408807715543282`.

The t statistic and p-value are unchanged by the common positive rescaling.

Inference scope:

This paired inference is over the matched item population for two frozen
checkpoints. It is not an inference over independently trained model seeds and
does not establish a monotonic model-size scaling law or a universal
parameter-count threshold.

## Validation and boundary evidence

The following results remain important main-text validation or boundary
evidence, but are not presented as independent headline novelty contributions.

### Causal specificity and robustness

The 130M causal role is supported by prospective specificity, necessity,
restoration, and small-epsilon robustness evidence.

The primary specificity control is not an arbitrary weak comparison.

The frozen response-blind control PP5 was selected as the principal plane with
maximum projector-separation magnitude:

`PP5 = argmax_i sin(theta_i)`.

On a fresh N=300 XG1 holdout:

- mean `D_SPEC = +2.3040673691981465e-08`;
- `t(299) = 7.655867341055923`;
- one-sided `p = 1.3366883647357252e-13`.

Thus the selected PP3 effect exceeds a prospectively frozen geometry-selected
hard control.

This does not establish dominance over every possible random direction or
arbitrary subspace, and the paper should not claim that it does.

At 1.4B, the separately frozen adjacent-site comparison provides an additional
site-specificity check without claiming a global layer optimum.

### Behavioral coupling

Direct behavioral interventions establish positive downstream coupling at
370M, while the same positive behavioral bridge is not established at 1.4B.

Use the frozen formulation:

`SCALE_SPECIFIC_BEHAVIORAL_BRIDGE_ONLY`.

Do not describe the three checkpoints as a monotonic behavioral scaling curve.

### External transfer and control boundary

The causal intervention transfers to natural-language AVeriTeC gold-evidence
inputs at 130M and 370M under the frozen narrow three-class setting.

The preregistered fixed-mirror steering intervention does not establish
practical utility.

These results should remain visible in the main paper through a compact table
or concise Results subsection rather than being hidden entirely in the
appendix.

They support a boundary statement:

`MECHANISTIC VALIDITY DOES NOT BY ITSELF GUARANTEE CONTROL UTILITY`.

This is a synthesis/boundary result, not a first-ever conceptual novelty claim.

## Results-first structure

### 3.1 Identifying a causal recurrent-state role

Introduce the 130M causal object.

Establish that the effect is not merely a consequence of choosing a
high-separation plane by presenting the prospective PP3-versus-max-separation
PP5 specificity result.

Then summarize necessity, restoration, and small-epsilon robustness.

Do not claim specificity against every arbitrary direction.

### 3.2 Causal roles recur while residual geometry reorganizes

Present the 370M and 1.4B scale extension.

Use the bounded formulation:

`CORE-STABLE / RESIDUAL-PLASTIC`.

Explicitly state that the principal-plane rank is scale-local:

- 370M dominant rank: P3;
- 1.4B dominant rank: P5.

The invariant is the causal role under the frozen procedure, not plane number.

Show that the residual mechanism reorganizes across the tested checkpoints in
signed profile, coefficient-mass distribution, dominant residual rank, and
generator-family coupling.

Do not perform or imply a cross-scale residual significance test that was not
frozen.

### 3.3 Cross-block transport only partially accounts for local functional change

Present the 1.4B prospectively fixed adjacent-site specificity result followed
by the transported-subspace intervention.

Use:

`geometric reorientation contributes causally to local functional change`

but not:

`geometric reorientation fully explains the change`.

The unexplained residual component is an explicit limitation and a target for
future work.

Do not report an "explained percentage" from ratios of mean effects.

### 3.4 Downstream readout reverses in the matched 370M-versus-1.4B comparison

Introduce the ownership distinction once:

Historical stored quantity:

`Delta_L_owned`.

Numerical-forward directional derivative for the frozen D-half arm:

`Delta_L_forward = 2 * Delta_L_owned`.

Then report the matched primary result prominently:

- N = 300 matched items;
- 370M mean owned readout:
  `+0.0005089563518854174`;
- 1.4B mean owned readout:
  `-0.0011954475058862238`;
- paired mean difference:
  `+0.001704403857771641`;
- paired `t(299) = 10.771333954019786`;
- one-sided `p = 2.2238330789610916e-23`.

Also report the corresponding forward-equivalent means:

- 370M:
  `+0.001017912703770835`;
- 1.4B:
  `-0.002390895011772448`.

Use 130M as a separately frozen contextual scale point, not as part of a
three-point trend test.

Explicit limitation:

The statistical unit is the matched item population for fixed checkpoints.
The result does not quantify variability across independently trained model
seeds.

### 3.5 Behavioral consequences are checkpoint-dependent

Present the direct behavioral bridge and stagewise localization.

The correct claim is:

internal causal relevance reaches the downstream decision margin at 370M,
while the same positive bridge is not established at 1.4B.

Do not claim that larger models weaken or reverse behavioral relevance in
general.

### 3.6 External transfer and control boundary

Keep this section compact but in the main text.

Present:

- narrow AVeriTeC gold-evidence transfer at 130M and 370M;
- 1.4B transfer not established;
- fixed-mirror steering utility not established.

A compact table should be present in the main text even if detailed numbers and
diagnostics move to the appendix.

## Main figure and table spine

Figure 1:
causal reconstruction/intervention schematic plus 130M causal object.

Figure 2:
prospective PP3-vs-hard-control PP5 specificity, necessity, restoration, and
small-epsilon robustness.

Figure 3:
cross-checkpoint CORE-STABLE / RESIDUAL-PLASTIC synthesis.

Figure 4:
1.4B adjacent-site specificity and cross-block transported response.

Figure 5:
three-checkpoint readout context with the matched 370M-versus-1.4B reversal as
the inferential centerpiece, plus direct behavioral coupling.

Main Table 1:
compact external-transfer / control-boundary table containing AVeriTeC and
fixed-mirror steering outcomes.

Detailed factor-2 diagnostics, residual decompositions, precursor nulls,
provenance, and secondary robustness material move to the appendix.

## Introduction positioning

The first page must acknowledge the closest prior families before stating the
gap.

Required conceptual sequence:

1. Prior work has causally localized and manipulated internal mechanisms in
   Mamba and other state-space models.

2. Other work has examined mechanistic similarity across architectures and has
   shown that representational signatures need not uniquely identify causal
   function.

3. These results do not answer whether a causal role that recurs across model
   checkpoints must retain either its geometric realization or its downstream
   functional alignment.

4. The present work tests exactly this relation.

5. Main answer:
   it does not, within the tested checkpoints and frozen intervention program.

Do not defer this distinction entirely to Related Work.

## Broader implication

The Discussion may state the following bounded implication:

A causal intervention direction should not be assumed to transfer reliably
across model scales merely because an apparently analogous internal causal role
recurs.

This motivates explicit revalidation of model editing or steering directions
after scaling or checkpoint changes.

This is an implication of the frozen evidence, not a demonstrated general
failure of model editing or steering.

Do not frame the result as a broad safety guarantee or broad safety failure.

## Explicit paper boundaries

Do not claim:

- first mechanistic interpretation of Mamba;
- first causal intervention or causal subspace in Mamba;
- generic representation-function dissociation as the standalone novelty;
- same-coordinate identity across checkpoints;
- a universal scaling law;
- a model-size phase transition;
- a zero-crossing threshold;
- generalization over independently trained model seeds;
- complete mediation by geometric transport;
- dominance over every random direction or arbitrary subspace;
- useful general steering;
- general inability to steer Mamba;
- AVeriTeC benchmark superiority;
- a hallucination precursor;
- architecture-independent universality;
- factor-2 nonlinear amplification or backend defect.

## Writing order

1. Results Sections 3.1 and 3.2.
2. Results Sections 3.3 and 3.4.
3. Results Sections 3.5 and 3.6.
4. Figures and Main Table 1.
5. Methods needed to support the Results.
6. Discussion and limitations.
7. Related Work.
8. Introduction.
9. Final Abstract revision.

`MANUSCRIPT_CONSTRUCTION_SPINE_READY = YES`
