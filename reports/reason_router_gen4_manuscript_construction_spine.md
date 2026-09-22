# ContraMamba Gen4 Manuscript Construction Spine

## Status

- Status: STATIC MANUSCRIPT CONSTRUCTION
- Evidence HEAD: `4c2ff22f5fedd66c8d33ce60492b6c1d65566603`
- Scientific execution: CLOSED
- Claim source:
  `reports/reason_router_gen4_final_paper_claim_disposition.md`
- Novelty source:
  `reports/reason_router_gen4_paper_novelty_positioning_matrix.md`

This artifact fixes the working manuscript structure. It creates no new
scientific claim and authorizes no new experiment.

## Working title

Stable Causal Roles, Plastic Geometry, and Scale-Dependent Readout in Mamba

## Paper thesis

Across the tested Mamba scales, recurrence of a scale-local causal role does not
imply preservation of its geometric realization or downstream task-readout
orientation. Cross-block geometric transport accounts for part, but not all, of
local functional change, and causal validity does not by itself guarantee
uniform behavioral alignment or useful fixed-intervention control.

## Abstract v1

Mechanistic structure that recurs across model scales need not occupy stable
geometric coordinates or induce stable downstream behavior. We study this
separation in pretrained Mamba state-space models at 130M, 370M, and 1.4B
parameters using a fixed causal reconstruction and intervention program. Across
the tested scales, a scale-local dominant causal role recurs, while its
surrounding residual geometry reorganizes rather than preserving a common
coordinate identity. At 1.4B, transporting the selected local subspace across an
adjacent block partially shifts the intervention response toward the canonical
effect but does not recover it, indicating that cross-block geometric
reorientation contributes to, but does not fully determine, local functional
change. We next connect these internal mechanisms to downstream task readout.
After accounting for the model's explicit gradient-ownership semantics,
forward-equivalent local readout alignment is positive at 130M and 370M but
negative at 1.4B, with a supported matched sign reversal between 370M and 1.4B.
Direct behavioral interventions likewise show positive downstream coupling at
370M while the same positive bridge is not established at 1.4B. The causal
intervention also transfers to natural-language gold-evidence fact-verification
inputs at 130M and 370M, whereas a fixed mirror-steering intervention does not
establish practical utility. Together, these results separate causal role
recurrence, geometric realization, downstream readout, and controllability as
distinct properties of Mamba representations.

## Contribution structure

### Contribution 1 — Causal role versus geometry

A scale-local dominant causal role recurs across the tested Mamba scales under
the frozen reconstruction procedure, while its residual and geometric
realization reorganizes.

### Contribution 2 — Geometry versus downstream function

Despite recurrence of the internal causal role, downstream local task-readout
alignment changes with scale and reverses sign in the matched
370M-versus-1.4B comparison. Cross-block transport accounts for part, but not
all, of local functional change.

### Contribution 3 — Mechanism versus consequence

The identified causal structure has measurable downstream behavioral and
external fact-verification consequences, but causal validity does not imply
uniform behavioral alignment or useful fixed-intervention control.

## Results-first structure

### 3.1 Identifying a causal recurrent-state role

Establish the 130M causal object and validate it with the frozen
specificity/necessity/restoration evidence.

### 3.2 Causal roles recur while geometry reorganizes across scale

Present the 130M, 370M, and 1.4B cross-scale synthesis using the bounded
CORE-STABLE / RESIDUAL-PLASTIC formulation.

Do not equate same-numbered planes across model scales.

### 3.3 Cross-block transport only partially explains functional change

Present the 1.4B adjacent-site specificity result followed by the transported
response experiment.

State only that reorientation contributes causally to the local functional
change; do not claim complete explanation or mediation.

### 3.4 Downstream readout reverses across scale

Present the three-scale readout using the corrected ownership semantics.

Historical stored quantity:
`Delta_L_owned`

Forward-equivalent numerical derivative for the frozen D-half arm:
`Delta_L_forward = 2 * Delta_L_owned`

Use the matched 370M-versus-1.4B sign reversal as the principal cross-scale
readout result.

### 3.5 Behavioral relevance is scale-dependent

Present the direct behavioral bridge and stagewise localization.

Use:
`SCALE_SPECIFIC_BEHAVIORAL_BRIDGE_ONLY`

Do not claim uniformly increasing usefulness with scale.

### 3.6 Causal validity does not guarantee control utility

Briefly present the narrow AVeriTeC gold-evidence transfer at 130M and 370M and
the failed fixed-mirror steering utility test.

Do not claim benchmark superiority, retrieval performance, or general
unsteerability.

## Main figure spine

Figure 1:
method schematic plus 130M causal recurrent-state geometry.

Figure 2:
specificity, necessity, restoration, and small-epsilon robustness.

Figure 3:
cross-scale CORE-STABLE / RESIDUAL-PLASTIC geometry.

Figure 4:
1.4B adjacent-site specificity and cross-block transported response.

Figure 5:
three-scale corrected readout plus matched behavioral coupling.

AVeriTeC, steering nulls, detailed factor-2 diagnostics, residual decomposition,
precursor nulls, and provenance material should move to a compact main table or
appendix unless main-text space remains.

## Explicit paper boundaries

Do not claim:

- first mechanistic interpretation of Mamba;
- first causal intervention or causal subspace in Mamba;
- generic representation-function dissociation as the standalone novelty;
- same-coordinate identity across scales;
- a universal scaling law or phase transition;
- complete mediation by geometric transport;
- useful general steering;
- AVeriTeC benchmark superiority;
- a hallucination precursor;
- architecture-independent universality;
- factor-2 nonlinear amplification or backend defect.

## Writing order

1. Results.
2. Methods needed to support Results.
3. Figures and tables.
4. Discussion / limitations.
5. Related Work.
6. Introduction.
7. Abstract final revision.

`MANUSCRIPT_CONSTRUCTION_SPINE_READY = YES`
